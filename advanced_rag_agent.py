import os
from dotenv import load_dotenv
from typing import List, Dict, Any, TypedDict
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langgraph.graph import StateGraph, END
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from graph_RAG.graph_manager import get_ingredients_for_product, find_products_containing_ingredient, get_product_links
from models import Source

# --- Environment and API Key Setup ---
load_dotenv()

# --- LLM & Embeddings ---
llm = ChatOpenAI(model="gpt-4o", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# --- Vector Store ---
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# --- Structured Output Models (Pydantic V2) ---
class RouterDecision(BaseModel):
    is_general_question: bool = Field(description="True if the question is a general greeting or off-topic chit-chat.")
    needs_graph_search: bool = Field(description="Whether graph search is needed. False if is_general_question is True.")
    reasoning: str = Field(description="Why this decision was made")

class GraphResult(BaseModel):
    found_entities: List[str] = Field(description="Entities found in the graph")
    relationships: List[str] = Field(description="Relationships discovered")
    links: List[str] = Field(description="Product links from the graph")

class FinalAnswer(BaseModel):
    conversational_answer: str = Field(description="The conversational response to the user")
    sources: List[Source] = Field(description="List of source documents with metadata, only if the answer is about product recommendations")

# --- Agent State ---
class AgentState(TypedDict):
    question: str
    chat_history: List[BaseModel]
    router_decision: Dict[str, Any]
    graph_results: Dict[str, Any]
    vector_documents: str
    final_answer: Dict[str, Any]

# --- PROMPTS ---

# 1. Router Prompt
router_prompt_template = """
You are an expert at routing user questions. Your task is to analyze the user's question and classify it.

1.  **General Question**: If the question is a general greeting (e.g., "hello", "how are you?"), a question about you ("who are you?"), or simple off-topic chit-chat, classify it as a general question.
2.  **Tool-requiring Question**: If the question is about DuoLife products, ingredients, business model, or health, it requires tools.

For tool-requiring questions, you must also determine if it needs a specific graph search for:
- The ingredients of a specific product.
- A list of products that contain a specific ingredient.
- Links to specific products.

All other questions about DuoLife will use a vector search.

Question: {question}

{format_instructions}
"""
router_parser = PydanticOutputParser(pydantic_object=RouterDecision)
router_prompt = ChatPromptTemplate.from_template(router_prompt_template).partial(
    format_instructions=router_parser.get_format_instructions()
)

# 2. General Question Prompt
general_question_prompt_template = """
You are DuoBot, a helpful and friendly assistant for the DuoLife company.
The user has asked a general question or a question that is not related to supplements, health, DuoLife company, or product recommendations.

For greetings like "hello" or "who are you?", provide a brief, friendly response introducing yourself.
For off-topic questions, politely redirect them to DuoLife-related topics.

Question: {question}

Provide a brief and helpful response. If the question is off-topic, politely ask them to stick to DuoLife-related questions about supplements, health, products, or the company.


Your response:
"""
general_question_prompt = ChatPromptTemplate.from_template(general_question_prompt_template)

# 3. Answer Generation Prompt
generate_answer_prompt_template = """
You are DuoBot, a helpful and friendly assistant for the DuoLife company. 
You are DuoLife's assistant, representing them personally in all interactions.

CORE PERSONALITY & TONE:
- Always be warm, enthusiastic, and helpful
- Use friendly greetings like "Hi!", "Hello!", "Hey!"
- Include relevant emojis to make responses engaging (💪, ✨, 🌿, 😊, etc.)
- Address users directly and personally
- Show excitement about DuoLife products

RESPONSE GUIDELINES:

1. **MEDICAL DISCLAIMERS**: 
   - ALWAYS end health-related responses with: "*[This information is not medical advice. Always consult with a healthcare professional for any health concerns.]*"
   - For high-risk conditions or drug interactions, be extra cautious and recommend consulting a doctor

2. **LANGUAGE RESTRICTIONS**:
   - NEVER use medical action words like: "inhibit", "treat", "cure", "heal", "affects", "prevents"
   - Instead use supportive language: "supports", "contributes to", "helps maintain", "designed to support"
   - Replace "affects the brain" with "supports normal brain function"

3. **AFFILIATE LINKS & SOURCES**:
   - For product recommendations, ALWAYS include: "You can check it out here: [Affiliate Link]"
   - Only include sources array for product recommendation questions
   - For informational questions (ingredients, business, health benefits), provide direct answers without sources

4. **CONTRAINDICATIONS & SAFETY**:
   - If asked about product safety with medical conditions, always recommend consulting a healthcare professional
   - For sensitive skin products, suggest patch testing
   - Be specific about product guidelines and warnings

5. **FACTUAL ACCURACY**:
   - Provide specific details: percentages, gram amounts, ingredient names
   - Use exact product names and proprietary formula names (e.g., "BACTILARDII®", "ProRelaxin®")
   - Include naturalness indices when relevant (e.g., "96% natural")

6. **BUSINESS QUESTIONS**:
   - Provide specific figures for positions, requirements, and earnings
   - Include point values (P) and premium point (PP) information
   - Explain DuoLife career structure clearly

7. **SECURITY & GUARDRAILS**:
   - For off-topic questions: "I'm sorry, I can't assist with that. I can only answer questions related to DuoLife."
   - For system/personal information requests: "I'm sorry, I can't assist with that. I can only answer questions related to DuoLife and I do not have access to user data."
   - Stay focused on DuoLife-related topics only

8. **MULTILINGUAL SUPPORT**:
   - Respond in the language the user asks in
   - Maintain the same tone and guidelines across all languages

ANSWER STRUCTURE:
- Start with friendly greeting: "Hi! I'm [Name]'s assistant."
- Provide clear, enthusiastic answer
- Include specific details and facts
- Add relevant emoji
- Include affiliate link for product recommendations
- End with medical disclaimer if health-related

Context from Graph Search (specific facts and links):
{graph_context}

Context from Vector Search (detailed descriptions):
{vector_context}

Question: {question}

{format_instructions}

Remember: You represent DuoLife personally, so maintain their reputation with helpful, accurate, and compliant responses.
"""
answer_parser = PydanticOutputParser(pydantic_object=FinalAnswer)
generate_answer_prompt = ChatPromptTemplate.from_template(generate_answer_prompt_template).partial(
    format_instructions=answer_parser.get_format_instructions()
)

# --- Agent Nodes ---

def router_node(state: AgentState):
    """Determines the route for the question."""
    print("---ROUTING: ANALYZING QUESTION---")
    chain = router_prompt | llm | router_parser
    result = chain.invoke({"question": state["question"]})
    print(f"Router Decision: General? {result.is_general_question}, Graph? {result.needs_graph_search} - {result.reasoning}")
    return {"router_decision": result.model_dump()}

def general_question_node(state: AgentState):
    """Handles general questions with a simple response."""
    print("---HANDLING GENERAL QUESTION---")
    chain = general_question_prompt | llm
    result = chain.invoke({"question": state["question"]})
    answer = FinalAnswer(conversational_answer=result.content, sources=[])
    return {"final_answer": answer.model_dump()}

def graph_search_node(state: AgentState):
    """Queries the knowledge graph for specific information."""
    print("---GRAPH SEARCH---")
    question = state["question"].lower()
    found_entities, relationships, links = [], [], []

    if "ingredients in" in question or "what is in" in question:
        product_name = question.split("in ")[-1].replace("?", "").strip()
        ingredients = get_ingredients_for_product(product_name)
        product_links = get_product_links(product_name)
        found_entities.append(product_name)
        relationships.extend([f"{product_name} contains {ing}" for ing in ingredients])
        links.extend(product_links)
    elif "which products contain" in question:
        ingredient_name = question.split("contain ")[-1].replace("?", "").strip()
        products = find_products_containing_ingredient(ingredient_name)
        found_entities.append(ingredient_name)
        for product in products:
            relationships.append(f"{product} contains {ingredient_name}")
            links.extend(get_product_links(product))

    graph_result = GraphResult(found_entities=found_entities, relationships=relationships, links=links)
    print(f"Graph Results: {graph_result}")
    return {"graph_results": graph_result.model_dump()}

def vector_search_node(state: AgentState):
    """Retrieves documents from the vector store."""
    print("---VECTOR SEARCH---")
    question = state["question"]
    graph_results = state.get("graph_results", {})
    enhanced_query = question
    if graph_results.get("found_entities"):
        entities = " ".join(graph_results["found_entities"])
        enhanced_query = f"{question} {entities}"

    docs = retriever.invoke(enhanced_query)
    doc_details = [
        f"Source Name: {d.metadata.get('name', 'Unknown')}\nLink: {d.metadata.get('link', '')}\nType: {d.metadata.get('type', 'General')}\nCategory: {d.metadata.get('category', 'Uncategorized')}\nContent: {d.page_content}"
        for d in docs
    ]
    return {"vector_documents": "\n\n---\n\n".join(doc_details)}

def generate_answer_node(state: AgentState):
    """Generates the final answer from context."""
    print("---GENERATING STRUCTURED ANSWER---")
    graph_context = ""
    if state.get("graph_results"):
        gr = state["graph_results"]
        graph_context = f"Entities: {gr.get('found_entities', [])}\nRelationships: {gr.get('relationships', [])}\nLinks: {gr.get('links', [])}"
    
    vector_context = state.get("vector_documents", "")
    chain = generate_answer_prompt | llm | answer_parser
    
    try:
        result = chain.invoke({
            "question": state["question"],
            "graph_context": graph_context,
            "vector_context": vector_context
        })
        return {"final_answer": result.model_dump()}
    except Exception as e:
        print(f"Error in generation: {e}")
        fallback_answer = FinalAnswer(conversational_answer="I'm having trouble processing your request.", sources=[])
        return {"final_answer": fallback_answer.model_dump()}

# --- Conditional Logic ---
def route_question(state: AgentState):
    """Determines the next step based on the router's decision."""
    router_decision = state.get("router_decision", {})
    if router_decision.get("is_general_question", False):
        return "general_question"
    if router_decision.get("needs_graph_search", False):
        return "graph_search"
    return "vector_search"

# --- Graph Definition ---
def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("router", router_node)
    graph.add_node("general_question", general_question_node)
    graph.add_node("graph_search", graph_search_node)
    graph.add_node("vector_search", vector_search_node)
    graph.add_node("generate_answer", generate_answer_node)
    
    graph.set_entry_point("router")
    graph.add_conditional_edges("router", route_question, {
        "general_question": "general_question",
        "graph_search": "graph_search",
        "vector_search": "vector_search"
    })
    
    graph.add_edge("general_question", END)
    graph.add_edge("graph_search", "vector_search")
    graph.add_edge("vector_search", "generate_answer")
    graph.add_edge("generate_answer", END)
    
    return graph.compile()

# --- Functions for API Integration ---
store = {}

def get_chat_history(session_id: str) -> BaseChatMessageHistory:
    """Factory function to get a chat history object for a session."""
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def get_agent_executor():
    """Builds and returns the agent executor with chat history."""
    app = build_graph()
    agent_executor = RunnableWithMessageHistory(
        app,
        get_chat_history,
        input_messages_key="question",
        history_messages_key="chat_history",
        history_output_key="final_answer"
    )
    return agent_executor

# --- Main Execution Block for Testing ---
if __name__ == "__main__":
    print("Agent is ready. Running test cases...")
    agent = get_agent_executor()
    session_id = "test_session_123"

    # Test Case 1: General Question
    print("\n--- TEST CASE 1: GENERAL QUESTION ---")
    question1 = "Hello, how are you?"
    response1 = agent.invoke(
        {"question": question1},
        config={"configurable": {"session_id": session_id}}
    )
    print(f"Q: {question1}\nA: {response1['final_answer']}")

    # Test Case 2: Graph + Vector Search
    print("\n--- TEST CASE 2: GRAPH + VECTOR SEARCH ---")
    question2 = "What are the ingredients in DuoLife Collagen?"
    response2 = agent.invoke(
        {"question": question2},
        config={"configurable": {"session_id": session_id}}
    )
    print(f"Q: {question2}\nA: {response2['final_answer']}")

    # Test Case 3: Vector Search Only
    print("\n--- TEST CASE 3: VECTOR SEARCH ONLY ---")
    question3 = "What is DuoLife Collagen good for?"
    response3 = agent.invoke(
        {"question": question3},
        config={"configurable": {"session_id": session_id}}
    )
    print(f"Q: {question3}\nA: {response3['final_answer']}")
