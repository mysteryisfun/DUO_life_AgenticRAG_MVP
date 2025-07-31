import os
import re
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

# --- HELPER FUNCTIONS FOR ENHANCED GRAPH SEARCH ---
def extract_ingredient_from_info_query(question: str) -> str:
    """Extract ingredient name from information queries."""
    # Handle patterns like "What is resveratrol", "function of resveratrol"
    patterns = [
        r"what is (?:the )?(?:ingredient )?['\"]?(\w+)['\"]?",
        r"function of (?:the )?(?:ingredient )?['\"]?(\w+)['\"]?",
        r"benefits of (?:the )?(?:ingredient )?['\"]?(\w+)['\"]?",
        r"primary function of (?:the )?(?:ingredient )?['\"]?(\w+)['\"]?",
        r"tell me about (?:the )?(?:ingredient )?['\"]?(\w+)['\"]?",
        r"information about (?:the )?(?:ingredient )?['\"]?(\w+)['\"]?"
    ]
    
    question_lower = question.lower()
    
    for pattern in patterns:
        match = re.search(pattern, question_lower)
        if match:
            return match.group(1).capitalize()
    
    # Fallback: look for known ingredients in the question
    known_ingredients = [
        "resveratrol", "collagen", "biotin", "keratin", "vitamin c", "magnesium", 
        "zinc", "iron", "calcium", "vitamin d", "vitamin b", "omega", "curcumin",
        "coenzyme", "glucosamine", "chondroitin", "hyaluronic", "probiotics"
    ]
    for ingredient in known_ingredients:
        if ingredient.lower() in question_lower:
            return ingredient.capitalize()
    
    return None

def extract_product_name_enhanced(question: str) -> str:
    """Extract product name from question with better patterns."""
    # Look for product name after "in"
    if " in " in question:
        product = question.split(" in ")[-1].replace("?", "").strip()
        return product.title()
    
    # Look for DuoLife products
    if "duolife" in question.lower():
        words = question.split()
        duolife_index = -1
        for i, word in enumerate(words):
            if "duolife" in word.lower():
                duolife_index = i
                break
        
        if duolife_index != -1 and duolife_index + 1 < len(words):
            return f"DuoLife {words[duolife_index + 1].replace('?', '').strip()}"
    
    return None

def extract_ingredient_name_enhanced(question: str) -> str:
    """Extract ingredient name from question with better patterns."""
    if "contain " in question:
        ingredient = question.split("contain ")[-1].replace("?", "").strip()
        return ingredient.capitalize()
    
    if "with " in question:
        ingredient = question.split("with ")[-1].replace("?", "").strip()
        return ingredient.capitalize()
        
    return None

def extract_health_condition(question: str) -> str:
    """Extract health condition or benefit from question."""
    condition_mappings = {
        "hair loss": ["biotin", "keratin", "collagen"],
        "brittle nails": ["biotin", "keratin", "collagen"],
        "nail health": ["biotin", "keratin", "collagen"],
        "hair health": ["biotin", "keratin", "collagen"],
        "skin health": ["collagen", "vitamin c", "hyaluronic acid"],
        "joint pain": ["collagen", "glucosamine", "chondroitin"],
        "joint health": ["collagen", "glucosamine", "chondroitin"],
        "cardiovascular": ["resveratrol", "omega", "coenzyme q10"],
        "heart health": ["resveratrol", "omega", "coenzyme q10"],
        "immune system": ["vitamin c", "zinc", "vitamin d"],
        "energy": ["b complex", "iron", "coenzyme q10"],
        "brain function": ["omega", "resveratrol", "b complex"],
        "antioxidant": ["resveratrol", "vitamin c", "curcumin"]
    }
    
    question_lower = question.lower()
    for condition, ingredients in condition_mappings.items():
        if condition in question_lower:
            return condition
    
    return None

# --- PROMPTS ---

# 1. Enhanced Router Prompt
router_prompt_template = """
You are an expert at routing user questions for the DuoLife assistant.

QUESTION CATEGORIES:

1. **General Questions**: 
   - Greetings: "hello", "how are you?", "who are you?"
   - Off-topic questions not related to DuoLife
   - Set is_general_question=True, needs_graph_search=False

2. **Graph Search Required**:
   - Ingredient information: "What is resveratrol?", "function of biotin", "primary function of resveratrol"
   - Product ingredients: "What's in DuoLife Collagen?", "ingredients in DuoLife Day and Night"
   - Products containing ingredient: "Which products contain vitamin C?", "products with resveratrol"
   - Product recommendations: "What helps with hair loss?", "recommend for cardiovascular health"
   - Set is_general_question=False, needs_graph_search=True

3. **Vector Search Only**:
   - General DuoLife information without specific product queries
   - Business model questions
   - General health topics without specific ingredient/product requests
   - Set is_general_question=False, needs_graph_search=False

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

# 3. Enhanced Answer Generation Prompt
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

1. **PRODUCT RECOMMENDATIONS**:
   - When specific products are found in graph context, ALWAYS recommend them
   - Include affiliate links: "You can check it out here: [Product Link]"
   - Add products to sources array with proper metadata
   - Be specific about product names and their benefits

2. **INGREDIENT INFORMATION**:
   - Provide detailed information about the ingredient's function and benefits
   - If products containing the ingredient are found in graph context, recommend them
   - Include sources array when recommending products

3. **MEDICAL DISCLAIMERS**: 
   - ALWAYS end health-related responses with: "*[This information is not medical advice. Always consult with a healthcare professional for any health concerns.]*"

4. **LANGUAGE RESTRICTIONS**:
   - Use supportive language: "supports", "contributes to", "helps maintain", "designed to support"
   - Avoid: "treat", "cure", "heal", "prevents"

5. **SOURCES REQUIREMENT**:
   - Include sources array when:
     * Recommending specific products (use graph links and vector store metadata)
     * Answering "which products contain X"
     * Providing product information with links
   - Do NOT include sources for general ingredient information without product recommendations

6. **FACTUAL ACCURACY**:
   - Use exact product names and links from the graph context
   - Include specific details from vector context
   - Mention naturalness indices and proprietary formulas when available

Context from Graph Search (specific facts and links):
{graph_context}

Context from Vector Search (detailed descriptions):
{vector_context}

Question: {question}

Based on the context provided, give a comprehensive answer. If specific products and links are found in the graph context, include them in your response with affiliate links and add them to the sources array.

{format_instructions}
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
    """Enhanced graph search with multiple query patterns."""
    print("---GRAPH SEARCH---")
    question = state["question"].lower()
    found_entities, relationships, links = [], [], []

    # 1. Product ingredients query
    if any(phrase in question for phrase in ["ingredients in", "what is in", "contains what", "composition of"]):
        product_name = extract_product_name_enhanced(state["question"])
        if product_name:
            print(f"Looking for ingredients in: {product_name}")
            ingredients = get_ingredients_for_product(product_name)
            product_links = get_product_links(product_name)
            found_entities.append(product_name)
            relationships.extend([f"{product_name} contains {ing}" for ing in ingredients])
            links.extend(product_links)

    # 2. Products containing ingredient query
    elif any(phrase in question for phrase in ["which products contain", "products with", "find products containing", "products that have"]):
        ingredient_name = extract_ingredient_name_enhanced(state["question"])
        if ingredient_name:
            print(f"Looking for products containing: {ingredient_name}")
            products = find_products_containing_ingredient(ingredient_name)
            found_entities.append(ingredient_name)
            for product in products:
                relationships.append(f"{product} contains {ingredient_name}")
                links.extend(get_product_links(product))

    # 3. Ingredient information query (NEW - THIS WAS MISSING!)
    elif any(phrase in question for phrase in ["what is", "function of", "benefits of", "primary function", "tell me about", "information about"]):
        ingredient_name = extract_ingredient_from_info_query(state["question"])
        if ingredient_name:
            print(f"Looking for ingredient info and products containing: {ingredient_name}")
            # Find products containing this ingredient
            products = find_products_containing_ingredient(ingredient_name)
            found_entities.append(ingredient_name)
            for product in products:
                relationships.append(f"{product} contains {ingredient_name}")
                links.extend(get_product_links(product))

    # 4. Product recommendation query (NEW)
    elif any(phrase in question for phrase in ["recommend", "help with", "good for", "best for", "what should i take"]):
        condition = extract_health_condition(state["question"])
        if condition:
            print(f"Looking for products for condition: {condition}")
            # This would need to be implemented in graph_manager.py
            # For now, we'll let vector search handle this
            pass

    graph_result = GraphResult(found_entities=found_entities, relationships=relationships, links=links)
    print(f"Graph Results: found_entities={found_entities} relationships={relationships[:3]} links={links[:3]}")
    return {"graph_results": graph_result.model_dump()}

def vector_search_node(state: AgentState):
    """Retrieves documents from the vector store."""
    print("---VECTOR SEARCH---")
    question = state["question"]
    graph_results = state.get("graph_results", {})
    enhanced_query = question
    
    # Enhance query with entities found in graph search
    if graph_results.get("found_entities"):
        entities = " ".join(graph_results["found_entities"])
        enhanced_query = f"{question} {entities}"
        print(f"Enhanced query: {enhanced_query}")

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

    # Test Case 3: Ingredient Information (NEW TEST)
    print("\n--- TEST CASE 3: INGREDIENT INFORMATION ---")
    question3 = "What is the primary function of resveratrol?"
    response3 = agent.invoke(
        {"question": question3},
        config={"configurable": {"session_id": session_id}}
    )
    print(f"Q: {question3}\nA: {response3['final_answer']}")

    # Test Case 4: Vector Search Only
    print("\n--- TEST CASE 4: VECTOR SEARCH ONLY ---")
    question4 = "What is DuoLife's business model?"
    response4 = agent.invoke(
        {"question": question4},
        config={"configurable": {"session_id": session_id}}
    )
    print(f"Q: {question4}\nA: {response4['final_answer']}")