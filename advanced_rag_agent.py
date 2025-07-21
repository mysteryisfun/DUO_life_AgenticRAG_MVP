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
    needs_graph_search: bool = Field(description="Whether graph search is needed for this question")
    reasoning: str = Field(description="Why this decision was made")

class GraphResult(BaseModel):
    found_entities: List[str] = Field(description="Entities found in the graph")
    relationships: List[str] = Field(description="Relationships discovered")
    links: List[str] = Field(description="Product links from the graph")

class FinalAnswer(BaseModel):
    conversational_answer: str = Field(description="The conversational response to the user")
    sources: List[Source] = Field(description="List of source documents with metadata")

# --- Agent State ---
class AgentState(TypedDict):
    question: str
    chat_history: List[BaseModel]
    router_decision: Dict[str, Any]
    graph_results: Dict[str, Any]
    vector_documents: str
    final_answer: Dict[str, Any]

# --- PROMPTS ---

# 1. Router Prompt with structured output
router_prompt_template = """
You are an expert at determining if a question requires specific graph database information.

Analyze the user's question and determine if it needs graph search for:
- Specific product ingredients
- Which products contain specific ingredients  
- Product relationships and links

Questions about general benefits, usage, business model, or company info do NOT need graph search.

Question: {question}

{format_instructions}
"""

router_parser = PydanticOutputParser(pydantic_object=RouterDecision)
router_prompt = ChatPromptTemplate.from_template(router_prompt_template).partial(
    format_instructions=router_parser.get_format_instructions()
)

# 2. Enhanced context prompt for final answer
generate_answer_prompt_template = """
You are a helpful assistant for the DuoLife company.
Answer the user's question based on the provided context from both graph and vector searches.

Graph Results (specific facts and links): {graph_context}
Vector Search Results (detailed descriptions): {vector_context}

Provide a conversational answer followed by structured sources using the exact format specified.

Question: {question}

{format_instructions}
"""

answer_parser = PydanticOutputParser(pydantic_object=FinalAnswer)
generate_answer_prompt = ChatPromptTemplate.from_template(generate_answer_prompt_template).partial(
    format_instructions=answer_parser.get_format_instructions()
)

# --- Agent Nodes ---

def router_node(state: AgentState):
    """Determines if graph search is needed."""
    print("---ROUTING: ANALYZING QUESTION---")
    chain = router_prompt | llm | router_parser
    result = chain.invoke({"question": state["question"]})
    print(f"Router Decision: {result.needs_graph_search} - {result.reasoning}")
    return {"router_decision": result.model_dump()}

def graph_search_node(state: AgentState):
    """Queries the knowledge graph for specific information and links."""
    print("---GRAPH SEARCH---")
    question = state["question"].lower()
    found_entities = []
    relationships = []
    links = []
    
    # Extract entities and get information
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
            product_links = get_product_links(product)
            links.extend(product_links)
    
    graph_result = GraphResult(
        found_entities=found_entities,
        relationships=relationships,
        links=links
    )
    
    print(f"Graph Results: {graph_result}")
    return {"graph_results": graph_result.model_dump()}

def vector_search_node(state: AgentState):
    """Retrieves documents from vector store, using graph context to enhance search."""
    print("---VECTOR SEARCH---")
    question = state["question"]
    graph_results = state.get("graph_results", {})
    
    # Enhance search query with graph entities if available
    enhanced_query = question
    if graph_results.get("found_entities"):
        entities = " ".join(graph_results["found_entities"])
        enhanced_query = f"{question} {entities}"
    
    docs = retriever.invoke(enhanced_query)
    
    # Format documents with metadata
    doc_details = []
    for d in docs:
        metadata = d.metadata
        name = metadata.get("name", "Unknown Source")
        link = metadata.get("link", "")
        doc_type = metadata.get("type", "General")
        category = metadata.get("category", "Uncategorized")
        snippet = d.page_content
        
        detail_str = (
            f"Source Name: {name}\n"
            f"Link: {link}\n"
            f"Type: {doc_type}\n"
            f"Category: {category}\n"
            f"Content: {snippet}"
        )
        doc_details.append(detail_str)
    
    formatted_docs = "\n\n---\n\n".join(doc_details)
    return {"vector_documents": formatted_docs}

def generate_answer_node(state: AgentState):
    """Generates structured final answer using both graph and vector context."""
    print("---GENERATING STRUCTURED ANSWER---")
    
    graph_context = ""
    if state.get("graph_results"):
        gr = state["graph_results"]
        graph_context = f"Entities: {gr.get('found_entities', [])}\n"
        graph_context += f"Relationships: {gr.get('relationships', [])}\n"
        graph_context += f"Links: {gr.get('links', [])}"
    
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
        print(f"Error in structured generation: {e}")
        # Fallback to simple response
        fallback_answer = FinalAnswer(
            conversational_answer="I apologize, but I'm having trouble processing your request right now.",
            sources=[]
        )
        return {"final_answer": fallback_answer.model_dump()}

# --- Conditional Logic ---
def should_do_graph_search(state: AgentState):
    """Determines next step based on router decision."""
    router_decision = state.get("router_decision", {})
    if router_decision.get("needs_graph_search", False):
        return "graph_search"
    return "vector_search"

# --- Graph Definition ---
def build_graph():
    graph = StateGraph(AgentState)
    
    # Add all nodes
    graph.add_node("router", router_node)
    graph.add_node("graph_search", graph_search_node)
    graph.add_node("vector_search", vector_search_node)
    graph.add_node("generate_answer", generate_answer_node)
    
    # Define the flow: router -> (optional graph_search) -> vector_search -> generate_answer
    graph.set_entry_point("router")
    
    graph.add_conditional_edges(
        "router",
        should_do_graph_search,
        {
            "graph_search": "graph_search",
            "vector_search": "vector_search"
        }
    )
    
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
    )
    return agent_executor

# --- Main Execution Block for Testing ---
if __name__ == "__main__":
    print("Agent is ready. Running test cases...")
    agent = get_agent_executor()
    session_id = "test_session_123"

    # Test Case 1: Graph + Vector Search
    print("\n--- TEST CASE 1: GRAPH + VECTOR SEARCH ---")
    question1 = "What are the ingredients in DuoLife Collagen?"
    response1 = agent.invoke(
        {"question": question1},
        config={"configurable": {"session_id": session_id}}
    )
    print(f"Q: {question1}\nA: {response1}")

    # Test Case 2: Vector Search Only
    print("\n--- TEST CASE 2: VECTOR SEARCH ONLY ---")
    question2 = "What is DuoLife Collagen good for?"
    response2 = agent.invoke(
        {"question": question2},
        config={"configurable": {"session_id": session_id}}
    )
    print(f"Q: {question2}\nA: {response2}")

    # Test Case 3: Follow-up question to test memory
    print("\n--- TEST CASE 3: FOLLOW-UP (TESTING MEMORY) ---")
    question3 = "Which of its ingredients helps with skin?"
    response3 = agent.invoke(
        {"question": question3},
        config={"configurable": {"session_id": session_id}}
    )
    print(f"Q: {question3}\nA: {response3}")
    print("\nNote: If the agent correctly answered the follow-up, session memory is working within a single process.")