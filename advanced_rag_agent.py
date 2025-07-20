import os
from dotenv import load_dotenv
from typing import List, Dict, Any, TypedDict
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from langchain_community.vectorstores import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory


from graph_RAG.graph_manager import get_ingredients_for_product, find_products_containing_ingredient

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

# --- Agent State ---
class AgentState(TypedDict):
    question: str
    chat_history: List[BaseModel]
    documents: List[str]
    graph_response: str
    answer: str

# --- PROMPTS ---

# 1. Router Prompt
router_prompt_template = """
You are an expert at routing a user's question to the correct tool.
Based on the user's question, determine whether it is better to use a vector database or a graph database.

- Use the 'vector_search' tool for questions about product descriptions, benefits, usage, business model, company info, or general queries.
- Use the 'graph_search' tool for specific questions about the ingredients of a particular product or which products contain a specific ingredient.

You must respond with ONLY the name of the tool to use, either 'vector_search' or 'graph_search'.

Question: {question}
"""
router_prompt = ChatPromptTemplate.from_template(router_prompt_template)

# 2. Generate Answer Prompt
generate_answer_prompt_template = """
You are a helpful assistant for the DuoLife company.
Answer the user's question based on the provided context, which may include information from a vector search, a graph database, and the chat history.

Keep your answer concise and directly address the question.
After your conversational answer, you MUST include a structured list of sources in a JSON format.
The JSON should be a list of objects, where each object represents a source document.

Start the JSON block with the separator '---JSON_SOURCES---' on a new line.

The final output should look like this:
<Your conversational answer here.>
---JSON_SOURCES---
[
  {{
    "name": "Source Name (e.g., Product Name)",
    "links": ["url_to_the_product_page_if_available"],
    "type": "Document Type (e.g., Product, Business_Model)",
    "category": "Document Category (e.g., Dietary Supplement, Marketing Plan)",
    "snippet": "A relevant short quote from the source document."
  }}
]

Context from vector search and/or graph search:
{context}

Question:
{question}
"""
generate_answer_prompt = ChatPromptTemplate.from_template(generate_answer_prompt_template)


# --- Agent Nodes ---

def router_node(state: AgentState):
    """Routes the question to the appropriate tool."""
    print("---ROUTING---")
    chain = router_prompt | llm | StrOutputParser()
    result = chain.invoke({"question": state["question"]})
    print(f"Route: {result}")
    if "vector_search" in result.lower():
        return "vector_search"
    return "graph_search"

def vector_search_node(state: AgentState):
    """Retrieves documents from the vector store and formats them with metadata."""
    print("---VECTOR SEARCH---")
    question = state["question"]
    docs = retriever.invoke(question)
    
    # Format documents with all metadata for the LLM
    doc_details = []
    for d in docs:
        metadata = d.metadata
        # Ensure all keys exist, provide defaults if not
        name = metadata.get("name", "Unknown Source")
        link = metadata.get("link", "")
        doc_type = metadata.get("type", "General")
        category = metadata.get("category", "Uncategorized")
        snippet = d.page_content
        
        # Create a detailed string for each document
        detail_str = (
            f"Source Name: {name}\n"
            f"Link: {link}\n"
            f"Type: {doc_type}\n"
            f"Category: {category}\n"
            f"Snippet: \"{snippet}\""
        )
        doc_details.append(detail_str)
        
    formatted_docs = "\n\n---\n\n".join(doc_details)
    return {"documents": formatted_docs, "graph_response": ""}

def graph_search_node(state: AgentState):
    """Queries the knowledge graph for specific ingredient/product relationships."""
    print("---GRAPH SEARCH---")
    question = state["question"].lower()
    # This is a simplified logic. A real implementation would use an LLM to extract entities.
    if "ingredients in" in question or "what is in" in question:
        product_name = question.split("in ")[-1].replace("?", "").strip()
        ingredients = get_ingredients_for_product(product_name)
        response = f"Ingredients for {product_name.title()}: {', '.join(ingredients) if ingredients else 'Not found.'}"
    elif "which products contain" in question:
        ingredient_name = question.split("contain ")[-1].replace("?", "").strip()
        products = find_products_containing_ingredient(ingredient_name)
        response = f"Products containing {ingredient_name.title()}: {', '.join(products) if products else 'Not found.'}"
    else:
        response = "Could not determine the specific graph query from the question."
    return {"documents": [], "graph_response": response}

def generate_answer_node(state: AgentState):
    """Generates the final answer based on the retrieved context."""
    print("---GENERATING ANSWER---")
    context = f"Vector Search Results:\n{state['documents']}\n\nGraph Search Results:\n{state['graph_response']}"
    chain = generate_answer_prompt | llm | StrOutputParser()
    answer = chain.invoke({
        "question": state["question"],
        "context": context.strip()
    })
    return {"answer": answer}


# --- Graph Definition ---
def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("vector_search", vector_search_node)
    graph.add_node("graph_search", graph_search_node)
    graph.add_node("generate_answer", generate_answer_node)

    graph.set_conditional_entry_point(
        router_node,
        {
            "vector_search": "vector_search",
            "graph_search": "graph_search",
        },
    )
    graph.add_edge("vector_search", "generate_answer")
    graph.add_edge("graph_search", "generate_answer")
    graph.add_edge("generate_answer", END)
    return graph.compile()

# --- Functions for API Integration ---

# Store for session histories
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

    # Test Case 1: Vector Search
    print("\n--- TEST CASE 1: VECTOR SEARCH ---")
    question1 = "What is DuoLife Collagen good for?"
    response1 = agent.invoke(
        {"question": question1},
        config={"configurable": {"session_id": session_id}}
    )
    print(f"Q: {question1}\nA: {response1['answer']}")

    # Test Case 2: Graph Search
    print("\n--- TEST CASE 2: GRAPH SEARCH ---")
    question2 = "What are the ingredients in DuoLife Collagen?"
    response2 = agent.invoke(
        {"question": question2},
        config={"configurable": {"session_id": session_id}}
    )
    print(f"Q: {question2}\nA: {response2['answer']}")

    # Test Case 3: Follow-up question (tests history)
    print("\n--- TEST CASE 3: FOLLOW-UP ---")
    question3 = "Which of those helps with skin?"
    response3 = agent.invoke(
        {"question": question3},
        config={"configurable": {"session_id": session_id}}
    )
    print(f"Q: {question3}\nA: {response3['answer']}")