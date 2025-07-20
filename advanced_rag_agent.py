# advanced_rag_agent.py

import os
import logging
from dotenv import load_dotenv
from typing import List, TypedDict, Literal

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma # <-- FIX: Using the modern, correct import
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
# Import from your existing, correct graph manager
from graph_RAG.graph_manager import duolife_kg, get_ingredients_for_product, find_products_containing_ingredient

# --- Configuration ---
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logging.getLogger("langgraph").setLevel(logging.INFO)

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CHROMA_PERSIST_DIRECTORY = os.path.join(PROJECT_ROOT, "chroma_db")
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o"

# (The rest of the code is structurally the same, but this ensures all parts work together)

# --- State Definition ---
class AgentState(TypedDict):
    question: str
    chat_history: List[BaseMessage]
    search_query: str
    filters: dict
    documents: List[Document]
    kg_query: dict  # Added to handle graph search queries
    kg_result: List[str]
    final_answer: str

# --- Tool-Calling Models for the Router ---
class VectorStoreSearch(BaseModel):
    """Routes a user query to the vector store, applying appropriate filters."""
    query: str = Field(description="The semantic query to search for in the vector store.")
    product_type: str = Field(
        description="The type of product, if specified (e.g., 'Liquid', 'Capsule', 'Powder'). Defaults to 'any'.",
        default="any"
    )
    domain: Literal["Products", "Cosmetics", "Business_Model", "Ranks", "Compensation", "Ingredients", "Unknown"] = Field(
        description="The primary domain the user is asking about."
    )

class KnowledgeGraphSearch(BaseModel):
    """Routes a user query to the knowledge graph for entity-specific questions."""
    entity_name: str = Field(description="The specific product, cosmetic, or ingredient name.")
    query_type: Literal["contains_ingredient", "get_ingredients"] = Field(
        description="The type of query to perform on the knowledge graph."
    )

# --- Graph Nodes ---
def router_node(state: AgentState):
    logging.info("[Router] Deciding next action.")
    llm_with_tools = ChatOpenAI(model=LLM_MODEL, temperature=0).bind_tools([VectorStoreSearch, KnowledgeGraphSearch])
    
    system_prompt = """You are an expert at routing a user's question to the appropriate tool.

Based on the user's question, you must decide whether to use the Vector Store or the Knowledge Graph.

**Vector Store (`VectorStoreSearch`):**
- Use for general questions about products, cosmetics, the business model, compensation plans, or ingredients.
- Use for questions that require searching for information based on concepts or descriptions (e.g., "products for skin health", "how to earn money").
- **Domain is crucial:**
    - 'Products': For questions about supplements, health benefits, usage, etc.
    - 'Cosmetics': For questions about creams, beauty products, etc.
    - 'Business_Model', 'Ranks', 'Compensation': For questions about the MLM structure, earning money, career levels.
    - 'Ingredients': For general questions about what an ingredient does.

**Knowledge Graph (`KnowledgeGraphSearch`):**
- Use ONLY for specific, direct questions about the relationships between entities.
- `get_ingredients`: Use when the user asks for the specific ingredients of a named product (e.g., "What is in DuoLife Collagen?").
- `contains_ingredient`: Use when the user asks which products contain a specific, named ingredient (e.g., "Which products have Shea Butter?").

If the user asks a general question like "Tell me about DuoLife Collagen", route it to the `VectorStoreSearch` with the domain 'Products'.
If the user asks a greeting or off-topic question, do not call any tool."""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("placeholder", "{chat_history}"),
        ("user", "{question}")
    ])
    
    router_chain = prompt | llm_with_tools
    tool_choice = router_chain.invoke({"question": state['question'], "chat_history": state['chat_history']})
    
    # --- FIX: If no tool is chosen, go to fallback to decide what to do next ---
    if not tool_choice.tool_calls:
        logging.info("  - Decision: No tool called, routing to fallback.")
        return {"next_node": "decide_fallback"}
        
    tool_name = tool_choice.tool_calls[0]['name']
    tool_args = tool_choice.tool_calls[0]['args']
    
    if tool_name == 'VectorStoreSearch':
        logging.info(f"  - Decision: Route to Vector Store Search with query '{tool_args['query']}'")
        filter_conditions = []
        if tool_args.get('domain'):
            filter_conditions.append({"domain": {"$eq": tool_args['domain']}})
        product_type_filter = tool_args.get('product_type', 'any').lower().strip()
        if product_type_filter != 'any':
            filter_conditions.append({"type": {"$eq": product_type_filter}})
        
        final_filters = {}
        if len(filter_conditions) > 1:
            final_filters = {"$and": filter_conditions}
        elif len(filter_conditions) == 1:
            final_filters = filter_conditions[0]

        return {"search_query": tool_args['query'], "filters": final_filters, "next_node": "vector_search"}
    
    elif tool_name == 'KnowledgeGraphSearch':
        logging.info(f"  - Decision: Route to Knowledge Graph with query: {tool_args}")
        return {"kg_query": tool_args, "next_node": "graph_search"}
        
    # Default fallback
    logging.warning("  - Warning: Router could not decide on a tool. Routing to error handler.")
    return {"next_node": "handle_error"}

def vector_search_node(state: AgentState):
    logging.info(f"[Vector Search] Searching for: '{state['search_query']}' with filters: {state.get('filters', {})}")
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = Chroma(persist_directory=CHROMA_PERSIST_DIRECTORY, embedding_function=embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4, "filter": state.get('filters', {})})
    documents = retriever.invoke(state['search_query'])
    logging.info(f"[Vector Search] Found {len(documents)} documents.")
    return {"documents": documents}

def graph_search_node(state: AgentState):
    logging.info(f"[Graph Search] Querying for entity: '{state['kg_query']['entity_name']}'")
    kg_query = state['kg_query']
    entity_name, query_type = kg_query['entity_name'], kg_query['query_type']
    if duolife_kg is None: 
        logging.warning("[Graph Search] Knowledge Graph not available.")
        return {"kg_result": ["Knowledge Graph is not available."]}
    if query_type == "get_ingredients":
        results = get_ingredients_for_product(duolife_kg, entity_name)
        res_str = f"The ingredients in {entity_name} are: " + ", ".join(results) if results else f"Could not find ingredients for {entity_name}."
    elif query_type == "contains_ingredient":
        results = find_products_containing_ingredient(duolife_kg, entity_name)
        names = [p.get('name', 'Unknown') for p in results]
        res_str = f"Products containing {entity_name} include: " + ", ".join(names) if names else f"Could not find any products containing {entity_name}."
    else: 
        res_str = "Unknown graph query type."
    logging.info(f"[Graph Search] Result: {res_str}")
    return {"kg_result": [res_str]}

def decide_fallback_node(state: AgentState):
    logging.info("[Fallback] Checking for retrieved information.")
    if not state.get('documents') and not state.get('kg_result'):
        logging.warning("[Fallback] No information found. Routing to error handler.")
        return {"next_node": "handle_error"}
    logging.info("[Fallback] Information found. Proceeding to generate answer.")
    return {"next_node": "generate_answer"}

def generate_answer_node(state: AgentState):
    logging.info("[Generator] Generating final answer.")
    context = []
    if state.get('documents'):
        for doc in state['documents']:
            content_str = doc.page_content
            if 'link' in doc.metadata:
                content_str += f"\nSource Link: {doc.metadata['link']}"
            context.append(content_str)
    
    context_str = "\n\n---\n\n".join(context)
    
    if state.get('kg_result'): context_str += "\n\n".join(state['kg_result'])
    if state.get('kg_result'): context += "\n\n".join(state['kg_result'])
    
    system_prompt = """You are the DuoLife Customer Helper, an expert on all DuoLife products and business opportunities.

Your goal is to provide clear, helpful, and concise answers based on the context provided.

- Answer the user's question directly using the information from the CONTEXT section.
- If the context contains links to products, you MUST include up to 3 of the most relevant links in your answer.
- Do not make up information. If the answer is not in the context, say so.
- Format your answers to be easy to read."""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("placeholder", "{chat_history}"),
        ("user", "Based on the following context...\n\nCONTEXT:\n{context}\n\nQUESTION:\n{question}")
    ])
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0.1)
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"context": context, "question": state['question'], "chat_history": state['chat_history']})
    return {"final_answer": answer}

def handle_error_node(state: AgentState):
    logging.info("[Error Handler] Providing generic response.")
    return {"final_answer": "I'm sorry, I couldn't find specific information about that. Could you rephrase your question?"}

# --- Build the Graph ---
workflow = StateGraph(AgentState)
workflow.add_node("router", router_node)
workflow.add_node("vector_search", vector_search_node)
workflow.add_node("graph_search", graph_search_node)
workflow.add_node("decide_fallback", decide_fallback_node)
workflow.add_node("generate_answer", generate_answer_node)
workflow.add_node("handle_error", handle_error_node)
workflow.set_entry_point("router")
workflow.add_conditional_edges(
    "router",
    lambda x: x["next_node"],
    {
        "vector_search": "vector_search",
        "graph_search": "graph_search",
        "decide_fallback": "decide_fallback",
        "handle_error": "handle_error",
    },
)
workflow.add_edge("vector_search", "decide_fallback")
workflow.add_edge("graph_search", "decide_fallback")
workflow.add_conditional_edges(
    "decide_fallback",
    lambda x: x["next_node"],
    {"generate_answer": "generate_answer", "handle_error": "handle_error"},
)
workflow.add_edge("generate_answer", END)
workflow.add_edge("handle_error", END)

advanced_rag_app = workflow.compile()

# --- Test Block ---
if __name__ == '__main__':
    print("--- Testing Advanced RAG Agent ---")
    if "OPENAI_API_KEY" not in os.environ:
        print("OpenAI API key not found. Cannot run tests.")
    else:
        chat_history = []
        
        print("\n\n--- TEST CASE 1: VECTOR SEARCH (Filtered) ---")
        question1 = "Are there any liquid products for skin health?"
        inputs1 = {"question": question1, "chat_history": chat_history}
        result1 = advanced_rag_app.invoke(inputs1)
        answer1 = result1['final_answer']
        print(f"Q: {question1}\nA: {answer1}")

        print("\n\n--- TEST CASE 2: KNOWLEDGE GRAPH SEARCH ---")
        question2 = "What are the ingredients in DuoLife Collagen?"
        inputs2 = {"question": question2, "chat_history": chat_history}
        result2 = advanced_rag_app.invoke(inputs2)
        answer2 = result2['final_answer']
        print(f"Q: {question2}\nA: {answer2}")