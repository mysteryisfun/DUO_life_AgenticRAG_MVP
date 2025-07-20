import os
import logging
from dotenv import load_dotenv
from typing import List, TypedDict

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langgraph.graph import StateGraph, END
from langchain_core.output_parsers import StrOutputParser   

# --- Configuration ---
load_dotenv()

CHROMA_PERSIST_DIRECTORY = "./chroma_db"
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4.1"

# --- 1. Define the State for our Graph ---
class GraphState(TypedDict):
    question: str
    search_query: str
    filters: dict
    documents: List[Document]
    answer: str

# --- 2. Define the Pydantic model for the Query Analyzer ---
class QueryFilters(BaseModel):
    search_query: str = Field(
        description="A concise, semantic search query that captures the user's main intent."
    )
    domain: str = Field(
        description="The primary domain the user is asking about. Must be one of: 'Products', 'Cosmetics', 'Business_Model', 'Ranks', 'Compensation', 'Ingredients', 'Unknown'.",
    )
    product_type: str = Field(
        description="The type of product, if specified (e.g., 'Liquid', 'Capsule', 'Powder'). Defaults to 'any'.",
        default="any"
    )

# --- 3. Define the Nodes of the Graph ---

def analyze_query_node(state: GraphState):
    """
    Analyzes the user's question to extract a search query and metadata filters.
    """
    logging.info("--- Node: analyze_query ---")
    question = state["question"]
    
    llm_with_tools = ChatOpenAI(model=LLM_MODEL, temperature=0).with_structured_output(QueryFilters)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert at analyzing user questions for a RAG system about the DuoLife company.
        Your task is to extract a semantic search query and structured filters based on the user's question.
        The 'domain' is the most important filter. If the user asks about products, supplements, or health, the domain is 'Products'.
        If they ask about creams or beauty, the domain is 'Cosmetics'.
        If they ask about how to earn money, MLM, or the business plan, the domain is 'Business_Model'.
        If they ask about career levels or positions, the domain is 'Ranks'."""),
        ("user", "User question: {question}")
    ])
    
    analyzer_chain = prompt | llm_with_tools
    structured_response = analyzer_chain.invoke({"question": question})
    
    filter_conditions = []
    if structured_response.domain != "Unknown":
        filter_conditions.append({"domain": {"$eq": structured_response.domain}})
    if structured_response.product_type != "any":
        filter_conditions.append({"type": {"$eq": structured_response.product_type}})
        
    final_filters = {}
    if len(filter_conditions) > 1:
        final_filters = {"$and": filter_conditions}
    elif len(filter_conditions) == 1:
        final_filters = filter_conditions[0]

        
    logging.info(f"  - Search Query: {structured_response.search_query}")
    logging.info(f"  - Constructed ChromaDB Filters: {final_filters}")

    return {
        "search_query": structured_response.search_query,
        "filters": final_filters # Pass the correctly formatted filters
    }

def retrieve_documents_node(state: GraphState):
    """
    Retrieves documents from the vector store based on the search query and filters.
    """
    logging.info("--- Node: retrieve_documents ---")
    search_query = state["search_query"]
    filters = state["filters"]
    
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = Chroma(persist_directory=CHROMA_PERSIST_DIRECTORY, embedding_function=embeddings)
    
    # The retriever will now correctly interpret the "$and" filter structure
    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 5, "filter": filters}
    )
    
    documents = retriever.invoke(search_query)
    logging.info(f"  - Found {len(documents)} documents after filtering.")
    
    return {"documents": documents}

def generate_answer_node(state: GraphState):
    """
    Generates a final answer using the retrieved documents and the original question.
    """
    logging.info("--- Node: generate_answer ---")
    question = state["question"]
    documents = state["documents"]
    
    prompt = ChatPromptTemplate.from_template("""
    You are the DuoLife Customer Helper, a friendly and knowledgeable assistant. Your goal is to help customers find the perfect DuoLife product that fits their needs and answer their questions clearly.

    Please use only the information provided in the CONTEXT below to answer the user's QUESTION. Do not make up information or use outside knowledge.

    - Speak directly to the customer in a supportive and helpful tone.
    - If the context describes a product that matches the customer's query, summarize its main benefits and explain how it could help them. Mention the product's name if it is available in the context.
    - If the context does not contain the answer, politely state that you couldn't find the specific information and that you're ready to assist with anything else.

    CONTEXT:
    {context}

    QUESTION:
    {question}
    """)
    
    context_str = "\n\n---\n\n".join([doc.page_content for doc in documents])
    
    llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
    
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({
        "context": context_str,
        "question": question
    })
    
    return {"answer": answer}

# --- 4. Build the Graph ---

def build_graph():
    """
    Builds and compiles the LangGraph for the advanced RAG process.
    """
    workflow = StateGraph(GraphState)

    workflow.add_node("analyze_query", analyze_query_node)
    workflow.add_node("retrieve_documents", retrieve_documents_node)
    workflow.add_node("generate_answer", generate_answer_node)

    workflow.set_entry_point("analyze_query")
    workflow.add_edge("analyze_query", "retrieve_documents")
    workflow.add_edge("retrieve_documents", "generate_answer")
    workflow.add_edge("generate_answer", END)

    app = workflow.compile()
    logging.info("LangGraph RAG application compiled successfully.")
    return app

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("--- Testing Advanced RAG with LangGraph ---")
    
    if "OPENAI_API_KEY" not in os.environ:
        print("OpenAI API key not found.")
    else:
        rag_app = build_graph()
        question = "Show me liquid supplements that are good for skin and hair"
        final_state = rag_app.invoke({"question": question})
        print("\n--- Final Answer ---")
        print(final_state['answer'])