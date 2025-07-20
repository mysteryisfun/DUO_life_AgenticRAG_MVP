# rag_retriever.py

import os
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- Configuration ---
# Load environment variables from .env file
load_dotenv()

# Define paths and models
CHROMA_PERSIST_DIRECTORY = "./chroma_db"
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4.1"

# --- RAG Prompt Template ---
RAG_PROMPT_TEMPLATE = """
CONTEXT:
{context}

QUESTION:
{question}

Based on the context provided, please provide a clear and concise answer to the question. If the context does not contain the answer, state that the information is not available in the provided documents.
"""

# Global variables to hold the chain and retriever so they are initialized only once.
_retriever = None
_rag_chain = None

def get_retriever():
    """
    Initializes and returns a Chroma vector store retriever.
    """
    global _retriever
    if _retriever is None:
        print("Initializing retriever...")
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        
        # Load the persistent database
        vectorstore = Chroma(
            persist_directory=CHROMA_PERSIST_DIRECTORY, 
            embedding_function=embeddings
        )
        
        # Create the retriever
        _retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
        print("Retriever initialized.")
    return _retriever

def create_rag_chain():
    """
    Creates and returns the full RAG chain.
    """
    global _rag_chain
    if _rag_chain is None:
        print("Creating RAG chain...")
        retriever = get_retriever()
        
        # Initialize the LLM
        llm = ChatOpenAI(model_name=LLM_MODEL, temperature=0)
        
        # Create the prompt template
        rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)
        
        # Create the RAG chain
        _rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | rag_prompt
            | llm
            | StrOutputParser()
        )
        print("RAG chain created.")
    return _rag_chain


if __name__ == '__main__':
    # This block is for testing the module directly.
    print("--- Testing RAG Retriever Module ---")

    # Ensure you have an OpenAI API key in your .env file
    if "OPENAI_API_KEY" not in os.environ:
        print("OpenAI API key not found. Please create a .env file and add your key.")
    else:
        # Get the RAG chain
        rag_chain = create_rag_chain()
    
        # --- Ask a question ---
        question = "What are the requirements for a Senior Manager and what benefits do they get?"
        print(f"\nQuerying the RAG chain with: '{question}'")
        
        answer = rag_chain.invoke(question)
        
        print("\nAnswer:")
        print(answer)