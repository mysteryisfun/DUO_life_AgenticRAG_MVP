# vector_RAG/build_vectorstore.py

import os
import json
from typing import List
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# --- Configuration ---
load_dotenv()

# --- UPDATED: Robust Path Construction ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JSON_DATA_PATH = os.path.join(PROJECT_ROOT, "data_filtering", "structured_rag_data.json")
CHROMA_PERSIST_DIRECTORY = os.path.join(PROJECT_ROOT, "chroma_db")
EMBEDDING_MODEL = "text-embedding-3-small"


def load_and_prepare_documents(json_path: str) -> List[Document]:
    """
    Loads data and prepares it for ingestion, ensuring metadata is clean.
    """
    print(f"Loading data from {json_path}...")
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{json_path}' was not found.")
        return []

    documents = []
    for item in data:
        metadata = item.get("metadata", {})
        
        # --- FIX: Standardize the 'type' field ---
        if 'type' in metadata and isinstance(metadata['type'], str):
            metadata['type'] = metadata['type'].lower().strip()
        
        # Convert any lists to strings for ChromaDB compatibility
        for key, value in metadata.items():
            if isinstance(value, list):
                metadata[key] = ", ".join(map(str, value))
        
        doc = Document(
            page_content=item.get("text_to_embed", ""),
            metadata=metadata
        )
        documents.append(doc)
    
    print(f"Successfully loaded and prepared {len(documents)} data chunks.")
    return documents

def create_and_persist_vectorstore(documents: List[Document], persist_directory: str):
    """
    Creates and saves the vector store.
    """
    if not documents:
        print("No documents to process.")
        return

    print("Initializing OpenAI embedding model...")
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    print(f"Creating and persisting vector store at '{persist_directory}'...")
    vectorstore = Chroma.from_documents(
        documents=documents, 
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    print(f"Successfully created vector store with {vectorstore._collection.count()} documents.")


if __name__ == "__main__":
    if "OPENAI_API_KEY" not in os.environ:
        print("OpenAI API key not found in .env file.")
    else:
        docs = load_and_prepare_documents(JSON_DATA_PATH)
        create_and_persist_vectorstore(docs, CHROMA_PERSIST_DIRECTORY)