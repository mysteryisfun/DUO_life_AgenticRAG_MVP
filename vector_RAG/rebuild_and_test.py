# rebuild_and_test.py

import os
import json
import shutil
from dotenv import load_dotenv
from typing import List

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# --- Configuration ---
load_dotenv()

# Define paths relative to the project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
JSON_DATA_PATH = "data_filtering/structured_rag_data.json"
CHROMA_PERSIST_DIRECTORY = "./chroma_db"
EMBEDDING_MODEL = "text-embedding-3-small"

def clean_and_prepare_documents(json_path: str) -> List[Document]:
    """
    Loads data and prepares it for ingestion, ensuring all metadata is standardized.
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
        
        # --- CRITICAL FIX: Standardize the 'type' field to lowercase ---
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
    
    print(f"Successfully loaded and standardized {len(documents)} data chunks.")
    return documents

def rebuild_vectorstore():
    """
    Deletes the old database and rebuilds it from scratch with clean data.
    """
    # 1. Delete the old database directory
    if os.path.exists(CHROMA_PERSIST_DIRECTORY):
        print(f"Deleting existing vector store at '{CHROMA_PERSIST_DIRECTORY}'...")
        shutil.rmtree(CHROMA_PERSIST_DIRECTORY)
        print("Old vector store deleted.")

    # 2. Load and prepare documents with standardized metadata
    documents = clean_and_prepare_documents(JSON_DATA_PATH)
    if not documents:
        print("No documents found. Aborting rebuild.")
        return

    # 3. Create and persist the new vector store
    print("Initializing OpenAI embedding model...")
    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    print(f"Rebuilding and persisting vector store at '{CHROMA_PERSIST_DIRECTORY}'...")
    Chroma.from_documents(
        documents=documents, 
        embedding=embeddings,
        persist_directory=CHROMA_PERSIST_DIRECTORY
    )
    print("Vector store rebuilt successfully!")


if __name__ == "__main__":
    if "OPENAI_API_KEY" not in os.environ:
        print("OpenAI API key not found in .env file.")
    else:
        # Run the entire process
        rebuild_vectorstore()
