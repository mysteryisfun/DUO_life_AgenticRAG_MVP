# build_vectorstore.py

import os
import json
from typing import List
from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# --- Configuration ---
load_dotenv()
JSON_DATA_PATH = "data_filtering/structured_rag_data.json"
CHROMA_PERSIST_DIRECTORY = "./chroma_db"
EMBEDDING_MODEL = "text-embedding-3-small"


def load_and_prepare_documents(json_path: str) -> List[Document]:
    """
    Loads structured data from a JSON file and converts it into LangChain Document objects.
    This version includes a fix to handle lists in metadata.
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
        for key, value in metadata.items():
            if isinstance(value, list):
                # Join list items into a single string
                metadata[key] = ", ".join(map(str, value))
        
        doc = Document(
            page_content=item.get("text_to_embed", ""),
            metadata=metadata  # Use the sanitized metadata
        )
        documents.append(doc)
    
    print(f"Successfully loaded and prepared {len(documents)} data chunks for ingestion.")
    return documents

def create_and_persist_vectorstore(documents: List[Document], persist_directory: str):
    """
    Creates embeddings for the documents and stores them in a persistent ChromaDB vector store.
    """
    if not documents:
        print("No documents to process. Skipping vector store creation.")
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
        print("OpenAI API key not found. Please create a .env file and add your key.")
    else:
        # Before running, it's a good idea to delete the old 'chroma_db' folder
        # if it was created with the error.
        print("Preparing to build the vector store...")
        docs = load_and_prepare_documents(JSON_DATA_PATH)
        create_and_persist_vectorstore(docs, CHROMA_PERSIST_DIRECTORY)