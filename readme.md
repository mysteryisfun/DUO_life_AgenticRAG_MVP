# DuoBot: An Advanced Hybrid RAG Agent for DuoLife

This repository contains the source code for DuoBot, a sophisticated conversational AI agent designed to answer questions about DuoLife products, business model, and health-related topics. The agent leverages an advanced Retrieval-Augmented Generation (RAG) architecture that combines the strengths of both Vector Search and Graph-based retrieval to provide accurate, context-aware, and factually grounded answers.

## Table of Contents
1.  [The Core Problem](#the-core-problem)
2.  [Our Solution: A Hybrid Graph + Vector RAG Approach](#our-solution-a-hybrid-graph--vector-rag-approach)
3.  [In-Depth: How Graph RAG Enhances Standard Vector RAG](#in-depth-how-graph-rag-enhances-standard-vector-rag)
4.  [Agent Workflow with LangGraph](#agent-workflow-with-langgraph)
5.  [Visualizing the Knowledge Graph](#visualizing-the-knowledge-graph)
6.  [Project Structure](#project-structure)
7.  [Setup and Installation](#setup-and-installation)

## The Core Problem

Standard chatbots often struggle with complex, domain-specific information. A simple Vector RAG system, while powerful, has limitations:
-   **Lack of Precision for Factual Queries**: Asking "What are the ingredients in DuoLife Collagen?" requires precise, structured data, not just a semantically similar text chunk which might be a long paragraph.
-   **Difficulty with Relational Questions**: Answering "Which products contain Vitamin C?" is a relational query that is difficult for vector search alone, as it requires connecting multiple pieces of information (products and ingredients).
-   **Hallucination of Facts**: Without a structured source of truth, the LLM might misinterpret or "hallucinate" specific details like product links or ingredient lists.

## Our Solution: A Hybrid Graph + Vector RAG Approach

To overcome these limitations, DuoBot uses a hybrid approach orchestrated by **LangGraph**. It intelligently decides how to best answer a user's question by leveraging two distinct data retrieval methods.

### 1. Vector RAG (The "What" and "Why")
This is the foundation. We store detailed product descriptions, health benefit articles, and business model documents in a **ChromaDB vector store**.
-   **Purpose**: To find documents that are *semantically similar* to the user's query.
-   **Best For**: Open-ended questions like "What is DuoLife Collagen good for?" or "Explain the DuoLife business model." It provides rich, descriptive context for the LLM to synthesize an answer.

### 2. Graph RAG (The "Who," "Which," and "How")
This is our enhancement. We model factual, interconnected data in a **Knowledge Graph** (using NetworkX). This graph contains nodes (e.g., `Product`, `Ingredient`) and edges that define their relationships (e.g., `CONTAINS_INGREDIENT`).
-   **Purpose**: To retrieve precise, factual, and interconnected data points.
-   **Best For**: Specific, factual questions like "What are the ingredients in DuoLife Collagen?" or "Which products contain Vitamin C?". It provides structured, unambiguous facts.

By combining these two, DuoBot gets the best of both worlds: the descriptive power of vector search and the factual precision of a knowledge graph.

## In-Depth: How Graph RAG Enhances Standard Vector RAG

Integrating a Knowledge Graph is not just an addition; it's a fundamental enhancement that elevates the agent's capabilities.

| Feature | Standard Vector RAG | Hybrid Graph + Vector RAG |
| :--- | :--- | :--- |
| **Precision** | Retrieves broad text chunks. May require the LLM to "find" the specific fact within a long paragraph. | Retrieves discrete, factual nodes and relationships (e.g., `['Vitamin C', 'L-Arginine']`). Highly precise. |
| **Factual Accuracy** | Higher risk of hallucination if the context is ambiguous or the LLM misinterprets the text. | Grounded in a structured source of truth. The agent gets facts, not just paragraphs, reducing hallucination. |
| **Relational Queries** | Struggles with queries like "Find all products with ingredient X." It can't easily traverse relationships. | Excels at relational queries. It can traverse the graph from an ingredient to all connected products. |
| **Query Understanding** | Relies solely on semantic similarity. | The agent first understands the *intent* of the query (e.g., "this is a request for ingredients") and then executes a targeted graph traversal. |
| **Context Quality** | The context provided to the LLM is unstructured text. | The context is a mix of structured facts (from the graph) and unstructured text (from vector search), giving the LLM a richer, more reliable foundation. |

## Agent Workflow with LangGraph

The agent's decision-making process is modeled as a state machine using LangGraph. This allows for a clear, auditable, and robust workflow.

1.  **Router Node**: The entry point. It analyzes the user's question using an LLM call and decides the best path forward based on the `RouterDecision` model.
    - Is it a general greeting?
    - Does it require a precise fact from the graph?
    - Or is it a general descriptive question?

2.  **Conditional Branching**: Based on the router's decision, the graph transitions to one of three nodes:
    - **`general_question_node`**: For simple greetings ("Hello"). Provides a direct, pre-defined response and ends the process.
    - **`graph_search_node`**: For factual queries. It executes a targeted function call (e.g., `get_ingredients_for_product()`) to retrieve data from the NetworkX graph.
    - **`vector_search_node`**: The default path for most company-related questions.

3.  **Data Retrieval & Enhancement**:
    - The `graph_search_node` populates the agent's state with structured results.
    - The `vector_search_node` then runs. It may use entities found in the graph search to create an "enhanced query" for more relevant document retrieval.

4.  **Answer Generation Node**: This is the final step for complex queries. It takes all the retrieved context—both from the graph and the vector store—and feeds it into the `generate_answer_prompt`. This prompt instructs the LLM to synthesize a comprehensive, conversational answer, ensuring it uses the provided facts and includes sources where appropriate.

## Visualizing the Knowledge Graph

Understanding and debugging a knowledge graph is much easier when you can see it. Yes, there are excellent tools to visualize NetworkX graphs.

A highly recommended tool for easy, interactive visualizations directly from your Python code is **Pyvis**. It creates interactive HTML files that you can pan, zoom, and click on.


**Other Powerful Visualization Tools:**
-   **Gephi**: A powerful, open-source desktop application for graph visualization and analysis. You can export your NetworkX graph to a format like GEXF and import it into Gephi for deep exploration.
-   **Cytoscape**: Another popular desktop tool, especially in bioinformatics, but excellent for any kind of network analysis.

## Project Structure

```
.
├── frontend/              # React frontend application
├── graph_RAG/             # Knowledge graph creation and management
│   ├── graph_manager.py
│   └── ...
├── chroma_db/             # Persistent vector store
├── .env                   # Environment variables (API keys)
├── advanced_rag_agent.py  # Core agent logic, prompts, and LangGraph definition
├── main.py                # FastAPI application to serve the agent
├── models.py              # Pydantic models for API requests/responses
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## Setup and Installation

1.  **Prerequisites**:
    -   Python 3.10+
    -   Node.js and npm (for the frontend)

2.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd <repository-folder>
    ```

3.  **Install Python dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up Environment Variables**:
    -   Create a file named `.env` in the root directory.
    -   Add your API keys to this file:
        ```
        OPENAI_API_KEY="sk-..."
        # Add any other required keys
        ```

5.  **Build the Frontend**:
    ```bash
    cd frontend
    npm install
    npm run build
    cd ..
    ```

6.  **Run the API Server**:
    ```bash
    uvicorn main:app --reload
    ```
    The API will be available at `http://127.0.0.1:8000`.
```# DuoBot: An Advanced Hybrid RAG Agent for DuoLife

This repository contains the source code for DuoBot, a sophisticated conversational AI agent designed to answer questions about DuoLife products, business model, and health-related topics. The agent leverages an advanced Retrieval-Augmented Generation (RAG) architecture that combines the strengths of both Vector Search and Graph-based retrieval to provide accurate, context-aware, and factually grounded answers.

## Table of Contents
1.  [The Core Problem](#the-core-problem)
2.  [Our Solution: A Hybrid Graph + Vector RAG Approach](#our-solution-a-hybrid-graph--vector-rag-approach)
3.  [In-Depth: How Graph RAG Enhances Standard Vector RAG](#in-depth-how-graph-rag-enhances-standard-vector-rag)
4.  [Agent Workflow with LangGraph](#agent-workflow-with-langgraph)
5.  [Visualizing the Knowledge Graph](#visualizing-the-knowledge-graph)
6.  [Project Structure](#project-structure)
7.  [Setup and Installation](#setup-and-installation)

## The Core Problem

Standard chatbots often struggle with complex, domain-specific information. A simple Vector RAG system, while powerful, has limitations:
-   **Lack of Precision for Factual Queries**: Asking "What are the ingredients in DuoLife Collagen?" requires precise, structured data, not just a semantically similar text chunk which might be a long paragraph.
-   **Difficulty with Relational Questions**: Answering "Which products contain Vitamin C?" is a relational query that is difficult for vector search alone, as it requires connecting multiple pieces of information (products and ingredients).
-   **Hallucination of Facts**: Without a structured source of truth, the LLM might misinterpret or "hallucinate" specific details like product links or ingredient lists.

## Our Solution: A Hybrid Graph + Vector RAG Approach

To overcome these limitations, DuoBot uses a hybrid approach orchestrated by **LangGraph**. It intelligently decides how to best answer a user's question by leveraging two distinct data retrieval methods.

### 1. Vector RAG (The "What" and "Why")
This is the foundation. We store detailed product descriptions, health benefit articles, and business model documents in a **ChromaDB vector store**.
-   **Purpose**: To find documents that are *semantically similar* to the user's query.
-   **Best For**: Open-ended questions like "What is DuoLife Collagen good for?" or "Explain the DuoLife business model." It provides rich, descriptive context for the LLM to synthesize an answer.

### 2. Graph RAG (The "Who," "Which," and "How")
This is our enhancement. We model factual, interconnected data in a **Knowledge Graph** (using NetworkX). This graph contains nodes (e.g., `Product`, `Ingredient`) and edges that define their relationships (e.g., `CONTAINS_INGREDIENT`).
-   **Purpose**: To retrieve precise, factual, and interconnected data points.
-   **Best For**: Specific, factual questions like "What are the ingredients in DuoLife Collagen?" or "Which products contain Vitamin C?". It provides structured, unambiguous facts.

By combining these two, DuoBot gets the best of both worlds: the descriptive power of vector search and the factual precision of a knowledge graph.

## In-Depth: How Graph RAG Enhances Standard Vector RAG

Integrating a Knowledge Graph is not just an addition; it's a fundamental enhancement that elevates the agent's capabilities.

| Feature | Standard Vector RAG | Hybrid Graph + Vector RAG |
| :--- | :--- | :--- |
| **Precision** | Retrieves broad text chunks. May require the LLM to "find" the specific fact within a long paragraph. | Retrieves discrete, factual nodes and relationships (e.g., `['Vitamin C', 'L-Arginine']`). Highly precise. |
| **Factual Accuracy** | Higher risk of hallucination if the context is ambiguous or the LLM misinterprets the text. | Grounded in a structured source of truth. The agent gets facts, not just paragraphs, reducing hallucination. |
| **Relational Queries** | Struggles with queries like "Find all products with ingredient X." It can't easily traverse relationships. | Excels at relational queries. It can traverse the graph from an ingredient to all connected products. |
| **Query Understanding** | Relies solely on semantic similarity. | The agent first understands the *intent* of the query (e.g., "this is a request for ingredients") and then executes a targeted graph traversal. |
| **Context Quality** | The context provided to the LLM is unstructured text. | The context is a mix of structured facts (from the graph) and unstructured text (from vector search), giving the LLM a richer, more reliable foundation. |

## Agent Workflow with LangGraph

The agent's decision-making process is modeled as a state machine using LangGraph. This allows for a clear, auditable, and robust workflow.

1.  **Router Node**: The entry point. It analyzes the user's question using an LLM call and decides the best path forward based on the `RouterDecision` model.
    - Is it a general greeting?
    - Does it require a precise fact from the graph?
    - Or is it a general descriptive question?

2.  **Conditional Branching**: Based on the router's decision, the graph transitions to one of three nodes:
    - **`general_question_node`**: For simple greetings ("Hello"). Provides a direct, pre-defined response and ends the process.
    - **`graph_search_node`**: For factual queries. It executes a targeted function call (e.g., `get_ingredients_for_product()`) to retrieve data from the NetworkX graph.
    - **`vector_search_node`**: The default path for most company-related questions.

3.  **Data Retrieval & Enhancement**:
    - The `graph_search_node` populates the agent's state with structured results.
    - The `vector_search_node` then runs. It may use entities found in the graph search to create an "enhanced query" for more relevant document retrieval.

4.  **Answer Generation Node**: This is the final step for complex queries. It takes all the retrieved context—both from the graph and the vector store—and feeds it into the `generate_answer_prompt`. This prompt instructs the LLM to synthesize a comprehensive, conversational answer, ensuring it uses the provided facts and includes sources where appropriate.

## Visualizing the Knowledge Graph

Understanding and debugging a knowledge graph is much easier when you can see it. Yes, there are excellent tools to visualize NetworkX graphs.

A highly recommended tool for easy, interactive visualizations directly from your Python code is **Pyvis**. It creates interactive HTML files that you can pan, zoom, and click on.

**Example Code to Generate a Visualization:**
```python
import networkx as nx
from pyvis.network import Network

# Assuming 'G' is your NetworkX graph object
# G = create_duolife_knowledge_graph() 

net = Network(notebook=True, height="750px", width="100%", bgcolor="#222222", font_color="white")

# Add nodes and edges from your NetworkX graph
net.from_nx(G)

# Customize physics for better layout
net.repulsion(node_distance=420, central_gravity=0.33, spring_length=110, spring_strength=0.10, damping=0.95)

# Show the graph
net.show("duolife_knowledge_graph.html")
```

**Other Powerful Visualization Tools:**
-   **Gephi**: A powerful, open-source desktop application for graph visualization and analysis. You can export your NetworkX graph to a format like GEXF and import it into Gephi for deep exploration.
-   **Cytoscape**: Another popular desktop tool, especially in bioinformatics, but excellent for any kind of network analysis.

## Project Structure

```
.
├── frontend/              # React frontend application
├── graph_RAG/             # Knowledge graph creation and management
│   ├── graph_manager.py
│   └── ...
├── chroma_db/             # Persistent vector store
├── .env                   # Environment variables (API keys)
├── advanced_rag_agent.py  # Core agent logic, prompts, and LangGraph definition
├── main.py                # FastAPI application to serve the agent
├── models.py              # Pydantic models for API requests/responses
├── requirements.txt       # Python dependencies
└── README.md              # This file
```

## Setup and Installation

1.  **Prerequisites**:
    -   Python 3.10+
    -   Node.js and npm (for the frontend)

2.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd <repository-folder>
    ```

3.  **Install Python dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up Environment Variables**:
    -   Create a file named `.env` in the root directory.
    -   Add your API keys to this file:
        ```
        OPENAI_API_KEY="sk-..."
        # Add any other required keys
        ```

5.  **Build the Frontend**:
    ```bash
    cd frontend
    npm install
    npm run build
    cd ..
    ```

6.  **Run the API