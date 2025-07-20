# graph_manager.py

import json
import os
import networkx as nx
from typing import List, Dict

# Define a standard file path for the saved graph
GRAPH_FILE_PATH = "graph_db/duolife_knowledge_graph.graphml"

def create_graph_from_files(nodes_path: str = 'graph_db/nodes.json', edges_path: str = 'graph_db/edges.json') -> nx.DiGraph:
    """
    Builds a NetworkX directed graph from JSON files.
    """
    print("Building Knowledge Graph from source JSON files...")
    G = nx.DiGraph()

    with open(nodes_path, 'r', encoding='utf-8') as f:
        nodes = json.load(f)
    for node_data in nodes:
        node_id = node_data['id']
        attributes = {k: v for k, v in node_data.items() if k != 'id'}
        G.add_node(node_id, **attributes)

    with open(edges_path, 'r', encoding='utf-8') as f:
        edges = json.load(f)
    for edge_data in edges:
        G.add_edge(edge_data['source'], edge_data['target'], label=edge_data['label'])
    
    print(f"Graph built successfully with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return G

def save_graph(graph: nx.DiGraph, path: str = GRAPH_FILE_PATH):
    """Saves the NetworkX graph to a file using GraphML format."""
    print(f"Saving graph to {path}...")
    nx.write_graphml(graph, path)
    print("Graph saved successfully.")

def load_graph(path: str = GRAPH_FILE_PATH) -> nx.DiGraph:
    """Loads a NetworkX graph from a GraphML file if it exists."""
    if os.path.exists(path):
        print(f"Loading graph from {path}...")
        return nx.read_graphml(path)
    else:
        print("Saved graph file not found.")
        return None

def get_ingredients_for_product(graph: nx.DiGraph, product_name: str) -> List[str]:
    """Finds a product by name and returns a list of its ingredients."""
    product_node_id = None
    # --- DEBUG ---
    print(f"\n[DEBUG] Searching for product with name: '{product_name}'")
    
    for node, data in graph.nodes(data=True):
        if data.get('name') and data['name'].lower() == product_name.lower():
            product_node_id = node
            # --- DEBUG ---
            print(f"[DEBUG] Found matching node ID: '{product_node_id}'")
            break

    if not product_node_id:
        return ["Product not found."]

    successors = list(graph.successors(product_node_id))
    # --- DEBUG ---
    print(f"[DEBUG] Found {len(successors)} ingredients (successors) for this node.")
    
    ingredients = [graph.nodes[successor]['name'] for successor in successors]
    return ingredients

def find_products_containing_ingredient(graph: nx.DiGraph, ingredient_name: str) -> List[Dict]:
    """Finds an ingredient by name and returns products/cosmetics that contain it."""
    ingredient_node_id = None
    search_name_lower = ingredient_name.lower()
    
    # --- DEBUG ---
    print(f"\n[DEBUG] Searching for ingredient with name: '{ingredient_name}' (normalized to: '{search_name_lower}')")
    
    for node, data in graph.nodes(data=True):
        if data.get('type') == 'Ingredient' and data.get('name', '').lower() == search_name_lower:
            ingredient_node_id = node
            # --- DEBUG ---
            print(f"[DEBUG] Found matching ingredient node ID: '{ingredient_node_id}'")
            break
            
    if not ingredient_node_id:
        return [{"message": "Ingredient not found."}]
        
    predecessors = list(graph.predecessors(ingredient_node_id))
    # --- DEBUG ---
    print(f"[DEBUG] Found {len(predecessors)} products (predecessors) containing this ingredient.")
    
    product_nodes = [graph.nodes[predecessor] for predecessor in predecessors]
    return product_nodes


# This block demonstrates the new workflow
if __name__ == '__main__':
    # --- New Workflow: Load first, build only if necessary ---
    duolife_kg = load_graph()
    
    if duolife_kg is None:
        # If no saved graph exists, build it from the JSON files
        duolife_kg = create_graph_from_files()
        # And save it for next time
        save_graph(duolife_kg)

    # --- Run the same tests with debugging ---
    print("\n--- Testing Graph Queries ---")

    # Test Case 1: Find ingredients for DuoLife Collagen
    ingredients = get_ingredients_for_product(duolife_kg, "DuoLife Collagen")
    print(f"\nIngredients in 'DuoLife Collagen':")
    print(ingredients if ingredients else "None found.")

    # Test Case 2: Find products containing Shea Butter (with debugging)
    products = find_products_containing_ingredient(duolife_kg, "Shea Butter")
    print(f"\nProducts containing 'Shea Butter':")
    print(products if products else "None found.")