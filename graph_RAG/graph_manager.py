# graph_RAG/graph_manager.py

import os
import networkx as nx
from typing import List, Dict

# --- Robust Path Construction ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRAPH_FILE_PATH = os.path.join(PROJECT_ROOT, "graph_db", "duolife_knowledge_graph.graphml")

def load_graph(path: str = GRAPH_FILE_PATH) -> nx.DiGraph:
    """Loads a NetworkX graph from a GraphML file if it exists."""
    if os.path.exists(path):
        print(f"Loading knowledge graph from {path}...")
        return nx.read_graphml(path)
    else:
        print(f"Error: Saved graph file not found at {path}. Please run your data prep notebook first.")
        return None

def get_ingredients_for_product(graph: nx.DiGraph, product_name: str) -> List[str]:
    """Finds a product by name and returns a list of its ingredients."""
    product_node_id = None
    for node, data in graph.nodes(data=True):
        if data.get('name') and data['name'].lower() == product_name.lower():
            product_node_id = node
            break
    if not product_node_id: return []
    return [graph.nodes[successor]['name'] for successor in graph.successors(product_node_id)]

def find_products_containing_ingredient(graph: nx.DiGraph, ingredient_name: str) -> List[Dict]:
    """Finds an ingredient by name and returns products/cosmetics that contain it."""
    ingredient_node_id = None
    for node, data in graph.nodes(data=True):
        if data.get('type') == 'Ingredient' and data.get('name', '').lower() == ingredient_name.lower():
            ingredient_node_id = node
            break
    if not ingredient_node_id: return []
    return [graph.nodes[predecessor] for predecessor in graph.predecessors(ingredient_node_id)]

# --- Global graph instance ---
duolife_kg = load_graph()

# =================================================================
# --- Test Block: Run this file directly to test the KG ---
# =================================================================
if __name__ == '__main__':
    print("\n--- Testing Knowledge Graph Manager ---")
    if duolife_kg is None:
        print("Could not run tests because the knowledge graph file was not found.")
    else:
        print(f"Graph loaded with {duolife_kg.number_of_nodes()} nodes and {duolife_kg.number_of_edges()} edges.")

        # --- Test Case 1: Find ingredients for a valid product ---
        product_to_find = "DuoLife Collagen"
        print(f"\n[TEST] Getting ingredients for: '{product_to_find}'")
        ingredients = get_ingredients_for_product(duolife_kg, product_to_find)
        if ingredients:
            print(f"[SUCCESS] Found {len(ingredients)} ingredients.")
            # print(ingredients)
        else:
            print("[FAIL] No ingredients found.")

        # --- Test Case 2: Find products containing a valid ingredient ---
        ingredient_to_find = "Shea Butter"
        print(f"\n[TEST] Finding products with: '{ingredient_to_find}'")
        products = find_products_containing_ingredient(duolife_kg, ingredient_to_find)
        if products:
            print(f"[SUCCESS] Found {len(products)} products.")
            # print(products)
        else:
            print("[FAIL] No products found.")
        
        # --- Test Case 3: Test a failing case ---
        bad_product = "Non-existent Product"
        print(f"\n[TEST] Getting ingredients for: '{bad_product}'")
        ingredients = get_ingredients_for_product(duolife_kg, bad_product)
        if not ingredients:
            print(f"[SUCCESS] Correctly found no ingredients for a non-existent product.")
        else:
            print("[FAIL] Found ingredients for a product that should not exist.")