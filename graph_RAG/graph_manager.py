import networkx as nx
from typing import List

# --- Load the graph once when the module is imported ---
GRAPH_FILE_PATH = "graph_db/duolife_knowledge_graph.graphml"
try:
    G = nx.read_graphml(GRAPH_FILE_PATH)
    print("Knowledge graph loaded successfully.")
except FileNotFoundError:
    G = nx.Graph()
    print(f"Warning: Knowledge graph file not found at {GRAPH_FILE_PATH}. A new empty graph has been created.")

def get_ingredients_for_product(product_name: str) -> List[str]:
    """Finds all ingredients for a given product."""
    ingredients = []
    normalized_product_name = product_name.lower()

    for node, data in G.nodes(data=True):
        if data.get('type') == 'Product' and data.get('name', '').lower() == normalized_product_name:
            for neighbor in G.neighbors(node):
                if G.get_edge_data(node, neighbor).get('relation') == 'CONTAINS':
                    ingredients.append(G.nodes[neighbor].get('name', 'Unknown Ingredient'))
            break
    return ingredients

def find_products_containing_ingredient(ingredient_name: str) -> List[str]:
    """Finds all products that contain a given ingredient."""
    products = []
    normalized_ingredient_name = ingredient_name.lower()

    for node, data in G.nodes(data=True):
        if data.get('type') == 'Ingredient' and data.get('name', '').lower() == normalized_ingredient_name:
            for neighbor in G.neighbors(node):
                if G.get_edge_data(neighbor, node).get('relation') == 'CONTAINS':
                    products.append(G.nodes[neighbor].get('name', 'Unknown Product'))
            break
    return products

def get_product_links(product_name: str) -> List[str]:
    """Gets the links for a specific product from the graph."""
    links = []
    normalized_product_name = product_name.lower()
    
    for node, data in G.nodes(data=True):
        if data.get('type') == 'Product' and data.get('name', '').lower() == normalized_product_name:
            link = data.get('link', '')
            if link:
                links.append(link)
            break
    return links