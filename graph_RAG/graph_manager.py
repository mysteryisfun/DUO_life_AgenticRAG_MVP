import networkx as nx
from typing import List

# --- Load the graph once when the module is imported ---
GRAPH_FILE_PATH = "graph_db/duolife_knowledge_graph.graphml"
try:
    G = nx.read_graphml(GRAPH_FILE_PATH)
    print("Knowledge graph loaded successfully.")
except FileNotFoundError:
    G = nx.Graph() # Create an empty graph if file not found
    print(f"Warning: Knowledge graph file not found at {GRAPH_FILE_PATH}. A new empty graph has been created.")


def get_ingredients_for_product(product_name: str) -> List[str]:
    """Finds all ingredients for a given product."""
    ingredients = []
    # Normalize the input product name to match the graph's 'name' attribute format
    normalized_product_name = product_name.lower()

    for node, data in G.nodes(data=True):
        # Check if the node is a product and its name matches
        if data.get('type') == 'Product' and data.get('name', '').lower() == normalized_product_name:
            # Find neighbors connected by a 'CONTAINS' edge
            for neighbor in G.neighbors(node):
                if G.get_edge_data(node, neighbor).get('relation') == 'CONTAINS':
                    ingredients.append(G.nodes[neighbor].get('name', 'Unknown Ingredient'))
            break # Stop after finding the product
    return ingredients

def find_products_containing_ingredient(ingredient_name: str) -> List[str]:
    """Finds all products that contain a given ingredient."""
    products = []
    # Normalize the input ingredient name
    normalized_ingredient_name = ingredient_name.lower()

    for node, data in G.nodes(data=True):
        # Check if the node is an ingredient and its name matches
        if data.get('type') == 'Ingredient' and data.get('name', '').lower() == normalized_ingredient_name:
            # Find neighbors connected by a 'CONTAINS' edge (reverse direction)
            for neighbor in G.neighbors(node):
                # The edge is from Product to Ingredient
                if G.get_edge_data(neighbor, node).get('relation') == 'CONTAINS':
                    products.append(G.nodes[neighbor].get('name', 'Unknown Product'))
            break # Stop after finding the ingredient
    return products