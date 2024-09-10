
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import itertools

def create_graph(nodes, w_edges):
    G = nx.Graph()
    G.add_nodes_from(nodes)
    G.add_weighted_edges_from(w_edges)
    return G

# Calculate the classical Max-Cut value (not using quantum here)
def classical_max_cut(G):
    # Extract nodes and their weights
    nodes = list(G.nodes())
    node_weights = nx.get_node_attributes(G, 'weight')
    
    # Initialize best cut
    best_cut_value = float('-inf')
    best_cut = (set(), set())
    
    # Try all possible partitions
    for cut in itertools.product([0, 1], repeat=len(nodes)):
        subset1 = {nodes[i] for i in range(len(nodes)) if cut[i] == 0}
        subset2 = {nodes[i] for i in range(len(nodes)) if cut[i] == 1}
        
        # Calculate cut value
        cut_value = 0
        for u, v in G.edges():
            if (u in subset1 and v in subset2) or (u in subset2 and v in subset1):
                edge_weight = G[u][v].get('weight', 1)
                node_weight_u = node_weights.get(u, 1)
                node_weight_v = node_weights.get(v, 1)
                cut_value += edge_weight * (node_weight_u + node_weight_v)
        
        # Update best cut
        if cut_value > best_cut_value:
            best_cut_value = cut_value
            best_cut = (subset1, subset2)
    
    return best_cut, best_cut_value


    return best_cut, best_cut_value

def visualize_graph(G, cut):
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray')
    cut_edges = [(u, v) for u, v in G.edges() if (u in cut) != (v in cut)]
    nx.draw_networkx_edges(G, pos, edgelist=cut_edges, edge_color='red', width=2)
    plt.show()


