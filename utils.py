
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import itertools
from qiskit_algorithms import SamplingVQE, NumPyMinimumEigensolver
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit_optimization.applications import Maxcut
from qiskit_optimization.algorithms.qrao import QuantumRandomAccessEncoding

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


def visualize_graph(G):
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightblue', edge_color='gray')
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels)

    plt.show()

def classical_max_cut_inbuilt(problem):
    exact = MinimumEigenOptimizer(NumPyMinimumEigensolver())
    result = exact.solve(problem)
    print(result.prettyprint())

def encode(G):
# Computing the weight matrix from the random graph
    nodes = list(G.nodes())
    n=len(nodes)
    w = np.zeros([n, n])
    for i in range(n):
        for j in range(n):
            temp = G.get_edge_data(i, j, default=0)
            if temp != 0:
                w[i, j] = temp["weight"]
    print(w)
    max_cut = Maxcut(w)
    max_cut_problem = max_cut.to_quadratic_program()
    print(max_cut_problem.prettyprint())
    return max_cut_problem
    

def problemHamiltonian(p):  #takes in problem made by encode
    qubitOp, offset = p.to_ising()
    print("Offset:", offset)
    print("Ising Hamiltonian:")
    print(str(qubitOp))