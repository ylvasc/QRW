#maxcut quantum circ
from qiskit import QuantumCircuit, QuantumRegister
from qiskit_aer import Aer
from qiskit import transpile, assemble
from qiskit.primitives import Sampler 
from qiskit.circuit import Parameter
from itertools import combinations
from qiskit_optimization.applications import Maxcut
from qiskit_optimization.algorithms.qrao import QuantumRandomAccessEncoding
import numpy as np
import networkx as nx
from qiskit.quantum_info import SparsePauliOp, Pauli
from scipy.sparse.linalg import expm
import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram
from mpl_toolkits.axes_grid1 import make_axes_locatable

from qiskit_aer import AerSimulator
from qiskit.quantum_info import Pauli, Statevector
from qiskit.circuit import QuantumCircuit, QuantumRegister


#make initial states
def plusStates(n):  #n= number of qubits
        q = QuantumRegister(n)
        circ = QuantumCircuit(q)
        circ.h(q)

def problemHamiltonian(p):  #takes in problem made by encode
    qubitOp, offset = p.to_ising() #outputs SparsePauliOp
    print("Offset:", offset)
    print("Ising Hamiltonian:")
    print(str(qubitOp))
    
    return qubitOp, offset
#makes an ising hamiltonian -> how to translate this together with hypercube hamiltonian in circuit?




def problemHamiltonian_qiskit(graph):
    """
    Calculate the Hamiltonian for a weighted Max-Cut problem as a SparsePauliOp for Qiskit.

    Parameters:
    graph (nx.Graph): A NetworkX graph with weights on the edges.

    Returns:
    SparsePauliOp: The Hamiltonian for the Max-Cut problem.
    """
    n = graph.number_of_nodes()
    
    # Initialize the Hamiltonian as an empty SparsePauliOp
    H_cost = SparsePauliOp.from_list([], num_qubits=n)  # Start with an empty SparsePauliOp
    
    # Iterate through each edge in the graph
    for i, j, data in graph.edges(data=True):
        weight = data.get('weight', 1.0)  # Default weight to 1 if not specified
        
        # Create the Pauli string for the edge (i, j)
        pauli = ['I'] * n
        pauli[i] = 'Z'  # Z operator on node i
        pauli[j] = 'Z'  # Z operator on node j
        pauli_str = ''.join(pauli)

        # Create the SparsePauliOp for the ZZ term
        zz_term = SparsePauliOp.from_list([(pauli_str, 1)])  # The ZZ term contribution

        # Create the identity term
        identity_term = SparsePauliOp.from_list([('I' * n, 1)])  # Identity term for n qubits
        
        # Create the Hamiltonian contribution as I - ZZ
        hamiltonian_term = identity_term - zz_term

        # Multiply by the weight and 0.5
        weighted_term = hamiltonian_term * (-0.5 * weight)
        
        # Add the current weighted term to the Hamiltonian
        H_cost += weighted_term

    return H_cost


    


# Updated walk (hypercube) Hamiltonian function for use in Qiskit circuits
def hypercubeHamiltonian_qiskit(n):
    """
    Create the walk Hamiltonian as a SparsePauliOp for Qiskit based on a hypercube with dimension 2**n.

    Parameters:
    n: Number of qubits/nodes.

    Returns:
    SparsePauliOp: The walk Hamiltonian for the hypercube.
    """
    pauli_terms = []  # Initialize a list to hold the Pauli terms
    coeffs = []  # List to hold the coefficients for each Pauli term
    
    # Add the terms for the Pauli-X walk operator
    for i in range(n):
        pauli_terms.append("X" + "I" * (n - 1))  # Create Pauli term
        coeffs.append(-1)  # Coefficient for each term
    
    # Create the SparsePauliOp from the list of terms and coefficients
    H_walk = SparsePauliOp.from_list(list(zip(pauli_terms, coeffs)))

    return H_walk


def trotterized_qrw_circuit(G, t, gamma, n):
    """
    Creates a Trotterized QRW circuit for the MaxCut problem.

    Parameters:
    G (networkx.Graph): Weighted networkx graph representing the problem instance.
    t (float): Evolution time.
    gamma (float): Walk parameter.
    n (int): Number of Trotter steps.
    
    Returns:
    QuantumCircuit: The parameterized quantum circuit for the QRW MaxCut problem.
    """
    N_qubits = len(G.nodes)  
    q = QuantumRegister(N_qubits)
    qc = QuantumCircuit(q)
    #initial state
    for qubit in qc.qubits:
        qc.h(qubit)

    # Define the cost Hamiltonian
    def apply_cost_hamiltonian(qc, G, cost_param):
        for edge in G.edges():
            i, j = int(edge[0]), int(edge[1])
            w = G[edge[0]][edge[1]]["weight"]
            wg = w * cost_param
            qc.cx(q[i], q[j])
            qc.rz(wg, q[j])
            qc.cx(q[i], q[j])

    # Define the walk Hamiltonian
    def apply_walk_hamiltonian(qc, walk_param):
        qc.rx(-2 * walk_param, range(N_qubits))

    # Trotterization loop
    for i in range(n):
        apply_cost_hamiltonian(qc, G, t / n)      #
        qc.barrier()
        apply_walk_hamiltonian(qc, gamma * t / n)  
        qc.barrier()

    return qc


def compute_expectation_value(qc, G, H_cost):

    simulator = Aer.get_backend('statevector_simulator')
    result = simulator.run(qc).result()
    statevector = result.get_statevector(qc)
    
    # Now calculate the expectation value of the cost Hamiltonian
    expectation_value = np.real(np.dot(statevector.conj().T, np.dot(H_cost, statevector)))
    
    return expectation_value

def cost_landscape_trotterized_qw(G, gamma, t, n, H_cost, offset=0, fig=None, plot=True):
    exp_val = np.zeros((len(t), len(gamma)))
    
    # Loop over every combination of t and gamma and calculate exp_val
    for i in range(len(t)):
        for j in range(len(gamma)):
            t_value = t[i] # Current t value, made into array object
            gamma_value = gamma[j]  # Current gamma value, made into array object
            qc=trotterized_qrw_circuit(G, t_value, gamma_value, n)
            exp_val[i, j] = compute_expectation_value(qc, G, H_cost)
    if plot==True:
        if fig is None:
            fig = plt.figure(figsize=(6, 6), dpi=80, facecolor="w", edgecolor="k")
            
        ax = fig.gca()
        ax.set_xlabel(r"$t$")           
        ax.set_ylabel(r"$\gamma$")      
        ax.set_title("Cost Landscape")
        
        
        im = ax.imshow(exp_val+offset, interpolation="bicubic", origin="lower", 
                    extent=[t[0], t[-1], gamma[0], gamma[-1]], aspect='auto')

        # Add colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

        plt.show()
    return exp_val



