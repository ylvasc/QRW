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






def sample_cost_landscape(G, gamma, t, n, shots=1024, memory=False):

    for i in range(len(t)):
        for j in range(len(gamma)):
            gamma_i = gamma[j]
            t_i = t[i]

            circ = trotterized_qrw_circuit(G, t_1, gamma_i, n)
            

           
            job = Aer.get_backend('aer_simulator').run(
                circ,
                shots=shots,
                optimization_level=0,
                memory=memory,
            )

            
            stats = measurement_statistics(job)

    return 


import numpy as np

def measurement_statistics(job, stat, problem, memorysize, memory_lists):

    jres = job.result()
    counts_list = jres.get_counts()

    
    if memorysize > 0:
        for i, _ in enumerate(jres.results):
            memory_list = jres.get_memory(experiment=i)
            for measurement in memory_list:
                memory_lists.append(
                    [measurement, problem.cost(measurement[::-1])]
                )
                memorysize -= 1
                if memorysize < 1:
                    break

    expectations = []
    variances = []
    maxcosts = []
    mincosts = []

    if isinstance(counts_list, list):
        for i, counts in enumerate(counts_list):
            stat.reset()
            for string in counts:
                
                cost = problem.cost(string[::-1])
                stat.add_sample(cost, counts[string], string[::-1])
            expectations.append(stat.get_CVaR())
            variances.append(stat.get_Variance())
            maxcosts.append(stat.get_max())
            mincosts.append(stat.get_min())
        
        
        return {
            "Expectations": -np.array(expectations),
            "Variances": np.array(variances),
            "MaxCosts": -np.array(maxcosts),
            "MinCosts": -np.array(mincosts)
        }
    else:
        for string in counts_list:
            cost = problem.cost(string[::-1])
            stat.add_sample(cost, counts_list[string], string[::-1])
    
    
    return {
        "Expectations": [],
        "Variances": [],
        "MaxCosts": [],
        "MinCosts": []
    }
