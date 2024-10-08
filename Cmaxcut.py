#non circuit calculations
import numpy as np
import scipy.sparse.linalg as linalgs
import Qmaxcut
import importlib
from qiskit.quantum_info import SparsePauliOp
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
importlib.reload(Qmaxcut)

def problemHamiltonianFromIsing(p):  #using inbuilt .to_ising() function
    """
    Create the problem (cost) Hamiltonian H_c by making the problem into a qubo with inbuilt functions. 
    Then uses inbuilt function .to_ising() to create Ising Hamiltionian, translated into matrix form.

    Parameters:
    - p: Qubo problem

    Returns:
    - H_p: numpy array representing the cost Hamiltonian matrix
    """
    qubitOp, offset=Qmaxcut.problemHamiltonian(p)
    H_p = qubitOp.to_matrix()
    return H_p, offset



def problemHamiltonian(graph):
    """
    Calculate the Hamiltonian matrix for a weighted Max-Cut problem using SparsePauliOp.

    Parameters:
    graph (nx.Graph): A NetworkX graph with weights on the edges.

    Returns:
    np.ndarray: The Hamiltonian matrix for the Max-Cut problem.
    """
    n = graph.number_of_nodes()
    
    # Initialize the Hamiltonian as zero matrix
    hamiltonian_matrix = np.zeros((2**n,2**n), dtype=complex)
    #print(hamiltonian_matrix.shape)
    
    # Iterate through each edge in the graph
    for i, j, data in graph.edges(data=True):
        weight = data.get('weight', 1.0)  # Default weight to 1 if not specified
        
        # Create the Pauli string for the edge (i, j)
        pauli = ['I'] * n
        pauli[i] = 'Z'  # Z operator on node i
        pauli[j] = 'Z'  # Z operator on node j
        pauli_str = ''.join(pauli)

        # Create the SparsePauliOp for the current edge
        pauli_op = SparsePauliOp.from_list([(pauli_str, 1)])  # Coefficient is 1

        # Convert to matrix and multiply by the weight
        pauli_matrix = pauli_op.to_matrix()
        
        # Create identity matrix
        identity_matrix = np.eye(pauli_matrix.shape[0], dtype=complex)
        
        # Make I-ZZ term for qubits i and j
        matrix = identity_matrix - pauli_matrix
        
        # Create total matrix
        hamiltonian_matrix += matrix*0.5*weight

    return hamiltonian_matrix*(-1) #-1 for minimization


def hypercubeHamiltonian(n):  #hypercube walk hamiltonian
    """
    Create the walk Hamiltonian based on a hypercube with dimension 2**n x 2**n 

    Parameters:
    - n: number of nodes

    Returns:
    - H: numpy array representing the walk Hamiltonian matrix
    """
    X = np.array([[0, 1], [1, 0]])
    dim = 2**n
    H = np.zeros((dim, dim), dtype=np.float64)

    # Sum the tensor products for each term in the Hamiltonian
    for i in range(n):
        # Create the term: I^{⊗ i} ⊗ X ⊗ I^{⊗ (N-i-1)}
        # create the list of matrices
        matrices = [np.eye(2)] * n
        matrices[i] = X  # Replace the i-th entry with the Pauli-X matrix
        
        # Take the tensor product of the matrices
        term = matrices[0]
        for j in range(1, n):
            term = np.kron(term, matrices[j])
        H -= term
    return H

def initialState(n):  #make |psi_0> = |+> ^(⊗n)
	plus_state = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
	state = plus_state
	for i in range(1, n):
		state = np.kron(state, plus_state)
	return state
	
def QWStep(H_cost, H_walk, gamma, t, initial_state):
	output_state = linalgs.expm_multiply(-1j*(gamma*H_cost+H_walk)*t, initial_state) 
	return output_state

def QW(H_cost, H_walk, t, initial_state, gamma):  
    #QW with gamma array and time array as input, depth = length of t and gamma
	state = initial_state
	exp_val=np.zeros(len(gamma))
	for i in range(len(gamma)): #iterate over all gammas
		gamma_step = gamma[i]
		#for step in range(steps):
		t_step = t[i]
		state = QWStep(H_cost, H_walk, gamma_step, t_step, state)
		exp_val[i] = np.real(state.conj().T@H_cost@state)
        #return state?
	return exp_val

def costLandscape(H_cost, H_walk, t, initial_state, gamma, offset, fig=None): #only for depth 1!!!! 
    """
    Calculate and plot the cost landscape.

    Parameters:
    - t: 1D array of t values.
    - gamma: 1D array of gamma values.
    - H_cost: Problem Hamiltonian.
    - H_walk: Walk Hamiltonian
    - offset: Offset calculated from to_ising() function
    """
    # Create a 2D array to store exp_val
    exp_val = np.zeros((len(t), len(gamma)))
    
    # Loop over every combination of t and gamma and calculate exp_val
    for i in range(len(t)):
        for j in range(len(gamma)):
            t_value = np.array([t[i]]) # Current t value, made into array object
            gamma_value = np.array([gamma[j]])  # Current gamma value, made into array object
            exp_val[i, j] = QW(H_cost, H_walk, t_value, initial_state, gamma_value)

    if fig is None:
        fig = plt.figure(figsize=(6, 6), dpi=80, facecolor="w", edgecolor="k")
        
    ax = fig.gca()
    ax.set_xlabel(r"$\gamma$")
    ax.set_ylabel(r"$t$")
    ax.set_title("Cost Landscape")
    
    # Prepare the data for imshow
    im = ax.imshow(exp_val + offset, interpolation="bicubic", origin="lower", 
                   extent=[gamma[0], gamma[-1], t[0], t[-1]], aspect='auto')

    # Add colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)

    plt.show()

    # Create the heatmap plot
    #plt.figure(figsize=(8, 6))
    #plt.imshow(exp_val+offset, aspect='auto', origin='lower', 
               #extent=[t[0], t[-1], gamma[0], gamma[-1]], cmap='viridis')
    #plt.colorbar(label='Expectation Values')
    #plt.xlabel('t')
    #plt.ylabel('gamma')
    #plt.title('Cost Landscape')
    #plt.show()






