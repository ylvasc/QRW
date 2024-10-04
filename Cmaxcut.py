#non circuit calculations
import numpy as np
import scipy.sparse.linalg as linalgs
import Qmaxcut
import importlib
import matplotlib.pyplot as plt
importlib.reload(Qmaxcut)

def problemHamiltonian(p):  #using inbuilt .to_ising() function
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

def problemHamiltonianMatrix(G): 
    """
    Create the cost Hamiltonian H_c = 0.5 * sum_(i,j) w_(ij) (I - Z_i Z_j)
    from graph.

    Parameters:
    - G: NetworkX graph, where edges have weights

    Returns:
    - H_p: numpy array representing the cost Hamiltonian matrix
    """
    n = G.number_of_nodes()
    dim = 2 ** n  #Dimension of the Hamiltonian matrix

    H_p = np.zeros((dim, dim)) #Initialize matrix

    for (i, j, weight) in G.edges(data=True):
        weight = weight.get('weight', 1.0)  # Default to 1.0 if no weight is specified
        
        zz_term = np.zeros((dim, dim)) #Contribution of the ZZ term
        
        for k in range(dim):
            # Calculate the basis state
            state1 = (k >> i) & 1  #Value of the i-th qubit
            state2 = (k >> j) & 1  #Value of the j-th qubit
            
            if state1 == 0 and state2 == 0:
                zz_term[k, k] = 1  # |00> state
            elif state1 == 1 and state2 == 1:
                zz_term[k, k] = 1  # |11> state
            else:
                zz_term[k, k] = 0  # Mixed states |01> and |10> yield no contribution

        # Add the term to the Hamiltonian
        H_p += 0.5 * weight * (np.eye(dim) - zz_term)

    return H_p


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

def costLandscape(H_cost, H_walk, t, initial_state, gamma, offset): #only for depth 1!!!! 
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

    # Create the heatmap plot
    plt.figure(figsize=(8, 6))
    plt.imshow(exp_val+offset, aspect='auto', origin='lower', 
               extent=[t[0], t[-1], gamma[0], gamma[-1]], cmap='viridis')
    plt.colorbar(label='Expectation Values')
    plt.xlabel('t')
    plt.ylabel('gamma')
    plt.title('Cost Landscape')
    plt.show()






