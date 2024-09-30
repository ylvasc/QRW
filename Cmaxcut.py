#non circuit calculations
import numpy as np
import scipy.sparse.linalg as linalgs
import Qmaxcut
import importlib
importlib.reload(Qmaxcut)

def problemHamiltonian(p):
	qubitOp, offset=Qmaxcut.problemHamiltonian(p)
	hamiltonian_matrix = qubitOp.to_matrix()
	hamiltonian_matrix += hamiltonian_matrix + offset * np.eye(hamiltonian_matrix.shape[0])
	return hamiltonian_matrix

def hypercubeHamiltonian(n):  #hypercube walk hamiltonian
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

def QW(H_cost, H_walk, t, initial_state, gamma, steps):  #QW with gamma array and time array as input
	state = initial_state
	exp_val=np.zeros(len(gamma))
	for i in range(len(gamma)): #iterate over all gammas
		gamma_step = gamma[i]
		#for step in range(steps):
		t_step = t[i]
		state = QWStep(H_cost, H_walk, gamma_step, t_step, state)
		exp_val[i] = np.real(state.conj().T@H_cost@state)
	return state, exp_val
	



