#maxcut quantum circ
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import Parameter
from itertools import combinations
from qiskit_optimization.applications import Maxcut
from qiskit_optimization.algorithms.qrao import QuantumRandomAccessEncoding
import numpy as np
import networkx as nx
from qiskit.quantum_info import SparsePauliOp

#make initial states
def plus_states(n):  #n= number of qubits
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


#or use problem Hamiltonian from QAOA??


#define hypercube Hamiltonian
