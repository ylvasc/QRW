#non circuit calculations
import numpy as np
import scipy.sparse.linalg as linalgs
import Qmaxcut
import importlib
from qiskit.quantum_info import SparsePauliOp
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
importlib.reload(Qmaxcut)
import utils
import pyvista as pv



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
	output_state = linalgs.expm_multiply(-1j*(gamma*H_walk+H_cost)*t, initial_state) 
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

def costLandscape(H_cost, H_walk, t, initial_state, gamma, offset, fig=None, plot=True): #only for depth 1!!!! 
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
    if plot==True:
        if fig is None:
            fig = plt.figure(figsize=(6, 6), dpi=80, facecolor="w", edgecolor="k")
            
        ax = fig.gca()
        ax.set_xlabel(r"$t$")           
        ax.set_ylabel(r"$\gamma$")      
        ax.set_title("Cost Landscape")
        
        
        im = ax.imshow(exp_val + offset, interpolation="bicubic", origin="lower", 
                    extent=[t[0], t[-1], gamma[0], gamma[-1]], aspect='auto')

        # Add colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

        plt.show()
    return exp_val

        # Create the heatmap plot
        #plt.figure(figsize=(8, 6))
        #plt.imshow(exp_val+offset, aspect='auto', origin='lower', 
                #extent=[t[0], t[-1], gamma[0], gamma[-1]], cmap='viridis')
        #plt.colorbar(label='Expectation Values')
        #plt.xlabel('t')
        #plt.ylabel('gamma')
        #plt.title('Cost Landscape')
        #plt.show()

def trotterizedQRW(H_cost, H_walk, t, initial_state, gamma, n):
    U_cost = linalgs.expm(-1j * (H_cost) * (t / n))
    U_walk = linalgs.expm(-1j * (gamma * H_walk) * (t / n))

    # Initialize the output state as the initial state
    output_state = initial_state.copy()

    # Apply the Trotterized evolution
    for _ in range(n):
        output_state = U_cost.dot(output_state)
        output_state = U_walk.dot(output_state)

    exp_val = np.real(output_state.conj().T@H_cost@output_state)

    return exp_val

def trotterizedCostLandscape(H_cost, H_walk, t, initial_state, gamma, offset, n, fig=None, plot=True):
    # Create a 2D array to store exp_val
    exp_val = np.zeros((len(t), len(gamma)))
    
    # Loop over every combination of t and gamma and calculate exp_val
    for i in range(len(t)):
        for j in range(len(gamma)):
            t_value = np.array([t[i]]) # Current t value, made into array object
            gamma_value = np.array([gamma[j]])  # Current gamma value, made into array object
            exp_val[i, j] = trotterizedQRW(H_cost, H_walk, t_value, initial_state, gamma_value, n)
    if plot==True:
        if fig is None:
            fig = plt.figure(figsize=(6, 6), dpi=80, facecolor="w", edgecolor="k")
            
        ax = fig.gca()
        ax.set_xlabel(r"$t$")           
        ax.set_ylabel(r"$\gamma$")      
        ax.set_title("Cost Landscape")
        
        
        im = ax.imshow(exp_val + offset, interpolation="bicubic", origin="lower", 
                    extent=[t[0], t[-1], gamma[0], gamma[-1]], aspect='auto')

        # Add colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

        plt.show()
    return exp_val

def trotterizationConvergenceTest(num_graphs, num_nodes, n):
    #num graph: how many test graphs
    #num_nodes: how many nodes per test graph
    #n: how far to take trotterization
    norms= None
    x = np.arange(1,n+1)
    print(x.shape)
    print(x)
    for i in range(num_graphs):
        diff= np.zeros(n)
        G = utils.createRandomGraph(num_nodes, 0.5) #create a test graph
        H_cost = problemHamiltonian(G)
        H_walk = hypercubeHamiltonian(num_nodes)
        initial_state = initialState(num_nodes)
        gamma = np.linspace(0, 4, 100)
        t = np.linspace(0, 6, 100)
        test_exp_val = costLandscape(H_cost, H_walk, t, initial_state, gamma, 0, fig=None, plot=False) #calculate cost landscape for test graph
        for i in range(1,n+1):
            compare_exp_val = trotterizedCostLandscape(H_cost, H_walk, t, initial_state, gamma, 0, i, fig=None, plot=False)
            difference=test_exp_val-compare_exp_val
            diff_norm = np.linalg.norm(difference, ord='fro')
            diff[i-1]= diff_norm
        if norms == None:
             norms = diff
        else: 
            norms+=diff
    average_norm = norms/num_graphs
    plt.figure(figsize=(8, 6))
    
    # Plot the data with a muted dark green-blue color
    plt.plot(x, average_norm, color='#2a536b', linewidth=2)
    
    # Label the axes
    plt.xlabel('Trotterization depth', fontsize=12)
    plt.ylabel('Difference to QRW result, Frobenius norm', fontsize=12)
    
    # Set the title
    plt.title(f'Difference of Trotterized circuit to QRW circuit, average over {num_graphs} graphs', fontsize=14)
    
    # Show grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Display the plot
    plt.show()

def trotterizationConvergenceTest2(num_graphs, num_nodes, n):
    """
    Perform Trotterization convergence test over multiple graphs and plot the results with variance.

    Parameters:
    - num_graphs: Number of test graphs.
    - num_nodes: Number of nodes per test graph.
    - n: Maximum Trotterization depth to test.
    """
    # Initialize list to store norms for each graph
    all_norms = []
    
    # x-axis values for Trotterization depths
    x = np.arange(1, n+1)

    # Loop over num_graphs test graphs
    for _ in range(num_graphs):
        diff = np.zeros(n)
        
        # Create a random graph and associated Hamiltonians
        G = utils.createRandomGraph(num_nodes, 0.5)
        H_cost = problemHamiltonian(G)
        H_walk = hypercubeHamiltonian(num_nodes)
        initial_state = initialState(num_nodes)
        gamma = np.linspace(0, 4, 100)
        t = np.linspace(0, 6, 100)
        
        # Calculate the cost landscape for the test graph
        test_exp_val = costLandscape(H_cost, H_walk, t, initial_state, gamma, 0, fig=None, plot=False)
        
        # Loop over Trotterization depths
        for i in range(1, n+1):
            compare_exp_val = trotterizedCostLandscape(H_cost, H_walk, t, initial_state, gamma, 0, i, fig=None, plot=False)
            difference = test_exp_val - compare_exp_val
            diff_norm = np.linalg.norm(difference, ord='fro')
            diff[i-1] = diff_norm
        
        # Store the calculated norms
        all_norms.append(diff)
    
    # Convert all_norms list to a NumPy array for easier manipulation
    all_norms = np.array(all_norms)
    
    # Calculate the average and variance if more than 1 graph
    average_norm = np.mean(all_norms, axis=0)
    
    if num_graphs > 1:
        variance_norm = np.var(all_norms, axis=0)
        std_dev = np.sqrt(variance_norm)  # Calculate standard deviation for error bars
    
    # Plotting
    plt.figure(figsize=(8, 6))
    plt.plot(x, average_norm, color='#2a536b', label='Average Frobenius Norm')
    
    # If more than 1 graph, plot the variance (as a shaded area)
    if num_graphs > 1:
        plt.fill_between(x, average_norm - std_dev, average_norm + std_dev, color='#2a536b', alpha=0.3, label='±1 Std Dev')

    # Label the axes and set the title
    plt.xlabel('Trotterization depth', fontsize=12)
    plt.ylabel('Difference to QRW result, Frobenius norm', fontsize=12)
    plt.title(f'Difference of Trotterized circuit to QRW circuit, average over {num_graphs} graphs', fontsize=14)
    
    # Show grid, legend, and plot
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()

def trotterizationConvergenceTest3(num_graphs, num_nodes, n):
    """
    Perform Trotterization convergence test over multiple graphs and plot the results with variance.

    Parameters:
    - num_graphs: Number of test graphs.
    - num_nodes: Number of nodes per test graph.
    - n: Maximum Trotterization depth to test.
    """
    
    # Pre-allocate a list to store norms for each graph
    all_norms = []
    
    # x-axis values for Trotterization depths
    x = np.arange(1, n+1)
    
    # Constants outside the loop
    gamma = np.linspace(0, 4, 100)
    t = np.linspace(0, 6, 100)
    H_walk = hypercubeHamiltonian(num_nodes)
    initial_state = initialState(num_nodes)
    # Loop over num_graphs test graphs
    for _ in range(num_graphs):
        # Pre-allocate the diff array
        diff = np.zeros(n)
        
        # Create the graph and Hamiltonians once per graph
        G = utils.createRandomGraph(num_nodes, 0.5)
        H_cost = problemHamiltonian(G)
        
        
        # Calculate the cost landscape once per graph
        test_exp_val = costLandscape(H_cost, H_walk, t, initial_state, gamma, 0, fig=None, plot=False)
        
        # Loop over Trotterization depths
        for i in range(1, n+1):
            # Calculate the Trotterized cost landscape at depth i
            compare_exp_val = trotterizedCostLandscape(H_cost, H_walk, t, initial_state, gamma, 0, i, fig=None, plot=False)
            
            # Compute the Frobenius norm of the difference
            diff_norm = np.linalg.norm(test_exp_val - compare_exp_val, ord='fro')
            diff[i-1] = diff_norm  # Store the result
        
        # Store the calculated norms for this graph
        all_norms.append(diff)
    
    # Convert all_norms list to a NumPy array
    all_norms = np.array(all_norms)
    
    # Calculate the average norm and variance if more than 1 graph
    average_norm = np.mean(all_norms, axis=0)
    
    if num_graphs > 1:
        variance_norm = np.var(all_norms, axis=0)
        std_dev = np.sqrt(variance_norm)  # Calculate standard deviation for error bars
    
    # Plotting
    plt.figure(figsize=(8, 6))
    plt.plot(x, average_norm, color='#2a5b7b', label='Average Frobenius Norm')
    
    # If more than 1 graph, plot the variance (as a shaded area)
    if num_graphs > 1:
        plt.fill_between(x, average_norm - std_dev, average_norm + std_dev, color='#1f77b4', alpha=0.3, label='±1 Std Dev')

    # Label the axes and set the title
    plt.xlabel('Trotterization depth', fontsize=12)
    plt.ylabel('Difference to QRW result, Frobenius norm', fontsize=12)
    plt.title(f'Difference of Trotterized circuit to QRW circuit, average over {num_graphs} graphs', fontsize=14)
    
    # Show grid, legend, and plot
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.show()

def nonVarQWOA(H_cost, H_walk, t, initial_state, gamma, beta, p, std_dev):
    #beta = np.linspace(0,1,100)
    
    exp_val = np.zeros(len(gamma))
    U = np.eye(len(initial_state), dtype=complex)
    for i in range(p):
        gamma_i = (beta + (1- beta)*(i/(p-1)))*gamma
        t_i = (1 -(1-beta)*(i/(p-1)))*t
        U_cost = linalgs.expm(-1j * (H_cost) * (t_i))
        U_walk = linalgs.expm(-1j * (H_walk) * (gamma_i / std_dev))
        U = U @ U_cost @ U_walk
    
    output_state = U @ initial_state
    exp_val = np.real(output_state.conj().T@H_cost@output_state)

    return exp_val

def costLandscapeNonVarQWOA(H_cost, H_walk, t, initial_state, gamma, beta, p, std_dev):
    exp_val = np.zeros((len(t), len(gamma), len(beta)))
    for i in range(len(t)):
        for j in range(len(gamma)):
            for k in range(len(beta)):
                t_value = np.array([t[i]]) # Current t value, made into array object
                gamma_value = np.array([gamma[j]])  # Current gamma value, made into array object
                beta_value =np.array([beta[k]])
                exp_val[i, j, k] = nonVarQWOA(H_cost, H_walk, t_value, initial_state, gamma_value, beta_value, p, std_dev)
    
    print(exp_val.min(), exp_val.max())

    # Create the grid
    grid = pv.UniformGrid()
    grid.dimensions = exp_val.shape
    grid.spacing= (1,1,1)
    
    
    labels = dict(zlabel='beta', xlabel='t', ylabel='gamma')




    grid.spacing = (t.max() / (len(t) - 1), gamma.max() / (len(gamma) - 1), beta.max() / (len(beta) - 1))
    grid.point_data["Exp_val"] = exp_val.flatten(order="F")

    # Set up the plotter
    plotter = pv.Plotter()

    # Opacity mapping
    opacity_vals = [1.0, 0.5, 1.0]
    plotter.add_mesh(grid, scalars='Exp_val', cmap="viridis", opacity=opacity_vals, show_scalar_bar=True)
    # Set the axis limits based on your gamma, t, and beta ranges
    plotter.show_grid(**labels)
    #plotter.add_axes(**labels)
    
    # Show grid and labels
    #plotter.show_bounds(axes_ranges=[t.min(), t.max(), gamma.min(), gamma.max(), beta.min(), beta.max()],xlabel='Time',ylabel='Gamma',zlabel='Beta')
    #plotter.show_grid(color='black')
    plotter.add_title("Cost landscape", font_size=10)

    plotter.camera.position = (2, 4, 3)  # Adjust camera position as needed
    #plotter.camera.view_up = (0, 1, 0)
    #plotter.add_axes(color='black',xlabel='X',ylabel='Y',zlabel='Z')
    # Show the plot
    plotter.show(jupyter_backend='static')


