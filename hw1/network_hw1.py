import numpy as np
import scipy as sp
from scipy.sparse.linalg import eigsh
import pandas as pd


'''
arr = np array from which values are removed
relative_to = if this value is not None then the values from arr are removed where relative_to == 0
return = arr after removing 0 entries
'''
def remove_zeros(arr, relative_to = None):
    
    #type checking
    assert type(arr) == np.ndarray
    assert (relative_to is None) or (type(relative_to) == np.ndarray)

    if relative_to is None:
        relative_to = arr

    return arr[relative_to != 0]

'''
A = adjacency matrix can be csr_matrix from the scipy module(sparse matrices) or it can be a np matrix
a little test showed that the sparse matrix is 1000 times faster so i suggest using that one
direction = can be either "in" or "out" used to get indegree or outdegree
return = indegree or outdegree of A
'''
def get_degrees(A, direction = "out"):

    if direction == "out":
        axis = 0
    elif direction == "in":
        axis = 1
    else:
        raise Exception("Unsupported degree direction")

    if type(A) == sp.sparse.csr.csr_matrix:
        degrees = np.array(A.sum(axis = axis))
    elif type(A) == np.ndarray:
        degrees = np.sum(A, axis = axis)
    else:
        raise Exception("Unsupported matrix format")
    
    #returning column vector
    return degrees.reshape([max(degrees.shape), 1])


'''
N = number of nodes in the network
average_degree = mean of degrees in the network
'''
def get_average_distance(N, average_degree):
    return np.log(N)/np.log(average_degree)

'''
average_degree = mean of degrees in the network
N = number of nodes in the network
'''
def get_average_clustering_coef(average_degree, N):
    return average_degree / N


'''
N = number of nodes in the network
gamma = degree exponent of the network
'''
def get_natural_cutoff(N, gamma):
    return N**(1/(gamma-1))


'''
degrees = numpy array containing the indegrees or outdegrees of each node
return = the PDF of the degrees 
'''
def get_neighbours_pdf(degrees):

    assert type(degrees) == np.ndarray

    #removing nodes with no connections
    degrees = remove_zeros(degrees)
    
    #getting the degrees of the nodes and the number of occurrences of each value
    unique, counts = np.unique(degrees, return_counts = True)
    
    #calculating the pdf
    pdf = counts / np.sum(counts)

    return unique, pdf


'''
pdf = probability distribution function over which the ccdf is calculated, has to be a np array
return = the ccdf associated to pdf
'''
def ccdf(pdf):

    assert type(pdf) == np.ndarray

    return 1 - np.cumsum(pdf)


'''
degrees = array conatining the indegrees or outdegrees of each node
return = the estimated exponent of the power law and the constant c in front of the expression
'''
def estimate_power_law(degrees, kmin):

    assert (type(degrees) == np.ndarray) and (type(kmin) == int) and (kmin > 0)
    

    #removing unnecessary degrees
    degrees = degrees[degrees > kmin]
    
    #calculating the parameters of the power law
    gamma = 1 + len(degrees)/np.sum(np.log(degrees / kmin))
    c = (gamma - 1)*kmin**(gamma - 1)
    
    return gamma, c


'''
exp_step_size = how much the exponent of the x is increased at every step
degrees = degrees of the node of the network
return = the bins and the probability associated with each bin
'''
def log_binning_pdf(exp_step_size, degrees):
    
    assert (exp_step_size > 0) and (type(degrees) == np.ndarray)

    degrees = remove_zeros(degrees)

    #getting the x bins
    x = 10 ** np.arange(0, np.ceil(np.log10(np.max(degrees))), exp_step_size)
    
    #getting the probability of being in a bin
    log_bin_pdf, _ = np.histogram(degrees, x)
    log_bin_pdf = log_bin_pdf / np.sum(log_bin_pdf)

    return [x[:-1], log_bin_pdf]


'''
A = sparse matrix
return = sparse matrix that is undirected, this is done by adding an edge in the opposite direction 
where there is already one
'''
def get_undirected_network(A):
    
    #type checking
    assert type(A) == sp.sparse.csr_matrix

    return 1*((A + A.T) > 0)  


'''
A = sparse matrix in particular a csr_matrix from scipy.sparse
return = cleaned sparse csr matrix containing only nodes from the Giant Component
'''
def clean_network(A):

    assert type(A) == sp.sparse.csr_matrix

    #building an undirected network
    Au = get_undirected_network(A)
    N = A.shape[0]
    
    #isolating the GC

    #non visited nodes will have value 1
    not_visited = np.ones((N, 1))

    #size of the biggest component found until now
    biggest_component = 0
    best_e1 = None

    while np.sum(not_visited) > biggest_component:

        #get first non zero index
        index = np.where(not_visited)[0][0]
        
        e1 = np.zeros((N,1))
        
        #setting to 1 one of the node of the GC
        e1[index] = 1

        #exit condition
        ex = False

        while not ex:

            e1_old = e1
            
            #searching for nodes connected to the nodes in e1
            e1 = (Au * e1 + e1) > 0

            #checking if no new nodes were added to the list
            ex = not np.sum(e1 != e1_old)
        
        #setting all visited nodes = 0
        not_visited = not_visited - e1
        
        #select the best bigger component
        if np.sum(e1) > biggest_component:
            best_e1 = e1
            biggest_component = np.sum(e1)

    e1 = np.reshape(best_e1, (N))
    
    #this is apparently the most efficient way of slicing a sparse matrix
    A = A[e1, :][:, e1]

    return A

'''
A = density matrix as a csr_matrix
neigh_dir = direction in which to calculate the degree of the nodes can be "in" or "out"
knn_dir = direction in which the average neighbours degree is calculated can be "in" or "out"
return = [p, unique, knn, temp_knn] coefficients of the polinomial fitting
(the one at index 0 is the assortativity value), unique degrees and knn
'''
def get_assortativity_value(A, neigh_dir = "out", knn_dir = "out"):

    #type checking
    assert type(A) == sp.sparse.csr_matrix
    assert (neigh_dir in ["in", "out"]) and (knn_dir in ["in", "out"])
    
    #getting the degrees
    neigh_deg = get_degrees(A, neigh_dir)
    knn_deg = get_degrees(A, knn_dir)

    if neigh_dir == "out":
        A = A.T

    #done in order to evade division by zero, non meaningful results are removed later
    temp_neigh_deg = np.copy(neigh_deg)
    temp_neigh_deg[temp_neigh_deg == 0] = 1

    temp_knn = (A * sp.sparse.csr_matrix(knn_deg))/temp_neigh_deg
    
    #removing unmeaningful values
    temp_knn = remove_zeros(np.array(temp_knn), relative_to = neigh_deg)
    neigh_deg = remove_zeros(neigh_deg)

    #setting temp_knn as a column vector
    temp_knn = np.reshape(temp_knn, [max(temp_knn.shape), 1])

    #removing the 0 values
    unique = np.unique(neigh_deg)
    
    #df_data is needed to create a pandas DataFrame that will make next calculation way easier
    df_data = np.concatenate([neigh_deg[:, np.newaxis], temp_knn], axis = 1)

    #getting knn
    knn = pd.DataFrame(df_data, columns = ["degrees", "temp_knn"]).groupby("degrees").mean().values

    #removing 0 values that would go to infinity in the log
    unique = remove_zeros(unique[:, np.newaxis], relative_to = knn)
    knn = remove_zeros(knn)

    #fitting the data with a linear model
    p = sp.polyfit(np.log(unique), np.log(knn), 1)

    #returning the slope of the linear model
    return p, unique, knn, temp_knn
