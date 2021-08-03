import numpy as np
import scipy as sp
from scipy.sparse.linalg import eigsh
import pandas as pd

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
A = sparse matrix representing the network
c = damping factor
q = teleport vector
return = page rank ranking vector
'''

def page_rank_linear_system(A, c = 0.85, q = None):
    
    assert type(A) == sp.sparse.csr.csr_matrix
    assert (type(c) == float) and (c > 0) and (c < 1)
    assert (q is None) or type(q) == np.ndarray

    N = A.shape[0]
 
    #if no q is passed then a q with all probabilities equal to 1/N is created
    if q is None:
        q = np.ones((N, 1))/N

    d = 1/get_degrees(A)
    M = A * sp.sparse.diags(d[:, 0])

    #finding a and b such that ap = b
    a = (np.eye(N) - c * M)/(1-c)
    
    #this is faster than the sparse solver because p is not a sparse vector 
    #since a has at leas one positive entry for each row/column and b doesn't have any 0
    p = np.linalg.solve(a, q)
    p = p/np.sum(p)

    return p

'''
A = sparse matrix
return = sparse matrix where there are no single nodes and all dead ends are removed
'''
def remove_dead_ends(A):
    
    assert type(A) == sp.sparse.csr_matrix

    #exit condition
    ex = False

    while not ex:
        #if the outdegree of the node is 0 remove it
        pos = (get_degrees(A) != 0).reshape(-1)
        A = A[pos, :][:, pos]

        #checking if there are any bad nodes left
        ex = np.sum(get_degrees(A) == 0) == 0
    
    return A

'''
A = sparse matrix whose page rank is calculated
iter_num = number of iterations
c = damping factor
q = teleport vector
p_linear = real page ranking obtained with linear system solution
'''
def page_rank_power_iteration(A, iter_num = 35, c = 0.85, q = None, p_linear=None):
    
    #type checks
    assert type(A) == sp.sparse.csr.csr_matrix
    assert (type(c) == float) and (c > 0) and (c < 1)
    assert (q is None) or (type(q) == np.ndarray)
    assert (type(iter_num) == int) and (iter_num > 0)
    assert (p_linear is None) or (type(p_linear) == np.ndarray)

    N = A.shape[0]
 
    #if no q is passed then a q with all probabilities equal to 1/N is created
    if q is None:
        q = np.ones((N, 1))/N
    
    #ranking starting point, all nodes have the same rank
    pt = np.ones((N, 1))/N

    #calculating the M matrix
    d = 1/get_degrees(A)
    M = A * sp.sparse.diags(d[:, 0])
    
    errors = []

    for t in range(iter_num):

        #updating and normalizing the ranking
        pt = c * M * pt + (1 - c) * q
        pt = pt/np.sum(pt)
        
        #if the real rank was passed the error is computed
        if p_linear is not None:
            errors.append(np.linalg.norm(p_linear - pt)/ N ** 0.5)

    #returning the error if it was calculated
    if p_linear is not None:
        return pt, errors
    
    return pt

'''
A = sparse matrix
return = the 2 biggest eigenvalues of the matrix A
'''
def get_two_highest_eigenvalues(A):
    
    assert type(A) == sp.sparse.csr_matrix

    #needed for the sparse function to work
    A = A.astype(float)

    #getting the highes eigenvalues
    val, vec = sp.sparse.linalg.eigs(A, 2)

    return val

'''
A = sparse matrix
return = HITS rank of A obtained by finding the eigenvector relative to the second highest eigenvalue
'''
def hits_linear_system(A):

    assert type(A) == sp.sparse.csr_matrix

    #getting the M matrix used in HITS
    M = A * A.T

    #getting the highest eigenvalues and the relative eigenvectors
    M = M.astype(float)
    val, vec = sp.sparse.linalg.eigs(M, k = 2)
    
    #normalizing the eigenvector
    p = -vec[:, 0]/np.linalg.norm(vec[:, 0])

    #the abs is returned because the vector is complex with imaginary part 0
    #his is not a problem because p is a probability vector so all entries must be positive
    return np.abs(p)

'''
A = sparse matrix
iter_num = number of iterations
p_linear = real HITS rank
return = HITS rank found thanks to power iteration
'''
def hits_power_iteration(A, iter_num = 35, p_linear=None):
    
    assert type(A) == sp.sparse.csr.csr_matrix
    assert (type(iter_num) == int) and (iter_num > 0)
    assert (p_linear is None) or (type(p_linear) == np.ndarray)

    N = A.shape[0]
    
    #giving the same rank to every node
    pt = np.ones((N, 1))/N**0.5

    #finding the M matrix
    M = A*A.T
    
    errors = []

    for t in range(iter_num):

        #updating pt
        pt = M*pt
        #normalizing pt
        pt /= np.linalg.norm(pt)

        #if the true pt is provided the error is calculated
        if p_linear is not None:
            errors.append(np.linalg.norm(p_linear - pt.T)/ N ** 0.5)

    if p_linear is not None:
        return pt, errors
    
    return pt

'''
A = sparse matrix
return = normalized laplacian of A
'''
def get_normalized_laplacian(A):
    
    assert type(A) == sp.sparse.csr_matrix

    N = A.shape[0]

    D = get_D_matrix(A)
    L = sp.sparse.identity(N) - D*A*D

    return L

'''
A = sparse matrix
return = D sparse matrix give A
'''
def get_D_matrix(A):

    assert type(A) == sp.sparse.csr_matrix
    
    N = A.shape[0]

    d = get_degrees(A)
    inv_d = np.reshape(1/np.sqrt(d), N)
    D = sp.sparse.diags(inv_d.T, offsets=0)

    return D

'''
A = sparse matrix
reorder = vector according to whom A should be reordered
inverse = if False reorder is sorted from smaller to bigger if true the opposite is done
reuturn = A reordered according to reorder from it's smaller value to the biggest
'''
def reorder_nodes(A, reorder, inverse=False):

    assert type(A) == sp.sparse.csr_matrix
    
    N = A.shape[0]
    
    assert type(reorder) == np.ndarray
    assert type(inverse) == bool

    reorder = reorder.reshape(N)
    ids = np.argsort(reorder)
    
    if inverse:
        ids = ids[::-1]

    A1 = A[ids, :][:, ids]

    return A1, ids

'''
A = sparse matrix
return = fiedler vector and it's successor
'''
def get_fiedler_vector(A):

    assert type(A) == sp.sparse.csr_matrix

    N = A.shape[0]
    
    D = get_D_matrix(A)
    L = sp.sparse.identity(N) - D*A*D

    #getting the eigenvectors of L
    eig_val, eig_vec = sp.sparse.linalg.eigsh(L, k = 3, which="SM")
    
    #normalizing the eigenvectors
    eig_vec = D * eig_vec

    return eig_vec[:, 1], eig_vec[:, 2]

'''
A = sparse matrix
return = conductance array of matrix A
'''
def get_conductance(A):

    assert type(A) == sp.sparse.csr_matrix

    N = A.shape[0]

    #values needed1 to calculate cut and assoc
    a = np.asarray(sp.sparse.triu(A).sum(axis=0))
    b = np.asarray(sp.sparse.tril(A).sum(axis=0))
    d = get_degrees(A)

    cut = np.cumsum(b - a, axis = 1)

    assoc = np.cumsum(d, axis = 0)

    D = np.sum(d)

    denominator =  np.min(np.concatenate([assoc, D - assoc], axis = 1), axis = 1)

    #removing the last value of denominator because it is going to be 0
    conductance = cut.T[:-1].reshape(N-1) / denominator[:-1]

    return conductance.T

'''
A = sparse matrix
epsilon = precision
starting_node = starting node
c = damping factor
return = approximate page nibble
'''
def page_nibble_with_finite_precision(A, epsilon, starting_node = 0, c = 0.85):

    assert type(A) == sp.sparse.csr_matrix
    assert type(epsilon) == float

    N = A.shape[0]

    assert (type(starting_node) == int) and (starting_node < N)

    #getting basic parameters to perform the page nibble
    d = get_degrees(A)
    M = A * sp.sparse.diags(1/d[:, 0])
    D = np.sum(d)
    q = np.zeros((N,1))
    q[starting_node] = 1
    u = np.zeros((N, 1))
    v = q.copy()
    th = epsilon * d / D

    #while some values of v are bigger than the threshold 
    while np.sum(v > th) > 0:

        #compute the delta where the vector is bigger than threshold
        delta = v.copy()
        delta[v < th] = 0

        #updating u,v
        u += (1-c)*delta
        v = v - delta + c*M*delta
        
    return u


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
A = sparse matrix
return = sparse matrix that is undirected, this is done by adding an edge in the opposite direction 
where there is already one
'''
def get_undirected_network(A):
    
    #type checking
    assert type(A) == sp.sparse.csr_matrix

    return 1*((A + A.T) > 0)  
