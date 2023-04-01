"""
create the graphs to run simulations on. will be imported.
"""

import numpy as np

def generate_hierarchical(d=2,L=3):
    N = d*L+1
    # A is for adjancency matrix
    A = np.zeros((N,N)) 
    # W is for weighted adjacency matrix
    W = np.zeros((N,N))

    for l in range(1,L):
        for i in range(d):
            A[d*l+i,range(d*(l-1),d*l)] = 1
            W[d*l+i,range(d*(l-1),d*l)] = 1/np.sqrt(d)

            
    A[-1,range(d*(L-1),d*L)] = 1
    W[-1,range(d*(L-1),d*L)] = 1/np.sqrt(d)
    return A, W

def generate_hierarchical_aug(d=2,L=3):
    A, _ = generate_hierarchical(d,L)
    N = len(A)
    A_aug = np.zeros((N+1,N+1))
    A_aug[1:,1:] = A
    A_aug[1:,0] = 1
    return A_aug

def generate_enhanced_parallel(N):
    '''
    another candidate graph type:
        every node is a parent of reward node. all the other nodes have at most 1 parent, except the root node.
    '''
    A = np.zeros((N,N))
    A[-1,0] = 1
    for i in range(1,N-1):
        # each node is parent of reward node (last node)
        A[-1,i] = 1
        # randomly select one parent for each node i
        parent_i = np.random.choice(i)
        A[i,parent_i] = 1

    return A

def generate_enhanced_parallel_aug(N):
    A = generate_enhanced_parallel(N)
    A_aug = np.zeros((N+1,N+1))
    A_aug[1:,1:] = A
    A_aug[1:,0] = 1
    return A_aug

