"""
code-base for running simulations 'enhanced parallel bandits' in Section 6.
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import multiprocessing as mp
import os

from functions_main import causal_linear_sem_TS
from prepare_graphs import generate_enhanced_parallel_aug
from config import SIMULATIONS_ESTIMATED_FOLDER


#%%%%%%%%%
'SIMULATE ENHANCED PARALLEL BANDITS GRAPHS'

# various number of nodes
#N_list = [7]
# number of different graph structures to simulate
#n_random_graphs = 1
# number of times to sample from the true distribution prior
n_parameter_sampling = 5
# number of times to repeat the experiment for approximating data expectation
n_repeat_data = 10
# number of iterations, horizon.
T = 5000

# range of weights for given adjacency matrix.
min_weight_abs = 0.25 
max_weight_abs = 1.0
# variance for sampling from prior of true distribution
var_s = 0.01

# parameter for posterior update
var_p = 1.0
# if all nodes are available, set None
available_nodes = None
is_reward_intervenable = False

# run in parallel

def pool_run(N,idx):
    np.random.seed()
    Omega = np.ones(N)
    A = generate_enhanced_parallel_aug(N)
                
    Wobs = np.random.uniform(min_weight_abs,max_weight_abs,[len(A),len(A)])* np.random.choice([-1,1],size=[len(A),len(A)])
    Wobs[np.where(A==0)] = 0
    # set the interventional weights as negatives of observational weights
    Wint = - Wobs
    # parents info.
    parents = {}
    for i in range(N):
        parents[i] = list(np.where(Wobs[i])[0])

    res = causal_linear_sem_TS(parents,Wobs,Wint,Omega=Omega,T=T,known_dist=False,available_nodes=available_nodes,\
                                  is_reward_intervenable=False,n_parameter_sampling=n_parameter_sampling,\
                                      var_s=var_s,var_p=var_p,n_repeat_data=n_repeat_data)

    res['A'] = A
    
    f = open(SIMULATIONS_ESTIMATED_FOLDER+'/enhanced_parallel'+'/size_N_%d_graph_%d.pkl'%(N,idx),'wb')
    pkl.dump(res,f)
    f.close()    
    print('N=%d, graph number: %d is finished in %.2f seconds'%(N,idx,res['time']))

    return res

#%%t
# N = 5
args_5 = [(5,i) for i in range(1,6)]
# N = 6
args_6 = [(6,i) for i in range(1,6)]
# N = 7
args_7 = [(7,i) for i in range(1,6)]
# N = 8
args_8 = [(8,i) for i in range(1,6)]
# N = 9
args_9 = [(9,i) for i in range(1,6)]

args_p = args_5+args_6+args_7+args_8+args_9

if __name__ == "__main__":
    pool = mp.Pool(12)
    results = pool.starmap(pool_run,args_p)
    pool.close()

