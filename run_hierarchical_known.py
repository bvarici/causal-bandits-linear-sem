"""
code-base for running simulations 'hierarchical graphs' in Section 6,

only difference from run_hierarhical.py file is to
test the knowledge of 'intervetional distrubitons of the parents'
knowing them corresponds to knowing all the weights except vector for reward node.


"""

import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import multiprocessing as mp

from functions_main import causal_linear_sem_TS, causal_linear_sem_TS_known
from prepare_graphs import generate_hierarchical
from config import SIMULATIONS_ESTIMATED_FOLDER


#%%%%%%%%%
'SIMULATE HIERARCHICAL GRAPHS'

# number of different graph structures to simulate
#n_random_graphs = 1
# number of times to sample from the true distribution prior
n_parameter_sampling = 20
# number of times to repeat the experiment for approximating data expectation
n_repeat_data = 20
# number of iterations, horizon.
T = 5000

# range of weights for given adjacency matrix.
min_weight_abs = 0.25 
max_weight_abs = 1.0
# variance for sampling from prior of true distribution
var_s = 0.01

# parameter for posterior update
var_p = 1.0
available_nodes = None
is_reward_intervenable = False

def pool_run(d,L,idx):
    np.random.seed()
    A, W = generate_hierarchical(d,L)
    N = len(A)
    nu = np.ones(N)
    Omega = np.ones(N)
    Wobs = np.random.uniform(min_weight_abs,max_weight_abs,[len(A),len(A)])* np.random.choice([-1,1],size=[len(A),len(A)])
    Wobs[np.where(A==0)] = 0
    # set the interventional weights as negatives of observational weights
    Wint = - Wobs
    
    # parents info.
    parents = {}
    for i in range(N):
        parents[i] = list(np.where(Wobs[i])[0])

    res = causal_linear_sem_TS_known(parents,Wobs,Wint,nu=nu,Omega=Omega,T=T,available_nodes=available_nodes,\
                                  is_reward_intervenable=is_reward_intervenable,n_parameter_sampling=n_parameter_sampling,\
                                      var_s=var_s,var_p=var_p,n_repeat_data=n_repeat_data)

    res['A'] = A
    f = open(SIMULATIONS_ESTIMATED_FOLDER+'/hierarchical'+'/hierarchical_known_d_%d_l_%d_graph_%d.pkl'%(d,L,idx),'wb')
    pkl.dump(res,f)
    f.close()    
    print('Known: d=%d, L=%d, graph number:%d is finished in %.2f seconds'%(d,L,idx,res['time']))

    return res

#%%
# example simulations
# args = [(2,2,1),(2,2,2),(4,3,1),(4,3,2)]
# pool = mp.Pool(4)
# results = pool.starmap(pool_run,args)
# pool.close


