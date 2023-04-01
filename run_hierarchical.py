"""
code-base for running simulations 'hierarchical graphs' in Section 6.

"""
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
import multiprocessing as mp

from functions_main import causal_linear_sem_TS
from prepare_graphs import generate_hierarchical, generate_hierarchical_aug
from config import SIMULATIONS_ESTIMATED_FOLDER


#%%%%%%%%%
'SIMULATE HIERARCHICAL GRAPHS'

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
available_nodes = None
is_reward_intervenable = True

#%%

# run in parallel
def pool_run(d,L,idx):
    np.random.seed()
    A = generate_hierarchical_aug(d,L)
    N = len(A)
    Omega = np.ones(N)
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
    f = open(SIMULATIONS_ESTIMATED_FOLDER+'/hierarchical'+'/hierarchical_d_%d_l_%d_graph_%d.pkl'%(d,L,idx),'wb')
    pkl.dump(res,f)
    f.close()    
    print('d=%d, L=%d, graph number:%d is finished in %.2f seconds'%(d,L,idx,res['time']))

    return res


#%%
#example simulations
#args = [(2,2,1),(2,2,2),(4,3,1),(4,3,2)]

# for d=2, L=1
args_2_1 = [(2,1,i) for i in range(1,6)]
# for d=2, L=2
args_2_2 = [(2,2,i) for i in range(1,6)]
# for d=2, L=3
args_2_3 = [(2,3,i) for i in range(1,6)]
# for d=2, L=4
args_2_4 = [(2,4,i) for i in range(1,6)]
# for d=3, L=1
args_3_1 = [(3,1,i) for i in range(1,6)]
# for d=3, L=2
args_3_2 = [(3,2,i) for i in range(1,6)]
# for d=3, L=3
args_3_3 = [(3,3,i) for i in range(1,6)]
# for d=3, L=4
args_3_4 = [(3,4,i) for i in range(1,6)]
# for d=4, L=1
args_4_1 = [(4,1,i) for i in range(1,6)]
# for d=4, L=2
args_4_2 = [(4,2,i) for i in range(1,6)]
# for d=4, L=3
args_4_3 = [(4,3,i) for i in range(1,6)]
# for d=4, L=4
args_4_4 = [(4,4,i) for i in range(1,6)]

#%%
#args_h=args_2_2+args_2_3+args_2_4+args_3_2+args_3_3+args_3_4+args_4_2+args_4_3+args_4_4
#args_h=args_2_1+args_2_2+args_2_3+args_3_1+args_3_2+args_3_3+args_4_1+args_4_2+args_4_3
args_h=args_3_3

if __name__ == '__main__':
    pool = mp.Pool(4)
    results = pool.starmap(pool_run,args_h)
    pool.close()






