"""
code-base for running simulations 'enhanced parallel bandits' in Section 6.
"""

import numpy as np
#import matplotlib.pyplot as plt
import pickle as pkl
import multiprocessing as mp
#import os
import random

from functions_main import causal_linear_sem_TS
#from prepare_graphs import generate_enhanced_parallel_aug
from config import SIMULATIONS_ESTIMATED_FOLDER

def load_hierarchical_W(d,L,idx=0,return_all=False):
    res = pkl.load(open(SIMULATIONS_ESTIMATED_FOLDER+'/hierarchical'+'/hierarchical_d_%d_l_%d_graph_%d.pkl'%(d,L,idx), 'rb'))
    Wobs = np.mean(res['Wobs_s'],0)
    Wint = np.mean(res['Wint_s'],0)
    if return_all is True:
        return res['Wobs_s'], res['Wint_s']
    elif return_all is False:
        return Wobs, Wint
    
def load_enhanced_parallel_W(N,idx=0,return_all=False):
    res = pkl.load(open(SIMULATIONS_ESTIMATED_FOLDER+'/enhanced_parallel'+'/size_N_%d_graph_%d.pkl'%(N,idx), 'rb'))    
    Wobs = np.mean(res['Wobs_s'],0)
    Wint = np.mean(res['Wint_s'],0)
    if return_all is True:
        return res['Wobs_s'], res['Wint_s']
    elif return_all is False:
        return Wobs, Wint


n_parameter_sampling = 5
n_repeat_data = 3
T = 5000



#%%
def run_enhanced_fp_mis(N,idx,n_fp_edges=2):
    Wobs, Wint = load_enhanced_parallel_W(N,idx)
    Omega = np.ones(N)

    # parents info.
    parents = {}
    parents_mis = {}
    for i in range(N):
        parents[i] = list(np.where(Wobs[i])[0])
        parents_mis[i] = list(np.where(Wobs[i])[0]) 
    
    '''misspecification would be just disturbing the parents.
    if there are additional mis-parents, algo is still supposed to work.'''

    valid_nonedges = [(j,i) for i in range(N) for j in range(N) if Wobs[i,j]==0 and j < i]
    if n_fp_edges > len(valid_nonedges):
        n_fp_edges = len(valid_nonedges)
    
    fp_edges = random.sample(valid_nonedges, n_fp_edges)
    
    for fp_edge in fp_edges:
        parents_mis[fp_edge[1]].append(fp_edge[0])
        


    res = causal_linear_sem_TS(parents_mis,Wobs,Wint,Omega=Omega,T=T,known_dist=False,available_nodes=None,\
                                  is_reward_intervenable=True,n_parameter_sampling=n_parameter_sampling,\
                                      var_s=0.01,var_p=1.0,n_repeat_data=n_repeat_data)

    
    f = open(SIMULATIONS_ESTIMATED_FOLDER+'/misspecification_fp'+'/size_mis_fp_N_%d_graph_%d.pkl'%(N,idx),'wb')
    pkl.dump(res,f)
    f.close()    
    print('N=%d, graph number: %d is finished in %.2f seconds'%(N,idx,res['time']))

    return res

def run_enhanced_fn_mis(N,idx,n_fn_edges=2):
    Wobs, Wint = load_enhanced_parallel_W(N,idx)
    Omega = np.ones(N)

    # parents info.
    parents = {}
    parents_mis = {}
    for i in range(N):
        parents[i] = list(np.where(Wobs[i])[0])
        parents_mis[i] = list(np.where(Wobs[i])[0]) 
    
    '''misspecification would be just disturbing the parents.
    remove a couple edges, i.e., there are false negatives. '''

    all_edges = [(j,i) for i in range(N) for j in range(N) if Wobs[i,j]!=0 and j < i]

    if n_fn_edges > len(all_edges):
        n_fn_edges = len(all_edges)
    
    fn_edges = random.sample(all_edges, n_fn_edges)
    
    for fn_edge in fn_edges:
        parents_mis[fn_edge[1]].remove(fn_edge[0])
        
    res = causal_linear_sem_TS(parents_mis,Wobs,Wint,Omega=Omega,T=T,known_dist=False,available_nodes=None,\
                                  is_reward_intervenable=True,n_parameter_sampling=n_parameter_sampling,\
                                      var_s=0.01,var_p=1.0,n_repeat_data=n_repeat_data)
    
    f = open(SIMULATIONS_ESTIMATED_FOLDER+'/misspecification_fn'+'/size_mis_fn_N_%d_graph_%d.pkl'%(N,idx),'wb')
    pkl.dump(res,f)
    f.close()    
    print('N=%d, graph number: %d is finished in %.2f seconds'%(N,idx,res['time']))

    return res

def run_hierarchical_fn_mis(d,L,idx,n_fn_edges=2):
    Wobs, Wint = load_hierarchical_W(d,L,idx)
    N = Wobs.shape[-1]
    Omega = np.ones(N)

    # parents info.
    parents = {}
    parents_mis = {}
    for i in range(N):
        parents[i] = list(np.where(Wobs[i])[0])
        parents_mis[i] = list(np.where(Wobs[i])[0]) 
    
    '''misspecification would be just disturbing the parents.
    remove a couple edges, i.e., there are false negatives. '''

    all_edges = [(j,i) for i in range(N) for j in range(N) if Wobs[i,j]!=0 and j < i]

    if n_fn_edges > len(all_edges):
        n_fn_edges = len(all_edges)
    
    fn_edges = random.sample(all_edges, n_fn_edges)
    
    for fn_edge in fn_edges:
        parents_mis[fn_edge[1]].remove(fn_edge[0])
        
    res = causal_linear_sem_TS(parents_mis,Wobs,Wint,Omega=Omega,T=T,known_dist=False,available_nodes=None,\
                                  is_reward_intervenable=True,n_parameter_sampling=n_parameter_sampling,\
                                      var_s=0.01,var_p=1.0,n_repeat_data=n_repeat_data)

    f = open(SIMULATIONS_ESTIMATED_FOLDER+'/misspecification_fn'+'/hierarchical_mis_fn_d_%d_l_%d_graph_%d.pkl'%(d,L,idx),'wb')
    pkl.dump(res,f)
    f.close()    
    print('d=%d, L=%d, graph number:%d is finished in %.2f seconds'%(d,L,idx,res['time']))
    return res


def run_hierarchical_fp_mis(d,L,idx,n_fp_edges=2):
    Wobs, Wint = load_hierarchical_W(d,L,idx)
    N = Wobs.shape[-1]
    Omega = np.ones(N)

    # parents info.
    parents = {}
    parents_mis = {}
    for i in range(N):
        parents[i] = list(np.where(Wobs[i])[0])
        parents_mis[i] = list(np.where(Wobs[i])[0]) 
    
    '''misspecification would be just disturbing the parents.
    if there are additional mis-parents, algo is still supposed to work.'''

    n_fp_edges = 2    
    valid_nonedges = [(j,i) for i in range(N) for j in range(N) if Wobs[i,j]==0 and j < i]
    if n_fp_edges > len(valid_nonedges):
        n_fp_edges = len(valid_nonedges)
    
    fp_edges = random.sample(valid_nonedges, n_fp_edges)
    
    for fp_edge in fp_edges:
        parents_mis[fp_edge[1]].append(fp_edge[0])

    res = causal_linear_sem_TS(parents_mis,Wobs,Wint,Omega=Omega,T=T,known_dist=False,available_nodes=None,\
                                  is_reward_intervenable=True,n_parameter_sampling=n_parameter_sampling,\
                                      var_s=0.01,var_p=1.0,n_repeat_data=n_repeat_data)

    f = open(SIMULATIONS_ESTIMATED_FOLDER+'/misspecification_fp'+'/hierarchical_mis_fp_d_%d_l_%d_graph_%d.pkl'%(d,L,idx),'wb')
    pkl.dump(res,f)
    f.close()    
    print('d=%d, L=%d, graph number:%d is finished in %.2f seconds'%(d,L,idx,res['time']))
    return res

#%%t
# for d=2, L=1
args_2_1 = [(2,1,i) for i in range(1,6)]
# for d=2, L=2
args_2_2 = [(2,2,i) for i in range(1,6)]
# for d=2, L=3
args_2_3 = [(2,3,i) for i in range(1,6)]
# for d=3, L=1
args_3_1 = [(3,1,i) for i in range(1,6)]
# for d=3, L=2
args_3_2 = [(3,2,i) for i in range(1,6)]
# for d=3, L=3
args_3_3 = [(3,3,i) for i in range(1,6)]
# for d=4, L=1
args_4_1 = [(4,1,i) for i in range(1,6)]
# for d=4, L=2
args_4_2 = [(4,2,i) for i in range(1,6)]
# for d=4, L=3
args_4_3 = [(4,3,i) for i in range(1,6)]

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

#args_p = args_5+args_6+args_7+args_8+args_9
args_h = args_4_3


if __name__ == "__main__":
    pool = mp.Pool(4)
    #results = pool.starmap(run_enhanced_fp_mis,args_p)
    #results = pool.starmap(run_enhanced_fn_mis,args_p)
    results = pool.starmap(run_hierarchical_fp_mis,args_h)
    #results = pool.starmap(run_hierarchical_fn_mis,args_h)

    pool.close()

