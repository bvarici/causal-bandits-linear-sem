"""
run standard UCB on the graphs that we run our algorithm.

"""
import numpy as np
import pickle as pkl
import multiprocessing as mp
#import matplotlib.pyplot as plt

from functions_main import baseline_UCB
from config import SIMULATIONS_ESTIMATED_FOLDER


def load_enhanced_parallel_W(N,idx):
    res = pkl.load(open(SIMULATIONS_ESTIMATED_FOLDER+'/enhanced_parallel'+'/size_N_%d_graph_%d.pkl'%(N,idx), 'rb'))    
    Wobs = np.mean(res['Wobs_s'],0)
    Wint = np.mean(res['Wint_s'],0)
    return Wobs, Wint

def load_hierarchical_W(d,L,idx=0):
    res = pkl.load(open(SIMULATIONS_ESTIMATED_FOLDER+'/hierarchical'+'/hierarchical_d_%d_l_%d_graph_%d.pkl'%(d,L,idx), 'rb'))
    Wobs = np.mean(res['Wobs_s'],0)
    Wint = np.mean(res['Wint_s'],0)
    return Wobs, Wint


#%%
'run standard UCB on the graphs that we ran our algorithm'

# number of times to repeat the experiment for approximating data expectation
n_repeat_data = 50
# number of iterations, horizon.
T = 5000
delta = (1/T)**2
available_nodes = None
is_reward_intervenable = False


def pool_run_ucb_hierarchical(d,L,idx=0):
    np.random.seed()
    Wobs, Wint = load_hierarchical_W(d,L,idx,mode)
    N = len(Wobs)
    nu = np.ones(N)
    Omega = np.ones(N)
    res = baseline_UCB(Wobs,Wint,nu,Omega,T,available_nodes,is_reward_intervenable,\
                       delta=delta,n_repeat_data=n_repeat_data)

    f = open(SIMULATIONS_ESTIMATED_FOLDER+'/hierarchical'+'/ucb_hierarchical_d_%d_l_%d_graph_%d.pkl'%(d,L,idx), 'wb')

    pkl.dump(res,f)
    f.close()     
    print('UCB. d=%d, L=%d, graph number: %d is finished in %.2f seconds'%(d,L,idx,res['time']))
    return res


def pool_run_ucb_enhanced_parallel(N,idx):
    np.random.seed()
    Wobs, Wint = load_type2_W(N,idx)
    nu = np.ones(N)
    Omega = np.ones(N)
    res = baseline_UCB(Wobs,Wint,nu,Omega,T,available_nodes,is_reward_intervenable,\
                       delta=delta,n_repeat_data=n_repeat_data)    
    f = open(SIMULATIONS_ESTIMATED_FOLDER+'/enhanced_parallel'+'/ucb_size_%d_graph_%d.pkl'%(N,idx),'wb')
    pkl.dump(res,f)
    f.close()     
    print('UCB. N=%d, graph number: %d is finished in %.2f seconds'%(N,idx,res['time']))
    return res


#%%
# example simulations
# args = [(5,1),(5,2),(5,3),(8,1),(8,2),(8,3)]
# pool = mp.Pool(6)
# results = pool.starmap(pool_run_ucb_enhanced_parallel,args)
# pool.close


# args = [(2,2,1),(2,2,2),(4,3,1),(4,3,2)]
# pool = mp.Pool(4)
# results = pool.starmap(pool_run_ucb_hierarchical,args)
# pool.close


