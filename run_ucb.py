"""
run standard UCB on the graphs that we run our algorithm.

"""
import numpy as np
import pickle as pkl
import multiprocessing as mp
#import matplotlib.pyplot as plt

from functions_main import baseline_UCB
from config import SIMULATIONS_ESTIMATED_FOLDER


def load_enhanced_parallel_W(N,idx=0):
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
n_repeat_data = 10
# number of iterations, horizon.
T = 5000
delta = (1/T)**2
available_nodes = None
is_reward_intervenable = True


def pool_run_ucb_hierarchical(d,L,idx=0):
    np.random.seed()
    Wobs, Wint = load_hierarchical_W(d,L,idx)
    N = len(Wobs)
    Omega = np.ones(N)
    res = baseline_UCB(Wobs,Wint,Omega,T,available_nodes,is_reward_intervenable,\
                       delta=delta,n_repeat_data=n_repeat_data)

    f = open(SIMULATIONS_ESTIMATED_FOLDER+'/hierarchical'+'/ucb_hierarchical_d_%d_l_%d_graph_%d.pkl'%(d,L,idx), 'wb')

    pkl.dump(res,f)
    f.close()     
    print('UCB. d=%d, L=%d, graph number: %d is finished in %.2f seconds'%(d,L,idx,res['time']))
    return res


def pool_run_ucb_enhanced_parallel(N,idx):
    np.random.seed()
    Wobs, Wint = load_enhanced_parallel_W(N,idx)
    Omega = np.ones(N)
    res = baseline_UCB(Wobs,Wint,Omega,T,available_nodes,is_reward_intervenable,\
                       delta=delta,n_repeat_data=n_repeat_data)    
    f = open(SIMULATIONS_ESTIMATED_FOLDER+'/enhanced_parallel'+'/ucb_size_N_%d_graph_%d.pkl'%(N,idx),'wb')
    pkl.dump(res,f)
    f.close()     
    print('UCB. N=%d, graph number: %d is finished in %.2f seconds'%(N,idx,res['time']))
    return res

#%%
#example simulations
#args = [(2,2,1),(2,2,2),(4,3,1),(4,3,2)]

# for d=2, L=2
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

# N =5
args_5 = [(5,i) for i in range(1,6)]
# N = 6
args_6 = [(6,i) for i in range(1,6)]
# N = 7
args_7 = [(7,i) for i in range(1,6)]
# N = 8
args_8 = [(8,i) for i in range(1,6)]
# N = 9
args_9 = [(9,i) for i in range(1,6)]


#%%
#args_h=args_2_2+args_2_3+args_2_4+args_3_2+args_3_3+args_3_4+args_4_2+args_4_3+args_4_4
#args_p=args_5+args_6+args_7+args_8+args_9
#args_h = args_4_1 + args_3_2
args_h = args_3_3 + args_4_2 + args_4_3


if __name__ == '__main__':
    pool = mp.Pool(4)
    results = pool.starmap(pool_run_ucb_hierarchical,args_h)
    #results = pool.starmap(pool_run_ucb_enhanced_parallel,args_p)    
    pool.close()



