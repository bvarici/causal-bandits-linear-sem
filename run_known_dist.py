
"""
run the algorithm when int. dist. are known.

"""
import numpy as np
import pickle as pkl
import multiprocessing as mp


from functions_main import causal_linear_sem_TS
from config import SIMULATIONS_ESTIMATED_FOLDER


def load_hierarchical_W(d,L,idx=0,return_all=False):
    res = pkl.load(open(SIMULATIONS_ESTIMATED_FOLDER+'/known_dist'+'/hierarchical_d_%d_l_%d_graph_%d.pkl'%(d,L,idx), 'rb'))
    Wobs = np.mean(res['Wobs_s'],0)
    Wint = np.mean(res['Wint_s'],0)
    if return_all is True:
        return res['Wobs_s'], res['Wint_s']
    elif return_all is False:
        return Wobs, Wint
    
def load_enhanced_parallel_W(N,idx=0,return_all=False):
    res = pkl.load(open(SIMULATIONS_ESTIMATED_FOLDER+'/known_dist'+'/size_N_%d_graph_%d.pkl'%(N,idx), 'rb'))    
    Wobs = np.mean(res['Wobs_s'],0)
    Wint = np.mean(res['Wint_s'],0)
    if return_all is True:
        return res['Wobs_s'], res['Wint_s']
    elif return_all is False:
        return Wobs, Wint



def hierarchical_known_dist(d,L,idx):
    Wobs, Wint = load_hierarchical_W(d,L,idx)
    N = Wobs.shape[-1]
    Omega = np.ones(N)
    parents = {}
    for i in range(N):
        parents[i] = list(np.where(Wobs[i])[0])

    res = causal_linear_sem_TS(parents,Wobs,Wint,Omega=Omega,T=5000,known_dist=True,available_nodes=None,\
                                  is_reward_intervenable=True,n_parameter_sampling=1,\
                                      var_s=0.01,var_p=1.0,n_repeat_data=10)

    f = open(SIMULATIONS_ESTIMATED_FOLDER+'/hierarchical_known_dist'+'/hierarchical_known_dist_d_%d_l_%d_graph_%d.pkl'%(d,L,idx),'wb')
    pkl.dump(res,f)
    f.close()    
    print('d=%d, L=%d, graph number:%d is finished in %.2f seconds'%(d,L,idx,res['time']))
    return res


def enhanced_parallel_known_dist(N,idx):
    Wobs, Wint = load_enhanced_parallel_W(N,idx)
    Omega = np.ones(N)
    parents = {}
    for i in range(N):
        parents[i] = list(np.where(Wobs[i])[0])

    res = causal_linear_sem_TS(parents,Wobs,Wint,Omega=Omega,T=5000,known_dist=True,available_nodes=None,\
                                  is_reward_intervenable=True,n_parameter_sampling=1,\
                                      var_s=0.01,var_p=1.0,n_repeat_data=10)

    
    f = open(SIMULATIONS_ESTIMATED_FOLDER+'/enhanced_parallel_known_dist'+'/known_dist_size_N_%d_graph_%d.pkl'%(N,idx),'wb')
    pkl.dump(res,f)
    f.close()    
    print('N=%d, graph number: %d is finished in %.2f seconds'%(N,idx,res['time']))

    return res


#%%
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

#args_h=args_2_2+args_2_3+args_2_4+args_3_2+args_3_3+args_3_4+args_4_2+args_4_3+args_4_4
#args_p=args_5+args_6+args_7+args_8+args_9
#%%
#args_h=args_3_1+args_3_2+args_4_1
args_h = args_3_3

if __name__ == '__main__':
    pool = mp.Pool(4)
    results = pool.starmap(hierarchical_known_dist,args_h)
    #results = pool.starmap(enhanced_parallel_known_dist,args_p)    
    pool.close()


