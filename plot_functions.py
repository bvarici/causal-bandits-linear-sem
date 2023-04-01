"""
functions for quick plotting of the results
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl

from config import SIMULATIONS_ESTIMATED_FOLDER, SIMULATIONS_FIGURES_FOLDER

xticks_size = 14
yticks_size = 14
xlabel_size = 18
ylabel_size = 18
legend_size = 12
legend_loc = 'upper left'
linewidth = 3
linestyle = '--'
markersize = 5

def load_hierarchical_general(d,L,idx,ucb=False,known_dist=False,known_noise=False,mis_fp=False,mis_fn=False):
    # UCB is only run for general case: unknown dist, unknown noise
    if ucb is True:
        res = pkl.load(open(SIMULATIONS_ESTIMATED_FOLDER+'/hierarchical'+'/ucb_hierarchical_d_%d_l_%d_graph_%d.pkl'%(d,L,idx), 'rb'))  
    # known_dist result
    elif known_dist is True:
        res = pkl.load(open(SIMULATIONS_ESTIMATED_FOLDER+'/known_dist'+'/hierarchical_known_dist_d_%d_l_%d_graph_%d.pkl'%(d,L,idx), 'rb'))  
    # known_noise result
    elif known_noise is True:
        res = pkl.load(open(SIMULATIONS_ESTIMATED_FOLDER+'/known_noise'+'/hierarchical_known_noise_d_%d_l_%d_graph_%d.pkl'%(d,L,idx), 'rb'))  
    # graph misspec: false positive edges
    elif mis_fp is True:
        res = pkl.load(open(SIMULATIONS_ESTIMATED_FOLDER+'/misspecification_fp'+'/hierarchical_mis_fp_d_%d_l_%d_graph_%d.pkl'%(d,L,idx), 'rb'))  
    # graph misspec: false negative edges
    elif mis_fn is True:
        res = pkl.load(open(SIMULATIONS_ESTIMATED_FOLDER+'/misspecification_fn'+'/hierarchical_mis_fn_d_%d_l_%d_graph_%d.pkl'%(d,L,idx), 'rb'))  
    # if no special case is given, return the main results
    else:
        res = pkl.load(open(SIMULATIONS_ESTIMATED_FOLDER+'/hierarchical'+'/hierarchical_d_%d_l_%d_graph_%d.pkl'%(d,L,idx), 'rb'))  

    cum_reg = res['cum_reg']
    avg_cum_reg = res['avg_cum_reg']
    
    return avg_cum_reg, cum_reg, res

def load_hierarchical_general_all(d,L,indices,ucb=False,known_dist=False,known_noise=False,mis_fp=False,mis_fn=False):

    all_avg_cum_reg = []
    
    for idx in indices:
        all_avg_cum_reg.append(load_hierarchical_general(d,L,idx,ucb,known_dist,known_noise,mis_fp,mis_fn)[0])
        
    return np.asarray(all_avg_cum_reg)    

def plot_hierarchical(d,L,indices,save=True,T=None,compare_ucb=False,known_dist=False,known_noise=False,mis_fp=False,mis_fn=False):

    # load our algo's results. take average regret over graph instances
    reg = np.mean(load_hierarchical_general_all(d,L,indices,False,known_dist,known_noise,mis_fp,mis_fn),0)

    # if want to compare to ucb, also load that. take average regret over graph instances
    if compare_ucb is True:
        reg_ucb = np.mean(load_hierarchical_general_all(d,L,indices,True,known_dist,known_noise,mis_fp,mis_fn),0)

    if compare_ucb is True:
        save_name = SIMULATIONS_FIGURES_FOLDER+'/hierarchical_comp_d_%d_l_%d_avg.eps'%(d,L)
    elif known_dist is True:
        save_name = SIMULATIONS_FIGURES_FOLDER+'/hierarchical_known_dist_d_%d_l_%d_avg.eps'%(d,L)
    elif known_noise is True:
        save_name = SIMULATIONS_FIGURES_FOLDER+'/hierarchical_known_noise_d_%d_l_%d_avg.eps'%(d,L)
    elif mis_fp is True:
        save_name = SIMULATIONS_FIGURES_FOLDER+'/hierarchical_mis_fp_d_%d_l_%d_avg.eps'%(d,L)
    elif mis_fn is True:
        save_name = SIMULATIONS_FIGURES_FOLDER+'/hierarchical_mis_fn_d_%d_l_%d_avg.eps'%(d,L)
    else:
        save_name = SIMULATIONS_FIGURES_FOLDER+'/hierarchical_d_%d_l_%d_avg.eps'%(d,L)

    if T is None:
        if compare_ucb is True:
            T = min(len(reg),len(reg_ucb))
        else:
            T = len(reg)

    plt.figure()
    # plot our results regardless
    plt.plot(reg[:T],'b',markersize=markersize,label='LinSEM-TS-Gaussian',linewidth=linewidth,linestyle=linestyle)
    if compare_ucb is True:
        plt.plot(reg_ucb[:T],'r',markersize=markersize,label='UCB',linewidth=linewidth,linestyle=linestyle)


    plt.xlabel('Number of Iterations',size=xlabel_size)
    plt.ylabel('Cumulative regret',size=ylabel_size)
    plt.xticks(fontsize=xticks_size)
    plt.yticks(fontsize=yticks_size)
    plt.legend(fontsize=legend_size,loc=legend_loc)
    plt.tight_layout()
    plt.grid()
    if save is True:
        plt.savefig(save_name)


def load_enhanced_parallel_general(N,idx,ucb=False,known_dist=False,known_noise=False,mis_fp=False,mis_fn=False):
    # UCB is only run for general case: unknown dist, unknown noise
    if ucb is True:
        res = pkl.load(open(SIMULATIONS_ESTIMATED_FOLDER+'/enhanced_parallel'+'/ucb_size_N_%d_graph_%d.pkl'%(N,idx), 'rb'))    
    # known_dist result
    elif known_dist is True:
        res = pkl.load(open(SIMULATIONS_ESTIMATED_FOLDER+'/known_dist'+'/known_dist_size_N_%d_graph_%d.pkl'%(N,idx), 'rb')) 
    # known_noise result
    elif known_noise is True:
        res = pkl.load(open(SIMULATIONS_ESTIMATED_FOLDER+'/known_dist'+'/known_noise_size_N_%d_graph_%d.pkl'%(N,idx), 'rb')) 
    # graph misspec: false positive edges
    elif mis_fp is True:
        res = pkl.load(open(SIMULATIONS_ESTIMATED_FOLDER+'/misspecification_fp'+'/size_mis_fp_N_%d_graph_%d.pkl'%(N,idx), 'rb')) 
    # graph misspec: false negative edges
    elif mis_fn is True:
        res = pkl.load(open(SIMULATIONS_ESTIMATED_FOLDER+'/misspecification_fn'+'/size_mis_fn_N_%d_graph_%d.pkl'%(N,idx), 'rb')) 
    # if no special case is given, return the main results
    else:
        res = pkl.load(open(SIMULATIONS_ESTIMATED_FOLDER+'/enhanced_parallel'+'/size_N_%d_graph_%d.pkl'%(N,idx), 'rb'))    

    cum_reg = res['cum_reg']
    avg_cum_reg = res['avg_cum_reg']
    
    return avg_cum_reg, cum_reg, res

def load_enhanced_parallel_general_all(N,indices,ucb=False,known_dist=False,known_noise=False,mis_fp=False,mis_fn=False):

    all_avg_cum_reg = []
    
    for idx in indices:
        all_avg_cum_reg.append(load_enhanced_parallel_general(N,idx,ucb,known_dist,known_noise,mis_fp,mis_fn)[0])
        
    return np.asarray(all_avg_cum_reg)    

def plot_enhanced_parallel(N,indices,save=True,T=None,compare_ucb=False,known_dist=False,known_noise=False,mis_fp=False,mis_fn=False):

    # load our algo's results. take average regret over graph instances
    reg = np.mean(load_enhanced_parallel_general_all(N,indices,False,known_dist,known_noise,mis_fp,mis_fn),0)

    # if want to compare to ucb, also load that. take average regret over graph instances
    if compare_ucb is True:
        reg_ucb = np.mean(load_enhanced_parallel_general_all(N,indices,True,known_dist,known_noise,mis_fp,mis_fn),0)

    if compare_ucb is True:
        save_name = SIMULATIONS_FIGURES_FOLDER+'/enhanced_comp_size_N_%d.eps'%(N)
    elif known_dist is True:
        save_name = SIMULATIONS_FIGURES_FOLDER+'/enhanced_known_dist_size_N_%d.eps'%(N)
    elif known_noise is True:
        save_name = SIMULATIONS_FIGURES_FOLDER+'/enhanced_known_noise_size_N_%d.eps'%(N)
    elif mis_fp is True:
        save_name = SIMULATIONS_FIGURES_FOLDER+'/enhanced_mis_fp_size_N_%d.eps'%(N)
    elif mis_fn is True:
        save_name = SIMULATIONS_FIGURES_FOLDER+'/enhanced_mis_fn_size_N_%d.eps'%(N)
    else:
        save_name = SIMULATIONS_FIGURES_FOLDER+'/enhanced_size_N_%d.eps'%(N)

    if T is None:
        if compare_ucb is True:
            T = min(len(reg),len(reg_ucb))
        else:
            T = len(reg)

    plt.figure()
    # plot our results regardless
    plt.plot(reg[:T],'b',markersize=markersize,label='LinSEM-TS-Gaussian',linewidth=linewidth,linestyle=linestyle)
    if compare_ucb is True:
        plt.plot(reg_ucb[:T],'r',markersize=markersize,label='UCB',linewidth=linewidth,linestyle=linestyle)


    plt.xlabel('Number of Iterations',size=xlabel_size)
    plt.ylabel('Cumulative regret',size=ylabel_size)
    plt.xticks(fontsize=xticks_size)
    plt.yticks(fontsize=yticks_size)
    plt.legend(fontsize=legend_size,loc=legend_loc)
    plt.tight_layout()
    plt.grid()
    if save is True:
        plt.savefig(save_name)





            



