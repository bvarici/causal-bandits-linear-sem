"""
functions for quick plotting of the results
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl

from config import SIMULATIONS_ESTIMATED_FOLDER, SIMULATIONS_FIGURES_FOLDER

#SIMULATIONS_ESTIMATED_FOLDER = '/Users/Burak/Library/Mobile Documents/com~apple~CloudDocs/LAB/TScode_drive/simulations/estimated'
#SIMULATIONS_ESTIMATED_FOLDER = '/Users/Burak/Desktop/TScode/simulations/estimated'

xticks_size = 14
yticks_size = 14
xlabel_size = 18
ylabel_size = 18
legend_size = 12
legend_loc = 'upper left'
linewidth = 3
linestyle = '--'
markersize = 5


def load_enhanced_parallel_res(N,idx,ucb=False):
    if ucb is False:
        res = pkl.load(open(SIMULATIONS_ESTIMATED_FOLDER+'/enhanced_parallel'+'/size_N_%d_graph_%d.pkl'%(N,idx), 'rb'))    
    elif ucb is True:
        res = pkl.load(open(SIMULATIONS_ESTIMATED_FOLDER+'/enhanced_parallel'+'/ucb_size_N_%d_graph_%d.pkl'%(N,idx), 'rb'))    
        
    #reg = res['reg']
    cum_reg = res['cum_reg']
    avg_cum_reg = res['avg_cum_reg']
        
    return avg_cum_reg, cum_reg, res

def load_enhanced_parallel_res_avg(N,indices,ucb=False):
    
    all_avg_cum_reg = []
    T = 1e6
    for idx in indices:
        res = load_enhanced_parallel_res(N, idx,ucb)[0]
        T = min(T,len(res))
        all_avg_cum_reg.append(res[:T])
        
    return np.asarray(all_avg_cum_reg)

def plot_enhanced_parallel_res(N,indices,save=True,T=None,method='ours'):
    
    if method in ['ours','all']:
        if type(indices) is list:
            all_avg_cum_reg = load_enhanced_parallel_res_avg(N, indices,ucb=False)
            avg_cum_reg = np.mean(all_avg_cum_reg,0)
        elif type(indices) is int:
            avg_cum_reg = load_enhanced_parallel_res_avg(N, indices,ucb=False)[0]


    if method in ['ucb','all']:
        if type(indices) is list:
            ucb_all_avg_cum_reg = load_enhanced_parallel_res_avg(N, indices,ucb=True)
            ucb_avg_cum_reg = np.mean(ucb_all_avg_cum_reg,0) 
        elif type(indices) is int:
            ucb_avg_cum_reg = load_enhanced_parallel_res_avg(N, indices,ucb=True)[0]       
        
    if method == 'all':
        if type(indices) is list:
            save_name = SIMULATIONS_FIGURES_FOLDER+'/enhanced_parallel'+'/comp_size_N_%d.eps'%(N)
        elif type(indices) is int:
            save_name = SIMULATIONS_FIGURES_FOLDER+'/enhanced_parallel'+'/comp_size_N_%d_trial_%d.eps'%(N,indices)
    elif method == 'ours':
        if type(indices) is list:
            save_name = SIMULATIONS_FIGURES_FOLDER+'/enhanced_parallel'+'/ours_size_N_%d.eps'%(N)
        elif type(indices) is int:
            save_name = SIMULATIONS_FIGURES_FOLDER+'/enhanced_parallel'+'/ours_size_N_%d_trial_%d.eps'%(N,indices)
    elif method == 'ucb':
        if type(indices) is list:
            save_name = SIMULATIONS_FIGURES_FOLDER+'/enhanced_parallel'+'/ucb_size_N_%d.eps'%(N)
        elif type(indices) is int:
            save_name = SIMULATIONS_FIGURES_FOLDER+'/enhanced_parallel'+'/ucb_size_N_%d_trial_%d.eps'%(N,indices)

    # plot horizon
    if T is None:
        if method == 'all':
            T = min(len(avg_cum_reg),len(ucb_avg_cum_reg))
        elif method == 'ours':
            T = len(avg_cum_reg)
        elif method == 'ucb':
            T = len(ucb_all_avg_cum_reg)
    
    plt.figure()
    if method in ['ours','all']:
        plt.plot(avg_cum_reg[:T],'b',markersize=markersize,label='LinSEM-TS-Gaussian',linewidth=linewidth,linestyle=linestyle)
        
    if method in ['ucb','all']:
        plt.plot(ucb_avg_cum_reg[:T],'r',markersize=markersize,label='UCB',linewidth=linewidth,linestyle=linestyle)

    plt.xlabel('Number of Iterations',size=xlabel_size)
    plt.ylabel('Cumulative regret',size=ylabel_size)
    plt.xticks(fontsize=xticks_size)
    plt.yticks(fontsize=yticks_size)
    plt.legend(fontsize=legend_size,loc=legend_loc)
    plt.tight_layout()
    if save is True:
        plt.savefig(save_name)
             
        

def load_hierarchical_res(d,L,idx=None,ucb=False):
    if ucb is False:
        res = pkl.load(open(SIMULATIONS_ESTIMATED_FOLDER+'/hierarchical'+'/hierarchical_d_%d_l_%d_graph_%d.pkl'%(d,L,idx), 'rb'))
    elif ucb is True:
        res = pkl.load(open(SIMULATIONS_ESTIMATED_FOLDER+'/hierarchical'+'/ucb_hierarchical_d_%d_l_%d_graph_%d.pkl'%(d,L,idx), 'rb'))  

    #reg = res['reg']
    cum_reg = res['cum_reg']
    avg_cum_reg = res['avg_cum_reg']
        
    return avg_cum_reg, cum_reg, res

def load_hierarchical_res_avg(d,L,indices,ucb=False):

    all_avg_cum_reg = []
    
    for idx in indices:
        all_avg_cum_reg.append(load_hierarchical_res(d, L, idx, mode, ucb)[0])
        
    return np.asarray(all_avg_cum_reg)


def plot_hierarchical_res(d,L,indices,save=True,T=None,method='ours'):
    
    if method in ['ours','all']:
        if type(indices) is list:
            all_avg_cum_reg = load_hierarchical_res_avg(d, L, indices,ucb=False)
            avg_cum_reg = np.mean(all_avg_cum_reg,0)
        elif type(indices) is int:
            avg_cum_reg = load_hierarchical_res_avg(d, L, indices,ucb=False)[0]

    if method in ['ucb','all']:
        if type(indices) is list:
            ucb_all_avg_cum_reg = load_hierarchical_res_avg(d, L, indices,ucb=True)
            ucb_avg_cum_reg = np.mean(ucb_all_avg_cum_reg,0) 
        elif type(indices) is int:
            ucb_avg_cum_reg = load_hierarchical_res_avg(d, L, indices,ucb=True)[0]       
        
    if method == 'all':
        if type(indices) is list:
            save_name = SIMULATIONS_FIGURES_FOLDER+'/hierarchical'+'/comp_hierarchical_d_%d_l_%d_avg.eps'%(d,L)
        elif type(indices) is int:
            save_name = SIMULATIONS_FIGURES_FOLDER+'/hierarchical'+'/comp_hierarchical_d_%d_l_%d_trial_%d.eps'%(d,L,indices)
    elif method == 'ours':
        if type(indices) is list:
            save_name = SIMULATIONS_FIGURES_FOLDER+'/hierarchical'+'/ours_hierarchical_d_%d_l_%d_avg.eps'%(d,L)
        elif type(indices) is int:
            save_name = SIMULATIONS_FIGURES_FOLDER+'/hierarchical'+'/ours_hierarchical_d_%d_l_%d_trial_%d.eps'%(d,L,indices)
    elif method == 'ucb':
        if type(indices) is list:
            save_name = SIMULATIONS_FIGURES_FOLDER+'/hierarchical'+'/ucb_hierarchical_d_%d_l_%d_avg.eps'%(d,L)
        elif type(indices) is int:
            save_name = SIMULATIONS_FIGURES_FOLDER+'/hierarchical'+'/ucb_hierarchical_d_%d_l_%d_trial_%d.eps'%(d,L,indices)

    # plot horizon
    if T is None:
        if method == 'all':
            T = min(len(avg_cum_reg),len(ucb_avg_cum_reg))
        elif method == 'ours':
            T = len(avg_cum_reg)
        elif method == 'ucb':
            T = len(ucb_all_avg_cum_reg)
    
    plt.figure()
    if method in ['ours','all']:
        plt.plot(avg_cum_reg[:T],'b',markersize=markersize,label='LinSEM-TS-Gaussian',linewidth=linewidth,linestyle=linestyle)
        
    if method in ['ucb','all']:
        plt.plot(ucb_avg_cum_reg[:T],'r',markersize=markersize,label='UCB',linewidth=linewidth,linestyle=linestyle)

    plt.xlabel('Number of Iterations',size=xlabel_size)
    plt.ylabel('Cumulative regret',size=ylabel_size)
    plt.xticks(fontsize=xticks_size)
    plt.yticks(fontsize=yticks_size)
    plt.legend(fontsize=legend_size,loc=legend_loc)
    plt.tight_layout()
    if save is True:
        plt.savefig(save_name)
        


