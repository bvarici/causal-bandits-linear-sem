"""
plot simulations results

"""

import numpy as np
import matplotlib.pyplot as plt
#import pickle as pkl

from config import SIMULATIONS_FIGURES_FOLDER
#from config import SIMULATIONS_ESTIMATED_FOLDER


from plot_functions import load_hierarchical_general_all
from plot_functions import load_enhanced_parallel_general_all
from plot_functions import plot_hierarchical, plot_enhanced_parallel

xticks_size = 14
yticks_size = 14
xlabel_size = 18
ylabel_size = 18
legend_size = 12
legend_loc = 'upper left'
linewidth = 3
linestyle = '-'
markersize = 5
marker = 'o'

#%%

'load every cumulative results we have'

idxs = [1,2,3,4,5]

## load results for main cases. LinSEM-TS-Gaussian
# hierarchical graphs
reg_2_1 = np.mean(load_hierarchical_general_all(2,1,idxs),0)
reg_2_2 = np.mean(load_hierarchical_general_all(2,2,idxs),0)
reg_2_3 = np.mean(load_hierarchical_general_all(2,3,idxs),0)
reg_3_1 = np.mean(load_hierarchical_general_all(3,1,idxs),0)
reg_3_2 = np.mean(load_hierarchical_general_all(3,2,idxs),0)
reg_3_3 = np.mean(load_hierarchical_general_all(3,3,idxs),0)
reg_4_1 = np.mean(load_hierarchical_general_all(4,1,idxs),0)
reg_4_2 = np.mean(load_hierarchical_general_all(4,2,idxs),0)
reg_4_3 = np.mean(load_hierarchical_general_all(4,3,idxs),0)
# enhanced parallel graphs
reg_5 = np.mean(load_enhanced_parallel_general_all(5,idxs),0)
reg_6 = np.mean(load_enhanced_parallel_general_all(6,idxs),0)
reg_7 = np.mean(load_enhanced_parallel_general_all(7,idxs),0)
reg_8 = np.mean(load_enhanced_parallel_general_all(8,idxs),0)
reg_9 = np.mean(load_enhanced_parallel_general_all(9,idxs),0)

## load results for UCB. 
# hierarchical graphs
reg_ucb_2_1 = np.mean(load_hierarchical_general_all(2,1,idxs,ucb=True),0)
reg_ucb_2_2 = np.mean(load_hierarchical_general_all(2,2,idxs,ucb=True),0)
reg_ucb_2_3 = np.mean(load_hierarchical_general_all(2,3,idxs,ucb=True),0)
reg_ucb_3_1 = np.mean(load_hierarchical_general_all(3,1,idxs,ucb=True),0)
reg_ucb_3_2 = np.mean(load_hierarchical_general_all(3,2,idxs,ucb=True),0)
reg_ucb_3_3 = np.mean(load_hierarchical_general_all(3,3,idxs,ucb=True),0)
reg_ucb_4_1 = np.mean(load_hierarchical_general_all(4,1,idxs,ucb=True),0)
reg_ucb_4_2 = np.mean(load_hierarchical_general_all(4,2,idxs,ucb=True),0)
reg_ucb_4_3 = np.mean(load_hierarchical_general_all(4,3,idxs,ucb=True),0)
# enhanced parallel graphs
reg_ucb_5 = np.mean(load_enhanced_parallel_general_all(5,idxs,ucb=True),0)
reg_ucb_6 = np.mean(load_enhanced_parallel_general_all(6,idxs,ucb=True),0)
reg_ucb_7 = np.mean(load_enhanced_parallel_general_all(7,idxs,ucb=True),0)
reg_ucb_8 = np.mean(load_enhanced_parallel_general_all(8,idxs,ucb=True),0)
reg_ucb_9 = np.mean(load_enhanced_parallel_general_all(9,idxs,ucb=True),0)

## load results for known int. dist.
# hierarchical graphs
reg_known_dist_2_2 = np.mean(load_hierarchical_general_all(2,2,idxs,known_dist=True),0)
reg_known_dist_2_3 = np.mean(load_hierarchical_general_all(2,3,idxs,known_dist=True),0)
reg_known_dist_3_2 = np.mean(load_hierarchical_general_all(3,2,idxs,known_dist=True),0)
reg_known_dist_3_3 = np.mean(load_hierarchical_general_all(3,3,idxs,known_dist=True),0)
reg_known_dist_4_2 = np.mean(load_hierarchical_general_all(4,2,idxs,known_dist=True),0)
reg_known_dist_4_3 = np.mean(load_hierarchical_general_all(4,3,idxs,known_dist=True),0)


# graph misspecification. False positive edges. hieararchical
reg_mis_fp_2_2 = np.mean(load_hierarchical_general_all(2,2,idxs,mis_fp=True),0)
reg_mis_fp_2_3 = np.mean(load_hierarchical_general_all(2,3,idxs,mis_fp=True),0)
reg_mis_fp_3_2 = np.mean(load_hierarchical_general_all(3,2,idxs,mis_fp=True),0)
reg_mis_fp_3_3 = np.mean(load_hierarchical_general_all(3,3,idxs,mis_fp=True),0)
reg_mis_fp_4_2 = np.mean(load_hierarchical_general_all(4,2,idxs,mis_fp=True),0)
reg_mis_fp_4_3 = np.mean(load_hierarchical_general_all(4,3,idxs,mis_fp=True),0)


#%%
idxs = [1,2,3,4,5]
dL_pairs = [(2,1),(2,2),(2,3),(3,1),(3,2),(3,3),(4,1),(4,2),(4,3)]
N_range = [5,6,7,8,9]

'PLOT OUR GENERAL RESULTS ONLY'
for (d,L) in dL_pairs:
    plot_hierarchical(d,L,idxs)
    
for N in N_range:
    plot_enhanced_parallel(N, idxs)
    
'PLOT OUR GENERAL RESULTS ALONG WITH UCB'
for (d,L) in dL_pairs:
    plot_hierarchical(d,L,idxs,compare_ucb=True)
    
for N in N_range:
    plot_enhanced_parallel(N,idxs, compare_ucb=True)

'PLOT OUR KNOWN DIST RESULTS ONLY'
for (d,L) in dL_pairs:
    plot_hierarchical(d,L,idxs,known_dist=True)
    
for N in N_range:
    plot_enhanced_parallel(N,idxs,known_dist=True)



#%%
'FOR OUR HIERARCHICAL RESULTS, SET L CONSTANT, VARY d'
reg_L1 = [reg_2_1[-1],reg_3_1[-1],reg_4_1[-1]]
reg_L2 = [reg_2_2[-1],reg_3_2[-1],reg_4_2[-1]]
reg_L3 = [reg_2_3[-1],reg_3_3[-1],reg_4_3[-1]]
reg_ucb_L1 = [reg_ucb_2_1[-1],reg_ucb_3_1[-1],reg_ucb_4_1[-1]]
reg_ucb_L2 = [reg_ucb_2_2[-1],reg_ucb_3_2[-1],reg_ucb_4_2[-1]]
reg_ucb_L3 = [reg_ucb_2_3[-1],reg_ucb_3_3[-1],reg_ucb_4_3[-1]]

reg_known_dist_L1 = [reg_known_dist_2_1[-1],reg_known_dist_3_1[-1],reg_known_dist_4_1[-1]]
reg_known_dist_L2 = [reg_known_dist_2_2[-1],reg_known_dist_3_2[-1],reg_known_dist_4_2[-1]]
reg_known_dist_L3 = [reg_known_dist_2_3[-1],reg_known_dist_3_3[-1],reg_known_dist_4_3[-1]]

d_range = [2,3,4]
'plot for just our general case'
plt.figure()
plt.plot(d_range,reg_L1,label='L=1',markersize=10,linewidth=linewidth,linestyle='-',marker='o')
plt.plot(d_range,reg_L2,label='L=2',markersize=10,linewidth=linewidth,linestyle='-',marker='o')
plt.plot(d_range,reg_L3,label='L=3',markersize=10,linewidth=linewidth,linestyle='-',marker='o')
plt.xticks(d_range,fontsize=xticks_size)
plt.yticks(fontsize=yticks_size)
plt.xlabel('Degree $d$',size=xlabel_size)
plt.ylabel('Cumulative regret',size=ylabel_size)
plt.ylim(bottom=0)
plt.legend(fontsize=legend_size)
plt.tight_layout()
plt.grid()
plt.savefig(SIMULATIONS_FIGURES_FOLDER+'/hierarchical_vary_d.eps')

'compare with known dist'
plt.figure()
#plt.plot(d_range,reg_L1,'o',label='L=1',markersize=10,linewidth=linewidth,linestyle='-')
plt.plot(d_range,reg_L2,'o',label='L=2',markersize=10,linewidth=linewidth,linestyle='-')
plt.plot(d_range,reg_L3,'o',label='L=3',markersize=10,linewidth=linewidth,linestyle='-')
#plt.plot(d_range,reg_known_dist_L1,'o',label='L=1: known dist',markersize=10,linewidth=linewidth,linestyle='dashed')
plt.plot(d_range,reg_known_dist_L2,'o',label='L=2: known dist',markersize=10,linewidth=linewidth,linestyle='dashed')
plt.plot(d_range,reg_known_dist_L3,'o',label='L=3: known dist',markersize=10,linewidth=linewidth,linestyle='dashed')
plt.xticks(d_range,fontsize=xticks_size)
plt.yticks(fontsize=yticks_size)
plt.xlabel('Degree $d$',size=xlabel_size)
plt.ylabel('Cumulative regret',size=ylabel_size)
plt.ylim(bottom=0)
plt.legend(fontsize=legend_size)
plt.tight_layout()
plt.grid()
plt.savefig(SIMULATIONS_FIGURES_FOLDER+'/hierarchical_vary_d_compare_known_dist.eps')

#%%
'NOW DO IT FOR ENHANCED PARALLEL CASE'
N_range = [5,6,7,8,9]
reg_N = [reg_5[-1],reg_6[-1],reg_7[-1],reg_8[-1],reg_9[-1]]
reg_ucb_N = [reg_ucb_5[-1],reg_ucb_6[-1],reg_ucb_7[-1],reg_ucb_8[-1],reg_ucb_9[-1]]
reg_known_dist_N = [reg_known_dist_5[-1],reg_known_dist_6[-1],reg_known_dist_7[-1],reg_known_dist_8[-1],reg_known_dist_9[-1]]

plt.figure()
plt.plot(N_range,reg_N,'bo',label='LinSEM-TS-Gaussian',markersize=10,linewidth=linewidth,linestyle='-')
plt.plot(N_range,reg_ucb_N,'ro',label='UCB',markersize=10,linewidth=linewidth,linestyle='-')
#plt.plot(N_range,reg_known_dist_N,'ro',label='given dist. knowledge',markersize=10,linewidth=linewidth,linestyle='-')
plt.xticks(N_range,fontsize=xticks_size)
plt.yticks(fontsize=yticks_size)
plt.ylim(bottom=-50)
plt.xlabel('Graph size $N$',size=xlabel_size)
plt.ylabel('Cumulative regret',size=ylabel_size)
#plt.yscale('log')
#plt.ylim(bottom=0)
plt.legend(fontsize=legend_size)
plt.tight_layout()
plt.grid()
plt.savefig(SIMULATIONS_FIGURES_FOLDER+'/enhanced_vary_N_compare_ucb.eps')

#%%
'FINALLY SHOWCASE SOME GRAPH MISSPECIFICATION STUFF'

reg_mis_fp_L1 = [reg_mis_fp_2_1[-1],reg_mis_fp_3_1[-1],reg_mis_fp_4_1[-1]]
reg_mis_fp_L2 = [reg_mis_fp_2_2[-1],reg_mis_fp_3_2[-1],reg_mis_fp_4_2[-1]]
reg_mis_fp_L3 = [reg_mis_fp_2_3[-1],reg_mis_fp_3_3[-1],reg_mis_fp_4_3[-1]]

#%%
d_range = [2,3,4]

plt.figure()
#plt.plot(d_range,reg_L1,'o',label='L=1',markersize=10,linewidth=linewidth,linestyle='-')
plt.plot(d_range,reg_L2,'o',label='L=2',markersize=10,linewidth=linewidth,linestyle='-')
plt.plot(d_range,reg_L3,'o',label='L=3',markersize=10,linewidth=linewidth,linestyle='-')
#plt.plot(d_range,reg_mis_fp_L1,'o',label='L=1: graph misspec.',markersize=10,linewidth=linewidth,linestyle='dashed')
plt.plot(d_range,reg_mis_fp_L2,'o',label='L=2: graph misspec.',markersize=10,linewidth=linewidth,linestyle='dashed')
plt.plot(d_range,reg_mis_fp_L3,'o',label='L=3: graph misspec.',markersize=10,linewidth=linewidth,linestyle='dashed')
plt.xticks(d_range,fontsize=xticks_size)
plt.yticks(fontsize=yticks_size)
plt.xlabel('Degree $d$',size=xlabel_size)
plt.ylabel('Cumulative regret',size=ylabel_size)
plt.ylim(bottom=0)
plt.legend(fontsize=legend_size)
plt.tight_layout()
plt.grid()
plt.savefig(SIMULATIONS_FIGURES_FOLDER+'/hierarchical_vary_d_compare_mis_fp.eps')

#%%

plt.figure()


plt.plot(reg_2_2,label='d=2: grpah misspec.',markersize=markersize,linewidth=linewidth,linestyle='-')
plt.plot(reg_3_2,label='d=3: grpah misspec.',markersize=markersize,linewidth=linewidth,linestyle='-')
plt.plot(reg_4_2,label='d=4: grpah misspec.',markersize=markersize,linewidth=linewidth,linestyle='-')

plt.plot(reg_mis_fp_2_2,label='d=2: grpah misspec.',markersize=10,linewidth=linewidth,linestyle='-')
plt.plot(reg_mis_fp_3_2,label='d=3: graph misspec.',markersize=10,linewidth=linewidth,linestyle='-')
plt.plot(reg_mis_fp_4_2,label='d=4: graph misspec.',markersize=10,linewidth=linewidth,linestyle='-')
