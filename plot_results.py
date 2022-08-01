"""
plot simulations results (Figures 1 and 2 in Section 6)

"""

import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl

#from config import SIMULATIONS_ESTIMATED_FOLDER, SIMULATIONS_FIGURES_FOLDER
from config import SIMULATIONS_FIGURES_FOLDER


from plot_functions import plot_enhanced_parallel_res, plot_hierarchical_res
from plot_functions import load_enhanced_parallel_res_avg, load_hierarchical_res_avg

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
'HIERARCHICAL GRAPHS'

dL_pairs = [(2,2),(2,3),(2,3),(2,4),(3,2),(3,3),(3,4),(4,2),(4,3),(4,4)]
for (d,L) in dL_pairs:
    plot_hierarchical_res(d,L,list(range(1,11)),True,method='ours')
    plot_hierarchical_res(d,L,list(range(1,11)),True,method='all')

#%%
avg_cum_reg_2_2 = np.mean(load_hierarchical_res_avg(2,2,list(range(1,11)),ucb=False),0)
avg_cum_reg_2_3 = np.mean(load_hierarchical_res_avg(2,3,list(range(1,11)),ucb=False),0)
avg_cum_reg_2_4 = np.mean(load_hierarchical_res_avg(2,4,list(range(1,11)),ucb=False),0)
avg_cum_reg_3_2 = np.mean(load_hierarchical_res_avg(3,2,list(range(1,11)),ucb=False),0)
avg_cum_reg_3_3 = np.mean(load_hierarchical_res_avg(3,3,list(range(1,11)),ucb=False),0)
avg_cum_reg_3_4 = np.mean(load_hierarchical_res_avg(3,4,list(range(1,11)),ucb=False),0)
avg_cum_reg_4_2 = np.mean(load_hierarchical_res_avg(4,2,list(range(1,11)),ucb=False),0)
avg_cum_reg_4_3 = np.mean(load_hierarchical_res_avg(4,3,list(range(1,11)),ucb=False),0)
avg_cum_reg_4_4 = np.mean(load_hierarchical_res_avg(4,4,list(range(1,11)),ucb=False),0)

        
ucb_avg_cum_reg_2_2 = np.mean(load_hierarchical_res_avg(2,2,list(range(1,11)),ucb=True),0)
ucb_avg_cum_reg_2_3 = np.mean(load_hierarchical_res_avg(2,3,list(range(1,11)),ucb=True),0)
ucb_avg_cum_reg_2_4 = np.mean(load_hierarchical_res_avg(2,4,list(range(1,11)),ucb=True),0)
ucb_avg_cum_reg_3_2 = np.mean(load_hierarchical_res_avg(3,2,list(range(1,11)),ucb=True),0)
ucb_avg_cum_reg_3_3 = np.mean(load_hierarchical_res_avg(3,3,list(range(1,11)),ucb=True),0)
ucb_avg_cum_reg_4_2 = np.mean(load_hierarchical_res_avg(4,2,list(range(1,11)),ucb=True),0)
ucb_avg_cum_reg_4_3 = np.mean(load_hierarchical_res_avg(4,3,list(range(1,11)),ucb=True),0)

#%%
plt.figure()
#plt.title('d=2, vary L')
plt.plot(avg_cum_reg_2_2,label='d=2, L=2',markersize=markersize,linewidth=linewidth,linestyle=linestyle)
plt.plot(avg_cum_reg_2_3,label='d=2, L=3',markersize=markersize,linewidth=linewidth,linestyle=linestyle)
plt.plot(avg_cum_reg_2_4,label='d=2, L=4',markersize=markersize,linewidth=linewidth,linestyle=linestyle)
plt.plot(avg_cum_reg_3_2,label='d=3, L=2',markersize=markersize,linewidth=linewidth,linestyle=linestyle)
plt.plot(avg_cum_reg_3_3,label='d=3, L=3',markersize=markersize,linewidth=linewidth,linestyle=linestyle)
plt.plot(avg_cum_reg_3_4,label='d=3, L=4',markersize=markersize,linewidth=linewidth,linestyle=linestyle)
plt.plot(avg_cum_reg_4_2,label='d=4, L=2',markersize=markersize,linewidth=linewidth,linestyle=linestyle)
plt.plot(avg_cum_reg_4_3,label='d=4, L=3',markersize=markersize,linewidth=linewidth,linestyle=linestyle)
plt.plot(avg_cum_reg_4_4,label='d=4, L=4',markersize=markersize,linewidth=linewidth,linestyle=linestyle)


plt.xlabel('Number of Iterations',size=xlabel_size)
plt.ylabel('Cumulative regret',size=ylabel_size)
plt.xticks(fontsize=xticks_size)
plt.yticks(fontsize=yticks_size)
plt.legend(fontsize=legend_size,loc=legend_loc)
plt.tight_layout()
plt.grid()
plt.savefig(SIMULATIONS_FIGURES_FOLDER+'/hierarchical/hierarchical_vary.eps')

#%%

# set L constant, vary D
ours_L2 = [avg_cum_reg_2_2[-1],avg_cum_reg_3_2[-1],avg_cum_reg_4_2[-1]]
ours_L3 = [avg_cum_reg_2_3[-1],avg_cum_reg_3_3[-1],avg_cum_reg_4_3[-1]]
ours_L4 = [avg_cum_reg_2_4[-1],avg_cum_reg_3_4[-1],avg_cum_reg_4_4[-1]]
ucb_L2 = [ucb_avg_cum_reg_2_2[-1],ucb_avg_cum_reg_3_2[-1],ucb_avg_cum_reg_4_2[-1]]
ucb_L3 = [ucb_avg_cum_reg_2_3[-1],ucb_avg_cum_reg_3_3[-1],ucb_avg_cum_reg_4_3[-1]]

d = [2,3,4]

#%%
plt.figure()
plt.plot(d,ours_L2,label='L=2',markersize=10,linewidth=linewidth,linestyle='-',marker='o')
plt.plot(d,ours_L3,label='L=3',markersize=10,linewidth=linewidth,linestyle='-',marker='o')
plt.plot(d,ours_L4,label='L=4',markersize=10,linewidth=linewidth,linestyle='-',marker='o')

plt.xticks(d,fontsize=xticks_size)
plt.yticks(fontsize=yticks_size)
plt.xlabel('Degree $d$',size=xlabel_size)
plt.ylabel('Cumulative regret',size=ylabel_size)
plt.ylim(bottom=0)
plt.legend(fontsize=legend_size)
plt.tight_layout()
plt.grid()
plt.savefig(SIMULATIONS_FIGURES_FOLDER+'/hierarchical/hierarchical_vary_d.eps')

#%%
plt.figure()
plt.plot(d,ours_L2,'bo',label='LinSEM-TS-Gaussian: L=2',markersize=10,linewidth=linewidth,linestyle='dashed')
plt.plot(d,ours_L3,'bo',label='LinSEM-TS-Gaussian: L=3',markersize=10,linewidth=linewidth,linestyle='dotted')
plt.plot(d,ucb_L2,'ro',label='UCB: L=2',markersize=10,linewidth=linewidth,linestyle='dashed')
plt.plot(d,ucb_L3,'ro',label='UCB: L=3',markersize=10,linewidth=linewidth,linestyle='dotted')
plt.xticks(d,fontsize=xticks_size)
plt.yticks(fontsize=yticks_size)
plt.xlabel('Degree $d$',size=xlabel_size)
plt.ylabel('Cumulative regret',size=ylabel_size)
#plt.ylim(bottom=0)
plt.legend(fontsize=legend_size)
plt.tight_layout()
plt.grid()
plt.savefig(SIMULATIONS_FIGURES_FOLDER+'/hierarchical/hierarchical_vary_d_compare.eps')

#%%
'ENHANCED PARALLEL'

plot_enhanced_parallel_res(5,list(range(1,7)),True,method='ours')
plot_enhanced_parallel_res(6,list(range(1,7)),True,method='ours')
plot_enhanced_parallel_res(7,list(range(1,7)),True,method='ours')
plot_enhanced_parallel_res(8,list(range(1,5)),True,method='ours')
plot_enhanced_parallel_res(9,list(range(1,4)),True,method='ours')

plot_enhanced_parallel_res(5,list(range(1,7)),True,method='all')
plot_enhanced_parallel_res(6,list(range(1,7)),True,method='all')
plot_enhanced_parallel_res(7,list(range(1,7)),True,method='all')
plot_enhanced_parallel_res(8,list(range(1,5)),True,method='all')
plot_enhanced_parallel_res(9,list(range(1,4)),True,method='all')

#%%
# vary N
avg_cum_reg_5 = np.mean(load_enhanced_parallel_res_avg(5,list(range(1,7)),ucb=False),0)
avg_cum_reg_6 = np.mean(load_enhanced_parallel_res_avg(6,list(range(1,7)),ucb=False),0)
avg_cum_reg_7 = np.mean(load_enhanced_parallel_res_avg(7,list(range(1,7)),ucb=False),0)
avg_cum_reg_8 = np.mean(load_enhanced_parallel_res_avg(8,list(range(1,5)),ucb=False),0)
avg_cum_reg_9 = np.mean(load_enhanced_parallel_res_avg(9,list(range(1,4)),ucb=False),0)

ucb_avg_cum_reg_5 = np.mean(load_enhanced_parallel_res_avg(5,list(range(1,7)),ucb=True),0)
ucb_avg_cum_reg_6 = np.mean(load_enhanced_parallel_res_avg(6,list(range(1,7)),ucb=True),0)
ucb_avg_cum_reg_7 = np.mean(load_enhanced_parallel_res_avg(7,list(range(1,7)),ucb=True),0)
ucb_avg_cum_reg_8 = np.mean(load_enhanced_parallel_res_avg(8,list(range(1,5)),ucb=True),0)
ucb_avg_cum_reg_9 = np.mean(load_enhanced_parallel_res_avg(9,list(range(1,4)),ucb=True),0)

#%%
plt.figure()
plt.plot(avg_cum_reg_5,label='N=5',markersize=markersize,linewidth=linewidth,linestyle=linestyle)
plt.plot(avg_cum_reg_6,label='N=6',markersize=markersize,linewidth=linewidth,linestyle=linestyle)
plt.plot(avg_cum_reg_7,label='N=7',markersize=markersize,linewidth=linewidth,linestyle=linestyle)
plt.plot(avg_cum_reg_8,label='N=8',markersize=markersize,linewidth=linewidth,linestyle=linestyle)
plt.plot(avg_cum_reg_9,label='N=9',markersize=markersize,linewidth=linewidth,linestyle=linestyle)

plt.xticks(fontsize=xticks_size)
plt.yticks(fontsize=yticks_size)
plt.ylim(bottom=0)
plt.xlabel('Number of Iterations',size=xlabel_size)
plt.ylabel('Cumulative regret',size=ylabel_size)
plt.legend(fontsize=legend_size)
plt.tight_layout()
plt.grid()
plt.savefig(SIMULATIONS_FIGURES_FOLDER+'/enhanced_parallel/enhanced_parallel_vary_N.eps')

#%%
N = [5,6,7,8,9]
ours_N = [avg_cum_reg_5[-1],avg_cum_reg_6[-1],avg_cum_reg_7[-1],avg_cum_reg_8[-1],\
          avg_cum_reg_9[-1]]
ucb_N = [ucb_avg_cum_reg_5[-1],ucb_avg_cum_reg_6[-1],ucb_avg_cum_reg_7[-1],\
         ucb_avg_cum_reg_8[-1],ucb_avg_cum_reg_9[-1]]


plt.figure()
plt.plot(N,ours_N,'bo',label='LinSEM-TS-Gaussian',markersize=10,linewidth=linewidth,linestyle='-')
plt.plot(N,ucb_N,'ro',label='UCB',markersize=10,linewidth=linewidth,linestyle='-')
plt.xticks(N,fontsize=xticks_size)
plt.yticks(fontsize=yticks_size)
plt.xlabel('Graph size $N$',size=xlabel_size)
plt.ylabel('Cumulative regret',size=ylabel_size)
#plt.yscale('log')
#plt.ylim(bottom=0)
plt.legend(fontsize=legend_size)
plt.tight_layout()
plt.grid()
plt.savefig(SIMULATIONS_FIGURES_FOLDER+'/enhanced_parallel/enhanced_parallel_vary_N_compare.eps')