'''
Main functions for causal bandit algorithms on Linear SEMs
'''

import numpy as np
import numpy.linalg as LA
import itertools as itr
import time

def sample_noise(means,variances):
    # assume that nodes are given in topological order
    p = len(means)
    noise = np.zeros(p)
    for ix, (mean,var) in enumerate(zip(means,variances)):
        noise[ix] = np.random.normal(loc=mean,scale=var ** .5)
        
    return noise

def sample_once(B,noise):
    # assume that nodes are given in topological order
    p = len(B)
    sample = np.zeros(p)

    for node in range(p):
        parents_node = np.where(B[node])[0]
        if len(parents_node)!=0:
            parents_vals = sample[parents_node]
            sample[node] = np.sum(parents_vals * B[node,parents_node]) + noise[node]
        else:
            sample[node] = noise[node]
            
    return sample        

def total_effect(W):
    l = 0
    N = len(W)
    Wsum = np.eye(N)
    Wpowerl = W.copy()
    # add powers W^l until the last column becomes zero
    while (not np.all(Wpowerl[-1,:]==0)) and l < N:
        l += 1
        Wsum += Wpowerl
        Wpowerl = Wpowerl@W
        #print(l, Wpowerl)
    #Z = Wsum[-1,:]
    #return Wsum, Z, l
    return Wsum


def causal_linear_sem_TS(parents,Wobs,Wint,nu,Omega,T=100,available_nodes=None,is_reward_intervenable=False,\
                         n_parameter_sampling=10,var_s=0.01,var_p=1.0,n_repeat_data=1):
    '''
    
    Wobs and Wint are 'true' parameters. In Bayesian sense, they are the 'means' of
    the prior of the true distribution (with variance var_s). If you want to investigate it in frequentiest sense,
    then just set n_parameter_sampling=1, and var_s=0.

    to approximate the bayesian regret, we sample the true parameters from this true distribution 
    n_parameter_sampling times to average the regrets in the end.

    for each sampled parameter, we also repeat the experiment n_repeat_data times, 
    to approximate the expectation over data.


    Parameters
    ----------
    parents : dict
        parents for each node, graph structure.
    Wobs : 2d np array
        (means for) true observational weights.
    Wint : 2d np array
        (means for) true interventional weights.
    nu : 1d array
        mean vector for noise terms.
    Omega : 2d array
        covariance matrix for noise terms.
    T : int, optional
        Horizon. The default is 100.
    available_nodes : list, optional
        list of intervenable nodes. The default is None, which is converted to [N].
    n_parameter_sampling : int, optional
        number of times to sample around the true weights.
    n_repeat_data : int, optional
        number of times to repeat the experiment for a chosen parameter set.
    var_s : float, optional
        variance for sampling the true weights. The default is 0.01.
    var_p : float, optional
        variance coefficient for TS posteriors. The default is 1.0.

    Returns
    -------
    res : disct
        returns everything.

    '''
    t0 = time.time()
    # return the necessary outputs
    res = {}
    # number of nodes
    N = len(parents)

    # if available nodes are not specified, all the non-root nodes are intervenable
    if available_nodes is None:
        root_nodes = [i for i in list(parents.keys()) if len(parents[i])==0]
        if is_reward_intervenable is True:
            available_nodes = list(set(np.arange(N))-set(root_nodes))
        else:
            available_nodes = list(set(np.arange(N-1))-set(root_nodes))
            
    # all possible interventions over intervenable nodes.
    Acal = []
    for size in range(0,len(available_nodes)+1):
        sized_sets = list(itr.combinations(available_nodes, size))
        for _ in range(len(sized_sets)):
            Acal.append(sized_sets[_])


    # for debugging
    Wobs_hat_final = np.zeros((n_parameter_sampling,n_repeat_data,N,N))
    Wint_hat_final = np.zeros((n_parameter_sampling,n_repeat_data,N,N))

    # for Bayesian approximation, sample true W from some Gaussian prior.
    # each Wobs(i) and Wint(i) is sampled from smth like N(Wobs(i),I x 0.1^2) 
    Wobs_s = np.zeros((n_parameter_sampling,N,N))
    Wint_s = np.zeros((n_parameter_sampling,N,N))

    # true expected rewards for each intervention. applies over repetitions for paramater_sampling
    mu = {}
    # expected reward under each intervention. no need to store over repetitions.
    mu_hat = {}
    # store the number of pulls for each intervention, over repetitions for parameter_sampling and data_repeats
    Acounts = {}
    # store the selected intervention, over repetitions for parameter_sampling and data_repeats
    #Ahistory = {}

    for idx_parameter_sampling in range(n_parameter_sampling):
        mu[idx_parameter_sampling] = {}
        Acounts[idx_parameter_sampling] = {}
        #Ahistory[idx_parameter_sampling] = {}

        for idx_repeat_data in range(n_repeat_data):
            Acounts[idx_parameter_sampling][idx_repeat_data] = {}
            #Ahistory[idx_parameter_sampling][idx_repeat_data] = []

            for A in Acal:
                Acounts[idx_parameter_sampling][idx_repeat_data][A] = 0


    # store the observed rewards
    observed_rewards = np.zeros((n_parameter_sampling,n_repeat_data,T))
    # store the optimal rewards
    optimal_rewards = np.zeros((n_parameter_sampling,n_repeat_data,T))
    # all data generating Wa matrices
    Wall = {}

    for idx_parameter_sampling in range(n_parameter_sampling):
        for i in range(N):
            if len(parents[i]) > 0:
                Wobs_s[idx_parameter_sampling,i][parents[i]] = np.random.multivariate_normal(Wobs[i][parents[i]], var_s*np.eye(len(parents[i])))
                Wint_s[idx_parameter_sampling,i][parents[i]] = np.random.multivariate_normal(Wint[i][parents[i]], var_s*np.eye(len(parents[i])))

        # fill out the data generating matrices dictironary
        for A in Acal:
            Wa = np.copy(Wobs_s[idx_parameter_sampling])
            for i in A:
                Wa[i] = Wint_s[idx_parameter_sampling,i]
                
            Wall[A] = Wa
            
        # true rewards for each intervention
        for A in Acal:
            mu_a = (total_effect(Wall[A])@nu)[-1]
            mu[idx_parameter_sampling][A] = mu_a
                    
        Astar = max(mu[idx_parameter_sampling], key=mu[idx_parameter_sampling].get)

        # now, repeat the experiment for the chosen parameters for n_repeat_data times
        for idx_repeat_data in range(n_repeat_data):
            # Wobs_hat, Wint_hat, Vobs, Vint, gobs, gint need to be re-initialized for each trial
            Wobs_hat = np.zeros((N,N))
            Wint_hat = np.zeros((N,N))
        
            Vobs = {}
            Vint = {}
            gobs = {}
            gint = {}
            Wobs_tilde = np.zeros((N,N))
            Wint_tilde = np.zeros((N,N))
        
            for i in range(N):
                Vobs[i] = np.eye(len(parents[i]))
                Vint[i] = np.eye(len(parents[i]))
                gobs[i] = np.zeros((len(parents[i]),1))
                gint[i] = np.zeros((len(parents[i]),1))

            for t in range(T):
                # sample weights for each linear problem. 
                for i in range(N):
                    # if no parents, parent weight vector is just all zeros.
                    if len(parents[i]) > 0:
                        Wobs_tilde[i][parents[i]] = np.random.multivariate_normal(Wobs_hat[i][parents[i]], var_p*LA.inv(Vobs[i]))
                        Wint_tilde[i][parents[i]] = np.random.multivariate_normal(Wint_hat[i][parents[i]], var_p*LA.inv(Vint[i]))


                # compute expected reward under each intervention
                for A in Acal:
                    Wa_tilde = np.copy(Wobs_tilde)
                    for i in A:
                        Wa_tilde[i] = Wint_tilde[i]
                
                    mu_a_hat = (total_effect(Wa_tilde)@nu)[-1]
                    mu_hat[A] = mu_a_hat
                    
                # select the best action for maximizing reward
                At = max(mu_hat, key=mu_hat.get)
                # recall the true Wat which generates the data for chosen At
                Wat = Wall[At]
                # play At, get a data sample from the linear SEM with Wat
                epsilon_t = sample_noise(nu,Omega)
                Xt = sample_once(Wat,epsilon_t)
                #D[idx_parameter_sampling,idx_repeat_data,t] = Xt      
                observed_rewards[idx_parameter_sampling,idx_repeat_data,t] = Xt[-1]  
                # also record the reward for optimal action Astar
                optimal_rewards[idx_parameter_sampling,idx_repeat_data,t] = sample_once(Wall[Astar],epsilon_t)[-1]
                
                # increase the arm count for At
                Acounts[idx_parameter_sampling][idx_repeat_data][At] += 1
                # store the selected arm
                #Ahistory[idx_parameter_sampling][idx_repeat_data].append(At)
                # update parameters
                for i in range(N):
                    # if no parents, no update to V,W,g.
                    if len(parents[i]) > 0:
                        # get the zero-padded Xt_pai vector
                        Xt_pai = np.zeros((N,1))
                        Xt_pai = Xt[parents[i]][:,np.newaxis]
                        # intervened nodes
                        if i in At:
                            Vint[i] += Xt_pai@Xt_pai.T
                            gint[i] += Xt_pai*(Xt[i]-nu[i])
                            Wint_hat[i,parents[i]] = (LA.inv(Vint[i])@gint[i])[:,0]
                        # non-intervened nodes
                        else:
                            Vobs[i] += Xt_pai@Xt_pai.T
                            gobs[i] += Xt_pai*(Xt[i]-nu[i])
                            Wobs_hat[i,parents[i]] = (LA.inv(Vobs[i])@gobs[i])[:,0]            


            print('N=%d, idx_parameter:%d, idx_repeat_data:%d'%(N,idx_parameter_sampling+1,idx_repeat_data+1))
            # for debugging
            Wobs_hat_final[idx_parameter_sampling,idx_repeat_data] = Wobs_hat
            Wint_hat_final[idx_parameter_sampling,idx_repeat_data] = Wint_hat

    t_past = time.time() - t0

    reg = optimal_rewards - observed_rewards
    cum_reg = np.cumsum(reg,2)
    avg_cum_reg = np.mean(cum_reg,(0,1))

    # for now, return everything
    res['Wobs_s'] = Wobs_s
    res['Wint_s'] = Wint_s
    res['Wobs_hat_final'] = Wobs_hat_final
    res['Wint_hat_final'] = Wint_hat_final
    res['Acal'] = Acal
    res['Acounts'] = Acounts
    #res['Ahistory'] = Ahistory
    #res['observed_rewards'] = observed_rewards
    #res['optimal_rewards'] = optimal_rewards
    res['mu'] = mu
    res['time'] = t_past
    res['reg'] = reg
    res['cum_reg'] = cum_reg
    res['avg_cum_reg'] = avg_cum_reg

    return res

def causal_linear_sem_TS_known(parents,Wobs,Wint,nu,Omega,T=100,available_nodes=None,is_reward_intervenable=False,\
                         n_parameter_sampling=10,var_s=0.01,var_p=1.0,n_repeat_data=1):
    '''
    
    Wobs and Wint are 'true' parameters. In Bayesian sense, they are the 'means' of
    the prior of the true distribution (with variance var_s). If you want to investigate it in frequentiest sense,
    then just set n_parameter_sampling=1, and var_s=0.

    to approximate the bayesian regret, we sample the true parameters from this true distribution 
    n_parameter_sampling times to average the regrets in the end.

    for each sampled parameter, we also repeat the experiment n_repeat_data times, 
    to approximate the expectation over data.


    Parameters
    ----------
    parents : dict
        parents for each node, graph structure.
    Wobs : 2d np array
        (means for) true observational weights.
    Wint : 2d np array
        (means for) true interventional weights.
    nu : 1d array
        mean vector for noise terms.
    Omega : 2d array
        covariance matrix for noise terms.
    T : int, optional
        Horizon. The default is 100.
    available_nodes : list, optional
        list of intervenable nodes. The default is None, which is converted to [N].
    n_parameter_sampling : int, optional
        number of times to sample around the true weights.
    n_repeat_data : int, optional
        number of times to repeat the experiment for a chosen parameter set.
    var_s : float, optional
        variance for sampling the true weights. The default is 0.01.
    var_p : float, optional
        variance coefficient for TS posteriors. The default is 1.0.

    Returns
    -------
    res : disct
        returns everything.

    '''

    t0 = time.time()
    # return the necessary outputs
    res = {}
    # number of nodes
    N = len(parents)

    # target weights to learn. for this case, only the reward node N, which has index N-1
    learn_nodes = [N-1]
    given_nodes = list(np.arange(N-1))


    # if available nodes are not specified, all the non-root nodes are intervenable
    if available_nodes is None:
        root_nodes = [i for i in list(parents.keys()) if len(parents[i])==0]
        if is_reward_intervenable is True:
            available_nodes = list(set(np.arange(N))-set(root_nodes))
        else:
            available_nodes = list(set(np.arange(N-1))-set(root_nodes))
            
    # all possible interventions over intervenable nodes.
    Acal = []
    for size in range(0,len(available_nodes)+1):
        sized_sets = list(itr.combinations(available_nodes, size))
        for _ in range(len(sized_sets)):
            Acal.append(sized_sets[_])


    # for debugging
    Wobs_hat_final = np.zeros((n_parameter_sampling,n_repeat_data,N,N))
    Wint_hat_final = np.zeros((n_parameter_sampling,n_repeat_data,N,N))

    # for Bayesian approximation, sample true W from some Gaussian prior.
    # each Wobs(i) and Wint(i) is sampled from smth like N(Wobs(i),I x 0.1^2) 
    Wobs_s = np.zeros((n_parameter_sampling,N,N))
    Wint_s = np.zeros((n_parameter_sampling,N,N))

    # true expected rewards for each intervention. applies over repetitions for paramater_sampling
    mu = {}
    # expected reward under each intervention. no need to store over repetitions.
    mu_hat = {}
    # store the number of pulls for each intervention, over repetitions for parameter_sampling and data_repeats
    Acounts = {}
    # store the selected intervention, over repetitions for parameter_sampling and data_repeats
    #Ahistory = {}

    for idx_parameter_sampling in range(n_parameter_sampling):
        mu[idx_parameter_sampling] = {}
        Acounts[idx_parameter_sampling] = {}
        #Ahistory[idx_parameter_sampling] = {}

        for idx_repeat_data in range(n_repeat_data):
            Acounts[idx_parameter_sampling][idx_repeat_data] = {}
            #Ahistory[idx_parameter_sampling][idx_repeat_data] = []

            for A in Acal:
                Acounts[idx_parameter_sampling][idx_repeat_data][A] = 0


    # store the observed rewards
    observed_rewards = np.zeros((n_parameter_sampling,n_repeat_data,T))
    # store the optimal rewards
    optimal_rewards = np.zeros((n_parameter_sampling,n_repeat_data,T))
    # all data generating Wa matrices
    Wall = {}

    for idx_parameter_sampling in range(n_parameter_sampling):
        for i in range(N):
            if len(parents[i]) > 0:
                Wobs_s[idx_parameter_sampling,i][parents[i]] = np.random.multivariate_normal(Wobs[i][parents[i]], var_s*np.eye(len(parents[i])))
                Wint_s[idx_parameter_sampling,i][parents[i]] = np.random.multivariate_normal(Wint[i][parents[i]], var_s*np.eye(len(parents[i])))

        # fill out the data generating matrices dictironary
        for A in Acal:
            Wa = np.copy(Wobs_s[idx_parameter_sampling])
            for i in A:
                Wa[i] = Wint_s[idx_parameter_sampling,i]
                
            Wall[A] = Wa
            
        # true rewards for each intervention
        for A in Acal:
            mu_a = (total_effect(Wall[A])@nu)[-1]
            mu[idx_parameter_sampling][A] = mu_a
                    
        Astar = max(mu[idx_parameter_sampling], key=mu[idx_parameter_sampling].get)

        # now, repeat the experiment for the chosen parameters for n_repeat_data times
        for idx_repeat_data in range(n_repeat_data):
            # Wobs_hat, Wint_hat, Vobs, Vint, gobs, gint need to be re-initialized for each trial
            # We only need to learn the weights of the reward node.
            # only one, theta vector

            Wobs_hat = np.zeros((N,N))
            Wint_hat = np.zeros((N,N))
        
            Vobs = {}
            Vint = {}
            gobs = {}
            gint = {}
            Wobs_tilde = np.zeros((N,N))
            Wint_tilde = np.zeros((N,N))

            # for i = 1 to N-1, it's dummy. Weights will be updated only for the reward node, i.e., node N   
            for i in given_nodes:
                Wobs_hat[i] = Wobs_s[idx_parameter_sampling][i]
                Wint_hat[i] = Wint_s[idx_parameter_sampling][i]
                Wobs_tilde[i] = Wobs_s[idx_parameter_sampling][i]
                Wint_tilde[i] = Wint_s[idx_parameter_sampling][i]
              
            for i in learn_nodes:
                Vobs[i] = np.eye(len(parents[i]))
                Vint[i] = np.eye(len(parents[i]))
                gobs[i] = np.zeros((len(parents[i]),1))
                gint[i] = np.zeros((len(parents[i]),1))

            for t in range(T):
                # sample weights for each linear problem. 
                for i in learn_nodes:
                    # if no parents, parent weight vector is just all zeros.
                    if len(parents[i]) > 0:
                        Wobs_tilde[i][parents[i]] = np.random.multivariate_normal(Wobs_hat[i][parents[i]], var_p*LA.inv(Vobs[i]))
                        Wint_tilde[i][parents[i]] = np.random.multivariate_normal(Wint_hat[i][parents[i]], var_p*LA.inv(Vint[i]))

                # compute expected reward under each intervention
                for A in Acal:
                    Wa_tilde = np.copy(Wobs_tilde)
                    for i in A:
                        Wa_tilde[i] = Wint_tilde[i]
                
                    mu_a_hat = (total_effect(Wa_tilde)@nu)[-1]
                    mu_hat[A] = mu_a_hat
                    
                # select the best action for maximizing reward
                At = max(mu_hat, key=mu_hat.get)
                # recall the true Wat which generates the data for chosen At
                Wat = Wall[At]
                # play At, get a data sample from the linear SEM with Wat
                epsilon_t = sample_noise(nu,Omega)
                Xt = sample_once(Wat,epsilon_t)
                #D[idx_parameter_sampling,idx_repeat_data,t] = Xt      
                observed_rewards[idx_parameter_sampling,idx_repeat_data,t] = Xt[-1]  
                # also record the reward for optimal action Astar
                optimal_rewards[idx_parameter_sampling,idx_repeat_data,t] = sample_once(Wall[Astar],epsilon_t)[-1]
                
                # increase the arm count for At
                Acounts[idx_parameter_sampling][idx_repeat_data][At] += 1
                # store the selected arm
                #Ahistory[idx_parameter_sampling][idx_repeat_data].append(At)
                # update parameters
                for i in learn_nodes:
                    # if no parents, no update to V,W,g.
                    if len(parents[i]) > 0:
                        # get the zero-padded Xt_pai vector
                        Xt_pai = np.zeros((N,1))
                        Xt_pai = Xt[parents[i]][:,np.newaxis]
                        # intervened nodes
                        if i in At:
                            Vint[i] += Xt_pai@Xt_pai.T
                            gint[i] += Xt_pai*(Xt[i]-nu[i])
                            Wint_hat[i,parents[i]] = (LA.inv(Vint[i])@gint[i])[:,0]
                        # non-intervened nodes
                        else:
                            Vobs[i] += Xt_pai@Xt_pai.T
                            gobs[i] += Xt_pai*(Xt[i]-nu[i])
                            Wobs_hat[i,parents[i]] = (LA.inv(Vobs[i])@gobs[i])[:,0]            


            print('N=%d, idx_parameter:%d, idx_repeat_data:%d'%(N,idx_parameter_sampling+1,idx_repeat_data+1))
            # for debugging
            Wobs_hat_final[idx_parameter_sampling,idx_repeat_data] = Wobs_hat
            Wint_hat_final[idx_parameter_sampling,idx_repeat_data] = Wint_hat

    t_past = time.time() - t0

    reg = optimal_rewards - observed_rewards
    cum_reg = np.cumsum(reg,2)
    avg_cum_reg = np.mean(cum_reg,(0,1))

    # for now, return everything
    res['Wobs_s'] = Wobs_s
    res['Wint_s'] = Wint_s
    res['Wobs_hat_final'] = Wobs_hat_final
    res['Wint_hat_final'] = Wint_hat_final
    res['Acal'] = Acal
    res['Acounts'] = Acounts
    #res['Ahistory'] = Ahistory
    #res['observed_rewards'] = observed_rewards
    #res['optimal_rewards'] = optimal_rewards
    res['mu'] = mu
    res['time'] = t_past
    res['reg'] = reg
    res['cum_reg'] = cum_reg
    res['avg_cum_reg'] = avg_cum_reg

    return res


def baseline_UCB(Wobs,Wint,nu,Omega,T=100,available_nodes=None,is_reward_intervenable=False,delta=0.1,n_repeat_data=1):
    t0 = time.time()
    # return the necessary outputs
    res = {}
    # number of nodes
    N = len(Wobs)
    
    # even if we don't use causal graph knowledge, to be fair, we should restrict
    # the set of arms, i.e., exclude root nodes
    # create parents info.
    parents = {}
    for i in range(N):
        parents[i] = list(np.where(Wobs[i])[0])
                            
    # if available nodes are not specified, all the non-root nodes are intervenable
    if available_nodes is None:
        root_nodes = [i for i in list(parents.keys()) if len(parents[i])==0]
        if is_reward_intervenable is True:
            available_nodes = list(set(np.arange(N))-set(root_nodes))
        else:
            available_nodes = list(set(np.arange(N-1))-set(root_nodes))
            

    # all possible interventions over intervenable nodes.
    Acal = []
    for size in range(0,len(available_nodes)+1):
        sized_sets = list(itr.combinations(available_nodes, size))
        for _ in range(len(sized_sets)):
            Acal.append(sized_sets[_])


    # true expected rewards for each intervention. 
    mu = {}
    # expected reward under each intervention. no need to store over repetitions.
    #mu_hat = {}
    # store the number of pulls for each intervention, over repetitions for data_repeats
    Acounts = {}
    # store the selected intervention, over repetitions for data_repeats
    #Ahistory = {}
    # expected reward under each intervention
    Ameans = {}
    ucb_bounds = {}
        
    # for regret computation later, true rewards for each intervention
    mu = {}

    for idx_repeat_data in range(n_repeat_data):
        Acounts[idx_repeat_data] = {}
        #Ahistory[idx_repeat_data] = []
        Ameans[idx_repeat_data] = {}
        ucb_bounds[idx_repeat_data] = {}
        for A in Acal:
            Acounts[idx_repeat_data][A] = 0
            Ameans[idx_repeat_data][A] = 0


    # store observed rewards
    observed_rewards = np.zeros((n_repeat_data,T))
    # store optimal rewards
    optimal_rewards = np.zeros((n_repeat_data,T))
    # all data generating Wa matrices
    Wall = {}
    for A in Acal:
        Wa = np.copy(Wobs)
        for i in A:
            Wa[i] = Wint[i]
            
        Wall[A] = Wa

    # true rewards for each intervention
    for A in Acal:
        mu_a = (total_effect(Wall[A])@nu)[-1]
        mu[A] = mu_a
                
    Astar = max(mu, key=mu.get)



    # now, repeat the experiment for n_repeat_data times
    for idx_repeat_data in range(n_repeat_data):
        for t in range(T):
            # compute UCB bounds for each intervention
            for A in Acal:
                ucb_bounds[idx_repeat_data][A] = Ameans[idx_repeat_data][A] + np.sqrt(2*np.log(1/delta)/max(Acounts[idx_repeat_data][A],1))
                
            # select the best action for maximizing reward
            At = max(ucb_bounds[idx_repeat_data], key=ucb_bounds[idx_repeat_data].get)
            # recall the true Wat
            Wat = Wall[At]
            # play At, get a data sample from the linear SEM with Wat
            epsilon_t = sample_noise(nu,Omega)
            Xt = sample_once(Wat,epsilon_t)
            observed_rewards[idx_repeat_data,t] = Xt[-1]
            # also record the eward for optimal action Astar
            optimal_rewards[idx_repeat_data,t] = sample_once(Wall[Astar],epsilon_t)[-1]
            # update arm means
            Ameans[idx_repeat_data][At] = (Ameans[idx_repeat_data][At]*Acounts[idx_repeat_data][At] + Xt[-1]) / (Acounts[idx_repeat_data][At]+1)
            # increase the arm count for At
            Acounts[idx_repeat_data][At] += 1
            # store the selected arm
            #Ahistory[idx_repeat_data].append(At)
            
        print('UCB N=%d, idx_repeat_data:%d'%(N,idx_repeat_data+1))

    t_past = time.time() - t0

    reg = optimal_rewards - observed_rewards
    cum_reg = np.cumsum(reg,1)
    avg_cum_reg = np.mean(cum_reg,0)

    # for now, return everything
    res['Wobs'] = Wobs
    res['Wint'] = Wint
    res['Wall'] = Wall
    res['Acal'] = Acal
    res['Acounts'] = Acounts
    #res['Ahistory'] = Ahistory
    res['observed_rewards'] = observed_rewards
    res['optimal_rewards'] = optimal_rewards
    res['mu'] = mu
    res['time'] = t_past
    res['Ameans'] = Ameans
    res['ucb_bounds'] = ucb_bounds
    res['reg'] = reg
    res['cum_reg'] = cum_reg
    res['avg_cum_reg'] = avg_cum_reg

    return res













