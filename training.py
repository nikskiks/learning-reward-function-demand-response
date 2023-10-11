# -*- coding: utf-8 -*-
import datetime
import time
import psutil
import os
import os.path
from os import path

from params import TIME_STEPS_TRAIN, PRICES
from utils.get_results import print_convergence_plots
# from memory_profiler import profile

from utils.value_func import (
    ValueFunctions
)

    
def train_function(
        env, 
        customer_agents, 
        num_episodes, 
        save_path, 
        log, 
        save,
        alphas,
        ext=None,
        irl_train=False,
        rand=False,
):
    """
    This function initiates training of agents in each episode and 
    updates the environment. 

    Parameters
    ----------
    env : object
        Environment object.
    customer_agents : object
        Customer agents object. 
    num_episodes : int
        Number of episodes for training.
    save_path : str
        Path for saving trained networks.
    log : bool
        Indicator whether to save reward values 
    save : bool
        Indicator wheter to save trained networks

    Returns
    -------
    None. 
    

    """
    memory_before = psutil.virtual_memory().used >> 20
    start = time.time()

    cust_rewards = {}
    val_functions = {}
    trajectories_optimal = {}
    
    if irl_train:
        IRL = irl_train
        ALP = alphas
    else:
        IRL = False
        ALP = None
        
    for agent in customer_agents:
        cust_rewards[agent.agent_id] = []
        val_functions[agent.agent_id] = ValueFunctions()
        if not(irl_train):
            trajectories_optimal[agent.agent_id] = []
            
    for episode in range(num_episodes):
        start_episode = time.time()
        # Reset environment and agents
        env.reset(max_steps=TIME_STEPS_TRAIN, irl=IRL, alphas=ALP)
        for agent in customer_agents:
            agent.reset()
        # Train single episode - 1 day
        while not env.done:
            for agent in customer_agents:
                agent.act(rand, train=True)
            env.step()

        # Calculate value function for each agent's trajectory  - input is the whole day
        for agent in customer_agents:
            if not(irl_train):
                trajectories_optimal[agent.agent_id].append(env.trajj[agent.agent_id])
            else:
                val_functions[agent.agent_id].calculate_value(env.trajj[agent.agent_id])
        if log:
            for agent in customer_agents:
                cust_rewards[agent.agent_id].append(
                    customer_agents[agent.agent_id].acc_reward
                )
                           
            
        if episode % 1 == 0:
            print('Episode %d/%d' %(episode, num_episodes))
            print('Day:', datetime.datetime.strptime('{} {}'
                                                     .format(env.day, 2018),
                                                     '%j %Y'))
            print('Episode run time:', (time.time() - start_episode), 'sec')
            print('Cumulated run time:', (time.time() - start), 'sec')
            print('Memory used: ' + str(psutil.virtual_memory().used >> 20) +
                  ' MB')
      
    # Calculate value function for each agent's trajectory
    vs = {}
    
    if not(irl_train):
        pass
    else:
        for agent in customer_agents:
            vs[agent.agent_id] = val_functions[agent.agent_id].get_value()
        
    print('Increase in memory from the beginning: %f MB' 
          %(-memory_before + (psutil.virtual_memory().used >> 20)))
    print('Total run time:', (time.time() - start))
            
    print('Training done')
    
    # Save trained networks
    if save:
        print('Saving networks...')
        save_trained_networks(save_path, customer_agents, ext) 
        
    # Print rewards through episodes
    if log:
        print_convergence_plots(
            save_path, 
            cust_rewards, 
            customer_agents
        )
    if not(irl_train):
        return trajectories_optimal
    else:
        return vs   

def save_trained_networks(save_path, customer_agents, ext=None): 
    """
    Trained customer agents' Q networks are saved.

    Parameters
    ----------
    save_path : str
        Path for saving trained networks.
    customer_agents : object
        Customer agents object. 

    Returns
    -------
    None.

    """
    if not(path.exists(save_path)):
        os.mkdir(save_path)
    for agent in customer_agents:
        agent.save(save_path, ext)