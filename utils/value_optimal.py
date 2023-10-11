# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 15:29:45 2023

@author: Korisnik
"""

from utils.value_func import (
    ValueFunctions
)


def value_optimal(
        env, 
        customer_agents,
        trajs
        ):
    
    cust_rewards = {}
    val_functions = {}
    vs = {}
    
    for agent in customer_agents:
        cust_rewards[agent.agent_id] = []
        val_functions[agent.agent_id] = ValueFunctions()
    
    for agent in customer_agents:
        trajectories_opt = trajs[agent.agent_id]
        for tr in trajectories_opt:
            val_functions[agent.agent_id].calculate_value(tr)

    for agent in customer_agents:
        vs[agent.agent_id] = val_functions[agent.agent_id].get_value()

    return vs