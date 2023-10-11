# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 21:20:21 2023

@author: Korisnik
"""
from gurobipy import *
from gurobipy import GRB, Model
import numpy as np

def objective_function(m, V_opt, V_pis, alp, ds, init_states):
    obj = 0
    z = {}
    for s_0 in init_states:
        for policy in range(0, len(V_pis)):
            diff = 0
            z[s_0, policy] = m.addVar(vtype=GRB.CONTINUOUS, name="z[(%f, %d)]" %(s_0, policy), lb=-np.inf)
            for d in range(0, ds):
                opt_val = V_pis[policy][0].get(s_0, -1)
                if opt_val == -1:
                    continue  
                else:
                    diff += alp[d]*(V_opt[0][s_0][0][d] - V_pis[policy][0][s_0][0][d])
            m.addConstr(z[s_0, policy] <= diff, name="bigM_constr1[(%d)]" %(policy))
            m.addConstr(z[s_0, policy] <= 2*diff, name="bigM_constr2[(%d)]" %(policy))
            obj += z[s_0, policy]
    return -obj

def get_results(m, ds, init_states):
    print('Objective function value:', -m.objVal)
    alp = []
    for i in range (0, ds):
        v = m.getVarByName('alpha[(%d)]' %(i+1)).X
        alp.append(v)
    return alp

def optimizing(V_opt, V_pis, d):
    m = Model('IRL_household_v1')
    alp = {}
    pom = {}
    init_states = list(V_opt[0].keys())
    for i in range(0, d):
        alp[i] = m.addVar(lb = -1, ub = 1, name='alpha[(%d)]' %(i+1))
            
    m.setObjective(
            objective_function(
                m, V_opt, V_pis, alp, d, init_states
            )
    )
    print('Optimizing...')
    m.optimize()
    print('Finished.')
    alphas = get_results(m, d, init_states)
    return alphas

