#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 23:09:05 2021

@author: ryanmarshall
"""
import pandas as pd
import cvxpy as cp
import numpy as np

# read in csv file
data = pd.read_csv("raptor.csv")


# convert salaries to type int
for i in range(len(data.salary)):
    data.loc[i, 'salary'] = int(float(data.salary[i].strip('$').replace(',', '')))

# 362 boolean decision vars
x = cp.Variable(len(data.player_name), boolean=True)

# total impact as obj_func
obj_func = x.T @ data.raptor_total

constraints = []
#choose 12 players
constraints.append(sum(x) == 12)
# total team salary constraint $110 mil
constraints.append(x.T @ data.salary <= 110000000)
# avg offense/defense raptor for roster at least 3.0
constraints.append(x.T @ data.raptor_offense >= 36)
constraints.append(x.T @ data.raptor_offense >= 36)
# select at least 4 guards, 4 forwards, 3 centers
constraints.append(x.T @ data.guard >= 4)
constraints.append(x.T @ data.forward >= 5)
constraints.append(x.T @ data.center >= 3)


problem = cp.Problem(cp.Maximize(obj_func), constraints)
problem.solve(solver=cp.GUROBI,verbose = True)

# match decision variables to player names
team = []
for i in range(len(data.player_name)):
    if x.value[i] == 1: 
        print(data.player_name[i], " ", data.position[i], " ", data.raptor_total[i])
        
#display roster
#print(team)


    