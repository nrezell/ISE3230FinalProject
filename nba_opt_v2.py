#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 29 22:24:35 2021

@author: ryanmarshall
"""

import pandas as pd
import cvxpy as cp
import numpy as np
from prettytable import PrettyTable

'''
Data based on 2020-2021 NBA season
Includes player impact (offensive/defensive/total) as estimated by 538's RAPTOR
along with position and salary for 20-21 season. Includes only players with
at least 500 minutes played in 2020-21.
'''
data = pd.read_csv("raptor.csv")

# convert salaries to type int
for i in range(len(data.salary)):
    data.loc[i, 'salary'] = int(float(data.salary[i].strip('$').replace(',', '')))

# 362 boolean decision vars
s = cp.Variable(len(data.player_name), boolean=True)
b = cp.Variable(len(data.player_name), boolean=True)

# total impact as obj_func 
# assuming 36 minutes for starters and 12 for bench players
obj_func = 36*(s.T @ data.raptor_total) + 12*(b.T @ data.raptor_total)

constraints = []
# select 12 players – 5 starters + 7 bench
constraints.append(sum(s) == 5)
constraints.append(sum(b) == 7)

# total team salary constraint $109 mil
# based on 
constraints.append((s.T + b.T) @ data.salary <= 109000000)

# combined bench RAPTOR must be nonnegative for offense/defense
constraints.append(b.T @ data.raptor_defense >= 0)
constraints.append(b.T @ data.raptor_offense >= 0)

# starters to include 2 guards, 2 forwards, 1 center 
constraints.append(s.T @ data.guard == 2)
constraints.append(s.T @ data.forward == 2)
constraints.append(s.T @ data.center == 1)

# bench to include at least 2 guards, at least 2 forwards, and exactly 2 centers
constraints.append(b.T @ data.guard >= 2)
constraints.append(b.T @ data.forward >= 2)
constraints.append(b.T @ data.center == 2)

# ensure player can't be a starter and bench player 
for i in range(len(data.player_name)):
    constraints.append(s[i] + b[i] <= 1)

# build optimal roster
problem = cp.Problem(cp.Maximize(obj_func), constraints)
problem.solve(solver=cp.GUROBI,verbose = False)

#construct table to display optimal team
t = PrettyTable(['Player', 'Position', 'Offensive Raptor', 
                 'Defensive Raptor', 'Total Raptor', 'Role'])
for i in range(len(data.player_name)):
    if s.value[i] == 1: 
        t.add_row([data.player_name[i], data.position[i], data.raptor_offense[i], 
                   data.raptor_defense[i], data.raptor_total[i], "Starter"])
        t.add_row(['','','','','',''])

t.add_row(['','','','','',''])

for i in range(len(data.player_name)):
    if b.value[i] == 1:
        t.add_row([data.player_name[i], data.position[i], data.raptor_offense[i], 
                   data.raptor_defense[i], data.raptor_total[i], "Bench"])
        t.add_row(['','','','','',''])
print(t)



    