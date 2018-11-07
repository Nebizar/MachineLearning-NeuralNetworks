# -*- coding: utf-8 -*-
"""
Created on Mon Nov  5 22:05:34 2018

@author: Krzysztof Pasiewicz
"""
# Upper Confidence Bound

# Importing the libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from math import sqrt, log

# Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# UCB
N = dataset.shape[0]
d = dataset.shape[1]
ads_selected = []
numbers_of_selections = [0] * d
sums_of_rewards = [0] * d
total_reward = 0
for n in range(0, N):
    ad = 0
    max_ub = 0
    for i in range(0, d):
        if(numbers_of_selections[i] > 0):
            avg_reward = sums_of_rewards[i] / numbers_of_selections[i]
            delta = sqrt(3/2 * log(n+1) / numbers_of_selections[i])
            upper_bound = avg_reward + delta
        else:
            upper_bound = 1e400
        if (upper_bound > max_ub):
            max_ub = upper_bound
            ad = i
    ads_selected.append(ad)
    numbers_of_selections[ad] += 1
    reward = dataset.values[n,ad]
    sums_of_rewards[ad] += reward
    total_reward += reward
    
# Visualising the results
plt.hist(ads_selected)
plt.title('Best ads')
plt.xlabel('Advertisments')
plt.ylabel('Number of selections')
plt.show()
            