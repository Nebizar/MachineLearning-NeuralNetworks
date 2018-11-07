# -*- coding: utf-8 -*-
"""
Created on Tue Nov  6 16:02:28 2018

@author: Krzysztof Pasiewicz
"""
# Thompson Sampling

# Importing the libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import random

# Importing the dataset
dataset = pd.read_csv('Ads_CTR_Optimisation.csv')

# Thompson Sampling
N = dataset.shape[0]
d = dataset.shape[1]
ads_selected = []
numbers_of_rewards_1 = [0] * d
numbers_of_rewards_0 = [0] * d
total_reward = 0
for n in range(0, N):
    ad = 0
    max_rand = 0
    for i in range(0, d):
        rand_beta = random.betavariate(numbers_of_rewards_1[i] + 1, numbers_of_rewards_0[i] + 1)
        if (rand_beta > max_rand):
            max_rand = rand_beta
            ad = i
    ads_selected.append(ad)
    reward = dataset.values[n,ad]
    if (reward == 1):
        numbers_of_rewards_1[ad] += 1
    else:
        numbers_of_rewards_0[ad] += 1 
    total_reward += reward
    
# Visualising the results
plt.hist(ads_selected)
plt.title('Best ads')
plt.xlabel('Advertisments')
plt.ylabel('Number of selections')
plt.show()


