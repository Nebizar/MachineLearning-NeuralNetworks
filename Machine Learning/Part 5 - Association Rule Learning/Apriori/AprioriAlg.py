# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 16:49:21 2018

@author: Krzyszof Pasiewicz
"""
# Apriori Algorithm

# Importing the libraries
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

# Importing the dataset
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
transactions = []
for i in range(0,dataset.shape[0]):
    transactions.append([str(dataset.values[i,j]) for j in range(0,dataset.shape[1])])
    
# Training Apriori on the dataset
from apyori import apriori
rules = apriori(transactions, min_support = round(3*7/7500, 3), min_confidence = .2, min_lift = 3, min_length = 2)

# Visualising the Results
results = list(rules)
results_list = []
for i in range(0, len(results)):       
    results_list.append('RULE:\t' + str(results[i][0]) + '\nSUPPORT:\t'
                        + str(results[i][1])+ '\nLIFT:\t' + str(results[i][2]))