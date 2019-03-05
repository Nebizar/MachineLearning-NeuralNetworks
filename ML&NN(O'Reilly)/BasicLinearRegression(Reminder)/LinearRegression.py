# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 18:33:33 2019

@author: Krzysztof Pasiewicz
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn

oecd = pd.read_csv("country_hapinness.csv", thousands = ",")
pkb = pd.read_csv("country_PKB.csv",thousands=",", delimiter = ';', encoding = 'latin1', na_values = "bd")

data = pd.merge(oecd, pkb, on = "Country")

