# -*- coding: utf-8 -*-
"""
Created on Tue Oct  9 17:43:12 2018

@author: Krzysztof Pasiewicz
"""

## Polynomial Regression


# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values #remembered as matrix not a vector
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

# Fitting Linear Regression to data set
from sklearn.linear_model import LinearRegression
linReg = LinearRegression()
linReg.fit(X, y)

# Fitting Polynomial Regression to data set
from sklearn.preprocessing import PolynomialFeatures
polyReg = PolynomialFeatures(degree = 4)
X_poly = polyReg.fit_transform(X)
linReg2 = LinearRegression()
linReg2.fit(X_poly, y)

# Visualisation Simple Linear Regression
plt.scatter(X, y, color = 'red')
plt.plot(X, linReg.predict(X), color = 'blue')
plt.title('Truth or Bluff (Linear Regression)')
plt.xlabel('Position lvl')
plt.ylabel('Salary')
plt.show()

# Visualisation Polynomial Linear Regression
X_grid = np.arange(min(X),max(X), 0.1)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, linReg2.predict(polyReg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Predicting new result with SLR
linReg.predict(6.5)

# Predicting new result with PLR
linReg2.predict(polyReg.fit_transform(6.5))

