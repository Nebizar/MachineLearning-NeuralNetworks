# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 18:33:33 2019

@author: Krzysztof Pasiewicz
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

def data_preprocessing(file1, file2):
    oecd = pd.read_csv(file1, delimiter = ",")
    pkb = pd.read_csv(file2,thousands=",", delimiter = ';', encoding = 'latin1', na_values = "bd")
    
    data = pd.merge(oecd, pkb, on = "Country")
    
    data_refined = data[['Country','2015','Reference Period Code', 'INDICATOR']]
    data_refined = data_refined.loc[data_refined['INDICATOR'] == 'SW_LIFS']
    
    data_refined = data_refined.groupby('Country').mean().reset_index()
    
    X = data_refined.iloc[:, 1].values
    y = data_refined.iloc[:, -1].values
    X = X.reshape(-1,1)
    y = y.reshape(-1,1)
    
    return X, y

def trainLinear(X, y):
    regressor = LinearRegression()
    regressor.fit(X,y)
    
    return regressor

def trainPolynomial(X, y):
    poly = PolynomialFeatures(degree = 5)
    X_poly = poly.fit_transform(X)
    polynomialRegressor = LinearRegression()
    polynomialRegressor.fit(X_poly, y)
    
    return polynomialRegressor

def plotRegression(X,y,regressor, polynomialRegressor):
    fig = plt.figure()
    
    ax1 = fig.add_subplot(121)
    ax1.scatter(X, y, color = 'red')
    ax1.plot(X, regressor.predict(X), color = 'blue')
    ax1.set_title('PKB to Happinness (Linear Regression)')
    ax1.set_xlabel('PKB')
    ax1.set_ylabel('Happinness')
    
    X_grid = np.arange(min(X),max(X), 0.1)
    X_grid = X_grid.reshape(len(X_grid), 1)
    
    poly = PolynomialFeatures(degree = 5)
    ax2 = fig.add_subplot(122)
    ax2.scatter(X, y, color = 'red')
    ax2.plot(X_grid, polynomialRegressor.predict(poly.fit_transform(X_grid)), color = 'blue')
    ax2.set_title('PKB to Happinness (Polynomial Regression)')
    ax2.set_xlabel('PKB')
    plt.show()

    

def main():
    X, y = data_preprocessing("country_hapinness.csv", "country_PKB.csv")
    regressor1 = trainLinear(X, y)
    regressor2 = trainPolynomial(X,y)
    plotRegression(X,y,regressor1, regressor2)
    

if __name__ == '__main__':
    main()