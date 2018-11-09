# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 14:04:04 2018

@author: Krzysztof Pasiewicz
"""
# Artificial Neural Network

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X1 = LabelEncoder()
X[:, 1] = labelencoder_X1.fit_transform(X[:, 1])
labelencoder_X2 = LabelEncoder()
X[:, 2] = labelencoder_X2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1]) # Dummy variable
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:] #  Dummy Variable Trap taken care of

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Building an Artificial Neural Network

# Importing needed libraries
# import keras
from keras.models import Sequential
from keras.layers import Dense

# Init
classifier = Sequential()

# Input layer and first hidden layer
classifier.add(Dense(units = 6, activation = 'relu', kernel_initializer = 'uniform', input_dim = 11))

# Second hidden layer
classifier.add(Dense(units = 6, activation = 'relu', kernel_initializer = 'uniform'))

# Output layer
classifier.add(Dense(units = 1, activation = 'sigmoid', kernel_initializer = 'uniform'))

# Compile
classifier.compile(optimizer = 'Adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
# Adam - good and efective Stochastic Gradient Descent 
# for more categories = categorical_crossentropy

# Training the Neural Network
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

# Predicting the Test Set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > .5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

accuracy = (cm[0,0]+cm[1,1])/ sum(sum(cm))

