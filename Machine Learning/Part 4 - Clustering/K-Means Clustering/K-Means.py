# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 12:24:32 2018

@author: Krzysztof Pasiewicz
"""
# K-Means Clustering

#%reset -f

# Importing libraries
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Mall_Customers.csv')
X = dataset.iloc[:,[3,4]].values

# finding optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters=i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1,11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.show()

# Applying K-Means wih optimal clusters
kmeans = KMeans(n_clusters = 5, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(X)

# Visualising the clusters
plt.scatter(X[y_kmeans==0, 0], X[y_kmeans == 0 , 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans==1, 0], X[y_kmeans == 1 , 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans==2, 0], X[y_kmeans == 2 , 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans==3, 0], X[y_kmeans == 3 , 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_kmeans==4, 0], X[y_kmeans == 4 , 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of clients')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score')
plt.legend()
plt.show()
