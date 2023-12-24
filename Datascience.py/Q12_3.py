# How to Apply the Confidence Quotient Criterion Using R/Python
#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score
from scipy.stats import norm

#generate synthetic data
X, y = make_blobs(n_samples=1000, n_features=2, centers=4, cluster_std=1.5, random_state=1)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='rainbow')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Synthetic Dataset')
plt.show()

#define range of possible values for number of clusters
k_range = range(2, 11)

#initialize empty lists to store CQC values and variance ratios
cqc_values = []
variance_ratios = []

#define alpha parameter
alpha = 3

#loop over k_range
for k in k_range:
    #perform k-means clustering
    kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=300, random_state=1)
    kmeans.fit(X)
    
    #calculate variance ratio
    var_ratio = calinski_harabasz_score(X, kmeans.labels_) * len(X)
    
    #calculate CQC value
    cqc = var_ratio * (1 / (1 + alpha * np.log(len(X))))
    
    #append values to lists
    variance_ratios.append(var_ratio)
    cqc_values.append(cqc)

#plot CQC values
plt.plot(k_range, cqc_values, marker='o')
plt.xlabel('Number of clusters')
plt.ylabel('CQC value')
plt.title('CQC criterion')
plt.grid()
plt.show()

#find optimal number of clusters
opt_k = np.argmax(cqc_values) + 2
print('Optimal number of clusters:', opt_k)
 
 
