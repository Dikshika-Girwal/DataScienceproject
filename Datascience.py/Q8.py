#Demonstrate  application of k‐MEANS CLUSTERING Using R/Python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Generate synthetic data for demonstration
np.random.seed(42)
X, _ = make_blobs(n_samples=300, centers=4, random_state=42)

# Visualize the generated data
plt.scatter(X[:, 0], X[:, 1], s=50)
plt.title('Generated Data')
plt.show()

# Apply k-means clustering
kmeans = KMeans(n_clusters=4, random_state=42)
kmeans.fit(X)

# Get cluster centers and labels
centers = kmeans.cluster_centers_
labels = kmeans.labels_

# Visualize the clustered data
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50, alpha=0.8)
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, marker='X', label='Cluster Centers')
plt.title('K-means Clustering')
plt.legend()
plt.show()