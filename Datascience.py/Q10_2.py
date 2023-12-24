# Demonstrate HOW you’ll apply PRINCIPAL COMPONENTS ANALYSIS Using R/Python
 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

# Load a sample dataset (Iris dataset)
iris = load_iris()
X = iris.data
y = iris.target

# Standardize the data (optional but recommended for PCA)
from sklearn.preprocessing import StandardScaler
X_standardized = StandardScaler().fit_transform(X)

# Apply PCA
num_components = 2  # Choose the number of components to keep
pca = PCA(n_components=num_components)
X_pca = pca.fit_transform(X_standardized)

# Visualize the results
plt.figure(figsize=(8, 6))

for i in range(len(iris.target_names)):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], label=iris.target_names[i])

plt.title('PCA of Iris Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()

# Explained Variance
explained_variance_ratio = pca.explained_variance_ratio_
print(f"Explained Variance Ratio: {explained_variance_ratio}")