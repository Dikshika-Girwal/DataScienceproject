#How to Perform Binning Based on Predictive Value Using R/Python

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import KBinsDiscretizer
import matplotlib.pyplot as plt
 
np.random.seed(42)
X = np.random.rand(100, 1) * 10
y = 2 * X.squeeze() + np.random.randn(100) * 2  # Linear relationship with noise
 
model = LinearRegression()
model.fit(X, y)
 
predicted_values = model.predict(X)
 
k_bins_discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile')
binned_values = k_bins_discretizer.fit_transform(predicted_values.reshape(-1, 1))
 
plt.scatter(X, y, label='Original Data', color='blue')
plt.scatter(X, binned_values, label='Binned Data', color='red', marker='x')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.title('Binning Based on Predictive Value')
plt.show()
