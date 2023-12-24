#How to Standardise Numeric Fields Using R/Python

import numpy as np

data = np.array([
    [25, 50000],
    [30, 60000],
    [35, 75000],
    [40, 90000]
])

# (z-score normalization)
standardized_data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)

print(standardized_data)
