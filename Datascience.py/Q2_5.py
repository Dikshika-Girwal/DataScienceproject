#How to Identify Outliers Using R/Python

import numpy as np

def identify_outliers_iqr(data):
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return (data < lower_bound) | (data > upper_bound)

# Example usage:
data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 100])
outliers = identify_outliers_iqr(data)

print("Data:", data)
print("Outliers:", data[outliers])