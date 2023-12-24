# How to Construct Histograms with Overlay Using R/Python
import matplotlib.pyplot as plt
import numpy as np

data1 = np.random.normal(0, 1, 1000)  # Mean = 0, Standard Deviation = 1
data2 = np.random.normal(3, 1.5, 1000)  # Mean = 3, Standard Deviation = 1.5
 
plt.hist(data1, bins=30, alpha=0.5, label='Distribution 1', color='blue')
plt.hist(data2, bins=30, alpha=0.5, label='Distribution 2', color='orange')
 
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.title('Histograms with Overlay')

plt.legend()

plt.show()
