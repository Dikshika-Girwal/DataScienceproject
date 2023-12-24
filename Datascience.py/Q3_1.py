# How to Construct a Bar Graph with Overlay Using R/Python

import matplotlib.pyplot as plt
import numpy as np
 
categories = ['Category 1', 'Category 2', 'Category 3', 'Category 4']
data1 = [10, 15, 7, 12]
data2 = [8, 11, 9, 14]
 
bar_width = 0.35
index = np.arange(len(categories))

plt.bar(index, data1, bar_width, label='Data 1')
plt.bar(index + bar_width, data2, bar_width, label='Data 2')
 
plt.xlabel('Categories')
plt.ylabel('Values')
plt.title('Bar Graph with Overlay')
plt.xticks(index + bar_width / 2, categories)
plt.legend()

plt.show()