#How to Re Express Categorical Field Values Using R/Python
import pandas as pd
  

# Sample DataFrame
data = {'Category': ['A', 'B', 'C', 'A', 'B']}
df = pd.DataFrame(data)

# Mapping dictionary
mapping_dict = {'A': 'High', 'B': 'Medium', 'C': 'Low'}

# Reexpressing values
df['Category'] = df['Category'].map(mapping_dict)

# Display the updated DataFrame
print(df)