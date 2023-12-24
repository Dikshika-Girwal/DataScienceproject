# How to Construct Contingency Tables Using R/Python
import pandas as pd

# Sample dataset
data = {'Category': ['A', 'B', 'A', 'B', 'A', 'B', 'A', 'A'],
        'Outcome': ['Success', 'Failure', 'Success', 'Success', 'Failure', 'Success', 'Failure', 'Success']}
df = pd.DataFrame(data)

# Constructing the contingency table
contingency_table = pd.crosstab(df['Category'], df['Outcome'])

# Display the contingency table
print("Contingency Table:")
print(contingency_table)

# Adding margins for row and column totals
contingency_table_with_margins = pd.crosstab(df['Category'], df['Outcome'], margins=True, margins_name="Total")

# Display the contingency table with margins
print("\nContingency Table with Margins:")
print(contingency_table_with_margins)


