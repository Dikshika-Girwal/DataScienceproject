#How to Apply the Confidence Difference Criterion Using R/Python
import pandas as pd

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Example transaction data
transactions = [['item1', 'item2', 'item3'],
                ['item1', 'item2'],
                ['item1', 'item3'],
                ['item2', 'item3'],
                ['item1', 'item2', 'item3', 'item4']]

# Convert data to transaction format
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)

# Find frequent itemsets
frequent_itemsets = apriori(df, min_support=0.2, use_colnames=True)

# Generate association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

# Calculate confidence difference between two rules
rule1 = rules.iloc[0]  # Change the index as needed
rule2 = rules.iloc[1]  # Change the index as needed

confidence_difference = rule1['confidence'] - rule2['confidence']
print("Confidence Difference:", confidence_difference)