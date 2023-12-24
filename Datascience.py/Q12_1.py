#  How to Mine Association Rules Using R/Python
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Sample transaction data (replace this with your own dataset)
transactions = [
    ['bread', 'milk', 'eggs'],
    ['bread', 'butter', 'eggs'],
    ['milk', 'butter', 'eggs'],
    ['bread', 'milk', 'butter', 'eggs'],
    ['bread', 'milk']
]

# Convert transaction data to a one-hot encoded format
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df = pd.DataFrame(te_ary, columns=te.columns_)

# Apply the Apriori algorithm to find frequent itemsets
frequent_itemsets = apriori(df, min_support=0.2, use_colnames=True)

# Generate association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

# Display the resulting association rules
print(rules)
