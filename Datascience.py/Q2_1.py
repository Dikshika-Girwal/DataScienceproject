# How to Add an Index Field Using R/Python

from prettytable import PrettyTable

my_list = ['apple', 'banana', 'orange']

# Using enumerate to add index
indexed_list = list(enumerate(my_list))

# Create a PrettyTable instance
table = PrettyTable()
table.field_names = ["Index", "Value"]

# Add rows to the table
for index, value in indexed_list:
    table.add_row([index, value])

# Print the table
print(table)
