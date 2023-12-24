 # Sample data with misleading values
data = ['good', 'bad', 'average', 'excellent', 'poor', 'good']

# Misleading-to-correct mapping
misleading_to_correct = {
    'bad': 'neutral',
    'poor': 'neutral'
}

# Correct misleading values
corrected_data = [misleading_to_correct[val] if val in misleading_to_correct else val for val in data]

# Display the original and corrected data
print("Original Data:", data)
print("Corrected Data:", corrected_data)

