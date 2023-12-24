#Accounting for Unequal Error Costs Using R/Python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

# Load the Iris dataset as an example
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# Create a Decision Tree classifier
decision_tree_classifier = DecisionTreeClassifier(random_state=42)

# Train the classifier on the training data
decision_tree_classifier.fit(X_train, y_train)

# Make predictions on the test data
predictions = decision_tree_classifier.predict(X_test)

# Display the confusion matrix
conf_matrix = confusion_matrix(y_test, predictions)
print("Confusion Matrix:")
print(conf_matrix)

# Define the costs associated with different types of errors
cost_fp = 5  # Cost of a false positive (Type I error)
cost_fn = 10  # Cost of a false negative (Type II error)

# Calculate the total cost
total_cost = cost_fp * conf_matrix[0, 1] + cost_fn * conf_matrix[1, 0]
print("Total Cost:", total_cost)

# Adjusted accuracy accounting for unequal error costs
adjusted_accuracy = accuracy_score(y_test, predictions, sample_weight=[1 if label == 0 else cost_fn/cost_fp for label in y_test])
print("Adjusted Accuracy:", adjusted_accuracy)
