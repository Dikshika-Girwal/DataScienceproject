import numpy as np
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Generate a synthetic imbalanced dataset
X, y = make_classification(
    n_classes=2, class_sep=2, weights=[0.1, 0.9], n_informative=3,
    n_redundant=1, flip_y=0, n_features=20, n_clusters_per_class=1,
    n_samples=1000, random_state=42
)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display class distribution in the original dataset
print("Class distribution in the original dataset:")
print("Class 0: ", np.sum(y == 0))
print("Class 1: ", np.sum(y == 1))
print()

# Apply SMOTE to balance the class distribution in the training set
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Display class distribution in the balanced dataset
print("Class distribution in the balanced dataset:")
print("Class 0: ", np.sum(y_train_resampled == 0))
print("Class 1: ", np.sum(y_train_resampled == 1))
print()

# Train a RandomForest classifier on the balanced training set
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train_resampled, y_train_resampled)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

# Display the results
print("Model Accuracy: {:.2f}%".format(accuracy * 100))
print("\nClassification Report:\n", report)
