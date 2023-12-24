#How to Perform Model Evaluation Using R/Python

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
 
X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
 
df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
df['target_variable'] = y
 
X = df.drop('target_variable', axis=1)
y = df['target_variable']
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
 
model = LogisticRegression()
model.fit(X_train, y_train)
 
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
 
conf_matrix = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:')
print(conf_matrix)
 
class_report = classification_report(y_test, y_pred)
print('Classification Report:')
print(class_report)
 
cv_scores = cross_val_score(model, X, y, cv=5)
print(f'Cross-Validation Scores: {cv_scores}')
print(f'Mean CV Score: {np.mean(cv_scores)}')
 
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
 
grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print(f'Best Hyperparameters: {best_params}')

