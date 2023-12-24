#How to Perform Logistic Regression Using R/Python

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

#generate synthetic data
X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=1)
plt.scatter(X[:, 0], X[:, 1], marker='o', c=y, s=25, edgecolor='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Synthetic Dataset')
plt.show()

#split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

#scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#create logistic regression object
log_reg = LogisticRegression(solver='lbfgs', random_state=1)

#fit logistic regression on train set
log_reg.fit(X_train, y_train)

#predict probabilities on test set
y_pred_proba = log_reg.predict_proba(X_test)
print(y_pred_proba[:5])

#choose threshold and make predictions
threshold = 0.5
y_pred = np.where(y_pred_proba[:, 1] >= threshold, 1, 0)
print(y_pred[:5])

#evaluate performance
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)
conf_mat = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:\n', conf_mat)
