#Demonstrate Stepwise Regression Using R/Python

import pandas as pd
import numpy as np
import statsmodels.api as sm

# Generate some sample data
np.random.seed(0)
data = pd.DataFrame({
    'X1': np.random.rand(100),
    'X2': np.random.rand(100),
    'X3': np.random.rand(100),
    'Y': 2 * np.random.rand(100) + 3
})

# Add a random column that is not related to the target variable
data['X4'] = np.random.rand(100)

# Function for stepwise regression
def stepwise_regression(data, target_col, criteria='aic'):
    X = data.drop(columns=[target_col])
    y = data[target_col]

    # Stepwise regression
    included = []
    while True:
        changed = False
        excluded = list(set(X.columns) - set(included))
        best_pvalue = 1
        candidate = None

        for new_column in excluded:
            model = sm.OLS(y, sm.add_constant(X[included + [new_column]])).fit()
            if criteria == 'aic':
                criteria_value = model.aic
            elif criteria == 'bic':
                criteria_value = model.bic
            else:
                raise ValueError("Invalid criteria. Use 'aic' or 'bic'.")

            if criteria_value < best_pvalue:
                best_pvalue = criteria_value
                candidate = new_column

        if candidate is not None:
            included.append(candidate)
            changed = True

        if not changed:
            break

    return sm.OLS(y, sm.add_constant(X[included])).fit()

# Perform stepwise regression
result = stepwise_regression(data, target_col='Y', criteria='aic')

# Display the summary
print(result.summary())
