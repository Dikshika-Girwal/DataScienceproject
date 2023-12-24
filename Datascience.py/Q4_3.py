import statsmodels.api as sm
import numpy as np

# Generate some example data
np.random.seed(123)
X = np.random.rand(100, 2)  # Your independent variables
lambda_values = np.exp(2 + 0.5 * X[:, 0] + 1.5 * X[:, 1])  # Simulated Poisson rate
y = np.random.poisson(lambda_values)  # Simulated Poisson-distributed counts

# Add a constant term to the independent variables
X = sm.add_constant(X)

# Fit Poisson regression model
poisson_model = sm.GLM(y, X, family=sm.families.Poisson())
poisson_results = poisson_model.fit()

# Display summary
print(poisson_results.summary())