#How to Perform Poisson Regression Using R/Python
 #import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

#generate synthetic data
np.random.seed(1)
n = 1000
x = np.random.normal(size=n)
y = np.random.poisson(lam=np.exp(x))
data = pd.DataFrame({'x': x, 'y': y})
plt.scatter(x, y, alpha=0.5)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Synthetic Dataset')
plt.show()

#split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

#add polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train.reshape(-1, 1))
X_test_poly = poly.transform(X_test.reshape(-1, 1))

#create Poisson regression object
pois_reg = sm.GLM(y_train, X_train_poly, family=sm.families.Poisson())

#fit Poisson regression on train set
pois_reg_fit = pois_reg.fit(method='newton')
print(pois_reg_fit.summary())

#predict expected counts on test set
y_pred = pois_reg_fit.predict(X_test_poly)
print(y_pred[:5])

#evaluate performance
aic = pois_reg_fit.aic
bic = pois_reg_fit.bic
conf_int = pois_reg_fit.conf_int()
pvalues = pois_reg_fit.pvalues
deviance = pois_reg_fit.deviance
print('AIC:', aic)
print('BIC:', bic)
print('Confidence Intervals:\n', conf_int)
print('P-values:\n', pvalues)
print('Deviance:', deviance)
