#Demonstrate How you will  Identify Multicollinearity R/Python

import pandas as pd
from patsy import dmatrices
from statsmodels.stats.outliers_influence import variance_inflation_factor

#create DataFrame
df = pd.DataFrame(
    {'rating': [90, 85, 82, 88, 94, 90, 76, 75, 87, 86],
     'points': [25, 20, 14, 16, 27, 20, 12, 15, 14, 19],
     'assists': [5, 7, 7, 8, 5, 7, 6, 9, 9, 5],
     'rebounds': [11, 8, 10, 6, 6, 9, 6, 10, 10, 7]})

#find design matrix for regression model using 'rating' as response variable
y, X = dmatrices('rating ~ points + assists + rebounds', data=df, return_type='dataframe')

#create DataFrame to hold VIF values
vif_df = pd.DataFrame()
vif_df['variable'] = X.columns

#calculate VIF for each predictor variable in the model
vif_df['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

#view VIF for each predictor variable
print(vif_df)

 