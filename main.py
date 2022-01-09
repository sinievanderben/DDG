from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from helperfunctions import *
from cleanup_features import *
from bonus import *
from tuning import tuning

import pandas as pd

## Get the data

data = pd.read_csv("ship_data.csv", skiprows= 0)
X, y = separateFeatures_Output(data)

X = addFeatures(X)

## Clean the data
data = removeShipname(data)
data = objectToCategorical(data)

## Split the data

X_train, y_train, X_val, y_val, X_test, y_test = getTrainValTest(X,y, 0.8, 0.1)

## Feature selection 
# Based on information from notebook 

subset_X_train = X_train[['cabins',  'length', 'passenger_density',
        'length_per_cabin', 'total_people']]

    
subset_X_val = X_val[['cabins' ,  'length', 'passenger_density',
       'length_per_cabin', 'total_people']]

subset_X_test = X_test[['cabins' ,  'length', 'passenger_density',
       'length_per_cabin', 'total_people']]

## Train Algorithms 

# LR
lr_model = LinearRegression()
lr_model.fit(subset_X_train, y_train)

# RF
rf_model = RandomForestRegressor()
tuned_rf = tuning("RF", rf_model, subset_X_train, y_train)

# GB 
gb_model = GradientBoostingRegressor()
tuned_gb = tuning("GB", gb_model, subset_X_train, y_train)

## Test Algorithms 

model_list = [lr_model, tuned_rf, tuned_gb]

# Save the results for later 
name_list = ["LR", "RF", "GB"]
r2_list = []
rmse_list = []

for i in model_list:
    yhat = i.predict(subset_X_test)
    r2 = r2_score(y_test, yhat)
    rmse = mean_squared_error(y_test, yhat, squared = False)

    r2_list.append(r2)
    rmse_list.append(rmse)

## Get final model and scores

compareModelScores(name_list, r2_list, rmse_list)

## Compare to simple equation 
bonus = Bonus(X_test, y_test)
bonus.compare2equation(name_list, rmse_list)
    