from helperfunctions import *
from cleanup_features import *
from bonus import *
import pandas as pd

## Get the data

data = pd.read_csv("ship_data.csv", skiprows= 0)
X, y = separateFeatures_Output(data)

## Clean the data
data = removeShipname(data)
data = objectToCategorical(data)

## Split the data

X_train, X_val, X_test, y_train, y_val, y_test = getTrainValTest(X,y, 0.8, 0.1)

## Train Algorithms 

# LR

# RF

# GB 

## Test Algorithms 

# Save the results for later 
name_list = []
rmse_list = []

## Get final model 

## Compare to simple equation 
bonus = Bonus(X_test, y_test)

for i in range(3):
    Bonus.compare2equation(name_list, rmse_list)
    