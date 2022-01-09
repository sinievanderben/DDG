from helperfunctions import *
from cleanup_features import *
import pandas as pd

## Get the data

data = pd.read_csv("ship_data.csv", skiprows= 0)
X, y = separateFeatures_Output(data)

## Clean the data
data = removeShipname(data)
data = objectToCategorical(data)

## Split the data

X_train, X_val, X_test, y_train, y_val, y_test = getTrainValTest(X,y, 0.8, 0.1)

## Get a suited algorithm

## Tune the algorithm's hyperparameters 

## Test the algorithm 

## Compare to simple equation 