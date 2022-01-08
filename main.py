from helperfunctions import *
import pandas as pd

## Get the data, clean the data and split into train, val and test

data = pd.read_csv("ship_data.csv", skiprows= 0)

X = data.drop(columns = ['gross_tonnage']).copy()
y = data['gross_tonnage']

X_train, X_val, X_test, y_train, y_val, y_test = getTrainValTest(X,y, 0.8, 0.1)

## Get additional useful features 

## Get a suited algorithm

## Tune the algorithm's hyperparameters 

## Test the algorithm 

## Stack of models 