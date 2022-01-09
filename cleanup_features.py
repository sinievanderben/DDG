import pandas as pd

# Separate the features and the output 
def separateFeatures_Output(data):
    X = data.drop(columns = ['gross_tonnage']).copy()
    y = data['gross_tonnage']
    return X, y

# The ship name has no influence on the prediction 
def removeShipname(data):
    data = data.drop(columns = ['Ship_name']).copy()

# Change all object columns to categorical 
def objectToCategorical(data):
    for column in data:
        if column.dtype == "object":
            data.column.astype('category').cat.codes
    

    return data 



