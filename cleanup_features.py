import pandas as pd

# Separate the features and the output 
def separateFeatures_Output(data):
    X = data.drop(columns = ['gross_tonnage']).copy()
    y = data['gross_tonnage']
    return X, y

# The ship name has no influence on the prediction 
def removeShipname(data):
    data = data.drop(columns = ['Ship_name']).copy()

    print("Shipname removed")
    print("........")

    return data

# Change all object columns to categorical 
def objectToCategorical(data):

    count = 0 

    for column_name in data.columns:
        if data.dtypes[column_name] == "object":
            data[column_name].astype('category').cat.codes
            count = count + 1
    
    print("Number of converted columns: ", count)

    return data 

def addFeatures(X):
    for i in range(158):
        X["passenger_per_cabin"] = X["cabins"] / (X["passengers"] / 10)
        X["length_per_cabin"] = X["cabins"] / X["length"]
        X["passengers_per_length"] = X["length"] / X["passengers"]
        X["total_people"] = X["crew"] + X["passengers"]
    
    return X



