import sklearn as sk
from sklearn.model_selection import train_test_split


def getTrainValTest(X, y, train_part, val_part):
    """ Randomly shuffle the data for a given test and validation set size

    Parameters
    ----------
    X: pandas dataframe columns
    y: pandas dataframe columns 
    train_part: portion of the data used for the train set. Default = 0.8
    val_part: portion of the data used for validation set. Default = 0.1

    """

    X_train, X_rest, y_train, y_rest = train_test_split(X,y, train_size=train_part)

    test_part = 1 - train_part - val_part

    X_val, X_test, y_val, y_test = train_test_split(X_rest,y_rest, test_size=test_part)

    print("........")
    print("Splitted the data into train, validation and test set")
    print("........")

    return X_train, y_train, X_val, y_val, X_test, y_test

def compareModelScores(name_list, r2_list, rmse_list):
    max_r2 = max(r2_list)
    max_r2_index = r2_list.index(max_r2)

    print("........FINAL RESULTS.......")
    print("The ", name_list[max_r2_index], " has the highest R2 score of ", max_r2)
    
    min_rmse = min(rmse_list)
    min_rmse_index = rmse_list.index(min_rmse)

    print("The ", name_list[min_rmse_index], " has the lowest RMSE score of ", min_rmse)
