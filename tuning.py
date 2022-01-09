from sklearn.model_selection import RandomizedSearchCV

rf_grid = {'bootstrap': [True, False],
               'max_depth': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, None],
               'max_features': ['auto', 'sqrt', 1,2,3,4,5],
               'min_samples_leaf': [1, 2, 3, 4, 8, 10, ],
               'min_samples_split': [2, 5, 10],
               'n_estimators': [1,2,4,8,16,32,40, 50, 64,100, 120, 140, 150, 200]}
gb_grid = {
    'n_estimators':[5,50,100,250,500],
    'max_depth':[1,3,5,7,9],
    'learning_rate':[0.01,0.1,1],
    'min_samples_leaf': [1, 2, 3, 4, 8, 10],
    'min_samples_split': [2, 5, 8, 10],
    'max_features': ['auto', 'sqrt', 'log2', None, 1,2,3,4,5]
}

def tuning(name, model, X, y):
    if name == "RF":
        grid = rf_grid
        print("Started tuning Random Forest")
        print("........")
        
        
    if name == "GB":
        grid = gb_grid
        print("Started tuning Gradient Boosting")
        print("........")
        

    
    search = RandomizedSearchCV(model, grid, n_iter=100, scoring = 'neg_root_mean_squared_error', cv=5)
    print("End of tuning the model..")

    final_model = search.fit(X, y)

    return final_model
