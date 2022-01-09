# Compare equation to model outcomes and show which are higher 
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error

class Bonus:

    def __init__(self, X, y):
        self.X = X
        self.y = y
        
        # Directly perform this action 
        self.y_manual_prediction = (self.X["passengers"] * self.X["passenger_density"])/10
        self.rmse_manual = mean_squared_error(self.y, self.y_manual_prediction, squared = False)

    def compare2equation(self, model_name, model_rmse):
        if self.rmse_manual > model_rmse:
            print("Multiplying \'passenger' * \'passenger_density gives a higher RMSE compared to the RMSE of ", model_name)
        else:
            print(model_name, " outperforms multiplication of \'passenger' * \'passenger_density")

