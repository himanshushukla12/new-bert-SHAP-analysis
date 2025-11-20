from sklearn.linear_model import Ridge, Lasso
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_validate
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb
import pandas as pd
import numpy as np

class ModelTrainer:
    def __init__(self):
        self.models = {
            "Ridge": Ridge(),
            "Lasso": Lasso(),
            "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=10),
            "KNN": KNeighborsRegressor(),
            "LightGBM": lgb.LGBMRegressor(verbose=-1),
            # Naive Bayes is typically for classification, but we are doing regression (rating 1-5).
            # We can use BayesianRidge or just skip NB for regression tasks unless we bin the target.
            # The prompt asks for "Naive Bayes", likely implying classification or just a list of standard models.
            # We will stick to Regressors for "Satisfaction Prediction" (1-5 continuous-ish).
            # If classification is preferred, we'd change this. Let's assume Regression for metrics MSE/RMSE.
        }
        self.best_model = None
        self.best_model_name = ""
        self.results = {}

    def train_and_evaluate(self, X, y):
        best_rmse = float('inf')
        
        for name, model in self.models.items():
            # 5-fold CV
            cv_results = cross_validate(model, X, y, cv=5, scoring=['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2'])
            
            mse = -cv_results['test_neg_mean_squared_error'].mean()
            mae = -cv_results['test_neg_mean_absolute_error'].mean()
            r2 = cv_results['test_r2'].mean()
            rmse = np.sqrt(mse)
            
            self.results[name] = {
                "MSE": mse,
                "RMSE": rmse,
                "MAE": mae,
                "R2": r2
            }
            
            if rmse < best_rmse:
                best_rmse = rmse
                self.best_model = model
                self.best_model_name = name
        
        # Retrain best model on full data
        self.best_model.fit(X, y)
        return self.results, self.best_model_name
