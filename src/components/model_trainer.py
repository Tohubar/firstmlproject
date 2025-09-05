import os  
import sys   
import numpy as np  
import pandas as pd 
from dataclasses import dataclass
 
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path:str = os.path.join('artifacts', 'model.pkl')
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Stariting the model training")
            logging.info("splitting model train and test data")
            
            X_train, X_test, y_train, y_test = (
                train_array[:, : -1],
                test_array[:, : -1],
                train_array[:, -1],
                test_array[:, -1]
            )
            models = {
                "Linear Regression": LinearRegression(), 
                "Random Forest": RandomForestRegressor(),
                "KNearest Neighbour": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "XG Boost": XGBRegressor(),
                "Cat Boost": CatBoostRegressor(verbose= False),
                "Ada Boost": AdaBoostRegressor(),
                "Gradien Boost": GradientBoostingRegressor(),
            }
            param_grids = {
                "Linear Regression": {},
                "Random Forest": {
                    "n_estimators": [100, 200, 500],
                    "max_depth": [None, 10, 20, 30],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "max_features": ["auto", "sqrt", "log2"]
                },
                "KNearest Neighbour": {
                    "n_neighbors": [3, 5, 7, 9, 11],
                    "weights": ["uniform", "distance"],
                    "metric": ["euclidean", "manhattan", "minkowski"]
                },
                "Decision Tree": {
                    "criterion": ["squared_error", "friedman_mse", "absolute_error", "poisson"],
                    "max_depth": [None, 5, 10, 20, 30],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "splitter": ["best", "random"]
                },
                "XG Boost": {
                    "n_estimators": [100, 200, 500],
                    "learning_rate": [0.01, 0.05, 0.1, 0.2],
                    "max_depth": [3, 5, 7, 10],
                    "subsample": [0.6, 0.8, 1.0],
                    "colsample_bytree": [0.6, 0.8, 1.0]
                },
                "Cat Boost": {
                    "iterations": [200, 500],
                    "depth": [4, 6, 8, 10],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "l2_leaf_reg": [1, 3, 5, 7]
                },
                "Ada Boost": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.05, 0.1, 1.0],
                    "loss": ["linear", "square", "exponential"]
                },
                "Gradien Boost": {
                    "n_estimators": [100, 200, 500],
                    "learning_rate": [0.01, 0.05, 0.1, 0.2],
                    "max_depth": [3, 5, 7, 10],
                    "subsample": [0.6, 0.8, 1.0],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4]
                }
            }
            model_report:dict = evaluate_model(X_train= X_train, y_train= y_train, X_test= X_test, y_test= y_test, models= models, params = param_grids)
            
            best_model_score = max(list(model_report.values()))
            
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            
            if best_model_score < 0.6:
                raise CustomException("No Best model found")
            logging.info("Found bet model score of the traing and testing data")
            
            save_object(
                file_path= self.model_trainer_config.trained_model_file_path,
                obj= best_model
            )
            
            prediction = best_model.predict(X_test)
            score = r2_score(y_test, prediction)
            return (best_model_name, score)
        
        
        except Exception as e:
            raise CustomException(e, sys)
