import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def _get_models(self):
        """
        Defines and returns a dictionary of models to be trained.
        """
        return {
            "Random Forest": RandomForestRegressor(),
            "Decision Tree": DecisionTreeRegressor(),
            "Gradient Boosting": GradientBoostingRegressor(),
            "Linear Regression": LinearRegression(),
            "XGBRegressor": XGBRegressor(),
            "CatBoosting Regressor": CatBoostRegressor(verbose=False),
            "AdaBoost Regressor": AdaBoostRegressor(),
        }

    def _get_model_params(self):
        """
        Defines and returns a dictionary of hyperparameters for model tuning.
        """
        return {
            "Decision Tree": {
                'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
            },
            "Random Forest": {
                'n_estimators': [8, 16, 32, 64, 128, 256]
            },
            "Gradient Boosting": {
                'learning_rate': [.1, .01, .05, .001],
                'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9],
                'n_estimators': [8, 16, 32, 64, 128, 256]
            },
            "Linear Regression": {},
            "XGBRegressor": {
                'learning_rate': [.1, .01, .05, .001],
                'n_estimators': [8, 16, 32, 64, 128, 256]
            },
            "CatBoosting Regressor": {
                'depth': [6, 8, 10],
                'learning_rate': [0.01, 0.05, 0.1],
                'iterations': [30, 50, 100]
            },
            "AdaBoost Regressor": {
                'learning_rate': [.1, .01, 0.5, .001],
                'n_estimators': [8, 16, 32, 64, 128, 256]
            }
        }

    def _get_best_model(self, X_train, y_train, X_test, y_test):
        """
        Evaluates all models and returns the best performing model based on R2 score.
        """
        models = self._get_models()
        params = self._get_model_params()

        # Evaluate models and get a report of their R2 scores
        model_report = evaluate_models(
            X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
            models=models, param=params
        )

        # Identify the best model
        best_model_score = max(sorted(model_report.values()))
        best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
        
        best_model = models[best_model_name]

        logging.info(f"Best model found: {best_model_name} with R2 score: {best_model_score}")
        return best_model, best_model_score

    def initiate_model_trainer(self, train_array, test_array):
        """
        Trains the models, evaluates their performance, and selects the best model.
        """
        try:
            logging.info("Splitting training and test data into X and y.")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            # Get the best model
            best_model, best_model_score = self._get_best_model(X_train, y_train, X_test, y_test)

            # If the best model score is less than a threshold, raise an exception
            if best_model_score < 0.6:
                raise CustomException("No suitable model found with R2 score >= 0.6.")

            logging.info(f"Best model is {best_model} with R2 score: {best_model_score}")

            # Save the trained model
            save_object(file_path=self.model_trainer_config.trained_model_file_path, obj=best_model)

            # Make predictions with the best model
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)

            logging.info(f"Model training and evaluation completed with R2 score: {r2_square}")
            return r2_square

        except Exception as e:
            logging.error(f"Error in model training: {str(e)}")
            raise CustomException(e, sys)

