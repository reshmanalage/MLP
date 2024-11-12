import os
import sys
from dataclasses import dataclass

# Importing necessary regression models from different libraries
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
    """
    Configuration class for the model trainer component.
    Stores the path for saving the trained model.
    """
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        """
        Main function to initiate the model training and evaluation process.

        It splits the training and test data, then evaluates multiple machine learning models
        using GridSearchCV for hyperparameter tuning. The best performing model is selected
        and saved.

        Parameters:
        - train_array (numpy array): The training data.
        - test_array (numpy array): The test data.

        Returns:
        - r2_square (float): The R-squared score of the best model on the test data.

        Raises:
        - CustomException: If any error occurs during the model training process.
        """
        try:
            # Splitting the train and test data into features (X) and target (y)
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            # Dictionary of models to evaluate
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            # Hyperparameters for GridSearchCV
            params = {
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

            # Evaluate models using the evaluate_models function from utils
            model_report: dict = evaluate_models(
                X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test,
                models=models, param=params
            )
            
            # Get the best model score from the report
            best_model_score = max(sorted(model_report.values()))

            # Get the name of the best model
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            # Get the best model from the models dictionary
            best_model = models[best_model_name]

            # If the best model score is below a threshold, raise an exception
            if best_model_score < 0.6:
                raise CustomException("No best model found with acceptable score")

            # Log that the best model was found
            logging.info(f"Best model found: {best_model_name} with score: {best_model_score}")

            # Save the best model using the save_object function
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Predict on the test set using the best model
            predicted = best_model.predict(X_test)

            # Calculate R-squared score of the best model on the test set
            r2_square = r2_score(y_test, predicted)

            return r2_square

        except Exception as e:
            # Raise a custom exception if any error occurs
            raise CustomException(e, sys)
