import os
import sys
import dill  # Using dill for saving and loading complex Python objects
import numpy as np 
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException

def save_object(file_path, obj):
    """
    Saves a given object (e.g., trained model) to the specified file path.

    Parameters:
    - file_path (str): The path where the object will be saved.
    - obj: The object (model, preprocessor, etc.) to save.

    Raises:
    - CustomException: If any error occurs during the saving process.
    """
    try:
        # Create the parent directory for the file if it doesn't exist
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        # Open the file in write-binary mode and use dill to save the object
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        # Raising custom exception if there's any error
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    """
    Evaluates multiple models using GridSearchCV for hyperparameter tuning,
    trains them, and calculates their R2 score on test data.

    Parameters:
    - X_train (array-like): Training data features.
    - y_train (array-like): Training data labels.
    - X_test (array-like): Test data features.
    - y_test (array-like): Test data labels.
    - models (dict): Dictionary of models to evaluate.
    - param (dict): Dictionary of hyperparameters to tune for each model.

    Returns:
    - report (dict): Dictionary with models and their corresponding test R2 scores.

    Raises:
    - CustomException: If any error occurs during the evaluation process.
    """
    try:
        # Initialize an empty dictionary to store the model scores
        report = {}

        # Loop through each model and its associated hyperparameters
        for i in range(len(list(models))):
            model = list(models.values())[i]
            para = param.get(list(models.keys())[i], {})

            # Perform GridSearchCV for hyperparameter tuning
            gs = GridSearchCV(estimator=model, param_grid=para, cv=3, scoring='r2')
            gs.fit(X_train, y_train)

            # Set the best hyperparameters found by GridSearchCV
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            # Predict the target values on both training and testing data
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Calculate R2 score for both training and test data
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            # Store the test R2 score in the report dictionary
            report[list(models.keys())[i]] = test_model_score

        return report

    except Exception as e:
        # Raising custom exception if any error occurs during evaluation
        raise CustomException(e, sys)

def load_object(file_path):
    """
    Loads a previously saved object (e.g., a trained model or preprocessor) from the specified file path.

    Parameters:
    - file_path (str): The path from which the object will be loaded.

    Returns:
    - obj: The loaded object (model, preprocessor, etc.).

    Raises:
    - CustomException: If any error occurs during the loading process.
    """
    try:
        # Open the file in read-binary mode and use dill to load the object
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        # Raising custom exception if there's any error
        raise CustomException(e, sys)
