import os
import sys
import pickle
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException

# Set up logger for the utils module
logger = logging.getLogger("utils")
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler(sys.stdout)
file_handler = logging.FileHandler('utils.log')
log_formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(name)s - %(message)s")
console_handler.setFormatter(log_formatter)
file_handler.setFormatter(log_formatter)
logger.addHandler(console_handler)
logger.addHandler(file_handler)

def save_object(file_path, obj):
    """
    Saves a Python object to a file using pickle.
    
    Parameters:
    - file_path (str): The path where the object will be saved.
    - obj: The object (model, preprocessor, etc.) to save.
    
    Raises:
    - CustomException: If any error occurs during the saving process.
    """
    try:
        # Log the start of saving the object
        logger.info(f"Saving object to {file_path}...")

        # Ensure the directory exists
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        # Save the object to file
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        
        logger.info(f"Object successfully saved to {file_path}")

    except Exception as e:
        logger.error(f"Error saving object to {file_path}: {str(e)}")
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    """
    Evaluates multiple models using GridSearchCV for hyperparameter tuning and calculates R2 score.
    
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
        logger.info("Starting model evaluation...")

        report = {}
        for model_name, model in models.items():
            logger.info(f"Evaluating model: {model_name}")

            # Perform hyperparameter tuning using GridSearchCV
            para = param.get(model_name, {})
            gs = GridSearchCV(estimator=model, param_grid=para, cv=3, scoring='r2')
            gs.fit(X_train, y_train)

            # Set the best hyperparameters found by GridSearchCV
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            # Predict and calculate R2 score
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            # Log model performance
            logger.info(f"Training R2 Score for {model_name}: {train_model_score:.4f}")
            logger.info(f"Testing R2 Score for {model_name}: {test_model_score:.4f}")

            # Add model performance to report
            report[model_name] = test_model_score

        logger.info("Model evaluation completed.")

        return report

    except Exception as e:
        logger.error(f"Error during model evaluation: {str(e)}")
        raise CustomException(e, sys)

def load_object(file_path):
    """
    Loads a Python object from a file using pickle.
    
    Parameters:
    - file_path (str): The path from which the object will be loaded.
    
    Returns:
    - obj: The loaded object (model, preprocessor, etc.)
    
    Raises:
    - CustomException: If any error occurs during the loading process.
    """
    try:
        logger.info(f"Loading object from {file_path}...")

        with open(file_path, "rb") as file_obj:
            obj = pickle.load(file_obj)
        
        logger.info(f"Object successfully loaded from {file_path}")
        return obj

    except Exception as e:
        logger.error(f"Error loading object from {file_path}: {str(e)}")
        raise CustomException(e, sys)
