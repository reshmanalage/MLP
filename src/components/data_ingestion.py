import os
import sys
import logging
import pandas as pd
from src.exception import CustomException
from src.utils import save_object, load_object

# Set up logger for data ingestion module
logger = logging.getLogger("data_ingestion")
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler(sys.stdout)
file_handler = logging.FileHandler('data_ingestion.log')
log_formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(name)s - %(message)s")
console_handler.setFormatter(log_formatter)
file_handler.setFormatter(log_formatter)
logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads data from a given file path. Supports CSV and Excel formats.
    
    Parameters:
    - file_path (str): Path to the data file.
    
    Returns:
    - pd.DataFrame: Loaded data in DataFrame format.
    
    Raises:
    - CustomException: If the file format is unsupported or an error occurs while reading the file.
    """
    try:
        logger.info(f"Attempting to load data from {file_path}...")

        # Check file extension and read accordingly
        file_extension = os.path.splitext(file_path)[-1].lower()
        
        if file_extension == ".csv":
            df = pd.read_csv(file_path)
        elif file_extension in [".xls", ".xlsx"]:
            df = pd.read_excel(file_path)
        else:
            raise ValueError("Unsupported file format. Please provide a CSV or Excel file.")
        
        logger.info(f"Data loaded successfully from {file_path}")
        return df

    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {str(e)}")
        raise CustomException(e, sys)

def handle_missing_values(df: pd.DataFrame, strategy: str = "drop") -> pd.DataFrame:
    """
    Handles missing values in the DataFrame based on the given strategy.
    
    Parameters:
    - df (pd.DataFrame): The input DataFrame with potential missing values.
    - strategy (str): The strategy for handling missing values. Options are "drop" or "mean".
    
    Returns:
    - pd.DataFrame: DataFrame with missing values handled.
    
    Raises:
    - CustomException: If an invalid strategy is provided or if an error occurs.
    """
    try:
        logger.info(f"Handling missing values with strategy: {strategy}...")

        if strategy == "drop":
            df_cleaned = df.dropna()
        elif strategy == "mean":
            df_cleaned = df.fillna(df.mean())
        else:
            raise ValueError("Invalid strategy. Choose either 'drop' or 'mean'.")
        
        logger.info(f"Missing values handled using {strategy} strategy.")
        return df_cleaned

    except Exception as e:
        logger.error(f"Error handling missing values: {str(e)}")
        raise CustomException(e, sys)

def preprocess_data(df: pd.DataFrame, columns_to_drop: list = [], columns_to_encode: list = []) -> pd.DataFrame:
    """
    Preprocesses the DataFrame by dropping specified columns and encoding categorical columns.
    
    Parameters:
    - df (pd.DataFrame): The input DataFrame to preprocess.
    - columns_to_drop (list): List of columns to drop from the DataFrame.
    - columns_to_encode (list): List of categorical columns to encode using one-hot encoding.
    
    Returns:
    - pd.DataFrame: Preprocessed DataFrame.
    
    Raises:
    - CustomException: If an error occurs during preprocessing.
    """
    try:
        logger.info("Starting preprocessing of the data...")

        # Drop specified columns
        if columns_to_drop:
            df = df.drop(columns=columns_to_drop)
            logger.info(f"Dropped columns: {columns_to_drop}")
        
        # One-hot encode categorical columns
        if columns_to_encode:
            df = pd.get_dummies(df, columns=columns_to_encode)
            logger.info(f"One-hot encoded columns: {columns_to_encode}")
        
        logger.info("Data preprocessing completed successfully.")
        return df

    except Exception as e:
        logger.error(f"Error during data preprocessing: {str(e)}")
        raise CustomException(e, sys)

def save_cleaned_data(df: pd.DataFrame, output_path: str) -> None:
    """
    Saves the cleaned and preprocessed DataFrame to the specified output path.
    
    Parameters:
    - df (pd.DataFrame): The cleaned and preprocessed DataFrame.
    - output_path (str): The output path where the cleaned data will be saved.
    
    Raises:
    - CustomException: If an error occurs while saving the cleaned data.
    """
    try:
        logger.info(f"Saving cleaned data to {output_path}...")

        # Save the cleaned data to a CSV file
        df.to_csv(output_path, index=False)
        logger.info(f"Cleaned data saved successfully to {output_path}")

    except Exception as e:
        logger.error(f"Error saving cleaned data to {output_path}: {str(e)}")
        raise CustomException(e, sys)

if __name__ == "__main__":
    # Example usage
    try:
        input_file_path = "data/raw_data.csv"  # Example input path
        output_file_path = "data/cleaned_data.csv"  # Example output path
        
        # Load the raw data
        df = load_data(input_file_path)

        # Handle missing values
        df = handle_missing_values(df, strategy="mean")

        # Preprocess the data (e.g., drop unwanted columns, encode categorical variables)
        df = preprocess_data(df, columns_to_drop=["unwanted_column"], columns_to_encode=["category_column"])

        # Save the cleaned data
        save_cleaned_data(df, output_file_path)
    
    except Exception as e:
        logger.error(f"Error in the data ingestion pipeline: {str(e)}")
