import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    """
    Configuration class for the data ingestion component.
    Stores paths for the raw, training, and testing data.
    """
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")
    preprocessor_path: str = os.path.join('artifacts', 'preprocessor.pkl')  # Path for preprocessor

class DataIngestion:
    def __init__(self):
        """
        Initializes the DataIngestion class with the specified configuration paths.
        """
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        """
        Method to handle the data ingestion process.
        
        - Reads the raw data from the specified path.
        - Splits the data into training and testing datasets.
        - Saves the training and testing datasets to specified file paths.

        Returns:
        - Tuple: paths to the training and testing datasets.

        Raises:
        - CustomException: If any error occurs during the data ingestion process.
        """
        logging.info("Entered the data ingestion method or component")
        try:
            # Read the raw data from the source file
            df = pd.read_csv('notebook/data/stud.csv')  # Replace with your actual CSV file path
            logging.info('Read the dataset as dataframe')

            # Create the directories for saving files if they do not exist
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save the raw data to the specified path
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Raw data saved to: %s", self.ingestion_config.raw_data_path)

            logging.info("Train-test split initiated")
            # Split the data into training and testing datasets (80% training, 20% testing)
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Save the training and testing datasets to the specified paths
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data ingestion completed successfully")

            # Return the file paths for the training and testing datasets
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
                self.ingestion_config.preprocessor_path  # Return the preprocessor path
            )
        except Exception as e:
            # If an error occurs during ingestion, raise a custom exception
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    """
    Main execution block: Initializes the data ingestion, transformation, 
    and model training pipeline.
    """
    # Instantiate the DataIngestion class and initiate the data ingestion
    obj = DataIngestion()
    train_data, test_data, preprocessor_path = obj.initiate_data_ingestion()  # Get preprocessor path

    # Instantiate the DataTransformation class and initiate the data transformation
    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

    # Instantiate the ModelTrainer class and initiate model training
    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr, test_arr, preprocessor_path))  # Pass preprocessor path
