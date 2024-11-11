import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            # Fixed path issue: changed from backslash to forward slash
            # df = pd.read_csv('notebook\data\stud.csv')  # Old path with backslashes
            df = pd.read_csv('notebook/data/stud.csv')  # New path with forward slashes
            logging.info('Read the dataset as dataframe')
            
            # Ensure the parent directory exists before saving files
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)  # Ensures parent dir exists
            
            # Save the raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Train and test split started")
            
            # Perform the split
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            
            # Save the train and test sets
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Ingestion of the data is completed")
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            # Ensure CustomException is defined correctly
            raise CustomException(e, sys)

if __name__ == "__main__":
    # Ensure logging is configured correctly
    logging.basicConfig(level=logging.INFO)  # Added logging setup
    
    # Create DataIngestion object and start ingestion
    obj = DataIngestion()
    obj.initiate_data_ingestion()
