import logging
import sys
import os
from datetime import datetime

# Create a log filename with current timestamp using the format %m_%d_%Y_%H_%M_%S
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"  # Using %m_%d_%Y_%H_%M_%S format

# Define the logs directory path and ensure it exists
logs_path = os.path.join(os.getcwd(), "logs")  # Logs directory path (without file name)
os.makedirs(logs_path, exist_ok=True)  # Ensure the "logs" directory exists

# Define the full path to the log file
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)  # Combine logs directory with log file name

# Configure logging
logging.basicConfig(
    filename=LOG_FILE_PATH,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

class CustomException(Exception):
    """Custom exception class."""
    pass

if __name__ == "__main__":
    
    # Example: Add more logging messages to test the setup
    try:
        a = 1 / 0  # Will raise an error to demonstrate logging of exceptions
    except Exception as e:
        logging.exception("Divide by zero")  # Logs the exception with traceback
        raise CustomException(e,sys)  # Raises a custom exception

