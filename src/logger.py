import logging
import sys
import os
from datetime import datetime

# Create a timestamped log filename for each run
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# Define the logs directory path and ensure it exists
logs_path = os.path.join(os.getcwd(), "logs")  # Logs directory path
os.makedirs(logs_path, exist_ok=True)  # Ensure the "logs" directory exists

# Define the full path to the log file
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# Configure the logger with handlers
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Default log level is DEBUG for detailed logging

# Console Handler for outputting logs to the console
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)

# File Handler for logging into the log file
file_handler = logging.FileHandler(LOG_FILE_PATH)
file_handler.setLevel(logging.INFO)  # Log level for file output can be set separately

# Formatter to define the log message structure
log_formatter = logging.Formatter(
    "[%(asctime)s] %(levelname)s - %(name)s - %(message)s"
)

# Adding formatter to the handlers
console_handler.setFormatter(log_formatter)
file_handler.setFormatter(log_formatter)

# Adding the handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

class CustomException(Exception):
    """Custom Exception Class to handle and log errors."""

    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = self.format_error_message(error_message, error_detail)

    def format_error_message(self, error, error_detail: sys):
        """Format detailed error messages with file name and line number."""
        _, _, exc_tb = error_detail.exc_info()
        file_name = exc_tb.tb_frame.f_code.co_filename
        line_number = exc_tb.tb_lineno
        error_message = f"Error in file [{file_name}] at line [{line_number}] - {str(error)}"
        return error_message

    def __str__(self):
        return self.error_message


# Example of how to raise and handle exceptions with logging
if __name__ == "__main__":
    try:
        a = 1 / 0  # Deliberate error for testing
    except Exception as e:
        # Log the error here before raising the custom exception
        logger.error(f"An error occurred: {str(e)}")
        raise CustomException(e, sys)  # Raise the custom exception with logging
