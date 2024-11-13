import sys
import logging

# Setting up logger for exception logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

# Create a log file handler and formatter
log_handler = logging.FileHandler("error_log.log")
log_formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
log_handler.setFormatter(log_formatter)
logger.addHandler(log_handler)

def error_message_detail(error, error_detail: sys):
    """
    Extracts detailed error message including filename, line number and error description.
    """
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = f"Error occurred in python script [{file_name}] at line number [{exc_tb.tb_lineno}], error message: [{str(error)}]"
    return error_message

class CustomException(Exception):
    """
    Custom exception class that extends the base Exception class
    and logs detailed error information.
    """
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_detail(error_message, error_detail)
        
        # Log the exception with detailed information
        logger.error(self.error_message)

    def __str__(self):
        return self.error_message


# Example of how to raise and handle exceptions
if __name__ == "__main__":
    try:
        # Deliberate error for demonstration
        a = 1 / 0  # This will raise a division by zero error
    except Exception as e:
        raise CustomException(e, sys)  # Raise custom exception with logging
