# utils/logger.py
import logging
import sys

def setup_logger():
    """
    Configures a centralized logger for the entire system.
    Using a standardized logger with clear formatting is essential for debugging
    a complex, multi-threaded, or multi-process system like this.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] [%(module)s.%(funcName)s] %(message)s",
        handlers=[
            logging.FileHandler("mission.log"), # Log to a file
            logging.StreamHandler(sys.stdout)    # Log to the console
        ]
    )
    logging.info("Logger configured.")