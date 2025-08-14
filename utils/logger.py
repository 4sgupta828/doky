# utils/logger.py
import logging
import logging.config
import os
from typing import Dict, Any

def setup_logger(log_dir: str = "logs", default_level: int = logging.INFO) -> None:
    """
    Configures a centralized logger for the entire Sovereign Agent Collective system.

    This function sets up a logger that is both powerful and easy to use, forming the
    foundation of the system's debuggability. It uses a dictionary-based configuration
    for maximum flexibility.

    Key Features:
    - **Dual Output**: Logs are sent to both the console (for real-time monitoring)
      and a rotating file (for persistent history and post-mortem analysis).
    - **Rotating Files**: The log file (`mission.log`) will automatically rotate
      once it reaches a certain size (5MB), keeping the last 5 log files. This
      prevents log files from growing indefinitely.
    - **Structured Formatting**: Log messages are formatted to include a timestamp,
      log level, the module where the log originated, and the message itself,
      making it easy to trace the flow of execution.
    - **Configurable**: The logging level can be set globally. For production, you
      might use INFO, while for development, DEBUG would be more appropriate.

    Args:
        log_dir: The directory where log files will be stored. It will be created
                 if it does not exist.
        default_level: The default logging level for the root logger.
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_file_path = os.path.join(log_dir, "mission.log")

    # Dictionary-based configuration provides a clear and extensible way to set up logging.
    LOGGING_CONFIG: Dict[str, Any] = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s [%(levelname)s] [%(name)s:%(lineno)d] - %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "standard",
                "level": logging.INFO,  # Console logs are typically less verbose.
                "stream": "ext://sys.stdout",
            },
            "rotating_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "formatter": "standard",
                "level": logging.DEBUG, # File logs should capture everything.
                "filename": log_file_path,
                "maxBytes": 5 * 1024 * 1024,  # 5 MB
                "backupCount": 5,
                "encoding": "utf8",
            },
        },
        "loggers": {
            # Root logger: catches all logs from any module.
            "": {
                "handlers": ["console", "rotating_file"],
                "level": default_level,
                "propagate": False, # Prevents the root logger from passing messages to its own handlers again
            },
            # Example of a more specific logger configuration if needed later.
            # "agents.planner": {
            #     "handlers": ["rotating_file"],
            #     "level": logging.DEBUG,
            #     "propagate": False,
            # }
        },
    }

    logging.config.dictConfig(LOGGING_CONFIG)
    logging.info("Logger configured successfully. Logging to console and %s", log_file_path)


# --- Self-Testing Block ---
# This block demonstrates how to use the logger and serves as a simple test case.
# To run this test, execute the file directly: `python utils/logger.py`
if __name__ == "__main__":
    print("--- Testing Logger Configuration ---")
    setup_logger(default_level=logging.DEBUG)

    # Get a logger instance for the current module (__name__)
    # This is the standard practice for using logging in Python.
    logger = logging.getLogger(__name__)

    logger.debug("This is a detailed debug message. Useful for developers.")
    logger.info("This is an informational message. It indicates normal operation.")
    logger.warning("This is a warning. Something unexpected happened, but the system can continue.")
    logger.error("This is an error. A serious problem occurred that prevented a task from completing.")
    
    try:
        1 / 0
    except ZeroDivisionError:
        # The exc_info=True argument automatically captures and logs the stack trace.
        # This is invaluable for debugging errors.
        logger.critical("This is a critical error with a stack trace.", exc_info=True)

    print("\n--- Logger Test Complete ---")
    print("Check the console output and the 'logs/mission.log' file for the results.")