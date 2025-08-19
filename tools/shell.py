# tools/shell.py
import subprocess
import logging
from typing import Dict, Set

# Get a logger instance for this module
logger = logging.getLogger(__name__)

# --- NEW: Centralized Safety Configuration ---
DISALLOWED_COMMANDS: Set[str] = {
    "rm", "mv", "dd", "mkfs", "shutdown", "reboot", "sudo", "su"
}

def execute_shell_command(command: str, working_dir: str) -> Dict[str, any]:
    """
    Executes a shell command in a specified directory and returns structured output.
    This function includes a safety check to prevent dangerous commands.

    Args:
        command: The shell command to execute.
        working_dir: The directory in which to run the command.

    Returns:
        A dictionary containing the stdout, stderr, and exit code of the command.
    """
    logger.info(f"Executing shell command: '{command}' in '{working_dir}'")
    
    # --- NEW: Safety Check ---
    if not command.strip():
        logger.error("Empty command blocked for safety.")
        return {
            "stdout": "",
            "stderr": "Error: Empty command provided.",
            "exit_code": -1
        }
        
    first_word = command.strip().split()[0]
    if first_word in DISALLOWED_COMMANDS:
        error_msg = f"Error: Execution of disallowed command '{first_word}' was blocked for safety."
        logger.error(error_msg)
        return {
            "stdout": "",
            "stderr": error_msg,
            "exit_code": -1
        }
    # --- End of Safety Check ---

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            cwd=working_dir,
            timeout=120  # A 2-minute timeout for safety
        )
        
        if result.returncode != 0:
            logger.warning(f"Command failed with exit code {result.returncode}")
            logger.warning(f"STDERR: {result.stderr.strip()}")

        return {
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
            "exit_code": result.returncode
        }
    except FileNotFoundError:
        logger.error(f"Command not found: {command.split()[0]}")
        return {
            "stdout": "",
            "stderr": f"Error: Command not found '{command.split()[0]}'. Is it installed and in the system's PATH?",
            "exit_code": -1
        }
    except subprocess.TimeoutExpired:
        logger.error(f"Command timed out: {command}")
        return {
            "stdout": "",
            "stderr": f"Error: Command '{command}' timed out after 120 seconds.",
            "exit_code": -1
        }
    except Exception as e:
        logger.critical(f"An unexpected error occurred while executing command: {command}", exc_info=True)
        return {
            "stdout": "",
            "stderr": f"An unexpected error occurred: {e}",
            "exit_code": -1
        }
