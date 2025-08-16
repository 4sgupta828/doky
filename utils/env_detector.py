# utils/env_detector.py
import subprocess
import logging
from typing import Optional

logger = logging.getLogger(__name__)

def detect_python_command() -> str:
    """
    Automatically detect the correct Python command to use.
    
    Returns:
        str: The Python command ('python3', 'python', etc.) that works on this system
        
    Raises:
        RuntimeError: If no Python installation is found
    """
    # Try different Python commands in order of preference
    python_candidates = ['python3', 'python', 'python3.11', 'python3.12', 'python3.13']
    
    for python_cmd in python_candidates:
        try:
            result = subprocess.run(
                [python_cmd, '--version'], 
                capture_output=True, 
                text=True, 
                timeout=5
            )
            if result.returncode == 0:
                logger.debug(f"Found Python command: {python_cmd} -> {result.stdout.strip()}")
                return python_cmd
        except (subprocess.SubprocessError, FileNotFoundError):
            continue
    
    # If we get here, no Python was found
    raise RuntimeError("No Python installation found. Tried: " + ", ".join(python_candidates))

def get_python_version_command() -> str:
    """
    Get the complete Python version command for the current system.
    
    Returns:
        str: Complete command like "python3 --version" or "python --version"
    """
    python_cmd = detect_python_command()
    return f"{python_cmd} --version"

def get_pip_list_command() -> str:
    """
    Get the complete pip list command for the current system.
    
    Returns:
        str: Complete command like "python3 -m pip list" or "pip list"
    """
    try:
        python_cmd = detect_python_command()
        # Prefer using python -m pip for consistency
        return f"{python_cmd} -m pip list"
    except RuntimeError:
        # Fallback to just pip if Python detection fails
        return "pip list"