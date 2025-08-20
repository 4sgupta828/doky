# agents/environment_modifier.py
import logging
import os
from pathlib import Path
from typing import Dict, Any, List

# Foundational dependencies
from .base import BaseAgent
from core.context import GlobalContext
from core.models import AgentResult
from tools.environment_tools import EnvironmentTools

# Get a logger instance for this module
logger = logging.getLogger(__name__)


class EnvironmentModifierAgent(BaseAgent):
    """
    Infrastructure Tier: Environment setup and modification operations.
    
    This agent centralizes all environment modification functionality
    using structured inputs and leveraging environment tools.
    
    Responsibilities:
    - Virtual environment creation and management
    - Dependency installation from requirements.txt
    - Python path management
    - Environment variable setup
    - Isolated environment validation
    """

    def __init__(self):
        super().__init__(
            name="EnvironmentModifierAgent",
            description="Manages Python environments, virtual environments, and dependency installation using structured inputs."
        )

    def required_inputs(self) -> List[str]:
        """Required inputs for EnvironmentModifierAgent execution."""
        return ["workspace_path"]

    def optional_inputs(self) -> List[str]:
        """Optional inputs for EnvironmentModifierAgent execution."""
        return [
            "venv_name", 
            "requirements_file", 
            "python_version",
            "additional_packages",
            "environment_vars",
            "force_recreate"
        ]

    def execute_v2(self, goal: str, inputs: Dict[str, Any], global_context: GlobalContext) -> AgentResult:
        """
        Set up and manage Python environments using environment tools.
        """
        logger.info(f"EnvironmentModifierAgent executing: '{goal}'")
        
        # Validate inputs with graceful handling
        try:
            self.validate_inputs(inputs)
        except Exception as validation_error:
            return self.create_result(
                success=False,
                message=str(validation_error),
                error_details={"validation_error": str(validation_error)}
            )

        # Extract inputs
        workspace_path = Path(inputs["workspace_path"])
        venv_name = inputs.get("venv_name", "venv")
        requirements_file = inputs.get("requirements_file", "requirements.txt")
        python_version = inputs.get("python_version")
        additional_packages = inputs.get("additional_packages", [])
        environment_vars = inputs.get("environment_vars", {})
        force_recreate = inputs.get("force_recreate", False)

        try:
            self.report_progress("Setting up Python environment", f"Workspace: {workspace_path}")

            # Step 1: Create or validate virtual environment using tools
            venv_result = EnvironmentTools.setup_virtual_environment(
                workspace_path, venv_name, python_version, force_recreate
            )
            
            if not venv_result["success"]:
                return self.create_result(
                    success=False,
                    message=f"Virtual environment setup failed: {venv_result['message']}",
                    error_details=venv_result
                )

            venv_path = venv_result["venv_path"]
            venv_python = venv_result["python_executable"]
            venv_pip = venv_result["pip_executable"]

            # Step 2: Install dependencies using tools
            deps_result = EnvironmentTools.install_dependencies(
                workspace_path, venv_pip, requirements_file, additional_packages
            )
            
            if not deps_result["success"]:
                return self.create_result(
                    success=False,
                    message=f"Dependency installation failed: {deps_result.get('message', 'Unknown error')}",
                    error_details=deps_result
                )

            # Step 3: Validate environment using tools
            validation_result = EnvironmentTools.validate_environment(venv_python, additional_packages)
            
            if not validation_result["success"]:
                return self.create_result(
                    success=False,
                    message=f"Environment validation failed: {validation_result['message']}",
                    error_details=validation_result
                )

            # Step 4: Set up environment variables if provided
            env_vars_result = self._setup_environment_variables(environment_vars)

            self.report_progress("Environment setup complete", f"Virtual environment ready at {venv_path}")

            return self.create_result(
                success=True,
                message=f"Python environment successfully set up at {venv_path}",
                outputs={
                    "venv_path": str(venv_path),
                    "python_executable": str(venv_python),
                    "pip_executable": str(venv_pip),
                    "installed_packages": deps_result.get("installed_packages", []),
                    "environment_ready": True,
                    "environment_vars": environment_vars,
                    "validation_details": validation_result.get("details", [])
                }
            )

        except Exception as e:
            error_msg = f"EnvironmentModifierAgent execution failed: {e}"
            logger.error(error_msg, exc_info=True)
            return self.create_result(
                success=False,
                message=error_msg,
                error_details={"exception": str(e)}
            )

    def _setup_environment_variables(self, environment_vars: Dict[str, str]) -> Dict[str, Any]:
        """Set up environment variables for the current process."""
        if not environment_vars:
            return {"success": True, "message": "No environment variables to set"}

        for key, value in environment_vars.items():
            os.environ[key] = value
            logger.info(f"Set environment variable {key}")

        return {
            "success": True,
            "message": f"Set {len(environment_vars)} environment variables"
        }