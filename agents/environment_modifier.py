# agents/environment_modifier.py
import logging
import subprocess
import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Optional

# Foundational dependencies
from .base import BaseAgent
from core.context import GlobalContext
from core.models import AgentResponse, AgentResult, TaskNode

# Get a logger instance for this module
logger = logging.getLogger(__name__)


class EnvironmentModifierAgent(BaseAgent):
    """
    Infrastructure Tier: Environment setup and modification operations.
    
    This agent centralizes all environment modification functionality.
    
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
            description="Manages Python environments, virtual environments, and dependency installation."
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
        NEW INTERFACE: Set up and manage Python environments.
        """
        logger.info(f"EnvironmentManagerAgent executing: '{goal}'")
        
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

            # Step 1: Create or validate virtual environment
            venv_result = self._setup_virtual_environment(
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

            # Step 2: Install dependencies
            deps_result = self._install_dependencies(
                workspace_path, venv_pip, requirements_file, additional_packages
            )
            
            if not deps_result["success"]:
                return self.create_result(
                    success=False,
                    message=f"Dependency installation failed: {deps_result['message']}",
                    error_details=deps_result
                )

            # Step 3: Validate environment
            validation_result = self._validate_environment(venv_python, additional_packages)
            
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
            error_msg = f"EnvironmentManagerAgent execution failed: {e}"
            logger.error(error_msg, exc_info=True)
            return self.create_result(
                success=False,
                message=error_msg,
                error_details={"exception": str(e)}
            )

    def _setup_virtual_environment(self, workspace_path: Path, venv_name: str, 
                                 python_version: Optional[str], force_recreate: bool) -> Dict[str, Any]:
        """Create or validate virtual environment."""
        venv_path = workspace_path / venv_name
        
        # Check if virtual environment already exists
        if venv_path.exists() and not force_recreate:
            self.report_progress("Virtual environment exists", f"Using existing environment at {venv_path}")
        else:
            if force_recreate and venv_path.exists():
                self.report_progress("Recreating virtual environment", f"Removing existing environment at {venv_path}")
                import shutil
                shutil.rmtree(venv_path)
            
            # Create new virtual environment
            self.report_progress("Creating virtual environment", f"Setting up isolated Python environment at {venv_path}")
            
            python_cmd = python_version if python_version else sys.executable
            result = subprocess.run(
                [python_cmd, "-m", "venv", str(venv_path)], 
                capture_output=True, text=True, cwd=workspace_path
            )
            
            if result.returncode != 0:
                return {
                    "success": False, 
                    "message": f"Failed to create virtual environment: {result.stderr}",
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }

        # Determine executable paths
        if sys.platform == "win32":
            venv_python = venv_path / "Scripts" / "python.exe"
            venv_pip = venv_path / "Scripts" / "pip.exe"
        else:
            venv_python = venv_path / "bin" / "python"
            venv_pip = venv_path / "bin" / "pip"

        # Validate executables exist
        if not venv_python.exists():
            return {
                "success": False,
                "message": f"Python executable not found at {venv_python}"
            }

        if not venv_pip.exists():
            return {
                "success": False,
                "message": f"Pip executable not found at {venv_pip}"
            }

        return {
            "success": True,
            "venv_path": venv_path,
            "python_executable": venv_python,
            "pip_executable": venv_pip
        }

    def _install_dependencies(self, workspace_path: Path, pip_executable: Path, 
                            requirements_file: str, additional_packages: List[str]) -> Dict[str, Any]:
        """Install dependencies from requirements file and additional packages."""
        installed_packages = []
        
        # Install from requirements file if it exists
        requirements_path = workspace_path / requirements_file
        if requirements_path.exists():
            self.report_progress("Installing dependencies", f"Installing from {requirements_file}")
            
            result = subprocess.run(
                [str(pip_executable), "install", "-r", str(requirements_path)], 
                capture_output=True, text=True
            )
            
            if result.returncode != 0:
                return {
                    "success": False,
                    "message": f"Requirements installation failed: {result.stderr}",
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }
            
            installed_packages.append(f"requirements from {requirements_file}")

        # Install additional packages
        if additional_packages:
            self.report_progress("Installing additional packages", f"Installing {len(additional_packages)} additional packages")
            
            for package in additional_packages:
                result = subprocess.run(
                    [str(pip_executable), "install", package], 
                    capture_output=True, text=True
                )
                
                if result.returncode != 0:
                    logger.warning(f"Failed to install package {package}: {result.stderr}")
                else:
                    installed_packages.append(package)

        return {
            "success": True,
            "installed_packages": installed_packages
        }

    def _validate_environment(self, python_executable: Path, expected_packages: List[str]) -> Dict[str, Any]:
        """Validate that the environment is properly set up."""
        details = []
        
        # Test Python executable
        result = subprocess.run(
            [str(python_executable), "--version"], 
            capture_output=True, text=True
        )
        
        if result.returncode != 0:
            return {
                "success": False,
                "message": f"Python executable test failed: {result.stderr}"
            }
        
        python_version = result.stdout.strip()
        details.append(f"Python version: {python_version}")
        
        # Test package imports
        for package in expected_packages:
            result = subprocess.run(
                [str(python_executable), "-c", f"import {package.split('==')[0]}"], 
                capture_output=True, text=True
            )
            
            if result.returncode == 0:
                details.append(f"✓ Package {package} imports successfully")
            else:
                details.append(f"✗ Package {package} import failed: {result.stderr}")

        return {
            "success": True,
            "details": details
        }

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

    # Legacy execute method for backward compatibility
    def execute(self, goal: str, context: GlobalContext, current_task: TaskNode) -> AgentResponse:
        """Legacy execute method for backward compatibility."""
        inputs = {
            "workspace_path": str(context.workspace_path)
        }
        
        result = self.execute_v2(goal, inputs, context)
        
        return AgentResponse(
            success=result.success,
            message=result.message,
            artifacts_generated=[]
        )