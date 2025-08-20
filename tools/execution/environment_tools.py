# tools/execution/environment_tools.py
import logging
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class EnvironmentOperation(Enum):
    """Environment operations."""
    CREATE_VENV = "create_venv"
    ACTIVATE_VENV = "activate_venv"
    INSTALL_DEPS = "install_deps"
    VALIDATE_ENV = "validate_env"
    SET_ENV_VARS = "set_env_vars"
    ANALYZE_ENV = "analyze_env"
    CHECK_TOOLS = "check_tools"


class PythonVersion(Enum):
    """Python version options."""
    SYSTEM = "system"
    PYTHON3_8 = "python3.8"
    PYTHON3_9 = "python3.9"
    PYTHON3_10 = "python3.10"
    PYTHON3_11 = "python3.11"
    PYTHON3_12 = "python3.12"


@dataclass
class EnvironmentContext:
    """Configuration for environment operations."""
    operation: EnvironmentOperation
    workspace_path: Path
    venv_name: str = "venv"
    python_version: Optional[PythonVersion] = None
    requirements_file: Optional[str] = "requirements.txt"
    additional_packages: List[str] = field(default_factory=list)
    environment_vars: Dict[str, str] = field(default_factory=dict)
    force_recreate: bool = False
    timeout_seconds: int = 300
    validate_packages: List[str] = field(default_factory=list)


@dataclass
class EnvironmentResult:
    """Result of environment operations."""
    success: bool
    message: str
    operation: EnvironmentOperation
    workspace_path: Path
    venv_path: Optional[Path] = None
    python_executable: Optional[Path] = None
    pip_executable: Optional[Path] = None
    installed_packages: List[str] = field(default_factory=list)
    environment_vars: Dict[str, str] = field(default_factory=dict)
    analysis_results: Optional[Dict] = None
    validation_details: List[str] = field(default_factory=list)
    error_details: Optional[str] = None


def create_virtual_environment(context: EnvironmentContext) -> EnvironmentResult:
    """Create or validate virtual environment."""
    venv_path = context.workspace_path / context.venv_name
    
    try:
        # Check if virtual environment already exists
        if venv_path.exists() and not context.force_recreate:
            logger.info(f"Using existing virtual environment at {venv_path}")
        else:
            if context.force_recreate and venv_path.exists():
                logger.info(f"Recreating virtual environment at {venv_path}")
                shutil.rmtree(venv_path)
            
            # Create new virtual environment
            python_cmd = (
                context.python_version.value 
                if context.python_version and context.python_version != PythonVersion.SYSTEM 
                else sys.executable
            )
            
            result = subprocess.run(
                [python_cmd, "-m", "venv", str(venv_path)], 
                capture_output=True, 
                text=True, 
                cwd=context.workspace_path,
                timeout=context.timeout_seconds
            )
            
            if result.returncode != 0:
                return EnvironmentResult(
                    success=False,
                    message=f"Failed to create virtual environment: {result.stderr}",
                    operation=context.operation,
                    workspace_path=context.workspace_path,
                    error_details=f"Command failed with return code {result.returncode}: {result.stderr}"
                )

        # Determine executable paths
        if sys.platform == "win32":
            python_executable = venv_path / "Scripts" / "python.exe"
            pip_executable = venv_path / "Scripts" / "pip.exe"
        else:
            python_executable = venv_path / "bin" / "python"
            pip_executable = venv_path / "bin" / "pip"

        # Validate executables exist
        if not python_executable.exists():
            return EnvironmentResult(
                success=False,
                message=f"Python executable not found at {python_executable}",
                operation=context.operation,
                workspace_path=context.workspace_path,
                error_details="Virtual environment creation succeeded but Python executable is missing"
            )

        if not pip_executable.exists():
            return EnvironmentResult(
                success=False,
                message=f"Pip executable not found at {pip_executable}",
                operation=context.operation,
                workspace_path=context.workspace_path,
                error_details="Virtual environment creation succeeded but pip executable is missing"
            )

        return EnvironmentResult(
            success=True,
            message=f"Virtual environment ready at {venv_path}",
            operation=context.operation,
            workspace_path=context.workspace_path,
            venv_path=venv_path,
            python_executable=python_executable,
            pip_executable=pip_executable
        )

    except subprocess.TimeoutExpired:
        return EnvironmentResult(
            success=False,
            message="Virtual environment creation timed out",
            operation=context.operation,
            workspace_path=context.workspace_path,
            error_details=f"Operation timed out after {context.timeout_seconds} seconds"
        )
    except Exception as e:
        return EnvironmentResult(
            success=False,
            message=f"Virtual environment creation error: {e}",
            operation=context.operation,
            workspace_path=context.workspace_path,
            error_details=str(e)
        )


def install_dependencies(context: EnvironmentContext, pip_executable: Path) -> EnvironmentResult:
    """Install dependencies from requirements file and additional packages."""
    installed_packages = []
    
    try:
        # Install from requirements file if it exists
        if context.requirements_file:
            requirements_path = context.workspace_path / context.requirements_file
            if requirements_path.exists():
                result = subprocess.run(
                    [str(pip_executable), "install", "-r", str(requirements_path)], 
                    capture_output=True, 
                    text=True,
                    timeout=context.timeout_seconds
                )
                
                if result.returncode != 0:
                    return EnvironmentResult(
                        success=False,
                        message=f"Requirements installation failed: {result.stderr}",
                        operation=context.operation,
                        workspace_path=context.workspace_path,
                        error_details=f"Failed to install from {requirements_path}: {result.stderr}"
                    )
                
                installed_packages.append(f"requirements from {context.requirements_file}")

        # Install additional packages
        if context.additional_packages:
            for package in context.additional_packages:
                result = subprocess.run(
                    [str(pip_executable), "install", package], 
                    capture_output=True, 
                    text=True,
                    timeout=60  # Shorter timeout per package
                )
                
                if result.returncode != 0:
                    logger.warning(f"Failed to install package {package}: {result.stderr}")
                else:
                    installed_packages.append(package)

        return EnvironmentResult(
            success=True,
            message=f"Successfully installed {len(installed_packages)} package groups",
            operation=context.operation,
            workspace_path=context.workspace_path,
            installed_packages=installed_packages
        )

    except subprocess.TimeoutExpired:
        return EnvironmentResult(
            success=False,
            message="Dependency installation timed out",
            operation=context.operation,
            workspace_path=context.workspace_path,
            installed_packages=installed_packages,
            error_details=f"Operation timed out after {context.timeout_seconds} seconds"
        )
    except Exception as e:
        return EnvironmentResult(
            success=False,
            message=f"Dependency installation error: {e}",
            operation=context.operation,
            workspace_path=context.workspace_path,
            installed_packages=installed_packages,
            error_details=str(e)
        )


def validate_environment(context: EnvironmentContext, python_executable: Path) -> EnvironmentResult:
    """Validate that the environment is properly set up."""
    details = []
    
    try:
        # Test Python executable
        result = subprocess.run(
            [str(python_executable), "--version"], 
            capture_output=True, 
            text=True,
            timeout=30
        )
        
        if result.returncode != 0:
            return EnvironmentResult(
                success=False,
                message=f"Python executable test failed: {result.stderr}",
                operation=context.operation,
                workspace_path=context.workspace_path,
                error_details="Python executable is not working properly"
            )
        
        python_version = result.stdout.strip()
        details.append(f"Python version: {python_version}")
        
        # Test package imports for validation packages
        for package in context.validate_packages:
            package_name = package.split('==')[0]  # Handle versioned packages
            result = subprocess.run(
                [str(python_executable), "-c", f"import {package_name}"], 
                capture_output=True, 
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                details.append(f"✓ Package {package_name} imports successfully")
            else:
                details.append(f"✗ Package {package_name} import failed: {result.stderr}")

        return EnvironmentResult(
            success=True,
            message="Environment validation completed",
            operation=context.operation,
            workspace_path=context.workspace_path,
            python_executable=python_executable,
            validation_details=details
        )

    except subprocess.TimeoutExpired:
        return EnvironmentResult(
            success=False,
            message="Environment validation timed out",
            operation=context.operation,
            workspace_path=context.workspace_path,
            validation_details=details,
            error_details="Validation operations timed out"
        )
    except Exception as e:
        return EnvironmentResult(
            success=False,
            message=f"Environment validation error: {e}",
            operation=context.operation,
            workspace_path=context.workspace_path,
            validation_details=details,
            error_details=str(e)
        )


def set_environment_variables(context: EnvironmentContext) -> EnvironmentResult:
    """Set up environment variables for the current process."""
    if not context.environment_vars:
        return EnvironmentResult(
            success=True,
            message="No environment variables to set",
            operation=context.operation,
            workspace_path=context.workspace_path
        )

    try:
        set_vars = {}
        for key, value in context.environment_vars.items():
            os.environ[key] = value
            set_vars[key] = value
            logger.info(f"Set environment variable {key}")

        return EnvironmentResult(
            success=True,
            message=f"Set {len(set_vars)} environment variables",
            operation=context.operation,
            workspace_path=context.workspace_path,
            environment_vars=set_vars
        )

    except Exception as e:
        return EnvironmentResult(
            success=False,
            message=f"Failed to set environment variables: {e}",
            operation=context.operation,
            workspace_path=context.workspace_path,
            error_details=str(e)
        )


def analyze_environment(context: EnvironmentContext) -> EnvironmentResult:
    """Analyze the current environment setup."""
    
    try:
        analysis = {}
        
        # System information
        analysis["system_info"] = {
            "platform": platform.platform(),
            "system": platform.system(),
            "machine": platform.machine(),
            "python_version": sys.version,
            "python_executable": sys.executable
        }
        
        # Virtual environment status
        analysis["virtual_env"] = {
            "is_virtual_env": is_in_virtual_environment(),
            "virtual_env_path": os.environ.get("VIRTUAL_ENV"),
            "conda_env": os.environ.get("CONDA_DEFAULT_ENV")
        }
        
        # Environment files in workspace
        workspace_files = {
            "requirements.txt": (context.workspace_path / "requirements.txt").exists(),
            "pyproject.toml": (context.workspace_path / "pyproject.toml").exists(),
            "Pipfile": (context.workspace_path / "Pipfile").exists(),
            "environment.yml": (context.workspace_path / "environment.yml").exists(),
            "poetry.lock": (context.workspace_path / "poetry.lock").exists()
        }
        analysis["environment_files"] = workspace_files
        
        # Python packages (simplified)
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "list", "--format=freeze"],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                packages = [line.split('==')[0] for line in result.stdout.strip().split('\n') if '==' in line]
                analysis["installed_packages"] = {
                    "count": len(packages),
                    "packages": packages[:20]  # Limit to first 20 for brevity
                }
        except:
            analysis["installed_packages"] = {"error": "Could not list packages"}
        
        return EnvironmentResult(
            success=True,
            message="Environment analysis completed",
            operation=context.operation,
            workspace_path=context.workspace_path,
            analysis_results=analysis
        )

    except Exception as e:
        return EnvironmentResult(
            success=False,
            message=f"Environment analysis error: {e}",
            operation=context.operation,
            workspace_path=context.workspace_path,
            error_details=str(e)
        )


def check_development_tools(context: EnvironmentContext) -> EnvironmentResult:
    """Check availability of common development tools."""
    
    try:
        tools_to_check = [
            "git", "pip", "python", "pytest", "black", "flake8", "mypy",
            "isort", "poetry", "pipenv", "conda", "docker", "make"
        ]

        tool_status = {}
        available_count = 0

        for tool in tools_to_check:
            try:
                result = subprocess.run(
                    [tool, "--version"], 
                    capture_output=True, 
                    text=True, 
                    timeout=5
                )
                if result.returncode == 0:
                    version_output = result.stdout.strip() or result.stderr.strip()
                    tool_status[tool] = {
                        "available": True,
                        "version": version_output.split('\n')[0]
                    }
                    available_count += 1
                else:
                    tool_status[tool] = {"available": False, "error": "Command failed"}
            except (subprocess.TimeoutExpired, FileNotFoundError):
                tool_status[tool] = {"available": False, "error": "Command not found or timed out"}

        analysis = {
            "tools_checked": len(tools_to_check),
            "available_tools": available_count,
            "tool_details": tool_status
        }

        return EnvironmentResult(
            success=True,
            message=f"Development tools check completed ({available_count}/{len(tools_to_check)} available)",
            operation=context.operation,
            workspace_path=context.workspace_path,
            analysis_results=analysis
        )

    except Exception as e:
        return EnvironmentResult(
            success=False,
            message=f"Development tools check error: {e}",
            operation=context.operation,
            workspace_path=context.workspace_path,
            error_details=str(e)
        )


def is_in_virtual_environment() -> bool:
    """Check if running in a virtual environment."""
    return (
        hasattr(sys, 'real_prefix') or
        (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix) or
        os.environ.get('VIRTUAL_ENV') is not None or
        os.environ.get('CONDA_DEFAULT_ENV') is not None
    )


def execute_environment_operation(context: EnvironmentContext) -> EnvironmentResult:
    """Execute an environment operation based on the context."""
    
    logger.info(f"Executing {context.operation.value} operation")
    
    if context.operation == EnvironmentOperation.CREATE_VENV:
        return create_virtual_environment(context)
    elif context.operation == EnvironmentOperation.SET_ENV_VARS:
        return set_environment_variables(context)
    elif context.operation == EnvironmentOperation.ANALYZE_ENV:
        return analyze_environment(context)
    elif context.operation == EnvironmentOperation.CHECK_TOOLS:
        return check_development_tools(context)
    elif context.operation == EnvironmentOperation.VALIDATE_ENV:
        if not context.workspace_path:
            return EnvironmentResult(
                success=False,
                message="Python executable required for validation",
                operation=context.operation,
                workspace_path=context.workspace_path,
                error_details="Cannot validate without python_executable"
            )
        # For validation, we need to find the python executable
        venv_path = context.workspace_path / context.venv_name
        if sys.platform == "win32":
            python_executable = venv_path / "Scripts" / "python.exe"
        else:
            python_executable = venv_path / "bin" / "python"
            
        return validate_environment(context, python_executable)
    elif context.operation == EnvironmentOperation.INSTALL_DEPS:
        # For install deps, we need to find the pip executable
        venv_path = context.workspace_path / context.venv_name
        if sys.platform == "win32":
            pip_executable = venv_path / "Scripts" / "pip.exe"
        else:
            pip_executable = venv_path / "bin" / "pip"
            
        return install_dependencies(context, pip_executable)
    else:
        return EnvironmentResult(
            success=False,
            message=f"Unknown operation: {context.operation}",
            operation=context.operation,
            workspace_path=context.workspace_path,
            error_details=f"Supported operations: {[op.value for op in EnvironmentOperation]}"
        )