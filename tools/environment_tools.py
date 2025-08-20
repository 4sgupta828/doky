# tools/environment_tools.py
import logging
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional

# Get a logger instance for this module
logger = logging.getLogger(__name__)


class EnvironmentTools:
    """
    Agent-agnostic environment analysis and management utilities.
    
    This module provides low-level environment operations that can be used
    by any agent or component. It contains no agent-specific logic,
    only pure environment analysis and manipulation functions.
    """

    @staticmethod
    def analyze_system_info() -> Dict[str, Any]:
        """Analyze basic system information."""
        try:
            return {
                "platform": platform.platform(),
                "system": platform.system(),
                "release": platform.release(),
                "version": platform.version(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "architecture": platform.architecture(),
                "node": platform.node(),
                "python_implementation": platform.python_implementation(),
                "python_compiler": platform.python_compiler()
            }
        except Exception as e:
            return {"error": f"Failed to analyze system info: {e}"}

    @staticmethod
    def analyze_python_environment() -> Dict[str, Any]:
        """Analyze Python installation and environment."""
        try:
            python_info = {
                "executable": sys.executable,
                "version": sys.version,
                "version_info": {
                    "major": sys.version_info.major,
                    "minor": sys.version_info.minor,
                    "micro": sys.version_info.micro,
                    "releaselevel": sys.version_info.releaselevel,
                    "serial": sys.version_info.serial
                },
                "prefix": sys.prefix,
                "base_prefix": sys.base_prefix,
                "exec_prefix": sys.exec_prefix,
                "path": sys.path[:10],  # Limit to first 10 paths
                "modules": list(sys.modules.keys())[:50],  # Limit to first 50 modules
                "platform": sys.platform,
                "maxsize": sys.maxsize,
                "copyright": sys.copyright[:200]  # Truncate copyright
            }

            # Check if running in virtual environment
            python_info["in_virtual_env"] = EnvironmentTools.is_in_virtual_environment()
            
            # Check for common Python installations
            python_info["available_pythons"] = EnvironmentTools.find_python_installations()

            return python_info
        except Exception as e:
            return {"error": f"Failed to analyze Python environment: {e}"}

    @staticmethod
    def analyze_virtual_environment(working_directory: str = None) -> Dict[str, Any]:
        """Analyze virtual environment details."""
        try:
            working_directory = working_directory or os.getcwd()
            
            venv_info = {
                "is_virtual_env": EnvironmentTools.is_in_virtual_environment(),
                "virtual_env": os.environ.get("VIRTUAL_ENV"),
                "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
                "pipenv_active": os.environ.get("PIPENV_ACTIVE"),
                "poetry_active": "POETRY_ACTIVE" in os.environ
            }

            # Check for common virtual environment files
            working_path = Path(working_directory)
            venv_indicators = {
                "venv": (working_path / "venv").exists(),
                "env": (working_path / "env").exists(),
                ".venv": (working_path / ".venv").exists(),
                "Pipfile": (working_path / "Pipfile").exists(),
                "pyproject.toml": (working_path / "pyproject.toml").exists(),
                "environment.yml": (working_path / "environment.yml").exists(),
                "requirements.txt": (working_path / "requirements.txt").exists(),
                "poetry.lock": (working_path / "poetry.lock").exists()
            }
            venv_info["environment_files"] = venv_indicators

            # Check virtual environment health
            if venv_info["is_virtual_env"]:
                venv_info["health_check"] = EnvironmentTools.check_virtual_env_health()

            return venv_info
        except Exception as e:
            return {"error": f"Failed to analyze virtual environment: {e}"}

    @staticmethod
    def check_development_tools(detailed: bool = False) -> Dict[str, Any]:
        """Check availability of common development tools."""
        tools_to_check = [
            "git", "pip", "python", "pytest", "black", "flake8", "mypy",
            "isort", "poetry", "pipenv", "conda", "docker", "make", "curl", "wget"
        ]

        if detailed:
            tools_to_check.extend([
                "pre-commit", "tox", "coverage", "bandit", "safety",
                "jupyter", "ipython", "pylint", "autopep8", "yapf"
            ])

        tool_status = {}

        for tool in tools_to_check:
            try:
                result = subprocess.run(
                    [tool, "--version"], 
                    capture_output=True, 
                    text=True, 
                    timeout=5
                )
                if result.returncode == 0:
                    # Extract version from output
                    version_output = result.stdout.strip() or result.stderr.strip()
                    tool_status[tool] = {
                        "available": True,
                        "version": version_output.split('\n')[0],  # First line usually has version
                        "path": EnvironmentTools.find_tool_path(tool)
                    }
                else:
                    tool_status[tool] = {
                        "available": False,
                        "error": result.stderr.strip()
                    }
            except subprocess.TimeoutExpired:
                tool_status[tool] = {
                    "available": False,
                    "error": "Command timed out"
                }
            except FileNotFoundError:
                tool_status[tool] = {
                    "available": False,
                    "error": "Command not found"
                }
            except Exception as e:
                tool_status[tool] = {
                    "available": False,
                    "error": str(e)
                }

        return {
            "tools_checked": len(tools_to_check),
            "available_tools": len([t for t in tool_status.values() if t.get("available", False)]),
            "tool_details": tool_status
        }

    @staticmethod
    def analyze_python_packages(working_directory: str = None) -> Dict[str, Any]:
        """Analyze installed Python packages."""
        working_directory = working_directory or os.getcwd()
        
        try:
            # Get list of installed packages
            result = subprocess.run(
                [sys.executable, "-m", "pip", "list", "--format=freeze"],
                capture_output=True,
                text=True,
                cwd=working_directory,
                timeout=30
            )

            if result.returncode == 0:
                packages = []
                for line in result.stdout.strip().split('\n'):
                    if '==' in line:
                        name, version = line.split('==', 1)
                        packages.append({"name": name, "version": version})

                # Check for common development packages
                dev_packages = [
                    "pytest", "black", "flake8", "mypy", "isort", "coverage",
                    "pre-commit", "tox", "jupyter", "ipython", "requests",
                    "numpy", "pandas", "flask", "django", "fastapi"
                ]

                package_names = [pkg["name"].lower() for pkg in packages]
                dev_status = {pkg: pkg.lower() in package_names for pkg in dev_packages}

                return {
                    "total_packages": len(packages),
                    "packages": packages,
                    "common_dev_packages": dev_status,
                    "pip_list_success": True
                }
            else:
                return {
                    "error": f"Failed to list packages: {result.stderr}",
                    "pip_list_success": False
                }
        except Exception as e:
            return {
                "error": f"Failed to analyze Python packages: {e}",
                "pip_list_success": False
            }

    @staticmethod
    def check_system_dependencies() -> Dict[str, Any]:
        """Check system-level dependencies and libraries."""
        system_deps = {}

        # Check common system libraries and tools
        if platform.system() == "Darwin":  # macOS
            deps_to_check = ["brew", "xcode-select", "gcc", "clang"]
        elif platform.system() == "Linux":
            deps_to_check = ["apt", "yum", "dnf", "gcc", "g++", "make"]
        elif platform.system() == "Windows":
            deps_to_check = ["choco", "winget", "cl"]  # Visual Studio compiler
        else:
            deps_to_check = []

        for dep in deps_to_check:
            try:
                result = subprocess.run(
                    [dep, "--version"], 
                    capture_output=True, 
                    text=True, 
                    timeout=5
                )
                system_deps[dep] = {
                    "available": result.returncode == 0,
                    "output": result.stdout.strip()[:200]  # Limit output
                }
            except (FileNotFoundError, subprocess.TimeoutExpired):
                system_deps[dep] = {"available": False}

        return {
            "system": platform.system(),
            "dependencies": system_deps,
            "architecture": platform.machine(),
            "processor": platform.processor()
        }

    @staticmethod
    def analyze_environment_variables() -> Dict[str, Any]:
        """Analyze relevant environment variables."""
        relevant_vars = [
            "PATH", "PYTHONPATH", "VIRTUAL_ENV", "CONDA_DEFAULT_ENV",
            "HOME", "USER", "SHELL", "TERM", "LANG", "LC_ALL",
            "PIPENV_ACTIVE", "POETRY_ACTIVE", "NODE_ENV", "JAVA_HOME"
        ]

        env_analysis = {
            "total_variables": len(os.environ),
            "relevant_variables": {}
        }

        for var in relevant_vars:
            if var in os.environ:
                value = os.environ[var]
                # Truncate very long values (like PATH)
                if len(value) > 500:
                    value = value[:500] + "... (truncated)"
                env_analysis["relevant_variables"][var] = value

        return env_analysis

    @staticmethod
    def analyze_system_path() -> Dict[str, Any]:
        """Analyze system PATH for development tools."""
        path_env = os.environ.get("PATH", "")
        path_entries = path_env.split(os.pathsep)

        analysis = {
            "total_entries": len(path_entries),
            "entries": path_entries[:20],  # Limit to first 20 entries
            "python_paths": [],
            "development_paths": []
        }

        # Look for Python-related paths
        for entry in path_entries:
            if any(keyword in entry.lower() for keyword in ["python", "pip", "conda", "venv"]):
                analysis["python_paths"].append(entry)
            elif any(keyword in entry.lower() for keyword in ["bin", "usr/local", "homebrew", "tools"]):
                analysis["development_paths"].append(entry)

        return analysis

    @staticmethod
    def is_in_virtual_environment() -> bool:
        """Check if running in a virtual environment."""
        return (
            hasattr(sys, 'real_prefix') or
            (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix) or
            os.environ.get('VIRTUAL_ENV') is not None or
            os.environ.get('CONDA_DEFAULT_ENV') is not None
        )

    @staticmethod
    def check_virtual_env_health() -> Dict[str, Any]:
        """Check health of current virtual environment."""
        health = {
            "pip_available": True,
            "can_install_packages": False,
            "environment_isolated": EnvironmentTools.is_in_virtual_environment()
        }

        try:
            # Test pip functionality
            result = subprocess.run(
                [sys.executable, "-m", "pip", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            health["pip_available"] = result.returncode == 0
            
            # Test package installation capability (dry run)
            if health["pip_available"]:
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "--dry-run", "requests"],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                health["can_install_packages"] = result.returncode == 0

        except Exception as e:
            health["error"] = str(e)

        return health

    @staticmethod
    def find_python_installations() -> List[Dict[str, str]]:
        """Find available Python installations."""
        installations = []
        common_names = ["python", "python3", "python3.8", "python3.9", "python3.10", "python3.11", "python3.12"]

        for name in common_names:
            try:
                result = subprocess.run(
                    [name, "--version"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    installations.append({
                        "name": name,
                        "version": result.stdout.strip(),
                        "path": EnvironmentTools.find_tool_path(name)
                    })
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue

        return installations

    @staticmethod
    def find_tool_path(tool_name: str) -> Optional[str]:
        """Find the path of a tool."""
        try:
            if platform.system() == "Windows":
                result = subprocess.run(
                    ["where", tool_name],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
            else:
                result = subprocess.run(
                    ["which", tool_name],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
            
            if result.returncode == 0:
                return result.stdout.strip().split('\n')[0]  # First path if multiple
        except Exception:
            pass
        
        return None

    @staticmethod
    def setup_virtual_environment(workspace_path: Path, venv_name: str = "venv", 
                                 python_version: Optional[str] = None, 
                                 force_recreate: bool = False) -> Dict[str, Any]:
        """Create or validate virtual environment."""
        venv_path = workspace_path / venv_name
        
        # Check if virtual environment already exists
        if venv_path.exists() and not force_recreate:
            logger.debug(f"Using existing virtual environment at {venv_path}")
        else:
            if force_recreate and venv_path.exists():
                logger.debug(f"Recreating virtual environment at {venv_path}")
                import shutil
                shutil.rmtree(venv_path)
            
            # Create new virtual environment
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

    @staticmethod
    def install_dependencies(workspace_path: Path, pip_executable: Path, 
                           requirements_file: str = "requirements.txt", 
                           additional_packages: List[str] = None) -> Dict[str, Any]:
        """Install dependencies from requirements file and additional packages."""
        additional_packages = additional_packages or []
        installed_packages = []
        
        # Install from requirements file if it exists
        requirements_path = workspace_path / requirements_file
        if requirements_path.exists():
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

    @staticmethod
    def validate_environment(python_executable: Path, expected_packages: List[str] = None) -> Dict[str, Any]:
        """Validate that the environment is properly set up."""
        expected_packages = expected_packages or []
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

    @staticmethod
    def assess_environment_health(analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall environment health."""
        issues = []
        warnings = []
        good_points = []

        # Check Python version
        python_info = analysis_results.get("python_info", {})
        if "version_info" in python_info:
            version_info = python_info["version_info"]
            if version_info["major"] >= 3 and version_info["minor"] >= 8:
                good_points.append("Python version is modern (3.8+)")
            elif version_info["major"] >= 3 and version_info["minor"] >= 6:
                warnings.append("Python version is older but supported (3.6-3.7)")
            else:
                issues.append("Python version is very old and may not be supported")

        # Check virtual environment
        venv_info = analysis_results.get("virtual_env_info", {})
        if venv_info and venv_info.get("is_virtual_env"):
            good_points.append("Running in virtual environment")
        else:
            warnings.append("Not running in a virtual environment")

        # Check development tools
        dev_tools = analysis_results.get("development_tools", {})
        if dev_tools:
            available_count = dev_tools.get("available_tools", 0)
            total_count = dev_tools.get("tools_checked", 0)
            
            if available_count >= total_count * 0.8:
                good_points.append(f"Most development tools available ({available_count}/{total_count})")
            elif available_count >= total_count * 0.5:
                warnings.append(f"Some development tools missing ({available_count}/{total_count})")
            else:
                issues.append(f"Many development tools missing ({available_count}/{total_count})")

        # Overall assessment
        if len(issues) == 0 and len(warnings) <= 1:
            health_status = "healthy"
            summary = "Environment is in good condition"
        elif len(issues) == 0:
            health_status = "minor_issues"
            summary = "Environment has minor issues but is functional"
        elif len(issues) <= 2:
            health_status = "needs_attention"
            summary = "Environment needs attention to resolve issues"
        else:
            health_status = "problematic"
            summary = "Environment has significant issues"

        return {
            "health_status": health_status,
            "summary": summary,
            "issues": issues,
            "warnings": warnings,
            "good_points": good_points,
            "recommendations": EnvironmentTools._generate_recommendations(issues, warnings)
        }

    @staticmethod
    def _generate_recommendations(issues: List[str], warnings: List[str]) -> List[str]:
        """Generate recommendations based on identified issues."""
        recommendations = []

        if any("python version" in issue.lower() for issue in issues):
            recommendations.append("Consider upgrading to Python 3.8 or newer")

        if any("virtual environment" in warning.lower() for warning in warnings):
            recommendations.append("Set up a virtual environment for project isolation")

        if any("tools missing" in issue.lower() or "tools missing" in warning.lower() for issue in issues for warning in warnings):
            recommendations.append("Install missing development tools for better development experience")

        if not recommendations:
            recommendations.append("Environment looks good - consider regular maintenance")

        return recommendations