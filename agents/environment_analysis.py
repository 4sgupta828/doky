# agents/environment_analysis.py
import logging
import sys
import os
import platform
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional

# Foundational dependencies
from .base import BaseAgent
from core.context import GlobalContext
from core.models import AgentResponse, AgentResult, TaskNode

# Get a logger instance for this module
logger = logging.getLogger(__name__)


class EnvironmentAnalysisAgent(BaseAgent):
    """
    Analysis Tier: Environment state diagnosis ONLY.
    
    This agent provides read-only analysis of the development environment.
    
    Responsibilities:
    - Python version and virtual environment analysis
    - System dependency and PATH analysis
    - Environment configuration validation
    - Development tool availability assessment
    
    Does NOT: Create environments, install anything, modify configurations
    """

    def __init__(self):
        super().__init__(
            name="EnvironmentAnalysisAgent",
            description="Analyzes development environment state, Python installations, and tool availability."
        )

    def required_inputs(self) -> List[str]:
        """Required inputs for EnvironmentAnalysisAgent execution."""
        return []  # Can analyze without specific inputs

    def optional_inputs(self) -> List[str]:
        """Optional inputs for EnvironmentAnalysisAgent execution."""
        return [
            "working_directory",
            "check_tools",
            "check_python_packages",
            "analyze_virtual_env",
            "check_system_dependencies",
            "detailed_analysis"
        ]

    def execute_v2(self, goal: str, inputs: Dict[str, Any], global_context: GlobalContext) -> AgentResult:
        """
        NEW INTERFACE: Analyze development environment state.
        """
        logger.info(f"EnvironmentAnalysisAgent executing: '{goal}'")
        
        # Validate inputs
        try:
            self.validate_inputs(inputs)
        except Exception as validation_error:
            return self.create_result(
                success=False,
                message=str(validation_error),
                error_details={"validation_error": str(validation_error)}
            )

        # Extract inputs
        working_directory = inputs.get("working_directory", str(global_context.workspace_path))
        check_tools = inputs.get("check_tools", True)
        check_python_packages = inputs.get("check_python_packages", True)
        analyze_virtual_env = inputs.get("analyze_virtual_env", True)
        check_system_dependencies = inputs.get("check_system_dependencies", True)
        detailed_analysis = inputs.get("detailed_analysis", False)

        try:
            self.report_progress("Starting environment analysis", f"Working in {working_directory}")

            analysis_results = {
                "system_info": self._analyze_system_info(),
                "python_info": self._analyze_python_environment(working_directory),
                "virtual_env_info": None,
                "development_tools": None,
                "python_packages": None,
                "system_dependencies": None,
                "environment_variables": self._analyze_environment_variables(),
                "path_analysis": self._analyze_system_path()
            }

            # Optional detailed analyses
            if analyze_virtual_env:
                analysis_results["virtual_env_info"] = self._analyze_virtual_environment(working_directory)

            if check_tools:
                analysis_results["development_tools"] = self._check_development_tools(detailed_analysis)

            if check_python_packages:
                analysis_results["python_packages"] = self._analyze_python_packages(working_directory)

            if check_system_dependencies:
                analysis_results["system_dependencies"] = self._check_system_dependencies()

            # Overall environment health assessment
            health_assessment = self._assess_environment_health(analysis_results)

            self.report_progress("Environment analysis complete", health_assessment["summary"])

            return self.create_result(
                success=True,
                message=f"Environment analysis complete: {health_assessment['summary']}",
                outputs={
                    "analysis_results": analysis_results,
                    "health_assessment": health_assessment,
                    "working_directory": working_directory
                }
            )

        except Exception as e:
            error_msg = f"EnvironmentAnalysisAgent execution failed: {e}"
            logger.error(error_msg, exc_info=True)
            return self.create_result(
                success=False,
                message=error_msg,
                error_details={"exception": str(e)}
            )

    def _analyze_system_info(self) -> Dict[str, Any]:
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

    def _analyze_python_environment(self, working_directory: str) -> Dict[str, Any]:
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
            python_info["in_virtual_env"] = self._is_in_virtual_environment()
            
            # Check for common Python installations
            python_info["available_pythons"] = self._find_python_installations()

            return python_info

        except Exception as e:
            return {"error": f"Failed to analyze Python environment: {e}"}

    def _analyze_virtual_environment(self, working_directory: str) -> Dict[str, Any]:
        """Analyze virtual environment details."""
        
        try:
            venv_info = {
                "is_virtual_env": self._is_in_virtual_environment(),
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
                venv_info["health_check"] = self._check_virtual_env_health()

            return venv_info

        except Exception as e:
            return {"error": f"Failed to analyze virtual environment: {e}"}

    def _check_development_tools(self, detailed: bool = False) -> Dict[str, Any]:
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
                        "path": self._find_tool_path(tool)
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

    def _analyze_python_packages(self, working_directory: str) -> Dict[str, Any]:
        """Analyze installed Python packages."""
        
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

    def _check_system_dependencies(self) -> Dict[str, Any]:
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

    def _analyze_environment_variables(self) -> Dict[str, Any]:
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

    def _analyze_system_path(self) -> Dict[str, Any]:
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

    def _assess_environment_health(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
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
            "recommendations": self._generate_recommendations(issues, warnings)
        }

    def _generate_recommendations(self, issues: List[str], warnings: List[str]) -> List[str]:
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

    def _is_in_virtual_environment(self) -> bool:
        """Check if running in a virtual environment."""
        return (
            hasattr(sys, 'real_prefix') or
            (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix) or
            os.environ.get('VIRTUAL_ENV') is not None or
            os.environ.get('CONDA_DEFAULT_ENV') is not None
        )

    def _check_virtual_env_health(self) -> Dict[str, Any]:
        """Check health of current virtual environment."""
        
        health = {
            "pip_available": True,
            "can_install_packages": False,
            "environment_isolated": self._is_in_virtual_environment()
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

    def _find_python_installations(self) -> List[Dict[str, str]]:
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
                        "path": self._find_tool_path(name)
                    })
            except (FileNotFoundError, subprocess.TimeoutExpired):
                continue

        return installations

    def _find_tool_path(self, tool_name: str) -> Optional[str]:
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
        except:
            pass
        
        return None

    # Legacy execute method for backward compatibility
    def execute(self, goal: str, context: GlobalContext, current_task: TaskNode) -> AgentResponse:
        """Legacy execute method - converts to new interface."""
        inputs = {
            'working_directory': str(context.workspace_path),
            'detailed_analysis': False
        }
        
        result = self.execute_v2(goal, inputs, context)
        
        return AgentResponse(
            success=result.success,
            message=result.message,
            artifacts_generated=[]
        )