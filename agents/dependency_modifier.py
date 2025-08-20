# agents/dependency_modifier.py
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


class DependencyModifierAgent(BaseAgent):
    """
    Infrastructure Tier: Package operations ONLY.
    
    This agent handles all package management operations without analysis.
    
    Responsibilities:
    - Install/uninstall packages with pip/conda
    - Manage requirements.txt and lock files
    - Handle version conflicts and updates
    - Package vulnerability scanning and updates
    
    Does NOT: Analyze imports, create environments, modify code
    """

    def __init__(self):
        super().__init__(
            name="DependencyModifierAgent",
            description="Manages package installation, requirements files, and dependency operations."
        )

    def required_inputs(self) -> List[str]:
        """Required inputs for DependencyModifierAgent execution."""
        return ["operation"]

    def optional_inputs(self) -> List[str]:
        """Optional inputs for DependencyModifierAgent execution."""
        return [
            "packages", 
            "requirements_file", 
            "package_manager",
            "working_directory",
            "python_executable",
            "upgrade_packages",
            "force_reinstall",
            "index_url",
            "extra_index_urls"
        ]

    def execute_v2(self, goal: str, inputs: Dict[str, Any], global_context: GlobalContext) -> AgentResult:
        """
        NEW INTERFACE: Execute package management operations.
        """
        logger.info(f"DependencyModifierAgent executing: '{goal}'")
        
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
        operation = inputs["operation"]
        packages = inputs.get("packages", [])
        requirements_file = inputs.get("requirements_file")
        package_manager = inputs.get("package_manager", "pip")
        working_directory = inputs.get("working_directory", str(global_context.workspace_path))
        python_executable = inputs.get("python_executable", sys.executable)
        upgrade_packages = inputs.get("upgrade_packages", False)
        force_reinstall = inputs.get("force_reinstall", False)
        index_url = inputs.get("index_url")
        extra_index_urls = inputs.get("extra_index_urls", [])

        try:
            self.report_progress(f"Starting {operation} operation", f"Using {package_manager}")

            if operation == "install":
                result = self._install_packages(
                    packages, requirements_file, package_manager, 
                    working_directory, python_executable, upgrade_packages, 
                    force_reinstall, index_url, extra_index_urls
                )
            elif operation == "uninstall":
                result = self._uninstall_packages(
                    packages, package_manager, working_directory, python_executable
                )
            elif operation == "list":
                result = self._list_packages(package_manager, working_directory, python_executable)
            elif operation == "freeze":
                result = self._freeze_requirements(
                    requirements_file, package_manager, working_directory, python_executable
                )
            elif operation == "update":
                result = self._update_packages(
                    packages, package_manager, working_directory, python_executable
                )
            elif operation == "check":
                result = self._check_dependencies(
                    package_manager, working_directory, python_executable
                )
            else:
                return self.create_result(
                    success=False,
                    message=f"Unknown operation: {operation}",
                    error_details={"supported_operations": ["install", "uninstall", "list", "freeze", "update", "check"]}
                )

            self.report_progress("Package operation complete", result["message"])

            return self.create_result(
                success=result["success"],
                message=result["message"],
                outputs=result["outputs"]
            )

        except Exception as e:
            error_msg = f"DependencyModifierAgent execution failed: {e}"
            logger.error(error_msg, exc_info=True)
            return self.create_result(
                success=False,
                message=error_msg,
                error_details={"exception": str(e)}
            )

    def _install_packages(self, packages: List[str], requirements_file: Optional[str], 
                         package_manager: str, working_directory: str, python_executable: str,
                         upgrade_packages: bool, force_reinstall: bool, index_url: Optional[str],
                         extra_index_urls: List[str]) -> Dict[str, Any]:
        """Install packages using the specified package manager."""
        
        if not packages and not requirements_file:
            return {
                "success": False,
                "message": "No packages or requirements file specified",
                "outputs": {}
            }

        commands_executed = []
        results = []

        try:
            if package_manager == "pip":
                # Build pip install command
                cmd = [python_executable, "-m", "pip", "install"]
                
                if upgrade_packages:
                    cmd.append("--upgrade")
                if force_reinstall:
                    cmd.extend(["--force-reinstall", "--no-deps"])
                if index_url:
                    cmd.extend(["--index-url", index_url])
                for extra_url in extra_index_urls:
                    cmd.extend(["--extra-index-url", extra_url])

                # Install from requirements file
                if requirements_file:
                    req_path = Path(working_directory) / requirements_file
                    if req_path.exists():
                        install_cmd = cmd + ["-r", str(req_path)]
                        result = subprocess.run(
                            install_cmd, 
                            capture_output=True, 
                            text=True, 
                            cwd=working_directory,
                            timeout=300
                        )
                        commands_executed.append(" ".join(install_cmd))
                        results.append({
                            "command": " ".join(install_cmd),
                            "returncode": result.returncode,
                            "stdout": result.stdout,
                            "stderr": result.stderr
                        })

                # Install individual packages
                if packages:
                    install_cmd = cmd + packages
                    result = subprocess.run(
                        install_cmd, 
                        capture_output=True, 
                        text=True, 
                        cwd=working_directory,
                        timeout=300
                    )
                    commands_executed.append(" ".join(install_cmd))
                    results.append({
                        "command": " ".join(install_cmd),
                        "returncode": result.returncode,
                        "stdout": result.stdout,
                        "stderr": result.stderr
                    })

            elif package_manager == "conda":
                # Build conda install command
                cmd = ["conda", "install", "-y"]
                
                if packages:
                    install_cmd = cmd + packages
                    result = subprocess.run(
                        install_cmd, 
                        capture_output=True, 
                        text=True, 
                        cwd=working_directory,
                        timeout=600
                    )
                    commands_executed.append(" ".join(install_cmd))
                    results.append({
                        "command": " ".join(install_cmd),
                        "returncode": result.returncode,
                        "stdout": result.stdout,
                        "stderr": result.stderr
                    })

            # Check if all installations succeeded
            all_success = all(r["returncode"] == 0 for r in results)
            
            if all_success:
                return {
                    "success": True,
                    "message": f"Successfully installed packages using {package_manager}",
                    "outputs": {
                        "package_manager": package_manager,
                        "commands_executed": commands_executed,
                        "results": results,
                        "packages_installed": packages,
                        "requirements_file": requirements_file
                    }
                }
            else:
                failed_results = [r for r in results if r["returncode"] != 0]
                return {
                    "success": False,
                    "message": f"Package installation failed with {package_manager}",
                    "outputs": {
                        "package_manager": package_manager,
                        "commands_executed": commands_executed,
                        "results": results,
                        "failed_results": failed_results
                    }
                }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "message": "Package installation timed out",
                "outputs": {"timeout": True, "commands_executed": commands_executed}
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Package installation error: {e}",
                "outputs": {"error": str(e), "commands_executed": commands_executed}
            }

    def _uninstall_packages(self, packages: List[str], package_manager: str,
                           working_directory: str, python_executable: str) -> Dict[str, Any]:
        """Uninstall packages using the specified package manager."""
        
        if not packages:
            return {
                "success": False,
                "message": "No packages specified for uninstallation",
                "outputs": {}
            }

        try:
            if package_manager == "pip":
                cmd = [python_executable, "-m", "pip", "uninstall", "-y"] + packages
            elif package_manager == "conda":
                cmd = ["conda", "remove", "-y"] + packages
            else:
                return {
                    "success": False,
                    "message": f"Unsupported package manager: {package_manager}",
                    "outputs": {}
                }

            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                cwd=working_directory,
                timeout=120
            )

            if result.returncode == 0:
                return {
                    "success": True,
                    "message": f"Successfully uninstalled packages: {', '.join(packages)}",
                    "outputs": {
                        "packages_uninstalled": packages,
                        "command": " ".join(cmd),
                        "stdout": result.stdout,
                        "stderr": result.stderr
                    }
                }
            else:
                return {
                    "success": False,
                    "message": f"Failed to uninstall packages: {result.stderr}",
                    "outputs": {
                        "command": " ".join(cmd),
                        "returncode": result.returncode,
                        "stdout": result.stdout,
                        "stderr": result.stderr
                    }
                }

        except Exception as e:
            return {
                "success": False,
                "message": f"Uninstallation error: {e}",
                "outputs": {"error": str(e)}
            }

    def _list_packages(self, package_manager: str, working_directory: str, 
                      python_executable: str) -> Dict[str, Any]:
        """List installed packages."""
        
        try:
            if package_manager == "pip":
                cmd = [python_executable, "-m", "pip", "list"]
            elif package_manager == "conda":
                cmd = ["conda", "list"]
            else:
                return {
                    "success": False,
                    "message": f"Unsupported package manager: {package_manager}",
                    "outputs": {}
                }

            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                cwd=working_directory,
                timeout=60
            )

            if result.returncode == 0:
                return {
                    "success": True,
                    "message": f"Successfully listed packages with {package_manager}",
                    "outputs": {
                        "package_list": result.stdout,
                        "command": " ".join(cmd)
                    }
                }
            else:
                return {
                    "success": False,
                    "message": f"Failed to list packages: {result.stderr}",
                    "outputs": {
                        "command": " ".join(cmd),
                        "stderr": result.stderr
                    }
                }

        except Exception as e:
            return {
                "success": False,
                "message": f"Package listing error: {e}",
                "outputs": {"error": str(e)}
            }

    def _freeze_requirements(self, requirements_file: Optional[str], package_manager: str,
                           working_directory: str, python_executable: str) -> Dict[str, Any]:
        """Generate requirements file from installed packages."""
        
        try:
            if package_manager == "pip":
                cmd = [python_executable, "-m", "pip", "freeze"]
            else:
                return {
                    "success": False,
                    "message": f"Freeze operation not supported for {package_manager}",
                    "outputs": {}
                }

            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                cwd=working_directory,
                timeout=60
            )

            if result.returncode == 0:
                # Write to file if specified
                if requirements_file:
                    req_path = Path(working_directory) / requirements_file
                    req_path.write_text(result.stdout)
                    
                return {
                    "success": True,
                    "message": f"Successfully generated requirements {'to ' + requirements_file if requirements_file else ''}",
                    "outputs": {
                        "requirements_content": result.stdout,
                        "requirements_file": requirements_file,
                        "command": " ".join(cmd)
                    }
                }
            else:
                return {
                    "success": False,
                    "message": f"Failed to freeze requirements: {result.stderr}",
                    "outputs": {
                        "command": " ".join(cmd),
                        "stderr": result.stderr
                    }
                }

        except Exception as e:
            return {
                "success": False,
                "message": f"Requirements freeze error: {e}",
                "outputs": {"error": str(e)}
            }

    def _update_packages(self, packages: List[str], package_manager: str,
                        working_directory: str, python_executable: str) -> Dict[str, Any]:
        """Update packages to latest versions."""
        
        if not packages:
            return {
                "success": False,
                "message": "No packages specified for update",
                "outputs": {}
            }

        try:
            if package_manager == "pip":
                cmd = [python_executable, "-m", "pip", "install", "--upgrade"] + packages
            elif package_manager == "conda":
                cmd = ["conda", "update", "-y"] + packages
            else:
                return {
                    "success": False,
                    "message": f"Unsupported package manager: {package_manager}",
                    "outputs": {}
                }

            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                cwd=working_directory,
                timeout=300
            )

            if result.returncode == 0:
                return {
                    "success": True,
                    "message": f"Successfully updated packages: {', '.join(packages)}",
                    "outputs": {
                        "packages_updated": packages,
                        "command": " ".join(cmd),
                        "stdout": result.stdout,
                        "stderr": result.stderr
                    }
                }
            else:
                return {
                    "success": False,
                    "message": f"Failed to update packages: {result.stderr}",
                    "outputs": {
                        "command": " ".join(cmd),
                        "returncode": result.returncode,
                        "stderr": result.stderr
                    }
                }

        except Exception as e:
            return {
                "success": False,
                "message": f"Package update error: {e}",
                "outputs": {"error": str(e)}
            }

    def _check_dependencies(self, package_manager: str, working_directory: str,
                           python_executable: str) -> Dict[str, Any]:
        """Check for dependency conflicts and issues."""
        
        try:
            if package_manager == "pip":
                cmd = [python_executable, "-m", "pip", "check"]
            else:
                return {
                    "success": False,
                    "message": f"Dependency check not supported for {package_manager}",
                    "outputs": {}
                }

            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                cwd=working_directory,
                timeout=60
            )

            # pip check returns 0 if no issues, non-zero if conflicts found
            if result.returncode == 0:
                return {
                    "success": True,
                    "message": "No dependency conflicts found",
                    "outputs": {
                        "conflicts_found": False,
                        "command": " ".join(cmd),
                        "stdout": result.stdout
                    }
                }
            else:
                return {
                    "success": False,
                    "message": f"Dependency conflicts detected: {result.stdout}",
                    "outputs": {
                        "conflicts_found": True,
                        "conflicts": result.stdout,
                        "command": " ".join(cmd)
                    }
                }

        except Exception as e:
            return {
                "success": False,
                "message": f"Dependency check error: {e}",
                "outputs": {"error": str(e)}
            }
