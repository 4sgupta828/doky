# agents/test_analysis.py
import logging
import subprocess
import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

# Foundational dependencies
from .base import BaseAgent
from core.context import GlobalContext
from core.models import AgentResponse, AgentResult, TaskNode

# Get a logger instance for this module
logger = logging.getLogger(__name__)


class TestAnalysisAgent(BaseAgent):
    """
    Analysis Tier: Read-only test result analysis.
    
    This agent provides read-only analysis of test results.
    
    Responsibilities:
    - Test result parsing and reporting
    - Test failure analysis
    - Test coverage analysis
    - Timeout and process management
    - Test discovery and filtering
    """

    def __init__(self):
        super().__init__(
            name="TestAnalysisAgent",
            description="Analyzes test results and provides comprehensive testing insights."
        )

    def required_inputs(self) -> List[str]:
        """Required inputs for TestExecutorAgent execution."""
        return ["test_target"]  # Can be files, directories, or test commands

    def optional_inputs(self) -> List[str]:
        """Optional inputs for TestExecutorAgent execution."""
        return [
            "test_framework",  # pytest, unittest, custom
            "python_executable", 
            "working_directory",
            "timeout_seconds",
            "test_patterns",
            "exclude_patterns",
            "output_format",  # json, xml, text
            "coverage_enabled",
            "additional_args"
        ]

    def execute_v2(self, goal: str, inputs: Dict[str, Any], global_context: GlobalContext) -> AgentResult:
        """
        NEW INTERFACE: Execute tests with comprehensive configuration options.
        """
        logger.info(f"TestExecutorAgent executing: '{goal}'")
        
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
        test_target = inputs["test_target"]
        test_framework = inputs.get("test_framework", "auto")  # auto-detect
        python_executable = inputs.get("python_executable", sys.executable)
        working_directory = inputs.get("working_directory", str(global_context.workspace_path))
        timeout_seconds = inputs.get("timeout_seconds", 300)  # 5 minutes default
        test_patterns = inputs.get("test_patterns", [])
        exclude_patterns = inputs.get("exclude_patterns", [])
        output_format = inputs.get("output_format", "json")
        coverage_enabled = inputs.get("coverage_enabled", False)
        additional_args = inputs.get("additional_args", [])

        try:
            self.report_progress("Starting test execution", f"Target: {test_target}")

            # Step 1: Detect or validate test framework
            framework_result = self._detect_test_framework(test_framework, test_target, working_directory)
            if not framework_result["success"]:
                return self.create_result(
                    success=False,
                    message=f"Test framework detection failed: {framework_result['message']}",
                    error_details=framework_result
                )

            detected_framework = framework_result["framework"]
            self.report_progress("Test framework detected", f"Using {detected_framework}")

            # Step 2: Discover and filter tests
            discovery_result = self._discover_tests(
                test_target, detected_framework, working_directory, 
                test_patterns, exclude_patterns
            )
            
            if not discovery_result["success"]:
                return self.create_result(
                    success=False,
                    message=f"Test discovery failed: {discovery_result['message']}",
                    error_details=discovery_result
                )

            test_files = discovery_result["test_files"]
            self.report_progress("Test discovery complete", f"Found {len(test_files)} test files")

            # Step 3: Execute tests
            execution_result = self._execute_tests(
                test_files, detected_framework, python_executable, 
                working_directory, timeout_seconds, coverage_enabled, 
                additional_args, output_format
            )

            # Step 4: Parse and analyze results
            analysis_result = self._analyze_test_results(execution_result, detected_framework)

            # Combine results
            final_result = {
                "test_framework": detected_framework,
                "test_files_discovered": len(test_files),
                "test_files_executed": len(execution_result.get("executed_files", [])),
                "execution_details": execution_result,
                "analysis": analysis_result,
                "success": execution_result.get("success", False) and analysis_result.get("success", False)
            }

            message = self._create_summary_message(final_result, analysis_result)

            self.report_progress("Test execution complete", message)

            return self.create_result(
                success=final_result["success"],
                message=message,
                outputs=final_result
            )

        except Exception as e:
            error_msg = f"TestExecutorAgent execution failed: {e}"
            logger.error(error_msg, exc_info=True)
            return self.create_result(
                success=False,
                message=error_msg,
                error_details={"exception": str(e)}
            )

    def _detect_test_framework(self, requested_framework: str, test_target: str, 
                             working_directory: str) -> Dict[str, Any]:
        """Detect or validate the test framework to use."""
        if requested_framework != "auto":
            # Validate requested framework is available
            if requested_framework == "pytest":
                try:
                    result = subprocess.run(
                        [sys.executable, "-c", "import pytest"], 
                        capture_output=True, text=True, cwd=working_directory
                    )
                    if result.returncode == 0:
                        return {"success": True, "framework": "pytest"}
                    else:
                        return {"success": False, "message": "pytest not available"}
                except:
                    return {"success": False, "message": "pytest validation failed"}
                    
            elif requested_framework == "unittest":
                return {"success": True, "framework": "unittest"}
            elif requested_framework == "custom":
                return {"success": True, "framework": "custom"}
            else:
                return {"success": False, "message": f"Unknown test framework: {requested_framework}"}

        # Auto-detect framework
        working_path = Path(working_directory)
        
        # Check for pytest indicators
        if (working_path / "pytest.ini").exists() or (working_path / "pyproject.toml").exists():
            try:
                result = subprocess.run(
                    [sys.executable, "-c", "import pytest"], 
                    capture_output=True, text=True, cwd=working_directory
                )
                if result.returncode == 0:
                    return {"success": True, "framework": "pytest"}
            except:
                pass

        # Check for unittest indicators
        if isinstance(test_target, str) and "test_" in test_target:
            return {"success": True, "framework": "unittest"}

        # Default to pytest if available, otherwise unittest
        try:
            result = subprocess.run(
                [sys.executable, "-c", "import pytest"], 
                capture_output=True, text=True
            )
            if result.returncode == 0:
                return {"success": True, "framework": "pytest"}
        except:
            pass

        return {"success": True, "framework": "unittest"}

    def _discover_tests(self, test_target: str, framework: str, working_directory: str,
                       test_patterns: List[str], exclude_patterns: List[str]) -> Dict[str, Any]:
        """Discover test files based on target and patterns."""
        working_path = Path(working_directory)
        test_files = []

        try:
            if isinstance(test_target, str):
                target_path = Path(test_target)
                
                if target_path.is_absolute():
                    search_path = target_path
                else:
                    search_path = working_path / target_path

                if search_path.is_file():
                    # Single file
                    test_files.append(str(search_path))
                elif search_path.is_dir():
                    # Directory - discover test files
                    if framework == "pytest":
                        # Pytest discovery patterns
                        patterns = test_patterns or ["test_*.py", "*_test.py"]
                    else:
                        # Unittest discovery patterns
                        patterns = test_patterns or ["test*.py"]

                    for pattern in patterns:
                        discovered = list(search_path.rglob(pattern))
                        test_files.extend([str(f) for f in discovered])
                else:
                    # Pattern or command
                    if framework == "pytest":
                        patterns = [test_target]
                    else:
                        patterns = test_patterns or ["test*.py"]
                    
                    for pattern in patterns:
                        discovered = list(working_path.rglob(pattern))
                        test_files.extend([str(f) for f in discovered])

            # Filter out excluded patterns
            if exclude_patterns:
                filtered_files = []
                for test_file in test_files:
                    excluded = any(pattern in test_file for pattern in exclude_patterns)
                    if not excluded:
                        filtered_files.append(test_file)
                test_files = filtered_files

            # Remove duplicates and ensure files exist
            test_files = list(set(test_files))
            test_files = [f for f in test_files if Path(f).exists()]

            return {
                "success": True,
                "test_files": test_files,
                "discovery_method": framework
            }

        except Exception as e:
            return {
                "success": False,
                "message": f"Test discovery failed: {e}",
                "test_files": []
            }

    def _execute_tests(self, test_files: List[str], framework: str, python_executable: str,
                      working_directory: str, timeout_seconds: int, coverage_enabled: bool,
                      additional_args: List[str], output_format: str) -> Dict[str, Any]:
        """Execute tests using the specified framework."""
        start_time = time.time()
        
        try:
            if framework == "pytest":
                return self._execute_pytest(
                    test_files, python_executable, working_directory, 
                    timeout_seconds, coverage_enabled, additional_args, output_format
                )
            elif framework == "unittest":
                return self._execute_unittest(
                    test_files, python_executable, working_directory, 
                    timeout_seconds, additional_args
                )
            elif framework == "custom":
                return self._execute_custom_tests(
                    test_files, python_executable, working_directory, 
                    timeout_seconds, additional_args
                )
            else:
                return {
                    "success": False,
                    "message": f"Unsupported test framework: {framework}",
                    "duration": time.time() - start_time
                }

        except Exception as e:
            return {
                "success": False,
                "message": f"Test execution failed: {e}",
                "duration": time.time() - start_time,
                "exception": str(e)
            }

    def _execute_pytest(self, test_files: List[str], python_executable: str, 
                       working_directory: str, timeout_seconds: int, coverage_enabled: bool,
                       additional_args: List[str], output_format: str) -> Dict[str, Any]:
        """Execute tests using pytest."""
        start_time = time.time()
        
        # Build pytest command
        cmd = [python_executable, "-m", "pytest"]
        
        # Add output format
        if output_format == "json":
            cmd.extend(["--json-report", "--json-report-file=/tmp/pytest_report.json"])
        elif output_format == "xml":
            cmd.extend(["--junit-xml=/tmp/pytest_report.xml"])
        
        # Add coverage if enabled
        if coverage_enabled:
            cmd.extend(["--cov=.", "--cov-report=term", "--cov-report=json:/tmp/coverage.json"])
        
        # Add additional arguments
        cmd.extend(additional_args)
        
        # Add test files
        cmd.extend(test_files)
        
        self.report_progress("Executing pytest", f"Command: {' '.join(cmd[-10:])}")
        
        try:
            result = subprocess.run(
                cmd,
                cwd=working_directory,
                capture_output=True,
                text=True,
                timeout=timeout_seconds
            )
            
            duration = time.time() - start_time
            
            # Parse pytest output
            test_results = self._parse_pytest_output(result, output_format)
            
            return {
                "success": result.returncode == 0,
                "command": cmd,
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "duration": duration,
                "executed_files": test_files,
                "parsed_results": test_results
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "message": f"Test execution timed out after {timeout_seconds} seconds",
                "duration": time.time() - start_time,
                "executed_files": test_files
            }

    def _execute_unittest(self, test_files: List[str], python_executable: str, 
                         working_directory: str, timeout_seconds: int, 
                         additional_args: List[str]) -> Dict[str, Any]:
        """Execute tests using unittest."""
        start_time = time.time()
        
        # Build unittest command
        cmd = [python_executable, "-m", "unittest"] + additional_args
        
        # Add test modules (convert file paths to module names)
        test_modules = []
        for test_file in test_files:
            rel_path = os.path.relpath(test_file, working_directory)
            if rel_path.endswith('.py'):
                module = rel_path[:-3].replace(os.sep, '.')
                test_modules.append(module)
        
        cmd.extend(test_modules)
        
        self.report_progress("Executing unittest", f"Command: {' '.join(cmd[-5:])}")
        
        try:
            result = subprocess.run(
                cmd,
                cwd=working_directory,
                capture_output=True,
                text=True,
                timeout=timeout_seconds
            )
            
            duration = time.time() - start_time
            
            # Parse unittest output
            test_results = self._parse_unittest_output(result)
            
            return {
                "success": result.returncode == 0,
                "command": cmd,
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "duration": duration,
                "executed_files": test_files,
                "parsed_results": test_results
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "message": f"Test execution timed out after {timeout_seconds} seconds",
                "duration": time.time() - start_time,
                "executed_files": test_files
            }

    def _execute_custom_tests(self, test_files: List[str], python_executable: str, 
                             working_directory: str, timeout_seconds: int, 
                             additional_args: List[str]) -> Dict[str, Any]:
        """Execute custom test scripts."""
        start_time = time.time()
        results = []
        
        for test_file in test_files:
            cmd = [python_executable, test_file] + additional_args
            
            try:
                result = subprocess.run(
                    cmd,
                    cwd=working_directory,
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds
                )
                
                results.append({
                    "file": test_file,
                    "success": result.returncode == 0,
                    "return_code": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr
                })
                
            except subprocess.TimeoutExpired:
                results.append({
                    "file": test_file,
                    "success": False,
                    "error": "timeout",
                    "message": f"Test timed out after {timeout_seconds} seconds"
                })
        
        overall_success = all(r["success"] for r in results)
        
        return {
            "success": overall_success,
            "duration": time.time() - start_time,
            "executed_files": test_files,
            "file_results": results
        }

    def _parse_pytest_output(self, result: subprocess.CompletedProcess, output_format: str) -> Dict[str, Any]:
        """Parse pytest output to extract test results."""
        # Simple parsing - in production this would be more sophisticated
        stdout = result.stdout
        
        if "failed" in stdout.lower():
            return {"status": "failed", "details": stdout}
        elif "passed" in stdout.lower():
            return {"status": "passed", "details": stdout}
        else:
            return {"status": "unknown", "details": stdout}

    def _parse_unittest_output(self, result: subprocess.CompletedProcess) -> Dict[str, Any]:
        """Parse unittest output to extract test results."""
        stdout = result.stdout + result.stderr
        
        if "FAILED" in stdout:
            return {"status": "failed", "details": stdout}
        elif "OK" in stdout:
            return {"status": "passed", "details": stdout}
        else:
            return {"status": "unknown", "details": stdout}

    def _analyze_test_results(self, execution_result: Dict[str, Any], framework: str) -> Dict[str, Any]:
        """Analyze test execution results."""
        if not execution_result.get("success", False):
            return {
                "success": False,
                "total_tests": 0,
                "passed_tests": 0,
                "failed_tests": 0,
                "summary": "Test execution failed",
                "details": execution_result.get("message", "Unknown error")
            }

        # Simple analysis - in production this would be more detailed
        parsed = execution_result.get("parsed_results", {})
        status = parsed.get("status", "unknown")
        
        return {
            "success": status == "passed",
            "framework": framework,
            "status": status,
            "execution_time": execution_result.get("duration", 0),
            "summary": f"Tests {status}",
            "details": parsed.get("details", "")
        }

    def _create_summary_message(self, final_result: Dict[str, Any], analysis_result: Dict[str, Any]) -> str:
        """Create a summary message for the test execution."""
        if final_result["success"]:
            return f"Test execution successful: {analysis_result.get('summary', 'Tests passed')}"
        else:
            return f"Test execution failed: {analysis_result.get('summary', 'Tests failed')}"

    # Legacy execute method for backward compatibility
    def execute(self, goal: str, context: GlobalContext, current_task: TaskNode) -> AgentResponse:
        """Legacy execute method for backward compatibility."""
        inputs = {
            "test_target": context.workspace_path,
            "working_directory": str(context.workspace_path)
        }
        
        result = self.execute_v2(goal, inputs, context)
        
        return AgentResponse(
            success=result.success,
            message=result.message,
            artifacts_generated=[]
        )