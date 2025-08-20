# tools/test_tools.py
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional

from core.context import GlobalContext

# Get a logger instance for this module
logger = logging.getLogger(__name__)


class TestTools:
    """
    Atomic test operations following the principle of structured tools.
    
    This module provides low-level test operations that can be used
    by agents or other components for test discovery, execution, and analysis.
    """

    @staticmethod
    def detect_test_framework(requested_framework: str = "auto", test_target: str = None, 
                             working_directory: str = None) -> Dict[str, Any]:
        """Detect or validate the test framework to use."""
        working_directory = working_directory or os.getcwd()
        
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
                except Exception:
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
            except Exception:
                pass

        # Check for unittest indicators
        if test_target and "test_" in str(test_target):
            return {"success": True, "framework": "unittest"}

        # Default to pytest if available, otherwise unittest
        try:
            result = subprocess.run(
                [sys.executable, "-c", "import pytest"], 
                capture_output=True, text=True
            )
            if result.returncode == 0:
                return {"success": True, "framework": "pytest"}
        except Exception:
            pass

        return {"success": True, "framework": "unittest"}

    @staticmethod
    def discover_test_files(test_target: str, framework: str = "auto", working_directory: str = None,
                           test_patterns: List[str] = None, exclude_patterns: List[str] = None) -> Dict[str, Any]:
        """Discover test files based on target and patterns."""
        working_directory = working_directory or os.getcwd()
        working_path = Path(working_directory)
        test_files = []

        try:
            if test_target:
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
                    if framework == "pytest" or framework == "auto":
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
                    patterns = [test_target] if framework == "pytest" else (test_patterns or ["test*.py"])
                    
                    for pattern in patterns:
                        discovered = list(working_path.rglob(pattern))
                        test_files.extend([str(f) for f in discovered])
            else:
                # Default patterns for workspace discovery
                if framework == "pytest" or framework == "auto":
                    patterns = test_patterns or ["test_*.py", "*_test.py"]
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

    @staticmethod
    def execute_tests(test_files: List[str], framework: str = "pytest", python_executable: str = None,
                     working_directory: str = None, timeout_seconds: int = 300, 
                     coverage_enabled: bool = False, additional_args: List[str] = None, 
                     output_format: str = "json") -> Dict[str, Any]:
        """Execute tests using the specified framework."""
        python_executable = python_executable or sys.executable
        working_directory = working_directory or os.getcwd()
        additional_args = additional_args or []
        start_time = time.time()
        
        try:
            if framework == "pytest":
                return TestTools._execute_pytest(
                    test_files, python_executable, working_directory, 
                    timeout_seconds, coverage_enabled, additional_args, output_format
                )
            elif framework == "unittest":
                return TestTools._execute_unittest(
                    test_files, python_executable, working_directory, 
                    timeout_seconds, additional_args
                )
            elif framework == "custom":
                return TestTools._execute_custom_tests(
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

    @staticmethod
    def parse_test_results(execution_result: Dict[str, Any], framework: str) -> Dict[str, Any]:
        """Parse and analyze test execution results."""
        if not execution_result.get("success", False):
            return {
                "success": False,
                "total_tests": 0,
                "passed_tests": 0,
                "failed_tests": 0,
                "summary": "Test execution failed",
                "details": execution_result.get("message", "Unknown error")
            }

        # Parse results based on framework
        if framework == "pytest":
            return TestTools._parse_pytest_results(execution_result)
        elif framework == "unittest":
            return TestTools._parse_unittest_results(execution_result)
        elif framework == "custom":
            return TestTools._parse_custom_results(execution_result)
        else:
            return {
                "success": False,
                "summary": f"Unknown framework: {framework}",
                "details": execution_result
            }

    @staticmethod
    def _execute_pytest(test_files: List[str], python_executable: str, 
                       working_directory: str, timeout_seconds: int, coverage_enabled: bool,
                       additional_args: List[str], output_format: str) -> Dict[str, Any]:
        """Execute tests using pytest."""
        start_time = time.time()
        
        # Build pytest command
        cmd = [python_executable, "-m", "pytest"]
        
        # Add output format
        if output_format == "json":
            cmd.extend(["--json-report", "--json-report-file=test_report.json"])
        elif output_format == "xml":
            cmd.extend(["--junit-xml=test_report.xml"])
        
        # Add coverage if enabled
        if coverage_enabled:
            cmd.extend(["--cov=.", "--cov-report=term", "--cov-report=json:coverage.json"])
        
        # Add additional arguments
        cmd.extend(additional_args)
        
        # Add test files
        cmd.extend(test_files)
        
        try:
            result = subprocess.run(
                cmd,
                cwd=working_directory,
                capture_output=True,
                text=True,
                timeout=timeout_seconds
            )
            
            duration = time.time() - start_time
            
            return {
                "success": result.returncode in [0, 1],  # 0=pass, 1=fail but valid
                "command": cmd,
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "duration": duration,
                "executed_files": test_files,
                "framework": "pytest"
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "message": f"Test execution timed out after {timeout_seconds} seconds",
                "duration": time.time() - start_time,
                "executed_files": test_files,
                "framework": "pytest"
            }

    @staticmethod
    def _execute_unittest(test_files: List[str], python_executable: str, 
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
        
        try:
            result = subprocess.run(
                cmd,
                cwd=working_directory,
                capture_output=True,
                text=True,
                timeout=timeout_seconds
            )
            
            duration = time.time() - start_time
            
            return {
                "success": result.returncode == 0,
                "command": cmd,
                "return_code": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "duration": duration,
                "executed_files": test_files,
                "framework": "unittest"
            }
            
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "message": f"Test execution timed out after {timeout_seconds} seconds",
                "duration": time.time() - start_time,
                "executed_files": test_files,
                "framework": "unittest"
            }

    @staticmethod
    def _execute_custom_tests(test_files: List[str], python_executable: str, 
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
            "file_results": results,
            "framework": "custom"
        }

    @staticmethod
    def _parse_pytest_results(execution_result: Dict[str, Any]) -> Dict[str, Any]:
        """Parse pytest execution results."""
        stdout = execution_result.get("stdout", "")
        return_code = execution_result.get("return_code", 1)
        
        # Try to read JSON report if it exists
        try:
            if Path("test_report.json").exists():
                with open("test_report.json", "r") as f:
                    report_data = json.load(f)
                summary = report_data.get("summary", {})
                return {
                    "success": return_code == 0,
                    "framework": "pytest",
                    "total_tests": summary.get("total", 0),
                    "passed_tests": summary.get("passed", 0),
                    "failed_tests": summary.get("failed", 0),
                    "execution_time": execution_result.get("duration", 0),
                    "summary": f"Tests {'passed' if return_code == 0 else 'failed'}",
                    "detailed_report": report_data
                }
        except Exception:
            pass
        
        # Fallback to stdout parsing
        if "failed" in stdout.lower():
            return {"success": False, "status": "failed", "details": stdout, "framework": "pytest"}
        elif "passed" in stdout.lower():
            return {"success": True, "status": "passed", "details": stdout, "framework": "pytest"}
        else:
            return {"success": return_code == 0, "status": "unknown", "details": stdout, "framework": "pytest"}

    @staticmethod
    def _parse_unittest_results(execution_result: Dict[str, Any]) -> Dict[str, Any]:
        """Parse unittest execution results."""
        stdout = execution_result.get("stdout", "") + execution_result.get("stderr", "")
        return_code = execution_result.get("return_code", 1)
        
        if "FAILED" in stdout:
            return {"success": False, "status": "failed", "details": stdout, "framework": "unittest"}
        elif "OK" in stdout:
            return {"success": True, "status": "passed", "details": stdout, "framework": "unittest"}
        else:
            return {"success": return_code == 0, "status": "unknown", "details": stdout, "framework": "unittest"}

    @staticmethod
    def _parse_custom_results(execution_result: Dict[str, Any]) -> Dict[str, Any]:
        """Parse custom test execution results."""
        file_results = execution_result.get("file_results", [])
        total_files = len(file_results)
        successful_files = sum(1 for r in file_results if r.get("success", False))
        
        return {
            "success": execution_result.get("success", False),
            "framework": "custom",
            "total_files": total_files,
            "successful_files": successful_files,
            "failed_files": total_files - successful_files,
            "execution_time": execution_result.get("duration", 0),
            "file_details": file_results
        }