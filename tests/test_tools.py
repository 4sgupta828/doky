# tools/execution/test_execution_tools.py
"""
Test execution tools for running tests, validating results, and managing test environments.
Extracted from TestRunnerAgent to provide atomic, reusable test execution capabilities.
"""

import logging
import subprocess
import sys
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

class TestFramework(Enum):
    """Supported test frameworks."""
    PYTEST = "pytest"
    UNITTEST = "unittest"
    NOSE = "nose"
    CUSTOM = "custom"
    AUTO = "auto"

class TestResult(Enum):
    """Test execution results."""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"

@dataclass
class TestExecutionContext:
    """Context for test execution."""
    test_target: Optional[str] = None
    test_framework: TestFramework = TestFramework.AUTO
    python_executable: Optional[str] = None
    working_directory: str = "."
    timeout_seconds: int = 300
    test_patterns: List[str] = None
    exclude_patterns: List[str] = None
    output_format: str = "json"
    coverage_enabled: bool = False
    additional_args: List[str] = None
    
    def __post_init__(self):
        if self.test_patterns is None:
            self.test_patterns = []
        if self.exclude_patterns is None:
            self.exclude_patterns = []
        if self.additional_args is None:
            self.additional_args = []

@dataclass
class TestExecutionResult:
    """Result of test execution."""
    success: bool
    framework: TestFramework
    test_files_discovered: int
    test_files_executed: int
    tests_run: int = 0
    tests_passed: int = 0
    tests_failed: int = 0
    tests_skipped: int = 0
    execution_time: float = 0.0
    coverage_percentage: Optional[float] = None
    output: str = ""
    error_output: str = ""
    detailed_results: List[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.detailed_results is None:
            self.detailed_results = []

def detect_test_framework(
    requested_framework: str = "auto",
    test_target: Optional[str] = None,
    working_directory: str = "."
) -> Dict[str, Any]:
    """Detect the appropriate test framework to use."""
    working_path = Path(working_directory)
    
    # If explicitly requested, validate and return
    if requested_framework != "auto":
        try:
            framework = TestFramework(requested_framework)
            return {
                "success": True,
                "framework": framework.value,
                "message": f"Using requested framework: {framework.value}"
            }
        except ValueError:
            return {
                "success": False,
                "message": f"Unknown test framework: {requested_framework}"
            }
    
    # Auto-detect based on project files and structure
    detection_results = []
    
    # Check for pytest
    if (working_path / "pytest.ini").exists() or (working_path / "pyproject.toml").exists():
        detection_results.append(("pytest", "pytest.ini or pyproject.toml found"))
    
    # Check for setup.cfg with pytest configuration
    setup_cfg = working_path / "setup.cfg"
    if setup_cfg.exists():
        try:
            content = setup_cfg.read_text()
            if "[tool:pytest]" in content or "[pytest]" in content:
                detection_results.append(("pytest", "pytest configuration in setup.cfg"))
        except Exception:
            pass
    
    # Check for nose configuration
    if (working_path / "nose.cfg").exists() or (working_path / ".noserc").exists():
        detection_results.append(("nose", "nose configuration file found"))
    
    # Check test file patterns to infer framework
    test_files = list(working_path.glob("**/test_*.py")) + list(working_path.glob("**/*_test.py"))
    if test_files:
        # Sample a few test files to check for framework-specific patterns
        pytest_indicators = 0
        unittest_indicators = 0
        
        for test_file in test_files[:5]:  # Check first 5 files
            try:
                content = test_file.read_text()
                if "import pytest" in content or "@pytest" in content or "def test_" in content:
                    pytest_indicators += 1
                if "import unittest" in content or "class.*TestCase" in content:
                    unittest_indicators += 1
            except Exception:
                continue
        
        if pytest_indicators > unittest_indicators:
            detection_results.append(("pytest", f"pytest patterns found in {pytest_indicators} test files"))
        elif unittest_indicators > 0:
            detection_results.append(("unittest", f"unittest patterns found in {unittest_indicators} test files"))
    
    # Default to pytest if available, otherwise unittest
    if detection_results:
        framework = detection_results[0][0]
        reasons = [result[1] for result in detection_results]
        return {
            "success": True,
            "framework": framework,
            "message": f"Auto-detected {framework}: {'; '.join(reasons)}"
        }
    
    # Fallback to unittest (built-in)
    return {
        "success": True,
        "framework": "unittest",
        "message": "No specific test framework detected, defaulting to unittest"
    }

def discover_test_files(
    test_target: str = ".",
    framework: str = "auto",
    working_directory: str = ".",
    test_patterns: List[str] = None,
    exclude_patterns: List[str] = None
) -> Dict[str, Any]:
    """Discover test files based on framework and patterns."""
    working_path = Path(working_directory)
    
    if test_patterns is None:
        # Default patterns based on framework
        if framework == "pytest":
            test_patterns = ["test_*.py", "*_test.py"]
        elif framework == "unittest":
            test_patterns = ["test*.py"]
        else:
            test_patterns = ["test_*.py", "*_test.py", "test*.py"]
    
    if exclude_patterns is None:
        exclude_patterns = ["__pycache__", "*.pyc", ".git", ".pytest_cache"]
    
    test_files = []
    
    try:
        # Handle specific file target
        if test_target and test_target != ".":
            target_path = working_path / test_target
            if target_path.is_file() and target_path.suffix == ".py":
                test_files = [str(target_path)]
            elif target_path.is_dir():
                # Search in specific directory
                for pattern in test_patterns:
                    test_files.extend([str(p) for p in target_path.rglob(pattern)])
            else:
                # Treat as a pattern
                test_files.extend([str(p) for p in working_path.rglob(test_target)])
        else:
            # Search in working directory
            for pattern in test_patterns:
                test_files.extend([str(p) for p in working_path.rglob(pattern)])
        
        # Apply exclusion patterns
        filtered_files = []
        for test_file in test_files:
            exclude = False
            for exclude_pattern in exclude_patterns:
                if exclude_pattern in test_file:
                    exclude = True
                    break
            if not exclude:
                filtered_files.append(test_file)
        
        return {
            "success": True,
            "test_files": filtered_files,
            "message": f"Discovered {len(filtered_files)} test files"
        }
        
    except Exception as e:
        return {
            "success": False,
            "test_files": [],
            "message": f"Test file discovery failed: {e}"
        }

def execute_tests(
    test_files: List[str],
    framework: str,
    python_executable: Optional[str] = None,
    working_directory: str = ".",
    timeout_seconds: int = 300,
    coverage_enabled: bool = False,
    additional_args: List[str] = None,
    output_format: str = "json"
) -> Dict[str, Any]:
    """Execute test files using the specified framework."""
    if not test_files:
        return {
            "success": True,
            "message": "No test files to execute",
            "executed_files": [],
            "output": "",
            "execution_time": 0.0
        }
    
    if python_executable is None:
        python_executable = sys.executable
    
    if additional_args is None:
        additional_args = []
    
    start_time = time.time()
    
    try:
        # Build command based on framework
        if framework == "pytest":
            cmd = [python_executable, "-m", "pytest"]
            
            # Add output format options
            if output_format == "json":
                cmd.extend(["--tb=short", "-v"])
            elif output_format == "xml":
                cmd.extend(["--junitxml=test-results.xml"])
            
            # Add coverage if enabled
            if coverage_enabled:
                cmd.extend(["--cov=.", "--cov-report=term-missing"])
            
            # Add additional args
            cmd.extend(additional_args)
            
            # Add test files
            cmd.extend(test_files)
            
        elif framework == "unittest":
            cmd = [python_executable, "-m", "unittest"]
            
            # Convert file paths to module names for unittest
            test_modules = []
            for test_file in test_files:
                # Convert path to module name (remove .py and replace / with .)
                module_name = test_file.replace("/", ".").replace("\\", ".").rstrip(".py")
                test_modules.append(module_name)
            
            cmd.extend(test_modules)
            
            if "-v" not in additional_args:
                cmd.append("-v")
            cmd.extend(additional_args)
            
        elif framework == "nose":
            cmd = [python_executable, "-m", "nose"]
            cmd.extend(additional_args)
            cmd.extend(test_files)
            
        else:
            return {
                "success": False,
                "message": f"Unsupported test framework: {framework}",
                "executed_files": []
            }
        
        # Execute the test command
        logger.info(f"Executing test command: {' '.join(cmd)}")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=working_directory,
            timeout=timeout_seconds
        )
        
        execution_time = time.time() - start_time
        
        return {
            "success": result.returncode == 0,
            "returncode": result.returncode,
            "executed_files": test_files,
            "command": " ".join(cmd),
            "output": result.stdout,
            "error_output": result.stderr,
            "execution_time": execution_time
        }
        
    except subprocess.TimeoutExpired:
        execution_time = time.time() - start_time
        return {
            "success": False,
            "message": f"Test execution timed out after {timeout_seconds} seconds",
            "executed_files": test_files,
            "execution_time": execution_time
        }
        
    except Exception as e:
        execution_time = time.time() - start_time
        return {
            "success": False,
            "message": f"Test execution failed: {e}",
            "executed_files": test_files,
            "execution_time": execution_time
        }

def parse_test_results(execution_result: Dict[str, Any], framework: str) -> Dict[str, Any]:
    """Parse test execution results based on framework."""
    if not execution_result.get("success", False):
        return {
            "success": False,
            "message": execution_result.get("message", "Test execution failed"),
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "skipped_tests": 0,
            "summary": "Tests failed to execute"
        }
    
    output = execution_result.get("output", "")
    error_output = execution_result.get("error_output", "")
    
    if framework == "pytest":
        return _parse_pytest_results(output, error_output)
    elif framework == "unittest":
        return _parse_unittest_results(output, error_output)
    elif framework == "nose":
        return _parse_nose_results(output, error_output)
    else:
        return {
            "success": True,
            "message": "Tests executed successfully",
            "summary": "Test execution completed",
            "raw_output": output
        }

def _parse_pytest_results(output: str, error_output: str) -> Dict[str, Any]:
    """Parse pytest output for test results."""
    import re
    
    # Look for the summary line: "= X failed, Y passed in Zs ="
    summary_pattern = r"=+ (?:(\d+) failed,?\s*)?(?:(\d+) passed,?\s*)?(?:(\d+) skipped,?\s*)?(?:(\d+) error,?\s*)?.*in ([\d.]+)s =+"
    match = re.search(summary_pattern, output)
    
    if match:
        failed = int(match.group(1) or 0)
        passed = int(match.group(2) or 0)
        skipped = int(match.group(3) or 0)
        errors = int(match.group(4) or 0)
        duration = float(match.group(5) or 0)
        
        total = failed + passed + skipped + errors
        success = failed == 0 and errors == 0
        
        return {
            "success": success,
            "total_tests": total,
            "passed_tests": passed,
            "failed_tests": failed + errors,
            "skipped_tests": skipped,
            "execution_time": duration,
            "summary": f"{passed}/{total} tests passed" if total > 0 else "No tests run",
            "raw_output": output
        }
    
    # Fallback parsing
    if "FAILED" in output or "ERROR" in output:
        return {
            "success": False,
            "message": "Tests failed (parsing incomplete)",
            "summary": "Some tests failed",
            "raw_output": output
        }
    else:
        return {
            "success": True,
            "message": "Tests passed",
            "summary": "All tests passed",
            "raw_output": output
        }

def _parse_unittest_results(output: str, error_output: str) -> Dict[str, Any]:
    """Parse unittest output for test results."""
    import re
    
    # Look for unittest summary: "Ran X tests in Ys"
    ran_pattern = r"Ran (\d+) tests? in ([\d.]+)s"
    match = re.search(ran_pattern, output)
    
    total_tests = int(match.group(1)) if match else 0
    duration = float(match.group(2)) if match else 0
    
    # Check for failures and errors
    failed = len(re.findall(r"FAIL:", output))
    errors = len(re.findall(r"ERROR:", output))
    skipped = len(re.findall(r"SKIP:", output))
    passed = total_tests - failed - errors - skipped
    
    success = failed == 0 and errors == 0
    
    return {
        "success": success,
        "total_tests": total_tests,
        "passed_tests": passed,
        "failed_tests": failed + errors,
        "skipped_tests": skipped,
        "execution_time": duration,
        "summary": f"{passed}/{total_tests} tests passed" if total_tests > 0 else "No tests run",
        "raw_output": output
    }

def _parse_nose_results(output: str, error_output: str) -> Dict[str, Any]:
    """Parse nose output for test results."""
    import re
    
    # Look for nose summary
    ran_pattern = r"Ran (\d+) tests? in ([\d.]+)s"
    match = re.search(ran_pattern, output)
    
    total_tests = int(match.group(1)) if match else 0
    duration = float(match.group(2)) if match else 0
    
    # Check for OK or failures
    if "OK" in output:
        return {
            "success": True,
            "total_tests": total_tests,
            "passed_tests": total_tests,
            "failed_tests": 0,
            "skipped_tests": 0,
            "execution_time": duration,
            "summary": f"All {total_tests} tests passed",
            "raw_output": output
        }
    else:
        # Parse failure details would require more complex parsing
        return {
            "success": False,
            "total_tests": total_tests,
            "summary": "Some tests failed",
            "raw_output": output
        }

def run_test_suite(context: TestExecutionContext) -> TestExecutionResult:
    """Run a complete test suite with discovery, execution, and analysis."""
    start_time = time.time()
    
    # Step 1: Detect test framework
    framework_result = detect_test_framework(
        context.test_framework.value if isinstance(context.test_framework, TestFramework) else context.test_framework,
        context.test_target,
        context.working_directory
    )
    
    if not framework_result["success"]:
        return TestExecutionResult(
            success=False,
            framework=TestFramework.AUTO,
            test_files_discovered=0,
            test_files_executed=0,
            output=framework_result["message"]
        )
    
    detected_framework = framework_result["framework"]
    
    # Step 2: Discover test files
    discovery_result = discover_test_files(
        context.test_target or ".",
        detected_framework,
        context.working_directory,
        context.test_patterns,
        context.exclude_patterns
    )
    
    if not discovery_result["success"]:
        return TestExecutionResult(
            success=False,
            framework=TestFramework(detected_framework),
            test_files_discovered=0,
            test_files_executed=0,
            output=discovery_result["message"]
        )
    
    test_files = discovery_result["test_files"]
    
    if not test_files:
        return TestExecutionResult(
            success=True,
            framework=TestFramework(detected_framework),
            test_files_discovered=0,
            test_files_executed=0,
            output="No test files found"
        )
    
    # Step 3: Execute tests
    execution_result = execute_tests(
        test_files,
        detected_framework,
        context.python_executable,
        context.working_directory,
        context.timeout_seconds,
        context.coverage_enabled,
        context.additional_args,
        context.output_format
    )
    
    # Step 4: Parse results
    analysis_result = parse_test_results(execution_result, detected_framework)
    
    # Step 5: Combine results
    total_time = time.time() - start_time
    
    return TestExecutionResult(
        success=execution_result.get("success", False) and analysis_result.get("success", True),
        framework=TestFramework(detected_framework),
        test_files_discovered=len(test_files),
        test_files_executed=len(execution_result.get("executed_files", [])),
        tests_run=analysis_result.get("total_tests", 0),
        tests_passed=analysis_result.get("passed_tests", 0),
        tests_failed=analysis_result.get("failed_tests", 0),
        tests_skipped=analysis_result.get("skipped_tests", 0),
        execution_time=total_time,
        coverage_percentage=None,  # Would need to parse coverage output
        output=execution_result.get("output", ""),
        error_output=execution_result.get("error_output", ""),
        detailed_results=[{
            "framework_detection": framework_result,
            "discovery": discovery_result,
            "execution": execution_result,
            "analysis": analysis_result
        }]
    )