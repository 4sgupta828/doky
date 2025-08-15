# agents/test_runner.py
import json
import logging
import subprocess
import sys
from pathlib import Path

# Foundational dependencies
from .base import BaseAgent
from core.context import GlobalContext
from core.models import AgentResponse, TaskNode

# Get a logger instance for this module
logger = logging.getLogger(__name__)

class TestRunnerAgent(BaseAgent):
    """
    Specialized in executing test suites and reporting the results. This agent
    can work independently by discovering and running tests in the workspace,
    or it can interpret existing pytest JSON reports if available.
    """

    def __init__(self):
        super().__init__(
            name="TestRunnerAgent",
            description="Executes test suites using pytest and reports structured results."
        )

    def execute(self, goal: str, context: GlobalContext, current_task: TaskNode) -> AgentResponse:
        logger.info(f"TestRunnerAgent executing with goal: '{goal}'")
        
        # Check if we have an existing test report artifact first
        test_report_key = current_task.input_artifact_keys[0] if current_task.input_artifact_keys else "pytest_output.json"
        test_results_str = context.get_artifact(test_report_key)
        
        if test_results_str:
            # We have an existing report, interpret it
            return self._interpret_test_report(test_results_str, context, current_task)
        else:
            # No existing report, run tests ourselves
            return self._run_tests_independently(context, current_task)

    def _interpret_test_report(self, test_results_str: str, context: GlobalContext, current_task: TaskNode) -> AgentResponse:
        """Interpret an existing pytest JSON report."""
        try:
            # Pytest can output a detailed JSON report. We parse it here.
            results = json.loads(test_results_str)
            summary = results.get("summary", {})
            total_tests = summary.get("total", 0)
            failed_tests = summary.get("failed", 0)

            if failed_tests > 0:
                msg = f"Test suite failed. {failed_tests} out of {total_tests} tests failed."
                logger.error(msg)
                # Save the detailed report for the DebuggingAgent to analyze
                context.add_artifact("failed_test_report.json", results, current_task.task_id)
                return AgentResponse(success=False, message=msg, artifacts_generated=["failed_test_report.json"])
            else:
                msg = f"Test suite passed. All {total_tests} tests were successful."
                logger.info(msg)
                return AgentResponse(success=True, message=msg)

        except (json.JSONDecodeError, KeyError) as e:
            msg = f"Failed to parse pytest JSON report. The test command may have failed or produced malformed output. Error: {e}. Report content: {test_results_str[:500]}"
            logger.error(msg)
            return AgentResponse(success=False, message=msg)

    def _run_tests_independently(self, context: GlobalContext, current_task: TaskNode) -> AgentResponse:
        """Run tests independently by discovering test files in the workspace."""
        workspace_path = Path(context.workspace_path)
        
        # Ensure test environment is set up
        env_setup_result = self._setup_test_environment(workspace_path)
        if not env_setup_result["success"]:
            return AgentResponse(success=False, message=env_setup_result["message"])
        
        # Look for tests in the workspace
        test_files = self._discover_test_files(workspace_path)
        
        if not test_files:
            msg = "No test files found in the workspace."
            logger.warning(msg)
            return AgentResponse(success=True, message=msg)
        
        logger.info(f"Found {len(test_files)} test files to run")
        
        # Try pytest first if available
        results = self._run_with_pytest(workspace_path, test_files)
        if results is None:
            # Fallback to running individual test files
            results = self._run_individual_tests(test_files)
        
        # Create summary report with actual test counts
        total_test_functions = sum(result.get('test_count', 1) for result in results)
        total_passed = sum(result.get('passed_count', 1 if result['success'] else 0) for result in results)
        total_failed = sum(result.get('failed_count', 0 if result['success'] else 1) for result in results)
        failed_files = sum(1 for result in results if not result['success'])
        
        summary = {
            "total_test_files": len(test_files),
            "total_test_functions": total_test_functions,
            "passed": total_passed,
            "failed": total_failed,
            "failed_files": failed_files,
            "details": results
        }
        
        # Save the report
        report_json = json.dumps(summary, indent=2)
        context.add_artifact("test_execution_report.json", report_json, current_task.task_id)
        
        if total_failed > 0 or failed_files > 0:
            msg = f"Test suite failed. {total_failed} out of {total_test_functions} tests failed across {failed_files} files."
            logger.error(msg)
            return AgentResponse(success=False, message=msg, artifacts_generated=["test_execution_report.json"])
        else:
            msg = f"Test suite passed. All {total_test_functions} tests passed across {len(test_files)} files."
            logger.info(msg)
            return AgentResponse(success=True, message=msg, artifacts_generated=["test_execution_report.json"])
    
    def _discover_test_files(self, workspace_path: Path) -> list:
        """Discover test files in the workspace."""
        test_files = []
        
        # Look for test files in standard locations
        for pattern in ["test_*.py", "*_test.py"]:
            # Look in tests/ directory
            tests_dir = workspace_path / "tests"
            if tests_dir.exists():
                test_files.extend(tests_dir.glob(pattern))
            
            # Look in root directory
            test_files.extend(workspace_path.glob(pattern))
            
            # Look recursively for test files
            test_files.extend(workspace_path.rglob(pattern))
        
        # Remove duplicates and filter out __pycache__ directories
        unique_files = []
        seen = set()
        for file_path in test_files:
            if "__pycache__" not in str(file_path) and file_path not in seen:
                unique_files.append(file_path)
                seen.add(file_path)
        
        return unique_files
    
    def _run_with_pytest(self, workspace_path: Path, test_files: list) -> list:
        """Try to run tests using pytest with JSON output."""
        try:
            # Use the virtual environment python if available
            python_exe = getattr(self, '_venv_python', sys.executable)
            
            # Check if pytest is available
            result = subprocess.run([python_exe, "-m", "pytest", "--version"], 
                                  capture_output=True, text=True, cwd=workspace_path)
            if result.returncode != 0:
                logger.info("pytest not available, falling back to individual test execution")
                return None
            
            # Run pytest with JSON report
            cmd = [python_exe, "-m", "pytest", "--tb=short", "-v"] + [str(f) for f in test_files]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=workspace_path)
            
            # Parse pytest output to create our own summary
            results = []
            output_text = result.stdout + result.stderr
            
            # Parse overall results first
            import re
            test_count = 0
            passed_count = 0
            failed_count = 0
            
            # Look for test collection info like "collected 14 items"
            collected_match = re.search(r'collected (\d+) items?', output_text)
            if collected_match:
                test_count = int(collected_match.group(1))
            
            # Look for final result line like "14 passed in 0.02s"
            passed_match = re.search(r'(\d+) passed', output_text)
            if passed_match:
                passed_count = int(passed_match.group(1))
            
            failed_match = re.search(r'(\d+) failed', output_text)
            if failed_match:
                failed_count = int(failed_match.group(1))
            
            # When running multiple files together, we get aggregate results
            # For simplicity, we'll create one entry per file and distribute the results
            file_success = result.returncode == 0
            
            if len(test_files) == 1:
                # Single file case - straightforward
                results.append({
                    'file': str(test_files[0]),
                    'success': file_success,
                    'output': output_text,
                    'test_count': test_count,
                    'passed_count': passed_count,
                    'failed_count': failed_count
                })
            else:
                # Multiple files - distribute counts evenly (approximation)
                for test_file in test_files:
                    results.append({
                        'file': str(test_file),
                        'success': file_success,
                        'output': output_text,
                        'test_count': test_count // len(test_files),
                        'passed_count': passed_count // len(test_files),
                        'failed_count': failed_count // len(test_files)
                    })
            
            return results
            
        except Exception as e:
            logger.info(f"Failed to run pytest: {e}, falling back to individual test execution")
            return None
    
    def _run_individual_tests(self, test_files: list) -> list:
        """Run test files individually using pytest."""
        results = []
        
        # Use the virtual environment python if available
        python_exe = getattr(self, '_venv_python', sys.executable)
        
        for test_file in test_files:
            try:
                logger.info(f"Running test file: {test_file}")
                
                # Try to run with pytest first
                result = subprocess.run(
                    [python_exe, "-m", "pytest", str(test_file), "-v", "--tb=short"],
                    capture_output=True,
                    text=True,
                    timeout=60,  # 60 second timeout per test file
                    cwd=test_file.parent.parent  # Set working directory to project root
                )
                
                # If pytest fails, try running as standalone script (fallback)
                if result.returncode != 0 and "No module named pytest" in result.stderr:
                    logger.info(f"Pytest not available for {test_file.name}, trying standalone execution")
                    result = subprocess.run(
                        [python_exe, str(test_file)],
                        capture_output=True,
                        text=True,
                        timeout=60,
                        cwd=test_file.parent.parent  # Set working directory to project root
                    )
                
                success = result.returncode == 0
                output = result.stdout + result.stderr  # Capture both stdout and stderr
                
                # Parse pytest output to get individual test counts
                test_count = 1  # Default to 1 if parsing fails
                passed_count = 0
                failed_count = 0
                
                # Always try to parse output for pytest patterns
                output_text = output
                import re
                
                # Look for test collection info like "collected 14 items"
                collected_match = re.search(r'collected (\d+) items?', output_text)
                if collected_match:
                    test_count = int(collected_match.group(1))
                
                # Look for final result line like "14 passed in 0.02s"
                passed_match = re.search(r'(\d+) passed', output_text)
                if passed_match:
                    passed_count = int(passed_match.group(1))
                
                failed_match = re.search(r'(\d+) failed', output_text)
                if failed_match:
                    failed_count = int(failed_match.group(1))
                
                # If we successfully parsed pytest output, use those counts
                if passed_count > 0 or failed_count > 0:
                    if success and passed_count > 0:
                        # Pytest succeeded
                        pass
                    elif not success and failed_count > 0:
                        # Pytest failed
                        pass
                else:
                    # Fallback: use success/failure of the file itself
                    if success:
                        passed_count = 1
                        failed_count = 0
                    else:
                        passed_count = 0
                        failed_count = 1
                
                results.append({
                    'file': str(test_file),
                    'success': success,
                    'output': output,
                    'return_code': result.returncode,
                    'test_count': test_count,
                    'passed_count': passed_count,
                    'failed_count': failed_count
                })
                
                if success:
                    logger.info(f"✅ {test_file.name} PASSED ({passed_count} tests)")
                else:
                    logger.error(f"❌ {test_file.name} FAILED (exit code {result.returncode}, {failed_count} failed tests)")
                    
            except subprocess.TimeoutExpired:
                logger.error(f"⏰ {test_file.name} TIMEOUT")
                results.append({
                    'file': str(test_file),
                    'success': False,
                    'output': "Test timed out after 60 seconds",
                    'return_code': -1
                })
            except Exception as e:
                logger.error(f"💥 {test_file.name} ERROR: {e}")
                results.append({
                    'file': str(test_file),
                    'success': False,
                    'output': str(e),
                    'return_code': -2
                })
        
        return results
    
    def _setup_test_environment(self, workspace_path: Path) -> dict:
        """Set up the test environment by ensuring required dependencies are installed."""
        try:
            # Check if we're in a virtual environment
            venv_path = workspace_path / "test_env"
            requirements_path = workspace_path / "requirements.txt"
            
            # If no virtual environment exists, create one
            if not venv_path.exists():
                logger.info("Creating virtual environment for testing...")
                result = subprocess.run([sys.executable, "-m", "venv", str(venv_path)], 
                                      capture_output=True, text=True, cwd=workspace_path)
                if result.returncode != 0:
                    return {"success": False, "message": f"Failed to create virtual environment: {result.stderr}"}
            
            # Determine the python executable in the venv
            if sys.platform == "win32":
                venv_python = venv_path / "Scripts" / "python.exe"
                pip_exe = venv_path / "Scripts" / "pip.exe"
            else:
                venv_python = venv_path / "bin" / "python"
                pip_exe = venv_path / "bin" / "pip"
            
            # Install pytest if not available
            result = subprocess.run([str(venv_python), "-c", "import pytest"], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                logger.info("Installing pytest in virtual environment...")
                result = subprocess.run([str(pip_exe), "install", "pytest"], 
                                      capture_output=True, text=True)
                if result.returncode != 0:
                    return {"success": False, "message": f"Failed to install pytest: {result.stderr}"}
            
            # Install requirements if available
            if requirements_path.exists():
                logger.info("Installing requirements in virtual environment...")
                result = subprocess.run([str(pip_exe), "install", "-r", str(requirements_path)], 
                                      capture_output=True, text=True)
                if result.returncode != 0:
                    logger.warning(f"Failed to install some requirements: {result.stderr}")
                    # Don't fail completely - some requirements might not be test-related
            
            # Store the python executable for later use
            self._venv_python = str(venv_python)
            return {"success": True, "message": "Test environment ready"}
            
        except Exception as e:
            return {"success": False, "message": f"Error setting up test environment: {e}"}

# --- Self-Testing Block ---
if __name__ == "__main__":
    import unittest
    import shutil
    from pathlib import Path
    from utils.logger import setup_logger

    setup_logger(default_level=logging.INFO)
    
    class TestTestRunnerAgent(unittest.TestCase):
        def setUp(self):
            self.test_workspace_path = Path("./temp_test_runner_ws")
            if self.test_workspace_path.exists():
                shutil.rmtree(self.test_workspace_path)
            self.context = GlobalContext(workspace_path=str(self.test_workspace_path))
            self.agent = TestRunnerAgent()
            self.task = TaskNode(goal="Run tests", assigned_agent="TestRunnerAgent", input_artifact_keys=["pytest_output.json"])

        def tearDown(self):
            shutil.rmtree(self.test_workspace_path)

        def test_test_runner_passed(self):
            print("\n--- [Test Case 1: TestRunnerAgent Success] ---")
            pytest_report = json.dumps({"summary": {"total": 5, "passed": 5, "failed": 0}})
            self.context.add_artifact("pytest_output.json", pytest_report, "task_tooling")
            
            response = self.agent.execute(self.task.goal, self.context, self.task)
            
            self.assertTrue(response.success)
            self.assertIn("Test suite passed", response.message)
            logger.info("✅ test_test_runner_passed: PASSED")

        def test_test_runner_failed(self):
            print("\n--- [Test Case 2: TestRunnerAgent Failure] ---")
            pytest_report = json.dumps({"summary": {"total": 5, "passed": 3, "failed": 2}})
            self.context.add_artifact("pytest_output.json", pytest_report, "task_tooling")
            
            response = self.agent.execute(self.task.goal, self.context, self.task)

            self.assertFalse(response.success)
            self.assertIn("Test suite failed", response.message)
            self.assertIsNotNone(self.context.get_artifact("failed_test_report.json"))
            logger.info("✅ test_test_runner_failed: PASSED")

        def test_test_runner_independent_execution(self):
            print("\n--- [Test Case 3: TestRunnerAgent Independent Execution] ---")
            # Create a simple test file in the workspace
            test_file = self.test_workspace_path / "test_example.py" 
            test_file.write_text('''
import unittest

class TestExample(unittest.TestCase):
    def test_simple(self):
        self.assertEqual(1 + 1, 2)

if __name__ == "__main__":
    unittest.main()
''')
            
            response = self.agent.execute(self.task.goal, self.context, self.task)
            
            self.assertTrue(response.success)
            self.assertIn("Test suite passed", response.message)
            # Should have generated a test execution report
            self.assertIn("test_execution_report.json", response.artifacts_generated or [])
            logger.info("✅ test_test_runner_independent_execution: PASSED")

        def test_test_discovery(self):
            print("\n--- [Test Case 4: Test File Discovery] ---")
            # Create multiple test files in different locations
            tests_dir = self.test_workspace_path / "tests"
            tests_dir.mkdir()
            
            (tests_dir / "test_unit.py").write_text("# Unit test file")
            (self.test_workspace_path / "test_integration.py").write_text("# Integration test file")
            (self.test_workspace_path / "feature_test.py").write_text("# Feature test file")
            
            # Test discovery
            discovered_files = self.agent._discover_test_files(self.test_workspace_path)
            
            self.assertGreaterEqual(len(discovered_files), 2)  # Should find test_unit.py and test_integration.py
            discovered_names = [f.name for f in discovered_files]
            self.assertIn("test_unit.py", discovered_names)
            self.assertIn("test_integration.py", discovered_names)
            logger.info("✅ test_test_discovery: PASSED")
            
    unittest.main(argv=['first-arg-is-ignored'], exit=False)