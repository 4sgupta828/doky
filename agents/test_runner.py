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
        
        # Create summary report
        total_tests = len(test_files)
        failed_tests = sum(1 for result in results if not result['success'])
        
        summary = {
            "total": total_tests,
            "passed": total_tests - failed_tests, 
            "failed": failed_tests,
            "details": results
        }
        
        # Save the report
        report_json = json.dumps(summary, indent=2)
        context.add_artifact("test_execution_report.json", report_json, current_task.task_id)
        
        if failed_tests > 0:
            msg = f"Test suite failed. {failed_tests} out of {total_tests} test files failed."
            logger.error(msg)
            return AgentResponse(success=False, message=msg, artifacts_generated=["test_execution_report.json"])
        else:
            msg = f"Test suite passed. All {total_tests} test files were successful."
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
            # Check if pytest is available
            result = subprocess.run([sys.executable, "-m", "pytest", "--version"], 
                                  capture_output=True, text=True, cwd=workspace_path)
            if result.returncode != 0:
                logger.info("pytest not available, falling back to individual test execution")
                return None
            
            # Run pytest with JSON report
            cmd = [sys.executable, "-m", "pytest", "--tb=short", "-v"] + [str(f) for f in test_files]
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=workspace_path)
            
            # Parse pytest output to create our own summary
            results = []
            for test_file in test_files:
                # Simple heuristic: if pytest succeeded overall and no failures mentioned this file
                file_success = result.returncode == 0 and str(test_file.name) not in result.stdout
                if not file_success and result.returncode != 0:
                    # Check if this specific file had issues
                    file_success = str(test_file.name) not in result.stderr
                
                results.append({
                    'file': str(test_file),
                    'success': file_success,
                    'output': result.stdout if file_success else result.stderr
                })
            
            return results
            
        except Exception as e:
            logger.info(f"Failed to run pytest: {e}, falling back to individual test execution")
            return None
    
    def _run_individual_tests(self, test_files: list) -> list:
        """Run test files individually."""
        results = []
        
        for test_file in test_files:
            try:
                logger.info(f"Running test file: {test_file}")
                result = subprocess.run(
                    [sys.executable, str(test_file)],
                    capture_output=True,
                    text=True,
                    timeout=60  # 60 second timeout per test file
                )
                
                success = result.returncode == 0
                output = result.stdout if success else result.stderr
                
                results.append({
                    'file': str(test_file),
                    'success': success,
                    'output': output,
                    'return_code': result.returncode
                })
                
                if success:
                    logger.info(f"✅ {test_file.name} PASSED")
                else:
                    logger.error(f"❌ {test_file.name} FAILED (exit code {result.returncode})")
                    
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