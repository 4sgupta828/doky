# agents/test_runner.py
import json
import logging

# Foundational dependencies
from .base import BaseAgent
from core.context import GlobalContext
from core.models import AgentResponse, TaskNode

# Get a logger instance for this module
logger = logging.getLogger(__name__)

class TestRunnerAgent(BaseAgent):
    """
    Specialized in executing test suites and reporting the results. It does not
    run the tests itself but interprets the structured output (JSON report) from
    a ToolingAgent task that ran the test command (e.g., pytest).
    """

    def __init__(self):
        super().__init__(
            name="TestRunnerAgent",
            description="Executes test suites using pytest and reports structured results."
        )

    def execute(self, goal: str, context: GlobalContext, current_task: TaskNode) -> AgentResponse:
        logger.info(f"TestRunnerAgent executing with goal: '{goal}'")
        
        # This agent's primary input is the JSON report from a pytest run.
        test_report_key = current_task.input_artifact_keys[0] if current_task.input_artifact_keys else "pytest_output.json"
        test_results_str = context.get_artifact(test_report_key)

        if not test_results_str:
            msg = f"Missing required artifact '{test_report_key}'. The plan should include a ToolingAgent task to run pytest and produce this report."
            return AgentResponse(success=False, message=msg)

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

        def test_test_runner_missing_artifact(self):
            print("\n--- [Test Case 3: TestRunnerAgent Missing Artifact] ---")
            response = self.agent.execute(self.task.goal, self.context, self.task)
            
            self.assertFalse(response.success)
            self.assertIn("Missing required artifact", response.message)
            logger.info("✅ test_test_runner_missing_artifact: PASSED")
            
    unittest.main(argv=['first-arg-is-ignored'], exit=False)