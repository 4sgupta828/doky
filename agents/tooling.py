# agents/tooling.py
import json
import logging
import subprocess
from typing import List

# Foundational dependencies
from .base import BaseAgent
from core.context import GlobalContext
from core.models import AgentResponse, TaskNode

# Get a logger instance for this module
logger = logging.getLogger(__name__)

class ToolingAgent(BaseAgent):
    """
    The team's DevOps specialist. It is the only agent authorized to run
    shell commands. This centralization ensures that all environment
    interactions are safe, logged, and controlled.
    """

    def __init__(self):
        super().__init__(
            name="ToolingAgent",
            description="Executes shell commands for build, dependency, and environment tasks."
        )
        # A set of commands that are not allowed for safety reasons.
        self.disallowed_commands = {"rm", "mv", "dd", "mkfs", "shutdown", "reboot"}

    def _is_command_safe(self, command: str) -> bool:
        """A simple safety check to prevent destructive commands."""
        if not command.strip():
            return False
        first_word = command.strip().split()[0]
        if first_word in self.disallowed_commands:
            logger.error(f"Execution of disallowed command '{first_word}' was blocked.")
            return False
        return True

    def execute(self, goal: str, context: GlobalContext, current_task: TaskNode) -> AgentResponse:
        # For this agent, the goal *is* the command to execute.
        command_to_run = goal
        logger.info(f"ToolingAgent executing command: '{command_to_run}'")

        if not self._is_command_safe(command_to_run):
            return AgentResponse(success=False, message=f"Command '{command_to_run}' is disallowed for safety reasons.")

        try:
            # The command is executed within the context of the agent's workspace.
            result = subprocess.run(
                command_to_run,
                shell=True,
                capture_output=True,
                text=True,
                cwd=str(context.workspace.repo_path),
                timeout=120  # A timeout is a crucial safety feature.
            )

            # Create an artifact with the combined output for easier debugging.
            output_content = f"--- STDOUT ---\n{result.stdout}\n\n--- STDERR ---\n{result.stderr}"
            output_artifact_key = f"{current_task.task_id}_output.txt"
            context.add_artifact(output_artifact_key, output_content, current_task.task_id)

            if result.returncode != 0:
                msg = f"Command failed with exit code {result.returncode}."
                logger.error(f"{msg}\n{output_content}")
                return AgentResponse(success=False, message=msg, artifacts_generated=[output_artifact_key])

            # Special handling for pytest JSON reports
            if "pytest" in command_to_run and "--json-report" in command_to_run:
                try:
                    # Assume the report is saved to a known file, then read it.
                    report_path = context.workspace.repo_path / ".report.json"
                    if report_path.exists():
                        with open(report_path, 'r', encoding='utf-8') as f:
                            # Use the raw string content, let the TestRunnerAgent parse it.
                            pytest_json_report_str = f.read()
                        context.add_artifact("pytest_output.json", pytest_json_report_str, current_task.task_id)
                        logger.info("Successfully captured pytest JSON report artifact.")
                except Exception as e:
                    logger.warning(f"Could not process pytest JSON report file: {e}")

            return AgentResponse(success=True, message="Command executed successfully.", artifacts_generated=[output_artifact_key])

        except subprocess.TimeoutExpired:
            msg = f"Command '{command_to_run}' timed out after 120 seconds."
            logger.error(msg)
            return AgentResponse(success=False, message=msg)
        except Exception as e:
            error_msg = f"An unexpected error occurred while running command: {e}"
            logger.critical(error_msg, exc_info=True)
            return AgentResponse(success=False, message=error_msg)


# --- Self-Testing Block ---
if __name__ == "__main__":
    import unittest
    from unittest.mock import patch
    import shutil
    from pathlib import Path
    from utils.logger import setup_logger

    setup_logger(default_level=logging.INFO)

    class TestToolingAgent(unittest.TestCase):

        def setUp(self):
            self.test_workspace_path = "./temp_tooling_test_ws"
            if Path(self.test_workspace_path).exists():
                shutil.rmtree(self.test_workspace_path)
            self.context = GlobalContext(workspace_path=self.test_workspace_path)
            self.agent = ToolingAgent()

        def tearDown(self):
            shutil.rmtree(self.test_workspace_path)

        @patch('subprocess.run')
        def test_tooling_agent_success(self, mock_subprocess_run):
            """Tests the ToolingAgent's successful command execution."""
            print("\n--- [Test Case 1: ToolingAgent Success] ---")
            mock_subprocess_run.return_value = subprocess.CompletedProcess(
                args=['ls', '-l'], returncode=0, stdout='total 0\n-rw-r--r-- 1 user group 0 Aug 13 18:20 file.txt', stderr=''
            )
            task = TaskNode(goal="ls -l", assigned_agent="ToolingAgent")

            response = self.agent.execute(task.goal, self.context, task)

            mock_subprocess_run.assert_called_once_with(
                "ls -l", shell=True, capture_output=True, text=True, cwd=str(self.context.workspace.repo_path), timeout=120
            )
            self.assertTrue(response.success)
            output_artifact = self.context.get_artifact(f"{task.task_id}_output.txt")
            self.assertIn("file.txt", output_artifact)
            logger.info("✅ test_tooling_agent_success: PASSED")

        @patch('subprocess.run')
        def test_tooling_agent_failure(self, mock_subprocess_run):
            """Tests the ToolingAgent when a command fails (non-zero exit code)."""
            print("\n--- [Test Case 2: ToolingAgent Command Failure] ---")
            mock_subprocess_run.return_value = subprocess.CompletedProcess(
                args=['cat', 'nonexistent.txt'], returncode=1, stdout='', stderr='cat: nonexistent.txt: No such file or directory'
            )
            task = TaskNode(goal="cat nonexistent.txt", assigned_agent="ToolingAgent")

            response = self.agent.execute(task.goal, self.context, task)

            self.assertFalse(response.success)
            self.assertIn("Command failed with exit code 1", response.message)
            output_artifact = self.context.get_artifact(f"{task.task_id}_output.txt")
            self.assertIn("No such file or directory", output_artifact)
            logger.info("✅ test_tooling_agent_failure: PASSED")

        def test_tooling_agent_disallowed_command(self):
            """Tests that the ToolingAgent blocks unsafe commands."""
            print("\n--- [Test Case 3: ToolingAgent Disallowed Command] ---")
            task = TaskNode(goal="rm -rf /", assigned_agent="ToolingAgent")
            response = self.agent.execute(task.goal, self.context, task)
            self.assertFalse(response.success)
            self.assertIn("disallowed for safety reasons", response.message)
            logger.info("✅ test_tooling_agent_disallowed_command: PASSED")

    unittest.main(argv=['first-arg-is-ignored'], exit=False)