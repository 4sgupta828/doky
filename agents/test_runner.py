# agents/test_runner.py
import json
import logging
from typing import Dict, Any, List

# Foundational dependencies
from .base import BaseAgent
from core.context import GlobalContext
from core.models import AgentResult, TaskNode

# Get a logger instance for this module
logger = logging.getLogger(__name__)

class TestRunnerAgent(BaseAgent):
    """
    Orchestrates the execution of test suites. It discovers test files using the
    FileSystemAgent and delegates command execution to the ToolingAgent.
    """

    def __init__(self, agent_registry: Dict[str, Any] = None):
        super().__init__(
            name="TestRunnerAgent",
            description="Discovers and executes test suites, then reports structured results."
        )
        self.agent_registry = agent_registry or {}

    def required_inputs(self) -> List[str]:
        """The TestRunnerAgent can run without specific inputs by discovering tests."""
        return []

    def optional_inputs(self) -> List[str]:
        return ["specific_test_files"]

    def execute_v2(self, goal: str, inputs: Dict[str, Any], global_context: GlobalContext) -> AgentResult:
        """
        Discovers, runs, and analyzes tests.
        """
        self.validate_inputs(inputs)
        
        file_system_agent = self.agent_registry.get("FileSystemAgent")
        tooling_agent = self.agent_registry.get("ToolingAgent")

        if not file_system_agent or not tooling_agent:
            return self.create_result(success=False, message="Required agents (FileSystemAgent, ToolingAgent) not available.")

        self.report_progress("Starting test execution", f"Goal: {goal}")

        # --- Step 1: Discover Test Files ---
        specific_files = inputs.get("specific_test_files")
        if specific_files:
            test_files = specific_files
            self.report_thinking(f"Running specific tests for {len(test_files)} provided files.")
        else:
            self.report_thinking("No specific test files provided. Discovering tests in the workspace.")
            discovery_result = self.call_agent_v2(
                target_agent=file_system_agent,
                goal="Discover all Python test files in the workspace.",
                inputs={
                    "operation": "discover",
                    "patterns": ["test_*.py", "*_test.py"],
                    "recursive": True
                },
                global_context=global_context
            )
            if not discovery_result.success:
                return self.create_result(success=False, message=f"Failed to discover test files: {discovery_result.message}")
            
            test_files = discovery_result.outputs.get("discovered_files", [])

        if not test_files:
            return self.create_result(success=True, message="No test files found to run.")
        
        self.report_progress(f"Discovered {len(test_files)} test files.", f"Examples: {test_files[:3]}")

        # --- Step 2: Execute Tests via ToolingAgent ---
        report_file = "test_report.json"
        test_command = f"python -m pytest {' '.join(test_files)} --json-report --json-report-file={report_file}"
        
        tooling_result = self.call_agent_v2(
            target_agent=tooling_agent,
            goal="Execute the test suite and generate a JSON report.",
            inputs={
                "commands": [test_command],
                "purpose": "Test Suite Execution",
                "ignore_errors": True # Capture the report even if tests fail (exit code 1)
            },
            global_context=global_context
        )

        # Pytest returns exit code 1 for failed tests, which is not an execution error for us.
        # Any other non-zero exit code is a real error.
        if not tooling_result.success and tooling_result.outputs.get("exit_code", 0) not in [0, 1]:
            return self.create_result(success=False, message=f"Test execution command failed: {tooling_result.message}")

        # --- Step 3: Analyze the Test Report ---
        return self._analyze_test_report(report_file, global_context)


    def _analyze_test_report(self, report_file: str, context: GlobalContext) -> AgentResult:
        """Analyzes the test_report.json file produced by pytest."""
        self.report_progress("Analyzing test report...")
        
        file_system_agent = self.agent_registry.get("FileSystemAgent")
        if not file_system_agent:
            return self.create_result(success=False, message="FileSystemAgent not available to read test report.")

        read_result = self.call_agent_v2(
            target_agent=file_system_agent,
            goal=f"Read the content of the test report file: {report_file}",
            inputs={"operation": "read", "target_path": report_file},
            global_context=context
        )

        if not read_result.success:
            return self.create_result(success=False, message=f"Could not read test report file '{report_file}'.")
        
        report_content = read_result.outputs.get("content")
        
        try:
            report_data = json.loads(report_content)
            summary = report_data.get("summary", {})
            failed_count = summary.get("failed", 0)
            passed_count = summary.get("passed", 0)
            total_count = summary.get("total", 0)

            if failed_count > 0:
                msg = f"Test suite failed: {failed_count} of {total_count} tests failed."
                logger.error(msg)
                return self.create_result(
                    success=False,
                    message=msg,
                    outputs={"failed_test_report": report_data}
                )
            else:
                msg = f"Test suite passed: All {passed_count} of {total_count} tests were successful."
                logger.info(msg)
                return self.create_result(
                    success=True,
                    message=msg,
                    outputs={"test_summary": summary}
                )

        except (json.JSONDecodeError, KeyError) as e:
            msg = f"Failed to parse test report. Error: {e}"
            return self.create_result(success=False, message=msg)
