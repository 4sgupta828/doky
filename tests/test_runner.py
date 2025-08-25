# agents/test_runner.py
import logging
from typing import Dict, Any, List

# Foundational dependencies
from .base import BaseAgent
from core.context import GlobalContext
from core.models import AgentResult, TaskNode, AgentResponse
from tools.test_tools import TestTools

# Get a logger instance for this module
logger = logging.getLogger(__name__)

class TestRunnerAgent(BaseAgent):
    """
    Comprehensive test execution agent that combines orchestration and execution capabilities.
    
    This agent handles test framework detection, discovery, execution, and analysis.
    It maintains the agent orchestration pattern while providing comprehensive test functionality.
    
    Responsibilities:
    - Multi-framework test support (pytest, unittest, custom)
    - Test discovery with advanced patterns and filtering
    - Test execution with coverage, timeout, and output options
    - Result parsing and comprehensive reporting
    - Environment validation and pre-flight checks
    """

    def __init__(self, agent_registry: Dict[str, Any] = None):
        super().__init__(
            name="TestRunnerAgent",
            description="Discovers and executes test suites, then reports structured results."
        )
        self.agent_registry = agent_registry or {}

    def required_inputs(self) -> List[str]:
        """Required inputs for TestRunnerAgent execution."""
        return []  # Can auto-discover tests

    def optional_inputs(self) -> List[str]:
        """Optional inputs for comprehensive test configuration."""
        return [
            "test_target",  # files, directories, or patterns
            "test_framework",  # pytest, unittest, custom, auto
            "python_executable", 
            "working_directory",
            "timeout_seconds",
            "test_patterns",
            "exclude_patterns",
            "output_format",  # json, xml, text
            "coverage_enabled",
            "additional_args",
            "specific_test_files"  # legacy support
        ]

    def execute_v2(self, goal: str, inputs: Dict[str, Any], global_context: GlobalContext) -> AgentResult:
        """
        Comprehensive test execution with framework detection, discovery, execution, and analysis.
        """
        logger.info(f"TestRunnerAgent executing: '{goal}'")
        
        # Validate inputs with graceful handling
        try:
            self.validate_inputs(inputs)
        except Exception as validation_error:
            return self.create_result(
                success=False,
                message=str(validation_error),
                error_details={"validation_error": str(validation_error)}
            )

        # Extract and set defaults for inputs
        test_target = inputs.get("test_target") or inputs.get("specific_test_files")
        test_framework = inputs.get("test_framework", "auto")
        python_executable = inputs.get("python_executable")
        working_directory = inputs.get("working_directory", str(global_context.workspace_path))
        timeout_seconds = inputs.get("timeout_seconds", 300)
        test_patterns = inputs.get("test_patterns", [])
        exclude_patterns = inputs.get("exclude_patterns", [])
        output_format = inputs.get("output_format", "json")
        coverage_enabled = inputs.get("coverage_enabled", False)
        additional_args = inputs.get("additional_args", [])

        try:
            self.report_progress("Starting comprehensive test execution", f"Target: {test_target or 'auto-discover'}")

            # Step 1: Detect test framework
            self.report_progress("Detecting test framework", f"Requested: {test_framework}")
            framework_result = TestTools.detect_test_framework(
                requested_framework=test_framework, 
                test_target=test_target, 
                working_directory=working_directory
            )
            
            if not framework_result["success"]:
                return self.create_result(
                    success=False,
                    message=f"Test framework detection failed: {framework_result['message']}",
                    error_details=framework_result
                )

            detected_framework = framework_result["framework"]
            self.report_progress("Framework detected", f"Using {detected_framework}")

            # Step 2: Discover test files
            self.report_progress("Discovering test files", "Searching for test files...")
            discovery_result = TestTools.discover_test_files(
                test_target=test_target or ".",
                framework=detected_framework,
                working_directory=working_directory,
                test_patterns=test_patterns,
                exclude_patterns=exclude_patterns
            )
            
            if not discovery_result["success"]:
                return self.create_result(
                    success=False,
                    message=f"Test discovery failed: {discovery_result['message']}",
                    error_details=discovery_result
                )

            test_files = discovery_result["test_files"]
            if not test_files:
                return self.create_result(
                    success=True, 
                    message="No test files found to run.",
                    outputs={"test_files_discovered": 0}
                )
            
            self.report_progress("Test discovery complete", f"Found {len(test_files)} test files")

            # Step 3: Execute tests using TestTools
            self.report_progress("Executing tests", f"Running {len(test_files)} test files with {detected_framework}")
            execution_result = TestTools.execute_tests(
                test_files=test_files,
                framework=detected_framework,
                python_executable=python_executable,
                working_directory=working_directory,
                timeout_seconds=timeout_seconds,
                coverage_enabled=coverage_enabled,
                additional_args=additional_args,
                output_format=output_format
            )

            # Step 4: Parse and analyze results
            analysis_result = TestTools.parse_test_results(execution_result, detected_framework)

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
            error_msg = f"TestRunnerAgent execution failed: {e}"
            logger.error(error_msg, exc_info=True)
            return self.create_result(
                success=False,
                message=error_msg,
                error_details={"exception": str(e)}
            )

    def _create_summary_message(self, final_result: Dict[str, Any], analysis_result: Dict[str, Any]) -> str:
        """Create a summary message for the test execution."""
        framework = final_result.get("test_framework", "unknown")
        files_count = final_result.get("test_files_discovered", 0)
        
        if final_result["success"]:
            if analysis_result.get("passed_tests"):
                passed = analysis_result.get("passed_tests", 0)
                total = analysis_result.get("total_tests", 0)
                return f"Test suite passed: {passed}/{total} tests successful using {framework} ({files_count} files)"
            else:
                return f"Test execution successful using {framework}: {analysis_result.get('summary', 'Tests passed')}"
        else:
            if analysis_result.get("failed_tests"):
                failed = analysis_result.get("failed_tests", 0)
                total = analysis_result.get("total_tests", 0)
                return f"Test suite failed: {failed}/{total} tests failed using {framework} ({files_count} files)"
            else:
                return f"Test execution failed using {framework}: {analysis_result.get('summary', 'Tests failed')}"
