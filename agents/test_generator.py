# agents/test_generator.py
import json
import logging
from typing import Dict, Any, List, Literal
from enum import Enum

# Foundational dependencies
from .base import BaseAgent
from core.context import GlobalContext
from core.models import AgentResponse, TaskNode

# Get a logger instance for this module
logger = logging.getLogger(__name__)


# --- Test Quality Levels ---
class TestQuality(Enum):
    """Defines different test quality levels for speed vs thoroughness trade-offs."""
    FAST = "fast"          # Quick, basic tests - prioritizes speed
    DECENT = "decent"      # Balanced approach - good coverage, reasonable detail (default)
    PRODUCTION = "production"  # Comprehensive, thorough tests with full edge case coverage


# --- Real LLM Integration Placeholder ---
class LLMClient:
    """A placeholder for a real LLM client (e.g., OpenAI, Gemini)."""
    def invoke(self, prompt: str) -> str:
        raise NotImplementedError("LLMClient.invoke must be implemented by a concrete class.")

# --- Agent Implementation ---

class TestGenerationAgent(BaseAgent):
    """
    Specialized in writing tests for source code. It can generate both
    focused unit tests and broader integration tests based on the task goal.
    """

    def __init__(self, llm_client: Any = None, default_quality: TestQuality = TestQuality.FAST):
        super().__init__(
            name="TestGenerationAgent",
            description="Generates unit and integration tests for application code."
        )
        self.llm_client = llm_client or LLMClient()
        self.default_quality = default_quality

    def _determine_test_type(self, goal: str) -> Literal["unit", "integration"]:
        """Analyzes the goal to determine what kind of test to write."""
        if "integration" in goal.lower():
            return "integration"
        # Default to unit tests for specificity and speed.
        return "unit"

    def _get_quality_instructions(self, quality: TestQuality, test_type: Literal["unit", "integration"]) -> Dict[str, str]:
        """Returns quality-specific instructions and descriptions."""
        if test_type == "unit":
            quality_configs = {
                TestQuality.FAST: {
                    "description": "basic unit tests with essential coverage",
                    "instructions": [
                        "Write focused pytest tests for main functions and classes",
                        "Test basic success cases and obvious error conditions",
                        "Use simple mocking for external dependencies",
                        "Keep test structure simple and straightforward"
                    ]
                },
                TestQuality.DECENT: {
                    "description": "comprehensive unit tests with good coverage",
                    "instructions": [
                        "Write thorough pytest unit tests for all functions and classes",
                        "Mock external dependencies using unittest.mock effectively",
                        "Test success cases, edge cases, and error conditions using pytest.raises",
                        "Include reasonable test data variety and boundary conditions"
                    ]
                },
                TestQuality.PRODUCTION: {
                    "description": "exhaustive unit tests with complete coverage",
                    "instructions": [
                        "Write comprehensive pytest unit tests covering all code paths",
                        "Implement sophisticated mocking strategies for complex dependencies",
                        "Test all success cases, edge cases, error conditions, and boundary values",
                        "Include parametrized tests for data variety and comprehensive fixtures",
                        "Add performance tests and memory usage considerations where appropriate"
                    ]
                }
            }
        else:  # integration
            quality_configs = {
                TestQuality.FAST: {
                    "description": "basic integration tests for key workflows",
                    "instructions": [
                        "Write pytest integration tests for main user workflows",
                        "Use test client for basic API endpoint testing",
                        "Focus on happy path scenarios",
                        "Keep test setup minimal and straightforward"
                    ]
                },
                TestQuality.DECENT: {
                    "description": "thorough integration tests with good workflow coverage",
                    "instructions": [
                        "Write comprehensive pytest integration tests for component interactions",
                        "Use test client (e.g., Flask's TestClient) for realistic API testing",
                        "Test end-to-end user flows including error scenarios",
                        "Include proper test data setup and cleanup"
                    ]
                },
                TestQuality.PRODUCTION: {
                    "description": "exhaustive integration tests with complete workflow coverage",
                    "instructions": [
                        "Write comprehensive integration tests covering all system interactions",
                        "Implement sophisticated test client usage with authentication and authorization",
                        "Test complete end-to-end workflows including edge cases and failure modes",
                        "Include database transaction testing, concurrent access scenarios",
                        "Add performance and load testing considerations for critical paths"
                    ]
                }
            }
        return quality_configs[quality]
    
    def _detect_quality_level(self, goal: str, context: GlobalContext) -> TestQuality:
        """Detects the desired test quality level from the goal and context."""
        goal_lower = goal.lower()
        
        # Check for explicit quality keywords in the goal
        if any(keyword in goal_lower for keyword in ['fast', 'quick', 'basic', 'simple', 'minimal']):
            logger.info("Detected FAST test quality level from goal keywords")
            self.report_thinking("Goal contains keywords suggesting FAST test quality - prioritizing speed over comprehensiveness.")
            return TestQuality.FAST
        elif any(keyword in goal_lower for keyword in ['decent', 'thorough', 'comprehensive', 'good', 'complete']):
            logger.info("Detected DECENT test quality level from goal keywords")
            self.report_thinking("Goal contains keywords suggesting DECENT test quality - balanced approach with good coverage.")
            return TestQuality.DECENT
        elif any(keyword in goal_lower for keyword in ['production', 'exhaustive', 'full', 'robust', 'enterprise']):
            logger.info("Detected PRODUCTION test quality level from goal keywords")
            self.report_thinking("Goal contains keywords suggesting PRODUCTION test quality - maximum coverage and robustness.")
            return TestQuality.PRODUCTION
        
        # Check context for quality preferences
        if hasattr(context, 'test_quality_preference'):
            self.report_thinking(f"Using context preference: {context.test_quality_preference.value} test quality.")
            return context.test_quality_preference
            
        # Default to FAST for speed optimization
        logger.info("Using default FAST test quality level")
        self.report_thinking("No explicit quality indicators found. Defaulting to FAST test quality for quick iteration.")
        return self.default_quality

    def _build_prompt(self, spec: str, code_to_test: Dict[str, str], test_type: Literal["unit", "integration"], quality: TestQuality = None) -> str:
        """Constructs a detailed prompt for the LLM to generate test code."""
        if quality is None:
            quality = self.default_quality
            
        quality_config = self._get_quality_instructions(quality, test_type)
        code_blocks = "\n\n".join(
            f"--- File: {path} ---\n```python\n{content}\n```"
            for path, content in code_to_test.items()
        )

        # Build quality-specific instructions
        quality_instructions = "\n        ".join([f"{i+1}. {instruction}" for i, instruction in enumerate(quality_config["instructions"])])
        
        test_instructions = f"""
        **Instructions for {test_type.title()} Tests ({quality.value.upper()} Quality):**
        {quality_instructions}
        """

        return f"""
        You are an expert QA Engineer specializing in Python. Your task is to write
        {quality_config["description"]} using the `pytest` framework for the provided source code,
        based on its technical specification.
        
        **Test Quality Level: {quality.value.upper()}**

        **Technical Specification:**
        ---
        {spec}
        ---

        **Source Code to Test:**
        ---
        {code_blocks}
        ---

        {test_instructions}

        **Final Output Requirement:**
        Your output MUST be a single, valid JSON object where keys are the test file paths
        (e.g., "tests/test_auth_integration.py") and values are the complete test code content as a string.
        """

    def execute(self, goal: str, context: GlobalContext, current_task: TaskNode) -> AgentResponse:
        logger.info(f"TestGenerationAgent executing with goal: '{goal}'")
        
        # Report meaningful progress
        self.report_progress("Analyzing requirements", f"Creating tests for: '{goal[:80]}...'")
        
        test_type = self._determine_test_type(goal)
        logger.info(f"Determined required test type: {test_type}")
        self.report_thinking(f"Determined test type: {test_type} based on goal analysis. This will influence the test structure and approach.")
        
        if test_type != "unit":
            self.report_thinking(f"I'll create {test_type} tests - these will be more comprehensive than simple unit tests.")

        spec = context.get_artifact("technical_spec.md")
        manifest = context.get_artifact("file_manifest.json")
        
        # Handle missing inputs gracefully 
        if not spec and not manifest:
            spec = f"User Request: {goal}"
            files_to_read = []
            logger.info("No spec or manifest found. Will generate tests based on goal and existing code.")
            self.report_thinking("No technical specification or manifest found. Will infer test requirements from the goal and discover code files automatically.")
        elif not spec:
            spec = f"User Request: {goal}"
            files_to_read = manifest.get("files_to_create", [])
            logger.info("No spec found. Using goal as specification for test generation.")
            self.report_thinking("Found file manifest but no technical specification. Using goal as spec and manifest for code discovery.")
        elif not manifest:
            spec = spec
            files_to_read = []
            logger.info("No manifest found. Will infer files to test from workspace.")
            self.report_thinking("Found technical specification but no manifest. Will discover code files in workspace automatically.")
        else:
            files_to_read = manifest.get("files_to_create", [])
            self.report_thinking(f"Found both specification and manifest. Will test {len(files_to_read)} files listed in manifest.")

        code_to_test = {}
        # files_to_read is already set above based on manifest availability
        for file_path in files_to_read:
            if not file_path.startswith("tests/"):
                content = context.workspace.get_file_content(file_path)
                if content:
                    code_to_test[file_path] = content
        
        # If no code was found from manifest, try to discover existing Python files
        if not code_to_test:
            logger.info("No code found from manifest. Attempting to discover existing Python files.")
            self.report_progress("Discovering code files", "Scanning workspace for Python files to test")
            self.report_thinking("No code files found from manifest. Initiating automatic code discovery by scanning workspace for Python files.")
            
            try:
                all_files = context.workspace.list_files(".")
                discovered_files = []
                for file_path in all_files:
                    if file_path.endswith(".py") and not file_path.startswith("tests/") and not file_path.startswith("test_"):
                        content = context.workspace.get_file_content(file_path)
                        if content:
                            code_to_test[file_path] = content
                            discovered_files.append(file_path)
                            logger.info(f"Discovered code file for testing: {file_path}")
                
                if discovered_files:
                    self.report_progress("Code discovery complete", f"Found {len(discovered_files)} Python files: {', '.join(discovered_files[:3])}{'...' if len(discovered_files) > 3 else ''}")
                else:
                    self.report_thinking("Code discovery found no Python files. This may be expected for a new project or indicate files are in unexpected locations.")
            except Exception as e:
                logger.warning(f"Failed to discover code files: {e}")
                self.report_thinking(f"Code discovery failed with error: {e}. Will proceed with available information.")
        
        if not code_to_test:
            return AgentResponse(success=True, message="No application code found to test.")

        # Define JSON schema for guaranteed structured response
        test_generation_schema = {
            "type": "object", 
            "properties": {},
            "additionalProperties": {"type": "string"},
            "description": "Dictionary mapping test file paths to their test code content"
        }

        try:
            # Detect quality level from goal and context
            quality_level = self._detect_quality_level(goal, context)
            logger.info(f"Using test quality level: {quality_level.value.upper()}")
            self.report_progress("Test quality determined", f"Using {quality_level.value.upper()} quality level for comprehensive {test_type} tests")
            self.report_thinking(f"Selected {quality_level.value.upper()} quality level based on goal analysis. This will determine test depth, coverage, and sophistication.")
            
            prompt = self._build_prompt(spec, code_to_test, test_type, quality_level)
            
            self.report_progress("Generating test code", f"Creating {quality_level.value} {test_type} tests with AI assistance")
            
            # Use regular invoke by default, fall back to function calling if needed
            logger.debug("Using regular invoke method (primary approach)")
            llm_response_str = self.llm_client.invoke(prompt)
            
            # Check if the response is valid JSON with actual content
            try:
                test_parse = json.loads(llm_response_str)
                if not isinstance(test_parse, dict) or not test_parse:
                    raise ValueError("Invalid or empty response")
                logger.debug("Regular invoke succeeded with valid JSON response")
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Regular invoke failed to produce valid JSON ({e}), trying function calling")
                if hasattr(self.llm_client, 'invoke_with_schema'):
                    try:
                        llm_response_str = self.llm_client.invoke_with_schema(prompt, test_generation_schema)
                        logger.debug("Function calling fallback succeeded")
                    except Exception as fallback_error:
                        logger.error(f"Function calling fallback also failed: {fallback_error}")
                        # Keep the original response from regular invoke for error reporting
                        pass
                else:
                    logger.warning("Function calling not available, keeping original response")
            
            generated_tests_map = json.loads(llm_response_str)
            if not isinstance(generated_tests_map, dict):
                raise ValueError("LLM response is not a valid dictionary of test files.")
        except Exception as e:
            return AgentResponse(success=False, message=f"An unexpected error occurred during test generation: {e}")

        # Report successful test generation with details
        total_lines = sum(len(content.split('\n')) for content in generated_tests_map.values())
        self.report_progress("Test generation complete", f"Generated {len(generated_tests_map)} test files with {total_lines} total lines")
        
        # Display the generated test code using enhanced UI
        if len(generated_tests_map) == 1:
            # Single test file - show as code snippet
            file_path, content = next(iter(generated_tests_map.items()))
            code_with_filename = {"content": content, "filename": file_path}
            self.report_intermediate_output("code_snippet", code_with_filename)
        else:
            # Multiple test files - show as code files
            self.report_intermediate_output("code_files", generated_tests_map)

        written_files = []
        self.report_progress("Writing test files", f"Saving {len(generated_tests_map)} test files to workspace")
        
        for file_path, code_content in generated_tests_map.items():
            context.workspace.write_file_content(file_path, code_content, current_task.task_id)
            written_files.append(file_path)
            logger.info(f"Successfully wrote test file: {file_path} ({len(code_content.split('\n'))} lines)")

        return AgentResponse(
            success=True,
            message=f"Successfully generated and wrote {len(written_files)} {test_type} test files.",
            artifacts_generated=written_files
        )


# --- Self-Testing Block ---
if __name__ == "__main__":
    import unittest
    from unittest.mock import MagicMock
    import shutil
    from pathlib import Path
    from utils.logger import setup_logger

    setup_logger(default_level=logging.INFO)

    class TestTestGenerationAgent(unittest.TestCase):
        def setUp(self):
            self.test_workspace_path = Path("./temp_test_gen_ws")
            if self.test_workspace_path.exists():
                shutil.rmtree(self.test_workspace_path)
            self.context = GlobalContext(workspace_path=str(self.test_workspace_path))
            self.mock_llm_client = MagicMock(spec=LLMClient)
            self.agent = TestGenerationAgent(llm_client=self.mock_llm_client)
            self.context.add_artifact("technical_spec.md", "Spec: `add(a, b)` function.", "task_spec")
            self.context.add_artifact("file_manifest.json", {"files_to_create": ["src/calc.py"]}, "task_manifest")
            self.context.workspace.write_file_content("src/calc.py", "def add(a, b): return a + b", "task_code")

        def tearDown(self):
            shutil.rmtree(self.test_workspace_path)

        def test_unit_test_generation(self):
            print("\n--- [Test Case 1: Unit Test Generation] ---")
            task = TaskNode(goal="Generate unit tests", assigned_agent="TestGenerationAgent")
            mock_test_code = json.dumps({"tests/test_unit.py": "def test_add(): assert add(2, 3) == 5"})
            self.mock_llm_client.invoke.return_value = mock_test_code
            
            response = self.agent.execute(task.goal, self.context, task)
            
            self.assertTrue(response.success)
            self.assertIn("unit test", response.message)
            prompt_call = self.mock_llm_client.invoke.call_args[0][0]
            self.assertIn("Instructions for Unit Tests", prompt_call)
            logger.info("✅ test_unit_test_generation: PASSED")

        def test_integration_test_generation(self):
            print("\n--- [Test Case 2: Integration Test Generation] ---")
            task = TaskNode(goal="Generate integration tests", assigned_agent="TestGenerationAgent")
            mock_test_code = json.dumps({"tests/test_integration.py": "def test_api(): ..."})
            self.mock_llm_client.invoke.return_value = mock_test_code

            response = self.agent.execute(task.goal, self.context, task)

            self.assertTrue(response.success)
            self.assertIn("integration test", response.message)
            prompt_call = self.mock_llm_client.invoke.call_args[0][0]
            self.assertIn("Instructions for Integration Tests", prompt_call)
            logger.info("✅ test_integration_test_generation: PASSED")

    unittest.main(argv=['first-arg-is-ignored'], exit=False)