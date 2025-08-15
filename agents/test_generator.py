# agents/test_generator.py
import json
import logging
from typing import Dict, Any, List, Literal

# Foundational dependencies
from .base import BaseAgent
from core.context import GlobalContext
from core.models import AgentResponse, TaskNode

# Get a logger instance for this module
logger = logging.getLogger(__name__)


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

    def __init__(self, llm_client: Any = None):
        super().__init__(
            name="TestGenerationAgent",
            description="Generates unit and integration tests for application code."
        )
        self.llm_client = llm_client or LLMClient()

    def _determine_test_type(self, goal: str) -> Literal["unit", "integration"]:
        """Analyzes the goal to determine what kind of test to write."""
        if "integration" in goal.lower():
            return "integration"
        # Default to unit tests for specificity and speed.
        return "unit"

    def _build_prompt(self, spec: str, code_to_test: Dict[str, str], test_type: Literal["unit", "integration"]) -> str:
        """Constructs a detailed prompt for the LLM to generate test code."""
        code_blocks = "\n\n".join(
            f"--- File: {path} ---\n```python\n{content}\n```"
            for path, content in code_to_test.items()
        )

        if test_type == "unit":
            test_instructions = """
        **Instructions for Unit Tests:**
        1.  Write focused `pytest` unit tests for individual functions and classes.
        2.  Mock all external dependencies (e.g., database, APIs) using `unittest.mock`.
        3.  Test success cases, edge cases, and error conditions (using `pytest.raises`).
        """
        else: # integration
            test_instructions = """
        **Instructions for Integration Tests:**
        1.  Write `pytest` integration tests that verify the interaction between multiple components.
        2.  Use a test client (e.g., Flask's `TestClient`) to make real requests to API endpoints.
        3.  Focus on end-to-end user flows (e.g., register, login, access protected route).
        """

        return f"""
        You are an expert QA Engineer specializing in Python. Your task is to write
        high-quality **{test_type} tests** using the `pytest` framework for the provided source code,
        based on its technical specification.

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
        
        test_type = self._determine_test_type(goal)
        logger.info(f"Determined required test type: {test_type}")

        spec = context.get_artifact("technical_spec.md")
        manifest = context.get_artifact("file_manifest.json")
        
        # Handle missing inputs gracefully 
        if not spec and not manifest:
            spec = f"User Request: {goal}"
            files_to_read = []
            logger.info("No spec or manifest found. Will generate tests based on goal and existing code.")
        elif not spec:
            spec = f"User Request: {goal}"
            files_to_read = manifest.get("files_to_create", [])
            logger.info("No spec found. Using goal as specification for test generation.")
        elif not manifest:
            spec = spec
            files_to_read = []
            logger.info("No manifest found. Will infer files to test from workspace.")
        else:
            files_to_read = manifest.get("files_to_create", [])

        code_to_test = {}
        # files_to_read is already set above based on manifest availability
        for file_path in files_to_read:
            if not file_path.startswith("tests/"):
                content = context.workspace.get_file_content(file_path)
                if content:
                    code_to_test[file_path] = content
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
            prompt = self._build_prompt(spec, code_to_test, test_type)
            
            # Use function calling for guaranteed JSON response
            if hasattr(self.llm_client, 'invoke_with_schema'):
                llm_response_str = self.llm_client.invoke_with_schema(prompt, test_generation_schema)
            else:
                # Fallback to regular invoke for backward compatibility
                llm_response_str = self.llm_client.invoke(prompt)
            
            generated_tests_map = json.loads(llm_response_str)
            if not isinstance(generated_tests_map, dict):
                raise ValueError("LLM response is not a valid dictionary of test files.")
        except Exception as e:
            return AgentResponse(success=False, message=f"An unexpected error occurred during test generation: {e}")

        written_files = []
        for file_path, code_content in generated_tests_map.items():
            context.workspace.write_file_content(file_path, code_content, current_task.task_id)
            written_files.append(file_path)

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