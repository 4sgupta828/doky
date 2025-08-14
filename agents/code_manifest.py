# agents/code_manifest.py
import json
import logging
from typing import Dict, Any, List

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

class CodeManifestAgent(BaseAgent):
    """
    Acts as the project's tech lead, planning the file structure by converting
    a technical specification into a JSON manifest of file paths.
    """

    def __init__(self, llm_client: Any = None):
        super().__init__(
            name="CodeManifestAgent",
            description="Defines the project's file and directory structure from a technical spec."
        )
        self.llm_client = llm_client or LLMClient()

    def _build_prompt(self, spec: str) -> str:
        """Constructs a precise prompt to guide the LLM in generating a file manifest."""
        return f"""
        You are an expert tech lead responsible for project structure. Based on the
        following technical specification, determine the optimal file and directory
        structure for a new Python project.

        **Technical Specification:**
        ---
        {spec}
        ---

        **Instructions:**
        1.  Analyze the components described in the spec (e.g., models, routes, tests).
        2.  Create a logical file structure that separates concerns (e.g., using a `src` layout).
        3.  The output MUST be a single, valid JSON object containing one key: "files_to_create",
            which holds a list of strings representing the file paths.
        4.  Do not include any other text, commentary, or markdown formatting.

        **Example Output:**
        {{
            "files_to_create": [
                "src/main.py",
                "src/models/user.py",
                "src/routes/auth.py",
                "tests/test_auth.py"
            ]
        }}
        """

    def execute(self, goal: str, context: GlobalContext, current_task: TaskNode) -> AgentResponse:
        logger.info(f"CodeManifestAgent executing with goal: '{goal}'")

        input_artifact_key = current_task.input_artifact_keys[0] if current_task.input_artifact_keys else "technical_spec.md"
        tech_spec = context.get_artifact(input_artifact_key)

        if not tech_spec:
            msg = f"Missing required artifact '{input_artifact_key}'. Cannot generate manifest."
            logger.error(msg)
            return AgentResponse(success=False, message=msg)
        
        try:
            prompt = self._build_prompt(tech_spec)
            manifest_json_str = self.llm_client.invoke(prompt)
            manifest_data = json.loads(manifest_json_str)

            if "files_to_create" not in manifest_data or not isinstance(manifest_data["files_to_create"], list):
                raise ValueError("LLM response for manifest is missing 'files_to_create' list.")

            output_artifact_key = "file_manifest.json"
            context.add_artifact(key=output_artifact_key, value=manifest_data, source_task_id=current_task.task_id)
            return AgentResponse(success=True, message="Successfully generated code manifest.", artifacts_generated=[output_artifact_key])

        except NotImplementedError:
            msg = "LLMClient is not implemented. Cannot generate manifest."
            logger.critical(msg)
            return AgentResponse(success=False, message=msg)
        except (json.JSONDecodeError, ValueError) as e:
            msg = f"Failed to parse LLM response for manifest as valid JSON with required keys. Error: {e}"
            logger.error(msg)
            return AgentResponse(success=False, message=msg)
        except Exception as e:
            error_msg = f"An unexpected error occurred while calling the LLM for manifest generation: {e}"
            logger.critical(error_msg, exc_info=True)
            return AgentResponse(success=False, message=error_msg)


# --- Self-Testing Block ---
if __name__ == "__main__":
    import unittest
    from unittest.mock import MagicMock
    import shutil
    from pathlib import Path
    from utils.logger import setup_logger

    setup_logger(default_level=logging.INFO)

    class TestCodeManifestAgent(unittest.TestCase):

        def setUp(self):
            self.test_workspace_path = "./temp_manifest_agent_test_ws"
            if Path(self.test_workspace_path).exists():
                shutil.rmtree(self.test_workspace_path)
            
            self.mock_llm_client = MagicMock(spec=LLMClient)
            self.context = GlobalContext(workspace_path=self.test_workspace_path)
            self.agent = CodeManifestAgent(llm_client=self.mock_llm_client)
            self.task = TaskNode(
                goal="Create a manifest",
                assigned_agent="CodeManifestAgent",
                input_artifact_keys=["technical_spec.md"]
            )
            self.context.add_artifact("technical_spec.md", "# Spec\n- GET /users", "task_spec")

        def tearDown(self):
            shutil.rmtree(self.test_workspace_path)

        def test_successful_manifest_generation(self):
            print("\n--- [Test Case 1: CodeManifestAgent Success] ---")
            mock_manifest_output = json.dumps({"files_to_create": ["src/main.py", "src/routes.py"]})
            self.mock_llm_client.invoke.return_value = mock_manifest_output

            response = self.agent.execute(self.task.goal, self.context, self.task)
            
            self.mock_llm_client.invoke.assert_called_once()
            self.assertTrue(response.success)
            generated_manifest = self.context.get_artifact("file_manifest.json")
            self.assertEqual(generated_manifest["files_to_create"], ["src/main.py", "src/routes.py"])
            logger.info("✅ test_successful_manifest_generation: PASSED")

        def test_failure_on_invalid_json(self):
            print("\n--- [Test Case 2: CodeManifestAgent Invalid JSON] ---")
            self.mock_llm_client.invoke.return_value = "this is not json"

            response = self.agent.execute(self.task.goal, self.context, self.task)

            self.assertFalse(response.success)
            self.assertIn("Failed to parse LLM response", response.message)
            logger.info("✅ test_failure_on_invalid_json: PASSED")

    unittest.main(argv=['first-arg-is-ignored'], exit=False)