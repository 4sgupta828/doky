# agents/coder.py
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

class CodeGenerationAgent(BaseAgent):
    """
    A specialized agent responsible for writing and modifying source code.
    It takes a detailed technical specification and a file manifest as input and
    produces the corresponding code, writing it to the workspace.
    """

    def __init__(self, llm_client: Any = None):
        super().__init__(
            name="CodeGenerationAgent",
            description="Writes, modifies, and refactors application code based on a spec."
        )
        self.llm_client = llm_client or LLMClient()

    def _build_prompt(self, files_to_generate: List[str], spec: str, existing_code: Dict[str, str]) -> str:
        """
        Constructs a detailed prompt to guide the LLM in generating code for a
        specific set of files.
        """
        existing_code_str = "\n".join(
            f"--- File: {path} ---\n```python\n{content}\n```"
            for path, content in existing_code.items()
        )
        
        return f"""
        You are an expert software developer. Your task is to write production-quality Python code
        for the following files based on the provided technical specification.

        **Technical Specification:**
        ---
        {spec}
        ---

        **Files to Generate/Modify:**
        - {"\n- ".join(files_to_generate)}

        **Existing Code for Context (if any):**
        ---
        {existing_code_str if existing_code else "No existing code provided. You are writing these files from scratch."}
        ---

        **Instructions:**
        1.  Adhere strictly to the technical specification.
        2.  Write clean, efficient, and well-commented code.
        3.  Ensure all necessary imports are included in each file.
        4.  Your output MUST be a single, valid JSON object. This object should be a dictionary
            where keys are the file paths (e.g., "src/main.py") and values are the complete
            code content for that file as a single string.

        **JSON Output Format Example:**
        {{
            "src/models/user.py": "from sqlalchemy import Column, String\\n\\nclass User(...):\\n    ...",
            "src/routes/auth.py": "from flask import Blueprint\\n\\nauth_bp = Blueprint(...)"
        }}

        Now, generate the code for the requested files.
        """

    def execute(self, goal: str, context: GlobalContext, current_task: TaskNode) -> AgentResponse:
        logger.info(f"CodeGenerationAgent executing with goal: '{goal}'")

        # 1. Retrieve necessary artifacts from the context.
        spec_key = "technical_spec.md"
        manifest_key = "file_manifest.json"
        
        tech_spec = context.get_artifact(spec_key)
        manifest = context.get_artifact(manifest_key)

        if not tech_spec or not manifest:
            msg = f"Missing required artifacts: needs '{spec_key}' and '{manifest_key}'."
            logger.error(msg)
            return AgentResponse(success=False, message=msg)
        
        files_to_generate = manifest.get("files_to_create", [])
        if not files_to_generate:
            return AgentResponse(success=True, message="Code manifest is empty. No files to generate.")

        # 2. Build context of existing code (if any files already exist).
        existing_code = {}
        for file_path in files_to_generate:
            content = context.workspace.get_file_content(file_path)
            if content:
                existing_code[file_path] = content

        # 3. Construct the prompt and invoke the LLM.
        # Define JSON schema for guaranteed structured response
        code_generation_schema = {
            "type": "object",
            "properties": {},
            "additionalProperties": {"type": "string"},
            "description": "Dictionary mapping file paths to their code content"
        }

        try:
            prompt = self._build_prompt(files_to_generate, tech_spec, existing_code)
            
            # Use function calling for guaranteed JSON response
            if hasattr(self.llm_client, 'invoke_with_schema'):
                llm_response_str = self.llm_client.invoke_with_schema(prompt, code_generation_schema)
            else:
                # Fallback to regular invoke for backward compatibility
                llm_response_str = self.llm_client.invoke(prompt)
            
            generated_code_map = json.loads(llm_response_str)

            if not isinstance(generated_code_map, dict):
                raise ValueError("LLM response is not a valid dictionary of file paths to code.")

        except NotImplementedError as e:
            msg = f"Cannot execute code generation: {e}"
            logger.critical(msg)
            return AgentResponse(success=False, message=msg)
        except (json.JSONDecodeError, ValueError) as e:
            msg = f"Failed to parse LLM response as valid JSON code map. Error: {e}"
            logger.error(msg)
            return AgentResponse(success=False, message=msg)
        except Exception as e:
            error_msg = f"An unexpected error occurred while calling the LLM for code generation: {e}"
            logger.critical(error_msg, exc_info=True)
            return AgentResponse(success=False, message=error_msg)

        # 4. Write the generated code to the workspace.
        written_files = []
        for file_path, code_content in generated_code_map.items():
            if not isinstance(code_content, str):
                logger.warning(f"Skipping file '{file_path}' due to invalid code content type.")
                continue
            
            try:
                context.workspace.write_file_content(file_path, code_content, current_task.task_id)
                written_files.append(file_path)
            except Exception as e:
                msg = f"Failed to write file '{file_path}' to workspace. Error: {e}"
                logger.error(msg, exc_info=True)
                # Return failure on the first write error.
                return AgentResponse(success=False, message=msg)

        if not written_files:
            return AgentResponse(success=False, message="LLM generated a response, but no valid files were written.")

        return AgentResponse(
            success=True,
            message=f"Successfully generated and wrote {len(written_files)} files to the workspace.",
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

    class TestCodeGenerationAgent(unittest.TestCase):

        def setUp(self):
            """Set up a clean environment for each test."""
            self.test_workspace_path = "./temp_coder_test_ws"
            if Path(self.test_workspace_path).exists():
                shutil.rmtree(self.test_workspace_path)
            
            self.mock_llm_client = MagicMock(spec=LLMClient)
            self.context = GlobalContext(workspace_path=self.test_workspace_path)
            self.code_task = TaskNode(goal="Implement API", assigned_agent="CodeGenerationAgent")
            self.agent = CodeGenerationAgent(llm_client=self.mock_llm_client)

            # Pre-populate context with required artifacts for most tests
            self.context.add_artifact("technical_spec.md", "Spec: Build an API.", "task_spec")
            self.context.add_artifact("file_manifest.json", {"files_to_create": ["src/main.py", "src/utils.py"]}, "task_manifest")

        def tearDown(self):
            """Clean up the environment after each test."""
            shutil.rmtree(self.test_workspace_path)

        def test_successful_code_generation(self):
            """Tests the ideal case where the LLM returns valid code for all files."""
            print("\n--- [Test Case 1: Successful Code Generation] ---")
            # Configure the mock LLM to return a valid code map.
            mock_code_output = json.dumps({
                "src/main.py": "import utils\n\nprint(utils.helper())",
                "src/utils.py": "def helper():\n    return 'Hello from helper'"
            })
            self.mock_llm_client.invoke.return_value = mock_code_output

            response = self.agent.execute(self.code_task.goal, self.context, self.code_task)

            self.mock_llm_client.invoke.assert_called_once()
            self.assertTrue(response.success)
            self.assertIn("Successfully generated and wrote 2 files", response.message)
            
            # Verify the files were actually written to the mock workspace.
            main_content = self.context.workspace.get_file_content("src/main.py")
            self.assertIn("import utils", main_content)
            logger.info("✅ test_successful_code_generation: PASSED")

        def test_failure_on_missing_artifacts(self):
            """Tests that the agent fails gracefully if prerequisites are not in the context."""
            print("\n--- [Test Case 2: Missing Artifacts] ---")
            empty_context = GlobalContext(workspace_path=self.test_workspace_path) # Use a fresh context
            
            response = self.agent.execute(self.code_task.goal, empty_context, self.code_task)

            self.assertFalse(response.success)
            self.assertIn("Missing required artifacts", response.message)
            self.mock_llm_client.invoke.assert_not_called() # LLM should not be called
            logger.info("✅ test_failure_on_missing_artifacts: PASSED")

        def test_llm_returns_invalid_json(self):
            """Tests how the agent handles a malformed JSON string from the LLM."""
            print("\n--- [Test Case 3: Invalid JSON from LLM] ---")
            self.mock_llm_client.invoke.return_value = "def main():\n  pass" # Not a JSON object

            response = self.agent.execute(self.code_task.goal, self.context, self.code_task)

            self.assertFalse(response.success)
            self.assertIn("Failed to parse LLM response", response.message)
            logger.info("✅ test_llm_returns_invalid_json: PASSED")

        def test_llm_returns_incomplete_map(self):
            """Tests when the LLM returns valid JSON but not for all requested files."""
            print("\n--- [Test Case 4: Incomplete Code Map from LLM] ---")
            # The manifest requests two files, but the LLM only returns one.
            mock_code_output = json.dumps({
                "src/main.py": "print('only one file')"
            })
            self.mock_llm_client.invoke.return_value = mock_code_output

            response = self.agent.execute(self.code_task.goal, self.context, self.code_task)

            self.assertTrue(response.success)
            self.assertIn("wrote 1 files", response.message) # Should still succeed with the files it got
            self.assertIsNotNone(self.context.workspace.get_file_content("src/main.py"))
            self.assertIsNone(self.context.workspace.get_file_content("src/utils.py")) # The other file should not exist
            logger.info("✅ test_llm_returns_incomplete_map: PASSED")

    # Run the tests
    unittest.main(argv=['first-arg-is-ignored'], exit=False)