# agents/spec_generator.py
import logging
from typing import Any

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

class SpecGenerationAgent(BaseAgent):
    """
    Acts as the project's software architect, translating clarified requirements
    into a formal technical specification using a Large Language Model.
    """

    def __init__(self, llm_client: Any = None):
        super().__init__(
            name="SpecGenerationAgent",
            description="Creates detailed technical specifications and API definitions from requirements."
        )
        self.llm_client = llm_client or LLMClient()

    def _build_prompt(self, requirements: str, is_fallback: bool = False) -> str:
        """Constructs a precise prompt to guide the LLM in generating a technical spec."""
        
        if is_fallback:
            content_type = "user goal"
            content_description = "the following user goal into a detailed technical specification"
        else:
            content_type = "clarified requirements"  
            content_description = "the following clarified user requirements into a detailed technical specification"
            
        return f"""
        You are an expert software architect. Your task is to convert {content_description} in Markdown format.

        **{content_type.title()}:**
        ---
        {requirements}
        ---

        **Instructions:**
        1.  Define the necessary **Data Models** with fields, types, and constraints (e.g., `user_id: UUID, primary_key`).
        2.  Define the **API Endpoints** with HTTP methods, paths, request bodies, and success/error responses.
        3.  Describe the core **Business Logic** for each endpoint in clear, unambiguous terms.
        4.  The output MUST be a single, well-formatted Markdown document. Do not include any other text or commentary.
        
        {'' if not is_fallback else 'Note: Since only the user goal was provided, make reasonable assumptions about requirements and clearly document any assumptions made in the specification.'}
        """

    def execute(self, goal: str, context: GlobalContext, current_task: TaskNode) -> AgentResponse:
        logger.info(f"SpecGenerationAgent executing with goal: '{goal}'")

        input_artifact_key = current_task.input_artifact_keys[0] if current_task.input_artifact_keys else "clarified_requirements.md"
        requirements_doc = context.get_artifact(input_artifact_key)

        # Handle missing inputs gracefully - work with what we have
        is_fallback = False
        if not requirements_doc:
            # No clarified requirements available - work directly from the goal
            requirements_doc = f"User Goal: {goal}\n\nNote: No clarified requirements were available, so generating spec directly from the user's goal."
            is_fallback = True
            logger.info(f"No artifact '{input_artifact_key}' found. Working directly from goal: '{goal}'")
            context.log_event("spec_generator_fallback", {
                "reason": "missing_clarified_requirements", 
                "working_from": "goal_only",
                "expected_artifact": input_artifact_key
            })

        try:
            prompt = self._build_prompt(requirements_doc, is_fallback=is_fallback)
            technical_spec = self.llm_client.invoke(prompt)

            if not technical_spec or not isinstance(technical_spec, str):
                 return AgentResponse(success=False, message="LLM returned an empty or invalid spec.")

            output_artifact_key = "technical_spec.md"
            context.add_artifact(key=output_artifact_key, value=technical_spec, source_task_id=current_task.task_id)
            return AgentResponse(success=True, message="Successfully generated technical specification.", artifacts_generated=[output_artifact_key])

        except NotImplementedError:
            msg = "LLMClient is not implemented. Cannot generate specification."
            logger.critical(msg)
            return AgentResponse(success=False, message=msg)
        except Exception as e:
            error_msg = f"An unexpected error occurred while calling the LLM for spec generation: {e}"
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

    class TestSpecGenerationAgent(unittest.TestCase):

        def setUp(self):
            self.test_workspace_path = "./temp_spec_agent_test_ws"
            if Path(self.test_workspace_path).exists():
                shutil.rmtree(self.test_workspace_path)
            
            self.mock_llm_client = MagicMock(spec=LLMClient)
            self.context = GlobalContext(workspace_path=self.test_workspace_path)
            self.agent = SpecGenerationAgent(llm_client=self.mock_llm_client)
            self.task = TaskNode(
                goal="Create a spec",
                assigned_agent="SpecGenerationAgent",
                input_artifact_keys=["clarified_requirements.md"]
            )
            self.context.add_artifact("clarified_requirements.md", "User wants a test API.", "task_clarify")

        def tearDown(self):
            shutil.rmtree(self.test_workspace_path)

        def test_successful_spec_generation(self):
            print("\n--- [Test Case 1: SpecGenerationAgent Success] ---")
            mock_spec_output = "# Technical Specification: Mock API\n## Data Models\n- user_id (UUID)"
            self.mock_llm_client.invoke.return_value = mock_spec_output

            response = self.agent.execute(self.task.goal, self.context, self.task)

            self.mock_llm_client.invoke.assert_called_once()
            self.assertTrue(response.success)
            self.assertEqual(self.context.get_artifact("technical_spec.md"), mock_spec_output)
            logger.info("✅ test_successful_spec_generation: PASSED")

        def test_fallback_on_missing_artifact(self):
            print("\n--- [Test Case 2: SpecGenerationAgent Fallback Behavior] ---")
            mock_spec_output = "# Technical Specification: Fallback Generated\n## Based on User Goal Only"
            self.mock_llm_client.invoke.return_value = mock_spec_output
            
            empty_context = GlobalContext(workspace_path=self.test_workspace_path)
            response = self.agent.execute(self.task.goal, empty_context, self.task)

            # Agent should now succeed by using fallback behavior
            self.assertTrue(response.success)
            self.assertIn("Successfully generated", response.message)
            self.mock_llm_client.invoke.assert_called_once()
            
            # Check that the prompt included fallback content
            called_prompt = self.mock_llm_client.invoke.call_args[0][0]
            self.assertIn("User Goal:", called_prompt)
            self.assertIn("reasonable assumptions", called_prompt)
            
            logger.info("✅ test_fallback_on_missing_artifact: PASSED")

    unittest.main(argv=['first-arg-is-ignored'], exit=False)