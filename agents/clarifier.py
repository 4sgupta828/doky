# agents/clarifier.py
import json
import logging
from typing import List, Dict, Any

# Foundational dependencies
from .base import BaseAgent
from core.context import GlobalContext
from core.models import AgentResponse, TaskNode

# Get a logger instance for this module
logger = logging.getLogger(__name__)


# --- Real Integration Placeholders ---
# These classes define the contracts for external services. In a real system,
# their concrete implementations would live in a dedicated `integrations` directory.
class LLMClient:
    """A placeholder for a real LLM client (e.g., OpenAI, Gemini)."""
    def invoke(self, prompt: str) -> str:
        raise NotImplementedError("LLMClient.invoke must be implemented by a concrete class.")

class CollaborationUI:
    """A placeholder for a real user interface (e.g., CLI, Web UI)."""
    def prompt_for_input(self, question: str) -> str:
        raise NotImplementedError("CollaborationUI.prompt_for_input must be implemented.")

# --- Agent Implementation ---

class IntentClarificationAgent(BaseAgent):
    """
    The team's business analyst. It uses an LLM to analyze the user's request,
    identify ambiguities, and engage in a dialogue to produce a set of clear,
    actionable requirements.
    """

    def __init__(self, llm_client: Any = None, ui_interface: Any = None):
        """
        Initializes the agent with its dependencies.

        Args:
            llm_client: An object that handles calls to a Large Language Model.
            ui_interface: An object that handles interaction with the user.
        """
        super().__init__(
            name="IntentClarificationAgent",
            description="Interacts with the user to refine goals and resolve ambiguity."
        )
        self.llm_client = llm_client or LLMClient()
        self.ui = ui_interface or CollaborationUI()

    def _build_prompt(self, goal: str) -> str:
        """Constructs a prompt to ask the LLM to act as a requirements analyst."""
        return f"""
        You are an expert software requirements analyst. Your task is to analyze the
        following user goal and identify any ambiguities, unstated assumptions, or missing
        technical details that a developer would need to know before starting work.

        User Goal: "{goal}"

        Generate a list of critical, concise questions to ask the user. Focus on technical
        specifics, scope, data models, and key features.

        Your response MUST be a single, valid JSON array of strings, where each string
        is a question. For example: ["What database should be used?", "What is the expected response format?"].
        If the goal is perfectly clear and requires no clarification, return an empty array: [].
        """

    def execute(self, goal: str, context: GlobalContext, current_task: TaskNode) -> AgentResponse:
        """
        Executes a clarification dialogue. It uses an LLM to generate questions,
        presents them to the user, and compiles a structured requirements document.
        """
        logger.info(f"ClarificationAgent executing with goal: '{goal}'")
        
        # Report meaningful progress  
        self.report_progress("Clarifying requirements", f"Analyzing goal: '{goal[:80]}...'")

        # Define JSON schema for guaranteed structured response
        questions_schema = {
            "type": "object",
            "properties": {
                "questions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Array of clarification questions to ask the user"
                }
            },
            "required": ["questions"]
        }

        try:
            prompt = self._build_prompt(goal)
            
            # Use regular invoke by default, fall back to function calling if needed
            logger.debug("Using regular invoke method (primary approach)")
            response_str = self.llm_client.invoke(prompt)
            
            # Check if the response is valid JSON with actual content
            try:
                test_parse = json.loads(response_str)
                if not isinstance(test_parse, dict) or not test_parse.get('questions'):
                    raise ValueError("Invalid or empty response")
                logger.debug("Regular invoke succeeded with valid JSON response")
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Regular invoke failed to produce valid JSON ({e}), trying function calling")
                if hasattr(self.llm_client, 'invoke_with_schema'):
                    try:
                        response_str = self.llm_client.invoke_with_schema(prompt, questions_schema)
                        logger.debug("Function calling fallback succeeded")
                    except Exception as fallback_error:
                        logger.error(f"Function calling fallback also failed: {fallback_error}")
                        # Keep the original response from regular invoke for error reporting
                        pass
                else:
                    logger.warning("Function calling not available, keeping original response")
            
            response_data = json.loads(response_str)
            questions_to_ask = response_data.get("questions", []) if isinstance(response_data, dict) else response_data

            if not isinstance(questions_to_ask, list):
                raise TypeError("LLM response for questions was not a valid list.")

        except NotImplementedError as e:
            msg = f"Cannot execute clarification: {e}"
            logger.critical(msg)
            return AgentResponse(success=False, message=msg)
        except (json.JSONDecodeError, TypeError) as e:
            error_msg = f"Failed to get a valid list of questions from LLM. Response was not valid JSON. Error: {e}"
            logger.error(error_msg)
            return AgentResponse(success=False, message=error_msg)
        except Exception as e:
            error_msg = f"An unexpected error occurred while generating questions: {e}"
            logger.critical(error_msg, exc_info=True)
            return AgentResponse(success=False, message=error_msg)

        if not questions_to_ask:
            logger.info("LLM determined the goal is clear. No clarification needed.")
            requirements_content = f"# Requirements\n\n## Original Goal\n\n{goal}\n\n*This goal was deemed sufficiently clear by the AI analyst.*"
        else:
            logger.info(f"LLM generated {len(questions_to_ask)} clarification questions. Starting user dialogue.")
            
            clarifications = []
            try:
                for question in questions_to_ask:
                    answer = self.ui.prompt_for_input(question)
                    clarifications.append({"question": question, "answer": answer})
            except NotImplementedError as e:
                msg = f"Cannot get user input: {e}"
                logger.critical(msg)
                return AgentResponse(success=False, message=msg)
            except Exception as e:
                error_msg = f"An error occurred during user interaction: {e}"
                logger.error(error_msg, exc_info=True)
                return AgentResponse(success=False, message=error_msg)

            # Compile the results into a structured markdown document.
            requirements_content = f"# Clarified Requirements\n\n## Original Goal\n\n{goal}\n\n"
            requirements_content += "## AI-Generated Questions & User Answers\n\n"
            for item in clarifications:
                requirements_content += f"- **Question:** {item['question']}\n"
                requirements_content += f"  - **User's Answer:** {item['answer']}\n"
        
        output_artifact_key = "clarified_requirements.md"
        context.add_artifact(
            key=output_artifact_key,
            value=requirements_content,
            source_task_id=current_task.task_id
        )

        return AgentResponse(
            success=True,
            message="Successfully clarified user intent and generated requirements document.",
            artifacts_generated=[output_artifact_key]
        )


# --- Self-Testing Block ---
if __name__ == "__main__":
    import unittest
    from unittest.mock import MagicMock
    import shutil
    from pathlib import Path
    from utils.logger import setup_logger

    setup_logger(default_level=logging.INFO)

    class TestIntentClarificationAgent(unittest.TestCase):

        def setUp(self):
            """Set up a clean environment for each test."""
            self.test_workspace_path = "./temp_clarifier_test_ws"
            if Path(self.test_workspace_path).exists():
                shutil.rmtree(self.test_workspace_path)
            
            self.mock_llm_client = MagicMock(spec=LLMClient)
            self.mock_ui = MagicMock(spec=CollaborationUI)
            self.context = GlobalContext(workspace_path=self.test_workspace_path)
            self.clarify_task = TaskNode(goal="Test Goal", assigned_agent="IntentClarificationAgent")
            self.agent = IntentClarificationAgent(llm_client=self.mock_llm_client, ui_interface=self.mock_ui)

        def tearDown(self):
            """Clean up the environment after each test."""
            shutil.rmtree(self.test_workspace_path)

        def test_ambiguous_goal_clarification(self):
            """Tests the full flow for an ambiguous goal requiring user interaction."""
            print("\n--- [Test Case 1: Ambiguous Goal] ---")
            # Configure mock LLM to return questions
            questions = [
                "What authentication method should be used?",
                "What user data fields are required?"
            ]
            self.mock_llm_client.invoke.return_value = json.dumps(questions)

            # Configure mock UI to return user answers
            self.mock_ui.prompt_for_input.side_effect = ["JWT", "email and password_hash"]

            response = self.agent.execute("Build a user auth API", self.context, self.clarify_task)

            self.mock_llm_client.invoke.assert_called_once()
            self.assertEqual(self.mock_ui.prompt_for_input.call_count, 2)
            self.assertTrue(response.success)
            
            generated_doc = self.context.get_artifact("clarified_requirements.md")
            self.assertIn("User's Answer: JWT", generated_doc)
            self.assertIn("User's Answer: email and password_hash", generated_doc)
            logger.info("✅ test_ambiguous_goal_clarification: PASSED")

        def test_clear_goal_no_clarification(self):
            """Tests the flow for a clear goal where the LLM returns no questions."""
            print("\n--- [Test Case 2: Clear Goal] ---")
            # Configure mock LLM to return an empty list
            self.mock_llm_client.invoke.return_value = json.dumps([])

            response = self.agent.execute("Calculate Fibonacci sequence", self.context, self.clarify_task)

            self.mock_llm_client.invoke.assert_called_once()
            self.mock_ui.prompt_for_input.assert_not_called() # No user interaction should occur
            self.assertTrue(response.success)

            generated_doc = self.context.get_artifact("clarified_requirements.md")
            self.assertIn("required no further clarification", generated_doc)
            logger.info("✅ test_clear_goal_no_clarification: PASSED")

        def test_llm_returns_invalid_json(self):
            """Tests graceful failure when the LLM returns a malformed string."""
            print("\n--- [Test Case 3: Invalid JSON from LLM] ---")
            self.mock_llm_client.invoke.return_value = "this is not json"

            response = self.agent.execute("A goal", self.context, self.clarify_task)

            self.assertFalse(response.success)
            self.assertIn("Failed to get a valid list of questions", response.message)
            logger.info("✅ test_llm_returns_invalid_json: PASSED")
            
        def test_unimplemented_dependencies(self):
            """Tests that the agent fails gracefully if its dependencies are not implemented."""
            print("\n--- [Test Case 4: Unimplemented Dependencies] ---")
            # Using real, unimplemented clients
            agent_with_real_deps = IntentClarificationAgent()
            response = agent_with_real_deps.execute("A goal", self.context, self.clarify_task)
            
            self.assertFalse(response.success)
            self.assertIn("LLMClient.invoke must be implemented", response.message)
            logger.info("✅ test_unimplemented_dependencies: PASSED")

    # Run the tests
    unittest.main(argv=['first-arg-is-ignored'], exit=False)