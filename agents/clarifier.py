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

class IntentValidationAgent(BaseAgent):
    """
    The team's business analyst. It uses an LLM to analyze the user's request,
    presents its understanding of the intent, self-answers potential questions,
    and asks the user to validate, refine, or correct its interpretation.
    """

    def __init__(self, llm_client: Any = None, ui_interface: Any = None):
        """
        Initializes the agent with its dependencies.

        Args:
            llm_client: An object that handles calls to a Large Language Model.
            ui_interface: An object that handles interaction with the user.
        """
        super().__init__(
            name="IntentValidationAgent",
            description="Presents understanding of user intent and validates assumptions with the user."
        )
        self.llm_client = llm_client or LLMClient()
        self.ui = ui_interface or CollaborationUI()

    def _build_prompt(self, goal: str) -> str:
        """Constructs a prompt to ask the LLM to present its understanding and assumptions."""
        return f"""
        You are an expert software requirements analyst. Your task is to analyze the
        following user goal and present your understanding of what they want to build,
        including reasonable assumptions and technical choices you would make.

        User Goal: "{goal}"

        Present your understanding in a structured format that covers:
        1. Core functionality interpretation
        2. Technical assumptions (database, framework, deployment, etc.)
        3. Scope boundaries (what's included/excluded)
        4. Key features and requirements
        5. Potential choices or alternatives where multiple options exist

        Your response MUST be a JSON object with this structure:
        {{
            "understanding": "Clear summary of what you believe the user wants to build",
            "core_functionality": ["List", "of", "main", "features"],
            "technical_assumptions": {{
                "database": "your assumption with rationale",
                "framework": "your assumption with rationale",
                "deployment": "your assumption with rationale",
                "authentication": "your assumption with rationale"
            }},
            "scope": {{
                "included": ["List", "of", "what's", "included"],
                "excluded": ["List", "of", "what's", "not", "included"]
            }},
            "key_choices": [
                {{
                    "decision": "Choice that needs to be made",
                    "options": ["Option 1", "Option 2"],
                    "recommendation": "Your recommended choice with rationale"
                }}
            ]
        }}
        """

    def _format_understanding_for_user(self, understanding_data: Dict) -> str:
        """Formats the LLM's understanding for user review."""
        formatted = "## My Understanding\n\n"
        formatted += f"**What you want to build:** {understanding_data.get('understanding', 'N/A')}\n\n"
        
        if understanding_data.get('core_functionality'):
            formatted += "**Core Features:**\n"
            for feature in understanding_data['core_functionality']:
                formatted += f"• {feature}\n"
            formatted += "\n"
            
        if understanding_data.get('technical_assumptions'):
            formatted += "**Technical Assumptions I'm Making:**\n"
            for key, value in understanding_data['technical_assumptions'].items():
                formatted += f"• **{key.title()}:** {value}\n"
            formatted += "\n"
            
        if understanding_data.get('scope'):
            scope = understanding_data['scope']
            if scope.get('included'):
                formatted += "**Included in Scope:**\n"
                for item in scope['included']:
                    formatted += f"• {item}\n"
                formatted += "\n"
            if scope.get('excluded'):
                formatted += "**Not Included (Out of Scope):**\n"
                for item in scope['excluded']:
                    formatted += f"• {item}\n"
                formatted += "\n"
                
        if understanding_data.get('key_choices'):
            formatted += "**Key Decisions & My Recommendations:**\n"
            for choice in understanding_data['key_choices']:
                formatted += f"• **{choice.get('decision', 'Decision')}**\n"
                if choice.get('options'):
                    formatted += f"  - Options: {', '.join(choice['options'])}\n"
                if choice.get('recommendation'):
                    formatted += f"  - My recommendation: {choice['recommendation']}\n"
            formatted += "\n"
            
        return formatted

    def _compile_requirements_document(self, original_goal: str, understanding_data: Dict) -> str:
        """Compiles the final requirements document."""
        content = "# Validated Requirements\n\n"
        content += f"## Original Goal\n\n{original_goal}\n\n"
        content += "## AI Understanding (User Validated)\n\n"
        content += f"**Intent:** {understanding_data.get('understanding', 'N/A')}\n\n"
        
        if understanding_data.get('core_functionality'):
            content += "### Core Functionality\n\n"
            for feature in understanding_data['core_functionality']:
                content += f"- {feature}\n"
            content += "\n"
            
        if understanding_data.get('technical_assumptions'):
            content += "### Technical Assumptions\n\n"
            for key, value in understanding_data['technical_assumptions'].items():
                content += f"- **{key.title()}:** {value}\n"
            content += "\n"
            
        if understanding_data.get('scope'):
            scope = understanding_data['scope']
            content += "### Scope\n\n"
            if scope.get('included'):
                content += "**Included:**\n"
                for item in scope['included']:
                    content += f"- {item}\n"
                content += "\n"
            if scope.get('excluded'):
                content += "**Excluded:**\n"
                for item in scope['excluded']:
                    content += f"- {item}\n"
                content += "\n"
                
        if understanding_data.get('key_choices'):
            content += "### Key Decisions\n\n"
            for choice in understanding_data['key_choices']:
                content += f"- **{choice.get('decision', 'Decision')}:** {choice.get('recommendation', 'TBD')}\n"
            content += "\n"
            
        return content

    def execute(self, goal: str, context: GlobalContext, current_task: TaskNode) -> AgentResponse:
        """
        Executes intent validation by presenting understanding to the user and
        getting their feedback to refine the interpretation.
        """
        logger.info(f"IntentValidationAgent executing with goal: '{goal}'")
        
        # Report meaningful progress  
        self.report_progress("Validating intent", f"Analyzing goal: '{goal[:80]}...'")

        # Define JSON schema for guaranteed structured response
        understanding_schema = {
            "type": "object",
            "properties": {
                "understanding": {"type": "string"},
                "core_functionality": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "technical_assumptions": {
                    "type": "object",
                    "properties": {
                        "database": {"type": "string"},
                        "framework": {"type": "string"},
                        "deployment": {"type": "string"},
                        "authentication": {"type": "string"}
                    }
                },
                "scope": {
                    "type": "object",
                    "properties": {
                        "included": {
                            "type": "array",
                            "items": {"type": "string"}
                        },
                        "excluded": {
                            "type": "array", 
                            "items": {"type": "string"}
                        }
                    }
                },
                "key_choices": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "decision": {"type": "string"},
                            "options": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "recommendation": {"type": "string"}
                        }
                    }
                }
            },
            "required": ["understanding", "core_functionality"]
        }

        # Validation loop - continues until user approves or provides empty input
        max_iterations = 3  # Prevent infinite loops
        iteration = 0
        
        while iteration < max_iterations:
            try:
                prompt = self._build_prompt(goal)
                
                # Use regular invoke by default, fall back to function calling if needed
                logger.debug("Using regular invoke method (primary approach)")
                response_str = self.llm_client.invoke(prompt)
                
                # Check if the response is valid JSON with actual content
                try:
                    test_parse = json.loads(response_str)
                    if not isinstance(test_parse, dict) or not test_parse.get('understanding'):
                        raise ValueError("Invalid or empty response")
                    logger.debug("Regular invoke succeeded with valid JSON response")
                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning(f"Regular invoke failed to produce valid JSON ({e}), trying function calling")
                    if hasattr(self.llm_client, 'invoke_with_schema'):
                        try:
                            response_str = self.llm_client.invoke_with_schema(prompt, understanding_schema)
                            logger.debug("Function calling fallback succeeded")
                        except Exception as fallback_error:
                            logger.error(f"Function calling fallback also failed: {fallback_error}")
                            # Keep the original response from regular invoke for error reporting
                            pass
                    else:
                        logger.warning("Function calling not available, keeping original response")
                
                understanding_data = json.loads(response_str)
                
                # Present the understanding to the user
                presentation = self._format_understanding_for_user(understanding_data)
                
                try:
                    # Ask user for validation/feedback - empty input means approval
                    user_feedback = self.ui.prompt_for_input(
                        f"{presentation}\n\n"
                        "Please review my understanding above. Press ENTER to approve and continue, "
                        "or provide feedback/corrections to refine my interpretation:"
                    )
                    
                    # Empty input (just Enter) means approval
                    if not user_feedback or user_feedback.strip() == "":
                        logger.info("User approved the intent understanding")
                        break
                    else:
                        # User provided feedback - update the goal for next iteration
                        logger.info(f"User provided feedback: {user_feedback[:100]}...")
                        goal = f"{goal}\n\nUser feedback: {user_feedback}"
                        iteration += 1
                        continue
                        
                except NotImplementedError as e:
                    msg = f"Cannot get user input: {e}"
                    logger.critical(msg)
                    return AgentResponse(success=False, message=msg)
                except Exception as e:
                    error_msg = f"An error occurred during user interaction: {e}"
                    logger.error(error_msg, exc_info=True)
                    return AgentResponse(success=False, message=error_msg)
                    
            except NotImplementedError as e:
                msg = f"Cannot execute validation: {e}"
                logger.critical(msg)
                return AgentResponse(success=False, message=msg)
            except (json.JSONDecodeError, TypeError) as e:
                error_msg = f"Failed to get valid understanding from LLM. Response was not valid JSON. Error: {e}"
                logger.error(error_msg)
                return AgentResponse(success=False, message=error_msg)
            except Exception as e:
                error_msg = f"An unexpected error occurred while generating understanding: {e}"
                logger.critical(error_msg, exc_info=True)
                return AgentResponse(success=False, message=error_msg)

        if iteration >= max_iterations:
            logger.warning("Max iterations reached in validation loop")
            
        # Compile the final requirements document
        requirements_content = self._compile_requirements_document(goal, understanding_data)
        
        output_artifact_key = "clarified_requirements.md"
        context.add_artifact(
            key=output_artifact_key,
            value=requirements_content,
            source_task_id=current_task.task_id
        )

        return AgentResponse(
            success=True,
            message="Successfully validated user intent and generated requirements document.",
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

    class TestIntentValidationAgent(unittest.TestCase):

        def setUp(self):
            """Set up a clean environment for each test."""
            self.test_workspace_path = "./temp_clarifier_test_ws"
            if Path(self.test_workspace_path).exists():
                shutil.rmtree(self.test_workspace_path)
            
            self.mock_llm_client = MagicMock(spec=LLMClient)
            self.mock_ui = MagicMock(spec=CollaborationUI)
            self.context = GlobalContext(workspace_path=self.test_workspace_path)
            self.clarify_task = TaskNode(goal="Test Goal", assigned_agent="IntentValidationAgent")
            self.agent = IntentValidationAgent(llm_client=self.mock_llm_client, ui_interface=self.mock_ui)

        def tearDown(self):
            """Clean up the environment after each test."""
            shutil.rmtree(self.test_workspace_path)

        def test_user_approves_understanding(self):
            """Tests the flow where user approves the agent's understanding on first try."""
            print("\n--- [Test Case 1: User Approves Understanding] ---")
            # Configure mock LLM to return understanding
            understanding = {
                "understanding": "A REST API for user authentication",
                "core_functionality": ["User registration", "User login", "JWT token management"],
                "technical_assumptions": {
                    "database": "PostgreSQL for user data storage",
                    "framework": "FastAPI for rapid development",
                    "deployment": "Docker containers",
                    "authentication": "JWT tokens with refresh mechanism"
                }
            }
            self.mock_llm_client.invoke.return_value = json.dumps(understanding)

            # Configure mock UI to return empty string (approval)
            self.mock_ui.prompt_for_input.return_value = ""

            response = self.agent.execute("Build a user auth API", self.context, self.clarify_task)

            self.mock_llm_client.invoke.assert_called_once()
            self.mock_ui.prompt_for_input.assert_called_once()
            self.assertTrue(response.success)
            
            generated_doc = self.context.get_artifact("clarified_requirements.md")
            self.assertIn("REST API for user authentication", generated_doc)
            self.assertIn("JWT tokens", generated_doc)
            logger.info("✅ test_user_approves_understanding: PASSED")

        def test_user_provides_feedback(self):
            """Tests the flow where user provides feedback to refine understanding."""
            print("\n--- [Test Case 2: User Provides Feedback] ---")
            # Configure mock LLM to return different responses for iterations
            understanding1 = {
                "understanding": "A simple calculator function",
                "core_functionality": ["Calculate Fibonacci numbers"]
            }
            understanding2 = {
                "understanding": "An optimized Fibonacci calculator with memoization",
                "core_functionality": ["Calculate Fibonacci numbers", "Memoization for performance", "Support large numbers"]
            }
            self.mock_llm_client.invoke.side_effect = [
                json.dumps(understanding1),
                json.dumps(understanding2)
            ]

            # First call: user provides feedback, second call: user approves  
            self.mock_ui.prompt_for_input.side_effect = [
                "I need it to be optimized and handle large numbers",
                ""  # Empty string means approval
            ]

            response = self.agent.execute("Calculate Fibonacci sequence", self.context, self.clarify_task)

            self.assertEqual(self.mock_llm_client.invoke.call_count, 2)
            self.assertEqual(self.mock_ui.prompt_for_input.call_count, 2) 
            self.assertTrue(response.success)

            generated_doc = self.context.get_artifact("clarified_requirements.md")
            self.assertIn("optimized", generated_doc)
            self.assertIn("memoization", generated_doc)
            logger.info("✅ test_user_provides_feedback: PASSED")

        def test_llm_returns_invalid_json(self):
            """Tests graceful failure when the LLM returns a malformed string."""
            print("\n--- [Test Case 3: Invalid JSON from LLM] ---")
            self.mock_llm_client.invoke.return_value = "this is not json"

            response = self.agent.execute("A goal", self.context, self.clarify_task)

            self.assertFalse(response.success)
            self.assertIn("Failed to get valid understanding", response.message)
            logger.info("✅ test_llm_returns_invalid_json: PASSED")
            
        def test_unimplemented_dependencies(self):
            """Tests that the agent fails gracefully if its dependencies are not implemented."""
            print("\n--- [Test Case 4: Unimplemented Dependencies] ---")
            # Using real, unimplemented clients
            agent_with_real_deps = IntentValidationAgent()
            response = agent_with_real_deps.execute("A goal", self.context, self.clarify_task)
            
            self.assertFalse(response.success)
            self.assertIn("LLMClient.invoke must be implemented", response.message)
            logger.info("✅ test_unimplemented_dependencies: PASSED")

    # Run the tests
    unittest.main(argv=['first-arg-is-ignored'], exit=False)