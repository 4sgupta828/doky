# agents/clarifier.py
import json
import logging
from typing import Dict, Any, List

# Foundational dependencies
from .base import BaseAgent
from core.context import GlobalContext
from core.models import AgentResult, TaskNode
from prompts.clarifier import build_clarification_prompt

# Get a logger instance for this module
logger = logging.getLogger(__name__)

class ClarifierAgent(BaseAgent):
    """
    The team's business analyst. It uses an LLM to analyze the user's request,
    presents its understanding, and validates its interpretation with the user.
    """

    def __init__(self, llm_client: Any = None, ui_interface: Any = None):
        super().__init__(
            name="ClarifierAgent",
            description="Presents understanding of user intent and validates assumptions with the user."
        )
        self.llm_client = llm_client
        self.ui = ui_interface

    def required_inputs(self) -> List[str]:
        return ["goal"]

    def execute_v2(self, goal: str, inputs: Dict[str, Any], global_context: GlobalContext) -> AgentResult:
        """
        Executes intent validation by presenting understanding to the user and
        getting their feedback to refine the interpretation.
        """
        self.validate_inputs(inputs)
        user_goal = inputs["goal"]
        
        if not self.ui:
            return self.create_result(success=False, message="UI interface is not available for user clarification.")

        self.report_progress("Validating intent", f"Analyzing goal: {user_goal[:80]}...")

        max_iterations = 3
        for i in range(max_iterations):
            try:
                prompt = build_clarification_prompt(user_goal)
                response_str = self.llm_client.invoke(prompt)
                understanding_data = json.loads(response_str)
                
                presentation = self._format_understanding_for_user(understanding_data)
                
                user_feedback = self.ui.prompt_for_input(
                    f"{presentation}\n\nPlease review my understanding. Press ENTER to approve, or provide feedback to refine it:"
                )

                if not user_feedback.strip():
                    logger.info("User approved the intent understanding.")
                    final_requirements = self._compile_requirements_document(user_goal, understanding_data)
                    return self.create_result(
                        success=True,
                        message="Successfully validated user intent.",
                        outputs={"clarified_requirements": final_requirements}
                    )
                else:
                    user_goal += f"\n\nUser Feedback (Iteration {i+1}): {user_feedback}"
                    self.report_thinking(f"Incorporating user feedback: {user_feedback[:100]}...")

            except (json.JSONDecodeError, KeyError) as e:
                return self.create_result(success=False, message=f"Failed to parse LLM response for clarification. Error: {e}")
            except Exception as e:
                return self.create_result(success=False, message=f"An unexpected error occurred during clarification: {e}")

        return self.create_result(success=False, message=f"Failed to get user approval after {max_iterations} iterations.")

    def _format_understanding_for_user(self, data: Dict) -> str:
        """Formats the LLM's understanding for user review."""
        formatted = "## My Understanding\n\n"
        formatted += f"**Intent:** {data.get('understanding', 'N/A')}\n\n"
        if data.get('core_functionality'):
            formatted += "**Core Features:**\n" + "\n".join(f"• {f}" for f in data['core_functionality']) + "\n"
        if data.get('technical_assumptions'):
            formatted += "\n**Technical Assumptions:**\n" + "\n".join(f"• **{k.title()}:** {v}" for k, v in data['technical_assumptions'].items()) + "\n"
        return formatted

    def _compile_requirements_document(self, original_goal: str, data: Dict) -> str:
        """Compiles the final, user-approved requirements document."""
        content = f"# Validated Requirements\n\n## Original Goal\n\n{original_goal}\n\n"
        content += f"## AI Understanding (User Validated)\n\n{self._format_understanding_for_user(data)}"
        return content
