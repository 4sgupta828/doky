# agents/clarifier.py
from .base import BaseAgent
from core.context import GlobalContext
from core.models import AgentResponse
from interfaces.collaboration_ui import CollaborationUI

class IntentClarificationAgent(BaseAgent):
    """
    The team's business analyst. It is responsible for the crucial first step:
    ensuring the user's initial request is clear, unambiguous, and complete
    before any planning or development begins. It acts as the primary interface
    between the user and the autonomous system.
    """

    def __init__(self):
        super().__init__(
            name="IntentClarificationAgent",
            description="Interacts with the user to refine goals and resolve ambiguity."
        )
        # This agent needs a way to communicate with the user.
        self.ui = CollaborationUI()

    def execute(self, goal: str, context: GlobalContext) -> AgentResponse:
        """
        Executes a clarification dialogue with the user.

        It will analyze the initial goal for vague terms or missing information.
        If ambiguity is found, it will use the CollaborationUI to ask the user
        targeted questions.

        Inputs from Context:
        -   Reads the initial 'mission_goal' from the user.

        Outputs to Context:
        -   Produces 'clarified_requirements.md', a detailed and unambiguous
            document that can be handed off to the SpecGenerationAgent.
        """
        pass