# agents/planner.py
from .base import BaseAgent
from core.context import GlobalContext
from core.models import AgentResponse

class PlannerAgent(BaseAgent):
    """
    The master strategist of the collective. Its sole responsibility is to take a
    high-level goal and break it down into a structured, logical TaskGraph that
    can be executed by the other specialized agents. It is the first agent invoked
    in any mission.
    """

    def __init__(self):
        super().__init__(
            name="PlannerAgent",
            description="Decomposes high-level goals into a detailed, executable TaskGraph."
        )

    def execute(self, goal: str, context: GlobalContext) -> AgentResponse:
        """
        Generates a TaskGraph based on the provided mission goal.

        This involves:
        -   Using an LLM to reason about the goal.
        -   Breaking the goal into logical, dependent steps.
        -   Assigning the correct specialized agent to each step.
        -   Defining the data dependencies (input/output artifacts) for each task.
        -   Adding the completed TaskGraph to the GlobalContext.

        Args:
            goal: The high-level mission objective.
            context: The shared GlobalContext.

        Returns:
            An AgentResponse indicating whether the plan was successfully created.
        """
        pass