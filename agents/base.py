# agents/base.py
from abc import ABC, abstractmethod
from core.context import GlobalContext
from core.models import AgentResponse

class BaseAgent(ABC):
    """
    The abstract base class for all specialized agents in the collective.
    It defines a standard interface that the Orchestrator can rely on, ensuring
    that any new agent can be seamlessly integrated into the system.
    """

    def __init__(self, name: str, description: str):
        """
        Initializes the agent with its identity.

        Args:
            name: The unique name of the agent (e.g., "PlannerAgent").
            description: A brief description of the agent's expertise and purpose.
        """
        self.name = name
        self.description = description

    @abstractmethod
    def execute(self, goal: str, context: GlobalContext) -> AgentResponse:
        """
        The primary method where the agent performs its specialized task.
        
        This method MUST be self-contained and idempotent where possible. It should
        rely solely on its 'goal' and the 'context' object for all its inputs and outputs.
        It should not hold its own state between calls.

        Args:
            goal: The specific sub-problem or objective this agent needs to solve.
            context: The shared GlobalContext object, providing access to the workspace,
                     artifacts, and the overall plan.

        Returns:
            An AgentResponse object detailing the outcome of its execution.
        """
        pass

    def get_capabilities(self) -> dict:
        """Returns a dictionary describing the agent's identity and skills."""
        return {"name": self.name, "description": self.description}