# agents/coder.py
from .base import BaseAgent
from core.context import GlobalContext
from core.models import AgentResponse

class CodeGenerationAgent(BaseAgent):
    """
    A specialized agent responsible for writing and modifying source code.
    It takes a detailed specification and a file manifest as input and produces
    the corresponding code, writing it to the workspace.
    """
    
    def __init__(self):
        super().__init__(
            name="CodeGenerationAgent",
            description="Writes, modifies, and refactors application code based on a spec."
        )

    def execute(self, goal: str, context: GlobalContext) -> AgentResponse:
        """
        Executes a coding task.

        This involves:
        -   Reading the technical specification from the context artifacts.
        -   Reading the file manifest to know which files to create or modify.
        -   Potentially reading existing code via the ContextBuilderAgent for context.
        -   Generating the new code using an LLM.
        -   Writing the generated code to the workspace via the WorkspaceManager.

        Args:
            goal: A specific coding task (e.g., "Implement the User model and login endpoint").
            context: The shared GlobalContext.

        Returns:
            An AgentResponse indicating the success or failure of the coding task.
        """
        pass