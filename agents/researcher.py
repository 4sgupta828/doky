# agents/researcher.py
from .base import BaseAgent
from core.context import GlobalContext
from core.models import AgentResponse

class ContextBuilderAgent(BaseAgent):
    """
    The team's research specialist. When another agent needs to understand
    an existing piece of code before modifying it, it invokes this agent.
    The ContextBuilderAgent reads the relevant files and provides a concise
    summary or the full content for the requesting agent.
    """

    def __init__(self):
        super().__init__(
            name="ContextBuilderAgent",
            description="Gathers relevant code snippets to provide context for other agents."
        )

    def execute(self, goal: str, context: GlobalContext) -> AgentResponse:
        """
        Executes the context-gathering task.

        Inputs from Context:
        -   The 'goal' will specify what context is needed (e.g., "Get the content
            of the User model to add a new field").
        -   Uses the WorkspaceManager to read files.

        Outputs to Context:
        -   Produces a 'targeted_context.txt' artifact containing the requested
            code snippets and summaries.
        """
        pass

class ToolingAgent(BaseAgent):
    """
    The team's DevOps specialist. It is the only agent authorized to run
    command-line tools. This centralization ensures that all environment
    interactions are safe, logged, and controlled.
    """

    def __init__(self):
        super().__init__(
            name="ToolingAgent",
            description="Executes shell commands for build, dependency, and environment tasks."
        )

    def execute(self, goal: str, context: GlobalContext) -> AgentResponse:
        """
        Executes a given shell command.

        Inputs from Context:
        -   The 'goal' is the command to be executed (e.g., "pip install -r requirements.txt").

        Outputs to Context:
        -   Produces 'command_output.txt' with the stdout and stderr of the command.
        -   Produces 'command_exit_code' to indicate success or failure.
        """
        pass