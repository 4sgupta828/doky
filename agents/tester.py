# agents/tester.py
from .base import BaseAgent
from core.context import GlobalContext
from core.models import AgentResponse

class TestGenerationAgent(BaseAgent):
    """
    Specialized in writing tests for source code. It analyzes application code
    and generates corresponding unit, integration, or e2e tests.
    """

    def __init__(self):
        super().__init__(
            name="TestGenerationAgent",
            description="Generates unit and integration tests for application code."
        )

    def execute(self, goal: str, context: GlobalContext) -> AgentResponse:
        """Generates test files and writes them to the workspace."""
        pass

class TestRunnerAgent(BaseAgent):
    """
    Specialized in executing test suites and reporting the results.
    It uses a command-line tool (like pytest or jest) via the ToolingAgent
    to run tests and then parses the output.
    """

    def __init__(self):
        super().__init__(
            name="TestRunnerAgent",
            description="Executes test suites and reports results."
        )

    def execute(self, goal: str, context: GlobalContext) -> AgentResponse:
        """
        Runs tests and reports the outcome.

        If tests fail, it ensures the detailed failure report is saved as an artifact
        so the DebuggingAgent and AdaptiveEngine can analyze it.
        """
        pass