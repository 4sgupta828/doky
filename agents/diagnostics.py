# agents/diagnostics.py
from .base import BaseAgent
from core.context import GlobalContext
from core.models import AgentResponse

class DebuggingAgent(BaseAgent):
    """
    The team's expert troubleshooter. When a test fails, this agent is
    activated to perform a root-cause analysis. It examines the code, the
    error message, and the stack trace to pinpoint the source of the problem.
    """

    def __init__(self):
        super().__init__(
            name="DebuggingAgent",
            description="Analyzes failed tests and stack traces to find and suggest fixes."
        )

    def execute(self, goal: str, context: GlobalContext) -> AgentResponse:
        """
        Executes a debugging session.

        Inputs from Context:
        -   Reads 'failed_test_results.json' and 'stack_trace.txt' artifacts.
        -   May use the ContextBuilderAgent to read the problematic source code.

        Outputs to Context:
        -   Produces 'root_cause_analysis.md', explaining the bug.
        -   Produces 'suggested_fix.diff', a code patch to resolve the issue,
            which can then be applied by the CodeGenerationAgent.
        """
        pass

class ChiefQualityOfficerAgent(BaseAgent):
    """
    The team's quality and security supervisor. This agent runs a suite of
    advanced checks to ensure the codebase is not just functional, but also
    secure, maintainable, and high-quality.
    """

    def __init__(self):
        super().__init__(
            name="ChiefQualityOfficerAgent",
            description="Performs static analysis, dependency audits, and security scans."
        )

    def execute(self, goal: str, context: GlobalContext) -> AgentResponse:
        """
        Executes a quality/security audit.

        Inputs from Context:
        -   Analyzes the entire codebase in the workspace.

        Outputs to Context:
        -   Produces a 'quality_report.json' artifact detailing any found issues,
            such as security vulnerabilities, code smells, or dependency risks.
        """
        pass