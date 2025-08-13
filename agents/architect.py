# agents/architect.py
from .base import BaseAgent
from core.context import GlobalContext
from core.models import AgentResponse

class SpecGenerationAgent(BaseAgent):
    """
    Acts as the project's software architect. Its primary function is to
    translate a set of clarified user requirements into a formal technical
    specification. This document serves as the blueprint for the CodeGenerationAgent.
    """

    def __init__(self):
        super().__init__(
            name="SpecGenerationAgent",
            description="Creates detailed technical specifications and API definitions."
        )

    def execute(self, goal: str, context: GlobalContext) -> AgentResponse:
        """
        Executes the specification generation task.

        Inputs from Context:
        -   Reads the 'clarified_requirements.md' artifact.

        Outputs to Context:
        -   Produces 'technical_spec.md', detailing class structures, function
            signatures, logic flow, and database schemas.
        -   May also produce 'api_definitions.json' (e.g., an OpenAPI spec) for
            API-based projects.
        """
        pass

class CodeManifestAgent(BaseAgent):
    """
    Acts as the project's tech lead, responsible for planning the file structure.
    Based on the technical specification, it determines which files need to be
    created or modified, creating a clear roadmap for the coder.
    """

    def __init__(self):
        super().__init__(
            name="CodeManifestAgent",
            description="Defines the project's file and directory structure."
        )

    def execute(self, goal: str, context: GlobalContext) -> AgentResponse:
        """
        Executes the manifest creation task.

        Inputs from Context:
        -   Reads the 'technical_spec.md' artifact.

        Outputs to Context:
        -   Produces 'file_manifest.json', a structured list of file paths that
            need to be created or modified to implement the spec.
        """
        pass