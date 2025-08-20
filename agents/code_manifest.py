# agents/code_manifest.py
import json
import logging
from typing import Dict, Any, List

# Foundational dependencies
from .base import BaseAgent
from core.context import GlobalContext
from core.models import AgentResult, TaskNode

# Import the externalized prompt builder
from prompts.code_manifest import build_manifest_generation_prompt

# Get a logger instance for this module
logger = logging.getLogger(__name__)

class CodeManifestAgent(BaseAgent):
    """
    Acts as the project's tech lead, planning the file structure by converting
    a technical specification into a JSON manifest of file paths.
    """

    def __init__(self, llm_client: Any = None):
        super().__init__(
            name="CodeManifestAgent",
            description="Defines the project's file and directory structure from a technical spec."
        )
        self.llm_client = llm_client

    def required_inputs(self) -> List[str]:
        """The CodeManifestAgent requires a technical specification."""
        return ["technical_spec"]

    def execute_v2(self, goal: str, inputs: Dict[str, Any], global_context: GlobalContext) -> AgentResult:
        """
        Generates a file manifest from a technical specification.
        """
        self.validate_inputs(inputs)
        
        tech_spec = inputs["technical_spec"]
        context_summary = {"existing_files": global_context.workspace.list_files()}

        self.report_progress("Generating file manifest", f"Analyzing spec: {tech_spec[:100]}...")

        try:
            prompt = build_manifest_generation_prompt(tech_spec, context_summary)
            response_str = self.llm_client.invoke(prompt)
            manifest_data = json.loads(response_str)

            if "files_to_create" not in manifest_data or not isinstance(manifest_data["files_to_create"], list):
                raise ValueError("LLM response for manifest is missing 'files_to_create' list.")

            self.report_intermediate_output("file_manifest", manifest_data)

            return self.create_result(
                success=True,
                message=f"Successfully generated a manifest with {len(manifest_data['files_to_create'])} files.",
                outputs={"file_manifest": manifest_data}
            )

        except (json.JSONDecodeError, ValueError) as e:
            return self.create_result(success=False, message=f"Failed to parse LLM response for manifest. Error: {e}")
        except Exception as e:
            return self.create_result(success=False, message=f"An unexpected error occurred while generating manifest: {e}")
