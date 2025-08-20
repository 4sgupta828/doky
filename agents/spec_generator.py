# agents/spec_generator.py
import json
import logging
from typing import Dict, Any, List

# Foundational dependencies
from .base import BaseAgent
from core.context import GlobalContext
from core.models import AgentResult, TaskNode

# Import the externalized prompt builder
from prompts.spec_generator import build_spec_generation_prompt

# Get a logger instance for this module
logger = logging.getLogger(__name__)

class SpecGeneratorAgent(BaseAgent):
    """
    Acts as the project's software architect, creating a detailed technical
    specification from a set of clarified user requirements.
    """

    def __init__(self, llm_client: Any = None, ui_interface: Any = None):
        super().__init__(
            name="SpecGeneratorAgent", 
            description="Creates detailed technical specifications from user requirements."
        )
        self.llm_client = llm_client
        self.ui = ui_interface # Retained for potential future user validation steps

    def required_inputs(self) -> List[str]:
        """The SpecGeneratorAgent requires the clarified requirements."""
        return ["clarified_requirements"]

    def execute_v2(self, goal: str, inputs: Dict[str, Any], global_context: GlobalContext) -> AgentResult:
        """
        Generates a technical specification based on the provided requirements.
        """
        self.validate_inputs(inputs)
        
        requirements_doc = inputs["clarified_requirements"]
        context_summary = {"files_in_workspace": global_context.workspace.list_files()}

        self.report_progress("Generating technical specification", f"Based on: {requirements_doc[:100]}...")

        try:
            prompt = build_spec_generation_prompt(requirements_doc, context_summary)
            response_str = self.llm_client.invoke(prompt)
            spec_data = json.loads(response_str)

            if not isinstance(spec_data, dict) or "architecture_summary" not in spec_data:
                raise ValueError("LLM response did not conform to the required JSON schema for a technical spec.")

            # Compile the final, user-readable markdown document from the structured JSON
            technical_spec_content = self._compile_specification_document(spec_data)
            
            self.report_intermediate_output("technical_specification", technical_spec_content)

            return self.create_result(
                success=True,
                message="Successfully generated technical specification.",
                outputs={"technical_spec": technical_spec_content}
            )

        except (json.JSONDecodeError, ValueError) as e:
            return self.create_result(success=False, message=f"Failed to parse LLM response for specification. Error: {e}")
        except Exception as e:
            return self.create_result(success=False, message=f"An unexpected error occurred while generating specification: {e}")

    def _compile_specification_document(self, spec_data: Dict) -> str:
        """Compiles the final specification document in markdown format."""
        content = "# Technical Specification\n\n"
        
        content += f"## Architecture Summary\n\n{spec_data.get('architecture_summary', 'N/A')}\n\n"
        
        if spec_data.get('data_models'):
            content += "## Data Models\n\n"
            for model in spec_data['data_models']:
                content += f"### {model.get('name', 'Unknown Model')}\n\n"
                content += f"{model.get('description', '')}\n\n"
                if model.get('fields'):
                    content += "**Fields:**\n"
                    for field, f_type in model['fields'].items():
                        content += f"- `{field}`: {f_type}\n"
                    content += "\n"
                    
        if spec_data.get('api_endpoints'):
            content += "## API Endpoints\n\n"
            for endpoint in spec_data['api_endpoints']:
                content += f"### `{endpoint.get('method', 'GET')} {endpoint.get('path', '/unknown')}`\n\n"
                content += f"**Description**: {endpoint.get('description', '')}\n\n"
                if endpoint.get('request_schema'):
                    content += f"**Request**: {endpoint['request_schema']}\n\n"
                if endpoint.get('response_schema'):
                    content += f"**Response**: {endpoint['response_schema']}\n\n"
                    
        if spec_data.get('core_business_logic'):
            content += "## Core Business Logic\n\n"
            for rule in spec_data['core_business_logic']:
                content += f"- {rule}\n"
            content += "\n"
                
        if spec_data.get('technical_decisions'):
            content += "## Technical Decisions\n\n"
            for key, value in spec_data['technical_decisions'].items():
                content += f"- **{key.title()}:** {value}\n"
            content += "\n"
            
        return content
