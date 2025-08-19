# agents/coder.py
import json
import logging
from typing import Dict, Any, List, Literal, Optional
from enum import Enum

# Foundational dependencies
from .base import BaseAgent
from core.context import GlobalContext
from core.models import AgentResult, TaskNode

# Get a logger instance for this module
logger = logging.getLogger(__name__)


# --- Code Quality Levels ---
class CodeQuality(Enum):
    """Defines different code quality levels for speed vs quality trade-offs."""
    FAST = "fast"
    DECENT = "decent"
    PRODUCTION = "production"


class CoderAgent(BaseAgent):
    """
    A specialized agent responsible for writing and modifying source code based on
    a detailed technical specification and a file manifest.
    """

    def __init__(self, llm_client: Any = None, default_quality: CodeQuality = CodeQuality.FAST):
        super().__init__(
            name="CoderAgent",
            description="Writes, modifies, and refactors application code based on a spec."
        )
        self.llm_client = llm_client
        self.default_quality = default_quality

    # --- V2 INTERFACE IMPLEMENTATION ---

    def required_inputs(self) -> List[str]:
        """The CoderAgent does not have strict required inputs, as it can fall back to the goal."""
        return []

    def optional_inputs(self) -> List[str]:
        """Optional inputs for the CoderAgent."""
        return ["technical_spec", "files_to_generate", "existing_code", "quality_level"]

    def execute_v2(self, goal: str, inputs: Dict[str, Any], global_context: GlobalContext) -> AgentResult:
        """
        Generates or modifies code based on a technical specification or a high-level goal.
        """
        logger.info(f"CoderAgent executing with goal: '{goal}'")
        self.validate_inputs(inputs) # Validates that no unknown inputs are passed

        # --- NEW: Flexible Input Handling ---
        # If a detailed spec isn't provided, use the goal as the primary instruction.
        tech_spec = inputs.get("technical_spec", goal)
        # If a file manifest isn't provided, the agent will ask the LLM to determine the files.
        files_to_generate = inputs.get("files_to_generate", [])
        # --- End of Flexible Input Handling ---

        existing_code = inputs.get("existing_code", {})
        quality_level_str = inputs.get("quality_level", self.default_quality.value)
        quality_level = CodeQuality(quality_level_str)

        self.report_progress("Generating code", f"Processing spec for {len(files_to_generate) if files_to_generate else 'inferred'} files...")
        self.report_thinking(f"I will generate {quality_level.value} quality code for the following files: {files_to_generate or 'To be determined by the LLM'}")

        try:
            prompt = self._build_prompt(files_to_generate, tech_spec, existing_code, quality_level)
            
            llm_response_str = self.llm_client.invoke(prompt)
            generated_code_map = json.loads(llm_response_str)

            if not isinstance(generated_code_map, dict) or not generated_code_map:
                raise ValueError("LLM returned empty or invalid code dictionary.")

            logger.info(f"LLM generated code for {len(generated_code_map)} files.")
            self.report_intermediate_output("code_files", generated_code_map)

            # Write the generated code to the workspace.
            written_files = []
            for file_path, code_content in generated_code_map.items():
                if not isinstance(code_content, str) or not code_content.strip():
                    logger.warning(f"Skipping empty or invalid code for file '{file_path}'")
                    continue
                
                global_context.workspace.write_file_content(file_path, code_content, "CoderAgent_task")
                written_files.append(file_path)

            if not written_files:
                return self.create_result(success=False, message="LLM generated code, but no valid files were written.")

            return self.create_result(
                success=True,
                message=f"Successfully generated and wrote {len(written_files)} files.",
                outputs={"artifacts_generated": written_files}
            )

        except (json.JSONDecodeError, ValueError) as e:
            msg = f"Failed to parse LLM response as valid JSON code map. Error: {e}"
            return self.create_result(success=False, message=msg)
        except Exception as e:
            error_msg = f"An unexpected error occurred during code generation: {e}"
            logger.critical(error_msg, exc_info=True)
            return self.create_result(success=False, message=error_msg)

    # --- PROMPT ENGINEERING ---

    def _get_quality_instructions(self, quality: CodeQuality) -> Dict[str, Any]:
        """Returns quality-specific instructions for the LLM."""
        quality_configs = {
            CodeQuality.FAST: {
                "description": "working Python code quickly",
                "instructions": [
                    "Focus on getting working code fast - don't over-engineer.",
                    "Minimal comments and basic error handling."
                ]
            },
            CodeQuality.DECENT: {
                "description": "clean, well-structured Python code",
                "instructions": [
                    "Write clean, readable code with reasonable comments.",
                    "Include basic error handling and validation."
                ]
            },
            CodeQuality.PRODUCTION: {
                "description": "production-quality, enterprise-ready Python code",
                "instructions": [
                    "Write robust, production-ready code with comprehensive error handling, detailed docstrings, and type hints.",
                    "Consider scalability, security, and maintainability."
                ]
            }
        }
        return quality_configs.get(quality, quality_configs[CodeQuality.DECENT])

    def _build_prompt(self, files_to_generate: List[str], spec: str, existing_code: Dict[str, str], quality: CodeQuality) -> str:
        """Constructs a detailed prompt to guide the LLM in generating code."""
        quality_config = self._get_quality_instructions(quality)
        existing_code_str = "\n".join(
            f"--- File: {path} ---\n```python\n{content}\n```"
            for path, content in existing_code.items()
        )

        if files_to_generate:
            files_section = f"**Files to Generate/Modify:**\n- {'\n- '.join(files_to_generate)}"
        else:
            files_section = "**Files to Generate/Modify:**\nYou must determine the appropriate file paths based on the specification (e.g., 'main.py', 'src/utils.py')."


        quality_instructions = "\n        ".join([f"- {inst}" for inst in quality_config["instructions"]])

        return f"""
        You are an expert software developer. Your task is to write {quality_config["description"]} based on the provided technical specification.
        
        **Code Quality Level: {quality.value.upper()}**
        
        **Technical Specification / Goal:**
        ---
        {spec}
        ---
        {files_section}

        **Existing Code for Context (if any):**
        ---
        {existing_code_str if existing_code else "No existing code provided. You are writing these files from scratch."}
        ---

        **Quality-Specific Instructions:**
        {quality_instructions}
        
        **Final Output Requirement:**
        Your output MUST be a single, valid JSON object where keys are the file paths and values are the complete code content as a string.

        **JSON Output Format Example:**
        {{
            "src/main.py": "def main():\\n    print('Hello')",
            "src/utils.py": "def helper():\\n    pass"
        }}

        Now, generate the code files.
        """
