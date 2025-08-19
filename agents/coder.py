# agents/coder.py
import json
import logging
from typing import Dict, Any, List, Literal, Optional
from enum import Enum

# Foundational dependencies
from .base import BaseAgent
from core.context import GlobalContext
from core.models import AgentResponse, TaskNode

# Get a logger instance for this module
logger = logging.getLogger(__name__)


# --- Code Quality Levels ---
class CodeQuality(Enum):
    """Defines different code quality levels for speed vs quality trade-offs."""
    FAST = "fast"          # Quick, working code - prioritizes speed
    DECENT = "decent"      # Balanced approach - good quality, reasonable speed (default)
    PRODUCTION = "production"  # High-quality, well-documented, robust code


class ExecutionMode(Enum):
    """Defines different execution modes for the code generation agent."""
    NEW_CODE = "new_code"        # Generate new code from specifications
    DESIGN_UPDATE = "design_update"  # Update existing code based on design changes
    REFACTOR = "refactor"        # Refactor existing code while preserving functionality


# --- Real LLM Integration Placeholder ---
class LLMClient:
    """A placeholder for a real LLM client (e.g., OpenAI, Gemini)."""
    def invoke(self, prompt: str) -> str:
        raise NotImplementedError("LLMClient.invoke must be implemented by a concrete class.")

# --- Agent Implementation ---

class CoderAgent(BaseAgent):
    """
    A specialized agent responsible for writing and modifying source code.
    It takes a detailed technical specification and a file manifest as input and
    produces the corresponding code, writing it to the workspace.
    """

    def __init__(self, llm_client: Any = None, default_quality: CodeQuality = CodeQuality.FAST):
        super().__init__(
            name="CoderAgent",
            description="Writes, modifies, and refactors application code based on a spec."
        )
        self.llm_client = llm_client or LLMClient()
        self.default_quality = default_quality

    def _get_quality_instructions(self, quality: CodeQuality) -> Dict[str, str]:
        """Returns quality-specific instructions and descriptions."""
        quality_configs = {
            CodeQuality.FAST: {
                "description": "working Python code quickly",
                "instructions": [
                    "Focus on getting working code fast - don't over-engineer",
                    "Use simple, straightforward implementations", 
                    "Minimal comments - only where absolutely necessary",
                    "Basic error handling (try/except where critical)",
                    "Prioritize functionality over optimization",
                    "Use standard library when possible to avoid dependencies"
                ]
            },
            CodeQuality.DECENT: {
                "description": "clean, well-structured Python code",
                "instructions": [
                    "Write clean, readable code with good structure",
                    "Include reasonable comments for complex logic",
                    "Use appropriate data structures and patterns",
                    "Add basic error handling and validation",
                    "Follow Python conventions (PEP 8 style)",
                    "Balance between speed and maintainability"
                ]
            },
            CodeQuality.PRODUCTION: {
                "description": "production-quality, enterprise-ready Python code",
                "instructions": [
                    "Write robust, production-ready code with comprehensive error handling",
                    "Include detailed docstrings and inline comments",
                    "Implement proper logging and monitoring hooks",
                    "Add input validation and edge case handling",
                    "Use type hints and follow strict PEP 8 compliance",
                    "Consider scalability, security, and maintainability",
                    "Include configuration management and environment handling"
                ]
            }
        }
        return quality_configs[quality]
    
    def _detect_quality_level(self, goal: str, context: GlobalContext) -> CodeQuality:
        """Detects the desired code quality level from the goal and context."""
        goal_lower = goal.lower()
        
        # Check for explicit quality keywords in the goal
        if any(keyword in goal_lower for keyword in ['fast', 'quick', 'rapid', 'prototype', 'draft']):
            logger.info("Detected FAST quality level from goal keywords")
            return CodeQuality.FAST
        elif any(keyword in goal_lower for keyword in ['decent', 'clean', 'readable', 'maintainable', 'structured', 'balanced']):
            logger.info("Detected DECENT quality level from goal keywords")
            return CodeQuality.DECENT
        elif any(keyword in goal_lower for keyword in ['production', 'enterprise', 'robust', 'comprehensive', 'high.quality']):
            logger.info("Detected PRODUCTION quality level from goal keywords")
            return CodeQuality.PRODUCTION
        
        # Check context for quality preferences (could be set by user commands or previous tasks)
        if hasattr(context, 'code_quality_preference'):
            return context.code_quality_preference
            
        # Default to FAST for speed optimization
        logger.info("Using default FAST quality level")
        return self.default_quality

    def _detect_execution_mode(self, goal: str, context: GlobalContext) -> ExecutionMode:
        """Detects the execution mode based on goal and available artifacts."""
        goal_lower = goal.lower()
        
        # Check for design update keywords
        if any(keyword in goal_lower for keyword in ['design update', 'design change', 'architectural change', 'redesign']):
            logger.info("Detected DESIGN_UPDATE mode from goal keywords")
            return ExecutionMode.DESIGN_UPDATE
        
        # Check for refactor keywords  
        if any(keyword in goal_lower for keyword in ['refactor', 'restructure', 'reorganize', 'improve design']):
            logger.info("Detected REFACTOR mode from goal keywords")
            return ExecutionMode.REFACTOR
        
        # Check if we have design change artifacts from debugging agent
        if context.get_artifact("design_change_request.json"):
            logger.info("Detected DESIGN_UPDATE mode from design change artifact")
            return ExecutionMode.DESIGN_UPDATE
        
        # Default to new code generation
        logger.info("Using default NEW_CODE mode")
        return ExecutionMode.NEW_CODE
    
    def _build_prompt(self, files_to_generate: List[str], spec: str, existing_code: Dict[str, str], quality: CodeQuality = None) -> str:
        """
        Constructs a detailed prompt to guide the LLM in generating code for a
        specific set of files, or to determine appropriate files if none specified.
        """
        if quality is None:
            quality = self.default_quality
            
        quality_config = self._get_quality_instructions(quality)
        existing_code_str = "\n".join(
            f"--- File: {path} ---\n```python\n{content}\n```"
            for path, content in existing_code.items()
        )
        
        if files_to_generate:
            files_section = f"""
        **Files to Generate/Modify:**
        - {"\n- ".join(files_to_generate)}"""
        else:
            files_section = """
        **Files to Generate/Modify:**
        No specific files specified. Please determine the appropriate file(s) to create based on the specification."""

        # Build quality-specific instructions
        quality_instructions = "\n        ".join([f"{i+1}.  {instruction}" for i, instruction in enumerate(quality_config["instructions"])])
        
        return f"""
        You are an expert software developer. Your task is to write {quality_config["description"]}
        based on the provided technical specification.
        
        **Code Quality Level: {quality.value.upper()}**
        
        **Technical Specification:**
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
        
        **General Requirements:**
        - Adhere strictly to the technical specification
        - Ensure all necessary imports are included in each file
        - Choose appropriate filenames if none were specified (e.g., "main.py", "utils.py", etc.)
        - You MUST generate at least one file with actual code content
        - Your output MUST be a single, valid JSON object with file paths as keys and code content as values
        - Do NOT return an empty dictionary {{}} - always generate at least one file

        **JSON Output Format Example:**
        {{
            "main.py": "def add_numbers(a, b):\\n    return a + b\\n\\nif __name__ == '__main__':\\n    print(add_numbers(2, 3))",
            "utils.py": "def helper_function():\\n    pass"
        }}

        **IMPORTANT:** You must generate actual code files. An empty response {{}} is not acceptable.
        
        Now, generate the appropriate code files based on the specification.
        """

    def _build_design_update_prompt(self, design_change_request: Dict[str, Any], existing_code: Dict[str, str], quality: CodeQuality) -> str:
        """
        Constructs a specialized prompt for design updates based on evidence and reasoning.
        """
        quality_config = self._get_quality_instructions(quality)
        existing_code_str = "\n".join(
            f"--- File: {path} ---\n```python\n{content}\n```"
            for path, content in existing_code.items()
        )
        
        # Extract design change details
        root_cause = design_change_request.get("root_cause_analysis", "Unknown issue")
        design_problem = design_change_request.get("design_problem", "Design issue identified")
        recommended_changes = design_change_request.get("recommended_changes", [])
        evidence = design_change_request.get("evidence", {})
        
        changes_text = "\n".join([f"- {change}" for change in recommended_changes])
        
        return f"""
        You are an expert software architect and developer. You need to update existing code
        based on a well-reasoned design analysis from a debugging investigation.

        **DESIGN ANALYSIS REPORT:**
        ---
        Root Cause: {root_cause}
        
        Design Problem: {design_problem}
        
        Recommended Changes:
        {changes_text}
        
        Evidence:
        {json.dumps(evidence, indent=2)}
        ---

        **EXISTING CODE TO UPDATE:**
        ---
        {existing_code_str if existing_code else "No existing code provided."}
        ---

        **Code Quality Level: {quality.value.upper()}**
        
        **Quality-Specific Instructions:**
        {chr(10).join([f"{i+1}.  {instruction}" for i, instruction in enumerate(quality_config["instructions"])])}
        
        **DESIGN UPDATE REQUIREMENTS:**
        - Implement the recommended design changes precisely
        - Preserve existing functionality unless explicitly changing it
        - Follow the evidence-based reasoning provided
        - Ensure the changes address the root cause identified
        - Maintain code consistency and style with existing codebase
        - Add appropriate comments explaining the design changes
        - Your output MUST be a single, valid JSON object with file paths as keys and updated code content as values
        
        **JSON Output Format Example:**
        {{
            "src/module.py": "# Updated design\\nclass ImprovedDesign:\\n    def __init__(self):\\n        # Design change: explanation\\n        pass",
            "src/utils.py": "# Supporting changes\\ndef new_helper_function():\\n    return 'updated'"
        }}

        **IMPORTANT:** Generate the updated code that implements the design changes based on the evidence and analysis provided.
        
        Now, implement the design updates.
        """

    def _handle_design_update(self, goal: str, context: GlobalContext, current_task: TaskNode) -> AgentResponse:
        """Handle design update execution mode with evidence-based changes."""
        logger.info("Executing design update mode")
        self.report_progress("Design update mode", "Processing evidence-based design changes")
        
        # 1. Get design change request from debugging agent
        design_change_request = context.get_artifact("design_change_request.json")
        if not design_change_request:
            return AgentResponse(
                success=False,
                message="No design change request found. Expected 'design_change_request.json' artifact from DebuggingAgent."
            )
        
        if isinstance(design_change_request, str):
            try:
                design_change_request = json.loads(design_change_request)
            except json.JSONDecodeError as e:
                return AgentResponse(
                    success=False,
                    message=f"Invalid design change request JSON: {e}"
                )
        
        self.report_intermediate_output("design_change_request", design_change_request)
        
        # 2. Get files to modify
        files_to_modify = design_change_request.get("files_to_modify", [])
        if not files_to_modify:
            return AgentResponse(
                success=False,
                message="No files specified for modification in design change request."
            )
        
        # 3. Load existing code for the files to modify
        existing_code = {}
        for file_path in files_to_modify:
            content = context.workspace.get_file_content(file_path)
            if content:
                existing_code[file_path] = content
            else:
                logger.warning(f"File {file_path} not found in workspace")
        
        if not existing_code:
            return AgentResponse(
                success=False,
                message="No existing code found for the specified files to modify."
            )
        
        self.report_thinking(f"Loaded {len(existing_code)} existing files for design update")
        
        # 4. Detect quality level
        quality_level = self._detect_quality_level(goal, context)
        logger.info(f"Using code quality level: {quality_level.value.upper()}")
        self.complete_step(f"Quality level: {quality_level.value.upper()}")
        
        # 5. Generate design update prompt and get LLM response
        try:
            self.report_progress("Building design update prompt", "Creating evidence-based update instructions")
            prompt = self._build_design_update_prompt(design_change_request, existing_code, quality_level)
            self.complete_step("Design update prompt constructed")
            
            logger.debug("Using regular invoke for design update")
            llm_response_str = self.llm_client.invoke(prompt)
            
            # Parse and validate response
            response_data = json.loads(llm_response_str)
            if not isinstance(response_data, dict):
                raise ValueError("LLM response is not a valid dictionary.")
            
            # Extract the files from response (handle both direct format and structured format)
            if "files" in response_data:
                updated_code_map = response_data["files"]
            else:
                updated_code_map = response_data
            
            if not isinstance(updated_code_map, dict) or not updated_code_map:
                raise ValueError("No valid code updates returned from LLM.")
            
            logger.info(f"LLM generated design updates for {len(updated_code_map)} files: {list(updated_code_map.keys())}")
            
            # Display the updated code
            if len(updated_code_map) == 1:
                file_path, content = next(iter(updated_code_map.items()))
                code_with_filename = {"content": content, "filename": file_path}
                self.report_intermediate_output("updated_code_snippet", code_with_filename)
            else:
                self.report_intermediate_output("updated_code_files", updated_code_map)
                
        except (json.JSONDecodeError, ValueError) as e:
            msg = f"Failed to parse LLM response for design update. Error: {e}"
            logger.error(msg)
            return AgentResponse(success=False, message=msg)
        except Exception as e:
            error_msg = f"Design update generation failed: {e}"
            logger.error(error_msg, exc_info=True)
            return AgentResponse(success=False, message=error_msg)
        
        # 6. Write the updated code to workspace
        written_files = []
        modified_files = {}
        
        for file_path, code_content in updated_code_map.items():
            if not isinstance(code_content, str) or not code_content.strip():
                logger.warning(f"Skipping invalid code for file '{file_path}'")
                continue
            
            try:
                # Get original content for diff
                original_content = context.workspace.get_file_content(file_path)
                
                # Write updated content
                context.workspace.write_file_content(file_path, code_content, current_task.task_id)
                written_files.append(file_path)
                logger.info(f"Successfully updated file '{file_path}' with design changes")
                
                # Track for diff display
                if original_content and original_content != code_content:
                    modified_files[file_path] = {
                        "old": original_content,
                        "new": code_content
                    }
                    
            except Exception as e:
                msg = f"Failed to write updated file '{file_path}' to workspace. Error: {e}"
                logger.error(msg, exc_info=True)
                return AgentResponse(success=False, message=msg)
        
        if not written_files:
            return AgentResponse(success=False, message="No files were successfully updated.")
        
        # Display diff for modified files
        if modified_files:
            self.report_intermediate_output("design_update_diff", modified_files)
            
        return AgentResponse(
            success=True,
            message=f"Successfully applied design updates to {len(written_files)} files based on debugging analysis.",
            artifacts_generated=written_files
        )

    def execute(self, goal: str, context: GlobalContext, current_task: TaskNode) -> AgentResponse:
        # Temporarily enable DEBUG logging for this agent
        original_level = logger.level
        logger.setLevel(logging.DEBUG)
        
        # Also enable DEBUG for the real_llm_client logger
        llm_logger = logging.getLogger('real_llm_client')
        original_llm_level = llm_logger.level
        llm_logger.setLevel(logging.DEBUG)
        
        # Also try the module name logger
        module_logger = logging.getLogger(__name__.replace('agents.coder', 'real_llm_client'))
        original_module_level = module_logger.level
        module_logger.setLevel(logging.DEBUG)
        
        try:
            logger.info(f"CodeGenerationAgent executing with goal: '{goal}'")
            
            # Report meaningful progress
            self.report_progress("Generating code", f"Processing request: '{goal[:80]}...'")

            # 0. Detect execution mode
            execution_mode = self._detect_execution_mode(goal, context)
            self.report_thinking(f"Detected execution mode: {execution_mode.value.upper()}")
            
            # Handle design update mode specially
            if execution_mode == ExecutionMode.DESIGN_UPDATE:
                return self._handle_design_update(goal, context, current_task)

            # 1. Retrieve necessary artifacts from the context (for normal code generation).
            spec_key = "technical_spec.md"
            manifest_key = "file_manifest.json"
            
            tech_spec = context.get_artifact(spec_key)
            manifest = context.get_artifact(manifest_key)

            # Handle missing inputs gracefully - work with what we have
            if not tech_spec and not manifest:
                # No artifacts available - work directly from the goal
                tech_spec = f"User Request: {goal}"
                files_to_generate = []
                logger.info("No spec or manifest found. Working directly from goal.")
                context.log_event("coder_fallback", {"reason": "no_artifacts", "working_from": "goal_only"})
                self.report_thinking("No technical spec or file manifest found. I'll determine what to build directly from the user's goal.")
            elif not tech_spec:
                # Have manifest but no spec - use goal as spec
                tech_spec = f"User Request: {goal}"
                files_to_generate = manifest.get("files_to_create", [])
                logger.info("No spec found. Using goal as specification.")
                self.report_intermediate_output("file_manifest", json.dumps(manifest, indent=2))
            elif not manifest:
                # Have spec but no manifest - infer files from spec and goal
                tech_spec = tech_spec
                files_to_generate = []
                logger.info("No manifest found. Will infer files to create from spec.")
                self.report_intermediate_output("technical_spec", tech_spec[:500])
            else:
                # Have both artifacts - show what we're working with
                files_to_generate = manifest.get("files_to_create", [])
                self.report_intermediate_output("technical_spec", tech_spec[:500])
                if len(files_to_generate) > 0:
                    self.report_thinking(f"I have both technical spec and file manifest. I'll create {len(files_to_generate)} files: {', '.join(files_to_generate[:3])}{'...' if len(files_to_generate) > 3 else ''}")
            
            # If no files specified, let the LLM decide what files to create based on the goal/spec
            if not files_to_generate:
                logger.info("No specific files to generate. LLM will determine appropriate files to create.")
                logger.debug(f"Working with goal: '{goal}' and spec: '{tech_spec[:100]}...'")  # Truncate for logging

            # 2. Build context of existing code (if any files already exist).
            existing_code = {}
            for file_path in files_to_generate:
                content = context.workspace.get_file_content(file_path)
                if content:
                    existing_code[file_path] = content
            
            if existing_code:
                self.report_thinking(f"Found {len(existing_code)} existing files that I'll modify/extend rather than overwrite.")

            # 3. Generate all files with single LLM call (sequential, reliable approach)
            # Define JSON schema for guaranteed structured response
            code_generation_schema = {
                "type": "object",
                "properties": {
                    "files": {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                        "description": "Dictionary mapping file paths to their code content",
                        "minProperties": 1
                    }
                },
                "required": ["files"],
                "description": "Object containing generated code files"
            }

            try:
                # Detect quality level from goal and context
                quality_level = self._detect_quality_level(goal, context)
                logger.info(f"Using code quality level: {quality_level.value.upper()}")
                self.complete_step(f"Quality level: {quality_level.value.upper()}")
                
                self.report_thinking(f"I'll generate {quality_level.value} quality code. This determines how much documentation, error handling, and optimization to include.")
                
                self.report_progress("Building LLM prompt", f"Creating {quality_level.value} quality prompt with specifications")
                prompt = self._build_prompt(files_to_generate, tech_spec, existing_code, quality_level)
                self.complete_step("Prompt constructed")
                logger.debug(f"Built prompt with {len(prompt)} characters")
                
                # Use regular invoke by default, fall back to function calling if needed
                logger.debug("Using regular invoke method (primary approach)")
                llm_response_str = self.llm_client.invoke(prompt)
                
                # Check if the response is valid JSON with actual content
                try:
                    test_parse = json.loads(llm_response_str)
                    if not isinstance(test_parse, dict) or not test_parse:
                        raise ValueError("Invalid or empty response")
                    logger.debug("Regular invoke succeeded with valid JSON response")
                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning(f"Regular invoke failed to produce valid JSON ({e}), trying function calling")
                    if hasattr(self.llm_client, 'invoke_with_schema'):
                        try:
                            llm_response_str = self.llm_client.invoke_with_schema(prompt, code_generation_schema)
                            logger.debug("Function calling fallback succeeded")
                        except Exception as fallback_error:
                            logger.error(f"Function calling fallback also failed: {fallback_error}")
                            # Keep the original response from regular invoke for error reporting
                            pass
                    else:
                        logger.warning("Function calling not available, keeping original response")
                
                logger.debug(f"LLM response length: {len(llm_response_str)} characters")
                logger.debug(f"LLM response preview: {llm_response_str[:200]}...")
                logger.debug(f"Full LLM response: {llm_response_str}")
                
                response_data = json.loads(llm_response_str)
                logger.debug(f"Parsed JSON type: {type(response_data)}")
                logger.debug(f"Parsed JSON keys: {list(response_data.keys()) if isinstance(response_data, dict) else 'Not a dict'}")

                if not isinstance(response_data, dict):
                    raise ValueError("LLM response is not a valid dictionary.")
                
                # Extract the files from the structured response
                if "files" in response_data:
                    generated_code_map = response_data["files"]
                else:
                    # Fallback: treat the whole response as the file map (backward compatibility)
                    generated_code_map = response_data
                
                logger.debug(f"Generated code map keys: {list(generated_code_map.keys()) if isinstance(generated_code_map, dict) else 'Not a dict'}")
                logger.debug(f"Generated code map length: {len(generated_code_map) if isinstance(generated_code_map, dict) else 'N/A'}")
                
                if not isinstance(generated_code_map, dict):
                    raise ValueError("Files section is not a valid dictionary of file paths to code.")
                
                if not generated_code_map:
                    raise ValueError("LLM returned empty code dictionary.")
                
                logger.info(f"LLM generated code for {len(generated_code_map)} files: {list(generated_code_map.keys())}")

                # Display the generated code using enhanced UI
                if len(generated_code_map) == 1:
                    # Single file - show as code snippet
                    file_path, content = next(iter(generated_code_map.items()))
                    # Create a special content wrapper that includes filename info
                    code_with_filename = {"content": content, "filename": file_path}
                    self.report_intermediate_output("code_snippet", code_with_filename)
                else:
                    # Multiple files - show as code files
                    self.report_intermediate_output("code_files", generated_code_map)

            except NotImplementedError as e:
                msg = f"Cannot execute code generation: {e}"
                logger.critical(msg)
                return AgentResponse(success=False, message=msg)
            except (json.JSONDecodeError, ValueError) as e:
                msg = f"Failed to parse LLM response as valid JSON code map. Error: {e}"
                logger.error(msg)
                logger.error(f"Raw LLM response causing the error: '{llm_response_str}'")
                logger.error(f"Prompt sent to LLM (first 500 chars): {prompt[:500]}...")
                return AgentResponse(success=False, message=msg)
            except Exception as e:
                error_msg = f"An unexpected error occurred while calling the LLM for code generation: {e}"
                logger.critical(error_msg, exc_info=True)
                return AgentResponse(success=False, message=error_msg)

            # 4. Write the generated code to the workspace.
            written_files = []
            skipped_files = []
            modified_files = {}
            
            for file_path, code_content in generated_code_map.items():
                if not isinstance(code_content, str):
                    logger.warning(f"Skipping file '{file_path}' due to invalid code content type: {type(code_content)}")
                    skipped_files.append(file_path)
                    continue
                
                if not code_content.strip():
                    logger.warning(f"Skipping file '{file_path}' due to empty code content")
                    skipped_files.append(file_path)
                    continue
                
                try:
                    # Check if this file already exists to create a diff
                    original_content = context.workspace.get_file_content(file_path)
                    
                    logger.debug(f"Writing file '{file_path}' with {len(code_content)} characters")
                    context.workspace.write_file_content(file_path, code_content, current_task.task_id)
                    written_files.append(file_path)
                    logger.info(f"Successfully wrote file '{file_path}'")
                    
                    # Track modified files for diff display
                    if original_content and original_content != code_content:
                        modified_files[file_path] = {
                            "old": original_content,
                            "new": code_content
                        }
                except Exception as e:
                    msg = f"Failed to write file '{file_path}' to workspace. Error: {e}"
                    logger.error(msg, exc_info=True)
                    # Return failure on the first write error.
                    return AgentResponse(success=False, message=msg)

            if not written_files:
                error_details = f"LLM generated {len(generated_code_map)} file(s) but none were valid for writing."
                if skipped_files:
                    error_details += f" Skipped files: {skipped_files}"
                logger.error(error_details)
                return AgentResponse(success=False, message=error_details)

            # Display diff for modified files
            if modified_files:
                self.report_intermediate_output("code_diff", modified_files)
                logger.info(f"Displayed diff for {len(modified_files)} modified files")

            # Run post-processing workflow with new helper agents
            self._run_post_processing_workflow(written_files, generated_code_map, context, current_task)
            
            return AgentResponse(
                success=True,
                message=f"Successfully generated and wrote {len(written_files)} files to the workspace.",
                artifacts_generated=written_files
            )
        
        finally:
            # Restore original logging levels
            logger.setLevel(original_level)
            llm_logger.setLevel(original_llm_level)
            module_logger.setLevel(original_module_level)
    
    def _run_post_processing_workflow(self, written_files: List[str], code_map: Dict[str, str], 
                                    context: GlobalContext, current_task: TaskNode):
        """Run post-processing workflow with helper agents."""
        try:
            self.report_progress("Post-processing", "Running dependency management and validation")
            
            # 1. Requirements Management
            self._call_requirements_manager(code_map, context, current_task)
            
            # 2. CLI Test Generation  
            test_script = self._call_cli_test_generator(code_map, context, current_task)
            
            # 3. Execution Validation
            self._call_execution_validator(code_map, test_script, context, current_task)
            
        except Exception as e:
            logger.warning(f"Post-processing workflow failed (non-critical): {e}")
    
    def _call_requirements_manager(self, code_map: Dict[str, str], context: GlobalContext, current_task: TaskNode):
        """Call RequirementsManagerAgent to manage dependencies."""
        try:
            from . import RequirementsManagerAgent
            
            requirements_agent = RequirementsManagerAgent()
            
            # Set up progress tracking
            if hasattr(self, 'progress_tracker') and self.progress_tracker:
                requirements_agent.set_progress_tracker(self.progress_tracker, f"{current_task.task_id}_requirements")
            
            result = requirements_agent.execute_v2(
                "Analyze generated code and update requirements.txt",
                {
                    'code_files': code_map,
                    'output_directory': context.workspace_path
                },
                context
            )
            
            if result.success:
                logger.info(f"Requirements management completed: {result.message}")
                if result.outputs.get('requirements_file'):
                    context.add_artifact("requirements.txt", result.outputs['requirements_file'], current_task.task_id)
            else:
                logger.warning(f"Requirements management failed: {result.message}")
                
        except Exception as e:
            logger.warning(f"Requirements management error: {e}")
    
    def _call_cli_test_generator(self, code_map: Dict[str, str], context: GlobalContext, current_task: TaskNode) -> str:
        """Call CLITestGeneratorAgent to create test script."""
        try:
            from . import CLITestGeneratorAgent
            
            test_agent = CLITestGeneratorAgent(llm_client=self.llm_client)
            
            # Set up progress tracking
            if hasattr(self, 'progress_tracker') and self.progress_tracker:
                test_agent.set_progress_tracker(self.progress_tracker, f"{current_task.task_id}_testing")
            
            result = test_agent.execute_v2(
                "Generate CLI test script for the generated code",
                {
                    'code_files': code_map,
                    'specification': current_task.goal,
                    'output_directory': context.workspace_path
                },
                context
            )
            
            if result.success:
                logger.info(f"CLI test generation completed: {result.message}")
                test_script = result.outputs.get('test_script')
                if test_script:
                    context.add_artifact("test_cli.py", test_script, current_task.task_id)
                return test_script
            else:
                logger.warning(f"CLI test generation failed: {result.message}")
                return None
                
        except Exception as e:
            logger.warning(f"CLI test generation error: {e}")
            return None
    
    def _call_execution_validator(self, code_map: Dict[str, str], test_script: str, 
                                context: GlobalContext, current_task: TaskNode):
        """Call ExecutionValidatorAgent to validate the code."""
        try:
            from . import ExecutionValidatorAgent
            
            validator_agent = ExecutionValidatorAgent()
            
            # Set up progress tracking
            if hasattr(self, 'progress_tracker') and self.progress_tracker:
                validator_agent.set_progress_tracker(self.progress_tracker, f"{current_task.task_id}_validation")
            
            inputs = {
                'code_files': code_map,
                'output_directory': context.workspace_path
            }
            
            if test_script:
                inputs['test_script'] = test_script
            
            result = validator_agent.execute_v2(
                "Validate generated code execution and functionality",
                inputs,
                context
            )
            
            if result.success:
                logger.info(f"Execution validation completed: {result.message}")
                # Store validation results as artifact
                context.add_artifact("validation_results.json", result.outputs, current_task.task_id)
            else:
                logger.warning(f"Execution validation failed: {result.message}")
                
        except Exception as e:
            logger.warning(f"Execution validation error: {e}")



# --- Self-Testing Block ---
if __name__ == "__main__":
    import unittest
    from unittest.mock import MagicMock
    import shutil
    from pathlib import Path
    from utils.logger import setup_logger

    setup_logger(default_level=logging.INFO)

    class TestCodeGenerationAgent(unittest.TestCase):

        def setUp(self):
            """Set up a clean environment for each test."""
            self.test_workspace_path = "./temp_coder_test_ws"
            if Path(self.test_workspace_path).exists():
                shutil.rmtree(self.test_workspace_path)
            
            self.mock_llm_client = MagicMock(spec=LLMClient)
            self.context = GlobalContext(workspace_path=self.test_workspace_path)
            self.code_task = TaskNode(goal="Implement API", assigned_agent="CodeGenerationAgent")
            self.agent = CodeGenerationAgent(llm_client=self.mock_llm_client)

            # Pre-populate context with required artifacts for most tests
            self.context.add_artifact("technical_spec.md", "Spec: Build an API.", "task_spec")
            self.context.add_artifact("file_manifest.json", {"files_to_create": ["src/main.py", "src/utils.py"]}, "task_manifest")

        def tearDown(self):
            """Clean up the environment after each test."""
            shutil.rmtree(self.test_workspace_path)

        def test_successful_code_generation(self):
            """Tests the ideal case where the LLM returns valid code for all files."""
            print("\n--- [Test Case 1: Successful Code Generation] ---")
            # Configure the mock LLM to return a valid code map.
            mock_code_output = json.dumps({
                "src/main.py": "import utils\n\nprint(utils.helper())",
                "src/utils.py": "def helper():\n    return 'Hello from helper'"
            })
            self.mock_llm_client.invoke.return_value = mock_code_output

            response = self.agent.execute(self.code_task.goal, self.context, self.code_task)

            self.mock_llm_client.invoke.assert_called_once()
            self.assertTrue(response.success)
            self.assertIn("Successfully generated and wrote 2 files", response.message)
            
            # Verify the files were actually written to the mock workspace.
            main_content = self.context.workspace.get_file_content("src/main.py")
            self.assertIn("import utils", main_content)
            logger.info("✅ test_successful_code_generation: PASSED")

        def test_failure_on_missing_artifacts(self):
            """Tests that the agent fails gracefully if prerequisites are not in the context."""
            print("\n--- [Test Case 2: Missing Artifacts] ---")
            empty_context = GlobalContext(workspace_path=self.test_workspace_path) # Use a fresh context
            
            response = self.agent.execute(self.code_task.goal, empty_context, self.code_task)

            self.assertFalse(response.success)
            self.assertIn("Missing required artifacts", response.message)
            self.mock_llm_client.invoke.assert_not_called() # LLM should not be called
            logger.info("✅ test_failure_on_missing_artifacts: PASSED")

        def test_llm_returns_invalid_json(self):
            """Tests how the agent handles a malformed JSON string from the LLM."""
            print("\n--- [Test Case 3: Invalid JSON from LLM] ---")
            self.mock_llm_client.invoke.return_value = "def main():\n  pass" # Not a JSON object

            response = self.agent.execute(self.code_task.goal, self.context, self.code_task)

            self.assertFalse(response.success)
            self.assertIn("Failed to parse LLM response", response.message)
            logger.info("✅ test_llm_returns_invalid_json: PASSED")

        def test_llm_returns_incomplete_map(self):
            """Tests when the LLM returns valid JSON but not for all requested files."""
            print("\n--- [Test Case 4: Incomplete Code Map from LLM] ---")
            # The manifest requests two files, but the LLM only returns one.
            mock_code_output = json.dumps({
                "src/main.py": "print('only one file')"
            })
            self.mock_llm_client.invoke.return_value = mock_code_output

            response = self.agent.execute(self.code_task.goal, self.context, self.code_task)

            self.assertTrue(response.success)
            self.assertIn("wrote 1 files", response.message) # Should still succeed with the files it got
            self.assertIsNotNone(self.context.workspace.get_file_content("src/main.py"))
            self.assertIsNone(self.context.workspace.get_file_content("src/utils.py")) # The other file should not exist
            logger.info("✅ test_llm_returns_incomplete_map: PASSED")

    # Run the tests
    unittest.main(argv=['first-arg-is-ignored'], exit=False)