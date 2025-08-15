# agents/coder.py
import json
import logging
from typing import Dict, Any, List, Literal
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


# --- Real LLM Integration Placeholder ---
class LLMClient:
    """A placeholder for a real LLM client (e.g., OpenAI, Gemini)."""
    def invoke(self, prompt: str) -> str:
        raise NotImplementedError("LLMClient.invoke must be implemented by a concrete class.")

# --- Agent Implementation ---

class CodeGenerationAgent(BaseAgent):
    """
    A specialized agent responsible for writing and modifying source code.
    It takes a detailed technical specification and a file manifest as input and
    produces the corresponding code, writing it to the workspace.
    """

    def __init__(self, llm_client: Any = None, default_quality: CodeQuality = CodeQuality.FAST):
        super().__init__(
            name="CodeGenerationAgent",
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

            # 1. Retrieve necessary artifacts from the context.
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

            # 3. Construct the prompt and invoke the LLM.
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
                    logger.debug(f"Writing file '{file_path}' with {len(code_content)} characters")
                    context.workspace.write_file_content(file_path, code_content, current_task.task_id)
                    written_files.append(file_path)
                    logger.info(f"Successfully wrote file '{file_path}'")
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