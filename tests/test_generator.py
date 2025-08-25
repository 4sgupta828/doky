# agents/test_generator.py
import json
import logging
from typing import Dict, Any, List

# Foundational dependencies
from .base import BaseAgent
from core.context import GlobalContext
from core.models import AgentResult
from tools.test_generation_tools import TestGenerationTools, TestQuality
from tools.file_system_tools import FileSystemTools

# Get a logger instance for this module
logger = logging.getLogger(__name__)


# --- Real LLM Integration Placeholder ---
class LLMClient:
    """A placeholder for a real LLM client (e.g., OpenAI, Gemini)."""
    def invoke(self, _prompt: str) -> str:
        raise NotImplementedError("LLMClient.invoke must be implemented by a concrete class.")

    def invoke_with_schema(self, prompt: str, _schema: Dict[str, Any]) -> str:
        """Optional structured output method."""
        return self.invoke(prompt)


class TestGenerationAgent(BaseAgent):
    """
    Comprehensive test generation agent that creates unit and integration tests.
    
    This agent uses structured inputs and outputs, leverages existing tools and agents,
    and maintains all test generation functionality while following modern patterns.
    
    Responsibilities:
    - Generate unit and integration tests based on specifications
    - Support multiple test quality levels (FAST, DECENT, PRODUCTION)
    - Discover and analyze source code for test generation
    - Leverage file system operations through existing agents
    - Provide structured outputs with validation
    """

    def __init__(self, llm_client: Any = None, agent_registry: Dict[str, Any] = None):
        super().__init__(
            name="TestGenerationAgent",
            description="Generates comprehensive unit and integration tests for application code using structured inputs and modern patterns."
        )
        self.llm_client = llm_client or LLMClient()
        self.agent_registry = agent_registry or {}

    def required_inputs(self) -> List[str]:
        """Required inputs for TestGenerationAgent execution."""
        return []  # Can work with minimal inputs by auto-discovery

    def optional_inputs(self) -> List[str]:
        """Optional inputs for comprehensive test generation configuration."""
        return [
            "test_type",  # "unit", "integration", "auto"
            "test_quality",  # "fast", "decent", "production"
            "source_files",  # List of specific files to test
            "specification",  # Technical specification or requirements
            "manifest_data",  # File manifest with files_to_create
            "output_directory",  # Where to write test files (default: tests/)
            "test_patterns",  # Custom test file naming patterns
            "coverage_requirements",  # Specific coverage requirements
            "additional_context"  # Extra context for test generation
        ]

    def execute_v2(self, goal: str, inputs: Dict[str, Any], global_context: GlobalContext) -> AgentResult:
        """
        Generate tests using structured inputs and leveraging existing tools and agents.
        """
        logger.info(f"TestGenerationAgent executing: '{goal}'")
        
        # Validate inputs with graceful handling
        try:
            self.validate_inputs(inputs)
        except Exception as validation_error:
            return self.create_result(
                success=False,
                message=str(validation_error),
                error_details={"validation_error": str(validation_error)}
            )

        try:
            self.report_progress("Analyzing test requirements", f"Processing goal: {goal[:80]}...")

            # Step 1: Extract and process inputs
            test_config = self._extract_test_configuration(inputs, goal)
            self.report_progress("Configuration determined", 
                               f"Type: {test_config['test_type']}, Quality: {test_config['quality'].value}")

            # Step 2: Discover source code to test
            self.report_progress("Discovering source code", "Finding code files to test...")
            code_discovery_result = self._discover_source_code(test_config, global_context)
            
            if not code_discovery_result["success"]:
                return self.create_result(
                    success=False,
                    message=f"Source code discovery failed: {code_discovery_result['message']}",
                    error_details=code_discovery_result
                )

            source_code = code_discovery_result["source_code"]
            if not source_code:
                return self.create_result(
                    success=True,
                    message="No source code files found to test.",
                    outputs={
                        "test_files_generated": 0,
                        "source_files_analyzed": 0,
                        "test_configuration": test_config
                    }
                )

            self.report_progress("Source code discovered", 
                               f"Found {len(source_code)} files: {list(source_code.keys())[:3]}...")

            # Step 3: Generate test specification and prompt
            specification = self._build_specification(test_config, global_context)
            test_prompt = self._build_test_prompt(
                spec=specification,
                code_to_test=source_code,
                test_type=test_config["test_type"],
                quality=test_config["quality"]
            )

            # Step 4: Generate tests using LLM
            self.report_progress("Generating test code", 
                               f"Creating {test_config['quality'].value} {test_config['test_type']} tests...")
            generation_result = self._generate_tests_with_llm(test_prompt)
            
            if not generation_result["success"]:
                return self.create_result(
                    success=False,
                    message=f"Test generation failed: {generation_result['message']}",
                    error_details=generation_result
                )

            generated_tests = generation_result["test_files"]

            # Step 5: Validate generated tests
            validation_result = TestGenerationTools.validate_test_files(generated_tests)
            if not validation_result["valid"]:
                logger.warning(f"Generated tests have validation errors: {validation_result['errors']}")
                self.report_progress("Validation warnings", 
                                   f"Tests generated with {len(validation_result['errors'])} errors")

            # Step 6: Write test files using FileSystemAgent
            self.report_progress("Writing test files", f"Saving {len(generated_tests)} test files...")
            write_result = self._write_test_files(generated_tests, test_config["output_directory"], global_context)
            
            if not write_result["success"]:
                return self.create_result(
                    success=False,
                    message=f"Failed to write test files: {write_result['message']}",
                    error_details=write_result
                )

            written_files = write_result["written_files"]

            # Step 7: Prepare structured output
            final_result = {
                "test_files_generated": len(written_files),
                "source_files_analyzed": len(source_code),
                "test_configuration": {
                    "test_type": test_config["test_type"],
                    "quality_level": test_config["quality"].value,
                    "output_directory": test_config["output_directory"]
                },
                "validation_results": validation_result,
                "generated_files": written_files,
                "source_files": list(source_code.keys())
            }

            message = self._create_summary_message(final_result)
            self.report_progress("Test generation complete", message)

            return self.create_result(
                success=True,
                message=message,
                outputs=final_result
            )

        except Exception as e:
            error_msg = f"TestGenerationAgent execution failed: {e}"
            logger.error(error_msg, exc_info=True)
            return self.create_result(
                success=False,
                message=error_msg,
                error_details={"exception": str(e)}
            )

    def _extract_test_configuration(self, inputs: Dict[str, Any], goal: str) -> Dict[str, Any]:
        """Extract and process test configuration from inputs."""
        # Determine test type
        test_type = inputs.get("test_type", "auto")
        if test_type == "auto":
            test_type = TestGenerationTools.determine_test_type(goal)

        # Determine quality level
        quality_str = inputs.get("test_quality", "auto")
        if quality_str == "auto":
            quality = TestGenerationTools.detect_quality_level(goal)
        else:
            try:
                quality = TestQuality(quality_str.lower())
            except ValueError:
                logger.warning(f"Invalid quality level '{quality_str}', using FAST")
                quality = TestQuality.FAST

        return {
            "test_type": test_type,
            "quality": quality,
            "source_files": inputs.get("source_files", []),
            "specification": inputs.get("specification"),
            "manifest_data": inputs.get("manifest_data"),
            "output_directory": inputs.get("output_directory", "tests/"),
            "test_patterns": inputs.get("test_patterns", []),
            "coverage_requirements": inputs.get("coverage_requirements"),
            "additional_context": inputs.get("additional_context", "")
        }

    def _discover_source_code(self, test_config: Dict[str, Any], global_context: GlobalContext) -> Dict[str, Any]:
        """Discover source code files to test using FileSystem tools and agents."""
        try:
            # Priority 1: Use explicitly provided source files
            if test_config["source_files"]:
                self.report_thinking("Using explicitly provided source files")
                source_code = TestGenerationTools.discover_code_files(
                    file_list=test_config["source_files"],
                    workspace_path=str(global_context.workspace_path)
                )
                if source_code:
                    return {"success": True, "source_code": source_code, "method": "explicit_files"}

            # Priority 2: Use manifest data if available
            manifest_data = test_config.get("manifest_data")
            if not manifest_data:
                # Try to get from global context artifacts
                try:
                    manifest_artifact = global_context.get_artifact("file_manifest.json")
                    if manifest_artifact:
                        if isinstance(manifest_artifact, str):
                            manifest_data = json.loads(manifest_artifact)
                        else:
                            manifest_data = manifest_artifact
                except Exception as e:
                    logger.debug(f"Failed to get manifest from context: {e}")

            if manifest_data and isinstance(manifest_data, dict):
                files_to_create = manifest_data.get("files_to_create", [])
                if files_to_create:
                    self.report_thinking(f"Using manifest files: {len(files_to_create)} files")
                    source_code = TestGenerationTools.discover_code_files(
                        file_list=files_to_create,
                        workspace_path=str(global_context.workspace_path)
                    )
                    if source_code:
                        return {"success": True, "source_code": source_code, "method": "manifest"}

            # Priority 3: Auto-discover using FileSystem agent
            file_system_agent = self.agent_registry.get("FileSystemAgent")
            if file_system_agent:
                self.report_thinking("Auto-discovering Python files using FileSystemAgent")
                discovery_result = self.call_agent_v2(
                    target_agent=file_system_agent,
                    goal="Discover Python source files for test generation",
                    inputs={
                        "operation": "discover",
                        "target_path": ".",
                        "patterns": ["*.py"],
                        "exclude_patterns": ["test_*.py", "*_test.py", "tests/", "__pycache__/", ".venv/"],
                        "recursive": True
                    },
                    global_context=global_context
                )
                
                if discovery_result.success:
                    discovered_files = discovery_result.outputs.get("discovered_files", [])
                    source_code = TestGenerationTools.discover_code_files(
                        file_list=discovered_files,
                        workspace_path=str(global_context.workspace_path)
                    )
                    if source_code:
                        return {"success": True, "source_code": source_code, "method": "auto_discovery"}

            # Priority 4: Fallback to direct file system discovery
            self.report_thinking("Fallback to direct workspace discovery")
            source_code = TestGenerationTools.discover_code_files(
                workspace_path=str(global_context.workspace_path)
            )
            
            return {
                "success": True, 
                "source_code": source_code, 
                "method": "direct_discovery"
            }

        except Exception as e:
            return {
                "success": False,
                "message": f"Source code discovery failed: {e}",
                "source_code": {}
            }

    def _build_specification(self, test_config: Dict[str, Any], global_context: GlobalContext) -> str:
        """Build test specification from available sources."""
        specification_parts = []

        # Add explicit specification if provided
        if test_config.get("specification"):
            specification_parts.append("**Provided Specification:**")
            specification_parts.append(test_config["specification"])

        # Try to get technical spec from context
        try:
            tech_spec = global_context.get_artifact("technical_spec.md")
            if tech_spec and isinstance(tech_spec, str) and tech_spec.strip():
                specification_parts.append("**Technical Specification:**")
                specification_parts.append(tech_spec)
        except Exception as e:
            logger.debug(f"No technical spec found in context: {e}")

        # Add coverage requirements if specified
        if test_config.get("coverage_requirements"):
            specification_parts.append("**Coverage Requirements:**")
            specification_parts.append(str(test_config["coverage_requirements"]))

        # Add additional context
        if test_config.get("additional_context"):
            specification_parts.append("**Additional Context:**")
            specification_parts.append(test_config["additional_context"])

        # Fallback if no specification found
        if not specification_parts:
            specification_parts.append("**Inferred Requirements:**")
            specification_parts.append(f"Generate {test_config['test_type']} tests for the provided source code.")

        return "\n\n".join(specification_parts)

    def _generate_tests_with_llm(self, prompt: str) -> Dict[str, Any]:
        """Generate tests using LLM with fallback strategies."""
        try:
            # Try regular invoke first
            logger.debug("Using regular invoke method for test generation")
            llm_response = self.llm_client.invoke(prompt)
            
            # Parse and validate response
            try:
                test_files = TestGenerationTools.parse_test_response(llm_response)
                return {
                    "success": True,
                    "test_files": test_files,
                    "method": "regular_invoke"
                }
            except ValueError as parse_error:
                logger.warning(f"Regular invoke parsing failed: {parse_error}")
                
                # Try structured invoke if available
                if hasattr(self.llm_client, 'invoke_with_schema'):
                    try:
                        logger.debug("Fallback to structured invoke method")
                        schema = self._get_json_schema()
                        structured_response = self.llm_client.invoke_with_schema(prompt, schema)
                        test_files = TestGenerationTools.parse_test_response(structured_response)
                        return {
                            "success": True,
                            "test_files": test_files,
                            "method": "structured_invoke"
                        }
                    except Exception as structured_error:
                        logger.error(f"Structured invoke also failed: {structured_error}")
                
                # If all parsing fails, return the original error
                return {
                    "success": False,
                    "message": f"Failed to parse LLM response: {parse_error}",
                    "raw_response": llm_response[:500] + "..." if len(llm_response) > 500 else llm_response
                }

        except Exception as e:
            return {
                "success": False,
                "message": f"LLM invocation failed: {e}",
                "exception": str(e)
            }

    def _write_test_files(self, test_files: Dict[str, str], output_directory: str, 
                         global_context: GlobalContext) -> Dict[str, Any]:
        """Write test files using FileSystemAgent or direct file operations."""
        written_files = []
        
        try:
            file_system_agent = self.agent_registry.get("FileSystemAgent")
            
            for file_path, content in test_files.items():
                # Ensure file path starts with output directory
                if not file_path.startswith(output_directory):
                    file_path = f"{output_directory.rstrip('/')}/{file_path.lstrip('/')}"
                
                if file_system_agent:
                    # Use FileSystemAgent for structured file operations
                    write_result = self.call_agent_v2(
                        target_agent=file_system_agent,
                        goal=f"Write test file: {file_path}",
                        inputs={
                            "operation": "write",
                            "target_path": file_path,
                            "content": content
                        },
                        global_context=global_context
                    )
                    
                    if write_result.success:
                        written_files.append(file_path)
                        logger.info(f"Successfully wrote test file via agent: {file_path}")
                    else:
                        logger.error(f"Failed to write test file via agent: {file_path} - {write_result.message}")
                        # Continue with other files
                else:
                    # Fallback to direct file system operations
                    write_result = FileSystemTools.write_file(
                        target_path=file_path,
                        content=content,
                        context=global_context
                    )
                    
                    if write_result["success"]:
                        written_files.append(file_path)
                        logger.info(f"Successfully wrote test file directly: {file_path}")
                    else:
                        logger.error(f"Failed to write test file directly: {file_path} - {write_result['message']}")

            return {
                "success": len(written_files) > 0,
                "written_files": written_files,
                "total_requested": len(test_files),
                "message": f"Successfully wrote {len(written_files)}/{len(test_files)} test files"
            }

        except Exception as e:
            return {
                "success": False,
                "message": f"File writing failed: {e}",
                "written_files": written_files
            }

    def _create_summary_message(self, result: Dict[str, Any]) -> str:
        """Create a comprehensive summary message."""
        files_generated = result["test_files_generated"]
        files_analyzed = result["source_files_analyzed"]
        test_type = result["test_configuration"]["test_type"]
        quality = result["test_configuration"]["quality_level"]
        
        validation = result.get("validation_results", {})
        warning_count = len(validation.get("warnings", []))
        error_count = len(validation.get("errors", []))
        
        base_message = f"Successfully generated {files_generated} {test_type} test files ({quality} quality) for {files_analyzed} source files."
        
        if warning_count > 0 or error_count > 0:
            base_message += f" Validation: {error_count} errors, {warning_count} warnings."
        
        return base_message

    def _build_test_prompt(self, spec: str, code_to_test: Dict[str, str], test_type: str, quality: TestQuality) -> str:
        """Build comprehensive test generation prompt using quality configuration."""
        quality_config = TestGenerationTools.get_quality_config(quality, test_type)
        
        # Build code blocks section
        if code_to_test:
            code_blocks = "\n\n".join(
                f"--- File: {path} ---\n```python\n{content}\n```"
                for path, content in code_to_test.items()
            )
        else:
            code_blocks = "No source code files provided. Generate tests based on specification only."

        # Build quality-specific instructions based on configuration
        test_instructions = self._build_quality_instructions(quality_config, test_type, quality)

        return f"""
        You are an expert QA Engineer specializing in Python. Your task is to write
        {quality_config["description"]} using the `pytest` framework for the provided source code,
        based on its technical specification.
        
        **Test Quality Level: {quality.value.upper()}**

        **Technical Specification:**
        ---
        {spec}
        ---

        **Source Code to Test:**
        ---
        {code_blocks}
        ---

        {test_instructions}

        **Final Output Requirement:**
        Your output MUST be a single, valid JSON object where keys are the test file paths
        (e.g., "tests/test_auth_integration.py") and values are the complete test code content as a string.
        """

    def _build_quality_instructions(self, quality_config: Dict[str, Any], test_type: str, quality: TestQuality) -> str:
        """Build quality-specific instructions for the LLM prompt."""
        # Map quality config to specific instructions
        instruction_map = {
            ("unit", "simple"): [
                "Write focused pytest tests for main functions and classes",
                "Test basic success cases and obvious error conditions",
                "Use simple mocking for external dependencies",
                "Keep test structure simple and straightforward"
            ],
            ("unit", "thorough"): [
                "Write thorough pytest unit tests for all functions and classes",
                "Mock external dependencies using unittest.mock effectively",
                "Test success cases, edge cases, and error conditions using pytest.raises",
                "Include reasonable test data variety and boundary conditions"
            ],
            ("unit", "comprehensive"): [
                "Write comprehensive pytest unit tests covering all code paths",
                "Implement sophisticated mocking strategies for complex dependencies",
                "Test all success cases, edge cases, error conditions, and boundary values",
                "Include parametrized tests for data variety and comprehensive fixtures",
                "Add performance tests and memory usage considerations where appropriate"
            ],
            ("integration", "simple"): [
                "Write pytest integration tests for main user workflows",
                "Use test client for basic API endpoint testing",
                "Focus on happy path scenarios",
                "Keep test setup minimal and straightforward"
            ],
            ("integration", "thorough"): [
                "Write comprehensive pytest integration tests for component interactions",
                "Use test client (e.g., Flask's TestClient) for realistic API testing",
                "Test end-to-end user flows including error scenarios",
                "Include proper test data setup and cleanup"
            ],
            ("integration", "comprehensive"): [
                "Write comprehensive integration tests covering all system interactions",
                "Implement sophisticated test client usage with authentication and authorization",
                "Test complete end-to-end workflows including edge cases and failure modes",
                "Include database transaction testing, concurrent access scenarios",
                "Add performance and load testing considerations for critical paths"
            ]
        }

        # Get instructions based on test type and complexity level
        key = (test_type, quality_config["complexity_level"])
        instructions = instruction_map.get(key, instruction_map[(test_type, "thorough")])
        
        # Format instructions
        formatted_instructions = "\n        ".join([f"{i+1}. {instruction}" for i, instruction in enumerate(instructions)])
        
        return f"""
        **Instructions for {test_type.title()} Tests ({quality.value.upper()} Quality):**
        {formatted_instructions}
        """

    def _get_json_schema(self) -> Dict[str, Any]:
        """Get JSON schema for structured LLM responses."""
        return {
            "type": "object", 
            "properties": {},
            "additionalProperties": {"type": "string"},
            "description": "Dictionary mapping test file paths to their test code content"
        }

