# agents/spec_generator.py
import json
import logging
from typing import Any, Dict
from enum import Enum

# Foundational dependencies
from .base import BaseAgent
from core.context import GlobalContext
from core.models import AgentResponse, TaskNode

# Get a logger instance for this module
logger = logging.getLogger(__name__)


# --- Specification Quality Levels ---
class SpecQuality(Enum):
    """Defines different specification quality levels for speed vs detail trade-offs."""
    FAST = "fast"          # Quick, basic specs - prioritizes speed
    DECENT = "decent"      # Balanced approach - good structure, reasonable detail (default)
    PRODUCTION = "production"  # Comprehensive, detailed specs with full considerations


# --- Real LLM Integration Placeholder ---
class LLMClient:
    """A placeholder for a real LLM client (e.g., OpenAI, Gemini)."""
    def invoke(self, prompt: str) -> str:
        raise NotImplementedError("LLMClient.invoke must be implemented by a concrete class.")

class CollaborationUI:
    """A placeholder for a real user interface (e.g., CLI, Web UI)."""
    def prompt_for_input(self, question: str) -> str:
        raise NotImplementedError("CollaborationUI.prompt_for_input must be implemented.")

# --- Agent Implementation ---

class SpecGeneratorAgent(BaseAgent):
    """
    Acts as the project's software architect, presenting its interpretation of requirements
    as a technical specification and validates the approach with the user before finalizing.
    """

    def __init__(self, llm_client: Any = None, ui_interface: Any = None, default_quality: SpecQuality = SpecQuality.FAST):
        super().__init__(
            name="SpecGeneratorAgent", 
            description="Creates detailed technical specifications and validates the approach with the user."
        )
        self.llm_client = llm_client or LLMClient()
        self.ui = ui_interface or CollaborationUI()
        self.default_quality = default_quality

    def _get_quality_instructions(self, quality: SpecQuality) -> Dict[str, str]:
        """Returns quality-specific instructions and descriptions."""
        quality_configs = {
            SpecQuality.FAST: {
                "description": "a basic, functional technical specification",
                "instructions": [
                    "Focus on core functionality and essential requirements",
                    "Define basic data models with key fields only",
                    "Outline main API endpoints without extensive detail",
                    "Include fundamental business logic descriptions",
                    "Keep documentation concise and to-the-point"
                ]
            },
            SpecQuality.DECENT: {
                "description": "a well-structured, comprehensive technical specification",
                "instructions": [
                    "Define detailed data models with proper types, constraints, and relationships",
                    "Specify complete API endpoints with request/response schemas",
                    "Include clear business logic descriptions with validation rules",
                    "Add reasonable error handling and edge case considerations",
                    "Structure the document logically with good organization"
                ]
            },
            SpecQuality.PRODUCTION: {
                "description": "a comprehensive, enterprise-ready technical specification",
                "instructions": [
                    "Create exhaustive data models with full validation, indexing, and relationship details",
                    "Define complete API specifications with authentication, authorization, and rate limiting",
                    "Include comprehensive business logic with all edge cases and error scenarios",
                    "Add security considerations, performance requirements, and scalability notes",
                    "Include deployment, monitoring, and operational considerations",
                    "Add detailed documentation standards and code quality requirements"
                ]
            }
        }
        return quality_configs[quality]
    
    def _detect_quality_level(self, goal: str, context: GlobalContext) -> SpecQuality:
        """Detects the desired specification quality level from the goal and context."""
        goal_lower = goal.lower()
        
        # Check for explicit quality keywords in the goal
        if any(keyword in goal_lower for keyword in ['fast', 'quick', 'basic', 'simple', 'minimal']):
            logger.info("Detected FAST spec quality level from goal keywords")
            return SpecQuality.FAST
        elif any(keyword in goal_lower for keyword in ['decent', 'detailed', 'comprehensive', 'structured', 'thorough']):
            logger.info("Detected DECENT spec quality level from goal keywords")
            return SpecQuality.DECENT
        elif any(keyword in goal_lower for keyword in ['production', 'enterprise', 'complete', 'exhaustive', 'robust']):
            logger.info("Detected PRODUCTION spec quality level from goal keywords")
            return SpecQuality.PRODUCTION
        
        # Check context for quality preferences
        if hasattr(context, 'spec_quality_preference'):
            return context.spec_quality_preference
            
        # Default to FAST for speed optimization
        logger.info("Using default FAST spec quality level")
        return self.default_quality

    def _build_prompt(self, requirements: str, is_fallback: bool = False, quality: SpecQuality = None) -> str:
        """Constructs a precise prompt to guide the LLM in generating a structured technical spec."""
        if quality is None:
            quality = self.default_quality
            
        quality_config = self._get_quality_instructions(quality)
        
        if is_fallback:
            content_type = "user goal"
            content_description = "the following user goal into a structured technical specification"
        else:
            content_type = "clarified requirements"  
            content_description = "the following clarified user requirements into a structured technical specification"
            
        # Build quality-specific instructions
        quality_instructions = "\n        ".join([f"{i+1}. {instruction}" for i, instruction in enumerate(quality_config["instructions"])])
        
        return f"""
        You are an expert software architect. Your task is to convert {content_description} and present your technical approach.
        
        **Specification Quality Level: {quality.value.upper()}**

        **{content_type.title()}:**
        ---
        {requirements}
        ---

        **Quality-Specific Instructions:**
        {quality_instructions}
        
        Your response MUST be a JSON object with this structure:
        {{
            "architecture_summary": "High-level description of the system architecture",
            "data_models": [
                {{
                    "name": "ModelName",
                    "description": "What this model represents",
                    "fields": ["field1: type", "field2: type"],
                    "relationships": ["Relationship descriptions"]
                }}
            ],
            "api_endpoints": [
                {{
                    "method": "GET|POST|PUT|DELETE",
                    "path": "/api/endpoint",
                    "description": "What this endpoint does",
                    "request_schema": "Request format description",
                    "response_schema": "Response format description"
                }}
            ],
            "technical_decisions": {{
                "database": "Choice and rationale",
                "framework": "Choice and rationale", 
                "deployment": "Choice and rationale",
                "authentication": "Choice and rationale"
            }},
            "implementation_phases": [
                {{
                    "phase": "Phase 1",
                    "description": "What gets built in this phase",
                    "deliverables": ["List", "of", "outputs"]
                }}
            ],
            "assumptions": ["List", "of", "assumptions", "made"],
            "alternatives_considered": [
                {{
                    "decision": "What decision was made",
                    "alternatives": ["Option A", "Option B"],
                    "rationale": "Why the chosen approach is preferred"
                }}
            ]
        }}
        
        {'' if not is_fallback else 'Note: Since only the user goal was provided, make reasonable assumptions about requirements and clearly document them in the assumptions array.'}
        """

    def _format_specification_for_user(self, spec_data: Dict) -> str:
        """Formats the LLM's specification for user review."""
        formatted = "## Technical Specification\n\n"
        formatted += f"**Architecture Summary:** {spec_data.get('architecture_summary', 'N/A')}\n\n"
        
        if spec_data.get('data_models'):
            formatted += "**Data Models:**\n"
            for model in spec_data['data_models']:
                formatted += f"• **{model.get('name', 'Unknown Model')}:** {model.get('description', '')}\n"
                if model.get('fields'):
                    formatted += f"  - Fields: {', '.join(model['fields'])}\n"
            formatted += "\n"
            
        if spec_data.get('api_endpoints'):
            formatted += "**API Endpoints:**\n"
            for endpoint in spec_data['api_endpoints']:
                formatted += f"• **{endpoint.get('method', 'GET')} {endpoint.get('path', '/unknown')}:** {endpoint.get('description', '')}\n"
            formatted += "\n"
            
        if spec_data.get('technical_decisions'):
            formatted += "**Technical Decisions:**\n"
            for key, value in spec_data['technical_decisions'].items():
                formatted += f"• **{key.title()}:** {value}\n"
            formatted += "\n"
            
        if spec_data.get('implementation_phases'):
            formatted += "**Implementation Phases:**\n"
            for i, phase in enumerate(spec_data['implementation_phases'], 1):
                phase_name = phase.get('phase', f'Phase {i}') if isinstance(phase, dict) else f'Phase {i}'
                phase_desc = phase.get('description', str(phase)) if isinstance(phase, dict) else str(phase)
                formatted += f"{i}. **{phase_name}:** {phase_desc}\n"
            formatted += "\n"
            
        if spec_data.get('assumptions'):
            formatted += "**Key Assumptions:**\n"
            for assumption in spec_data['assumptions']:
                formatted += f"• {assumption}\n"
            formatted += "\n"
            
        if spec_data.get('alternatives_considered'):
            formatted += "**Alternatives Considered:**\n"
            for alt in spec_data['alternatives_considered']:
                if isinstance(alt, dict):
                    formatted += f"• **{alt.get('decision', 'Decision')}:** {alt.get('rationale', 'No rationale provided')}\n"
                else:
                    formatted += f"• {alt}\n"
            formatted += "\n"
            
        return formatted

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
                    for field in model['fields']:
                        content += f"- {field}\n"
                    content += "\n"
                if model.get('relationships'):
                    content += "**Relationships:**\n"
                    for rel in model['relationships']:
                        content += f"- {rel}\n"
                    content += "\n"
                    
        if spec_data.get('api_endpoints'):
            content += "## API Endpoints\n\n"
            for endpoint in spec_data['api_endpoints']:
                content += f"### {endpoint.get('method', 'GET')} {endpoint.get('path', '/unknown')}\n\n"
                content += f"{endpoint.get('description', '')}\n\n"
                if endpoint.get('request_schema'):
                    content += f"**Request:** {endpoint['request_schema']}\n\n"
                if endpoint.get('response_schema'):
                    content += f"**Response:** {endpoint['response_schema']}\n\n"
                    
        if spec_data.get('technical_decisions'):
            content += "## Technical Decisions\n\n"
            for key, value in spec_data['technical_decisions'].items():
                content += f"**{key.title()}:** {value}\n\n"
                
        if spec_data.get('implementation_phases'):
            content += "## Implementation Plan\n\n"
            for i, phase in enumerate(spec_data['implementation_phases'], 1):
                phase_name = phase.get('phase', f'Phase {i}') if isinstance(phase, dict) else f'Phase {i}'
                phase_desc = phase.get('description', str(phase)) if isinstance(phase, dict) else str(phase)
                content += f"### {phase_name}\n\n{phase_desc}\n\n"
                if isinstance(phase, dict) and phase.get('deliverables'):
                    content += "**Deliverables:**\n"
                    for deliverable in phase['deliverables']:
                        content += f"- {deliverable}\n"
                    content += "\n"
                    
        if spec_data.get('assumptions'):
            content += "## Assumptions\n\n"
            for assumption in spec_data['assumptions']:
                content += f"- {assumption}\n"
            content += "\n"
            
        if spec_data.get('alternatives_considered'):
            content += "## Alternatives Considered\n\n"
            for alt in spec_data['alternatives_considered']:
                if isinstance(alt, dict):
                    content += f"**{alt.get('decision', 'Decision')}**\n\n"
                    if alt.get('alternatives'):
                        content += f"Options considered: {', '.join(alt['alternatives'])}\n\n"
                    content += f"Rationale: {alt.get('rationale', 'No rationale provided')}\n\n"
                else:
                    content += f"- {alt}\n"
            content += "\n"
            
        return content

    def execute(self, goal: str, context: GlobalContext, current_task: TaskNode) -> AgentResponse:
        logger.info(f"SpecValidationAgent executing with goal: '{goal}'")
        
        # Report meaningful progress
        self.report_progress("Validating specification", f"Processing goal: '{goal[:80]}...'")

        input_artifact_key = current_task.input_artifact_keys[0] if current_task.input_artifact_keys else "clarified_requirements.md"
        requirements_doc = context.get_artifact(input_artifact_key)

        # Handle missing inputs gracefully - work with what we have
        is_fallback = False
        if not requirements_doc:
            # No clarified requirements available - work directly from the goal
            requirements_doc = f"User Goal: {goal}\n\nNote: No clarified requirements were available, so generating spec directly from the user's goal."
            is_fallback = True
            logger.info(f"No artifact '{input_artifact_key}' found. Working directly from goal: '{goal}'")
            context.log_event("spec_generator_fallback", {
                "reason": "missing_clarified_requirements", 
                "working_from": "goal_only",
                "expected_artifact": input_artifact_key
            })
            self.report_thinking(f"No clarified requirements found. I'll work directly from the user's goal and make reasonable assumptions about the technical requirements.")
        else:
            # Show the requirements that were found
            self.report_intermediate_output("requirements", requirements_doc[:500])

        # Detect quality level from goal and context
        quality_level = self._detect_quality_level(goal, context)
        logger.info(f"Using spec quality level: {quality_level.value.upper()}")
        
        if quality_level != SpecQuality.FAST:
            self.report_thinking(f"I'll generate a {quality_level.value} quality specification - this means more detailed technical requirements and comprehensive coverage.")
        
        # Define JSON schema for structured specification
        spec_schema = {
            "type": "object",
            "properties": {
                "architecture_summary": {"type": "string"},
                "data_models": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "description": {"type": "string"},
                            "fields": {"type": "array", "items": {"type": "string"}},
                            "relationships": {"type": "array", "items": {"type": "string"}}
                        }
                    }
                },
                "api_endpoints": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "method": {"type": "string"},
                            "path": {"type": "string"},
                            "description": {"type": "string"},
                            "request_schema": {"type": "string"},
                            "response_schema": {"type": "string"}
                        }
                    }
                },
                "technical_decisions": {"type": "object"},
                "implementation_phases": {"type": "array"},
                "assumptions": {"type": "array", "items": {"type": "string"}},
                "alternatives_considered": {"type": "array"}
            },
            "required": ["architecture_summary"]
        }

        # Validation loop - continues until user approves or provides empty input
        max_iterations = 3  # Prevent infinite loops
        iteration = 0
        
        while iteration < max_iterations:
            try:
                # Generate the specification
                prompt = self._build_prompt(requirements_doc, is_fallback=is_fallback, quality=quality_level)
                
                # Use regular invoke by default, fall back to function calling if needed
                logger.debug("Using regular invoke method (primary approach)")
                response_str = self.llm_client.invoke(prompt)
                
                # Check if the response is valid JSON with actual content
                try:
                    test_parse = json.loads(response_str)
                    if not isinstance(test_parse, dict) or not test_parse.get('architecture_summary'):
                        raise ValueError("Invalid or empty response")
                    logger.debug("Regular invoke succeeded with valid JSON response")
                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning(f"Regular invoke failed to produce valid JSON ({e}), trying function calling")
                    if hasattr(self.llm_client, 'invoke_with_schema'):
                        try:
                            response_str = self.llm_client.invoke_with_schema(prompt, spec_schema)
                            logger.debug("Function calling fallback succeeded")
                        except Exception as fallback_error:
                            logger.error(f"Function calling fallback also failed: {fallback_error}")
                            # Keep the original response from regular invoke for error reporting
                            pass
                    else:
                        logger.warning("Function calling not available, keeping original response")
                
                spec_data = json.loads(response_str)
                
                # Present the specification to the user
                presentation = self._format_specification_for_user(spec_data)
                
                try:
                    # Ask user for validation/feedback - empty input means approval
                    user_feedback = self.ui.prompt_for_input(
                        f"{presentation}\n\n"
                        "Please review my technical specification above. Press ENTER to approve and continue, "
                        "or provide feedback/corrections to refine the specification:"
                    )
                    
                    # Empty input (just Enter) means approval
                    if not user_feedback or user_feedback.strip() == "":
                        logger.info("User approved the specification")
                        break
                    else:
                        # User provided feedback - update the requirements for next iteration
                        logger.info(f"User provided feedback: {user_feedback[:100]}...")
                        requirements_doc += f"\n\nUser feedback on specification: {user_feedback}"
                        iteration += 1
                        continue
                        
                except NotImplementedError as e:
                    msg = f"Cannot get user input: {e}"
                    logger.critical(msg)
                    return AgentResponse(success=False, message=msg)
                except Exception as e:
                    error_msg = f"An error occurred during user interaction: {e}"
                    logger.error(error_msg, exc_info=True)
                    return AgentResponse(success=False, message=error_msg)
                    
            except NotImplementedError as e:
                msg = f"Cannot execute validation: {e}"
                logger.critical(msg)
                return AgentResponse(success=False, message=msg)
            except (json.JSONDecodeError, TypeError) as e:
                error_msg = f"Failed to get valid specification from LLM. Response was not valid JSON. Error: {e}"
                logger.error(error_msg)
                return AgentResponse(success=False, message=error_msg)
            except Exception as e:
                error_msg = f"An unexpected error occurred while generating specification: {e}"
                logger.critical(error_msg, exc_info=True)
                return AgentResponse(success=False, message=error_msg)

        if iteration >= max_iterations:
            logger.warning("Max iterations reached in validation loop")
            
        # Compile the final specification document
        technical_spec_content = self._compile_specification_document(spec_data)
        
        # Store the specification
        output_artifact_key = "technical_spec.md"
        context.add_artifact(key=output_artifact_key, value=technical_spec_content, source_task_id=current_task.task_id)
        
        return AgentResponse(success=True, message="Successfully validated and generated technical specification.", artifacts_generated=[output_artifact_key])


# --- Self-Testing Block ---
if __name__ == "__main__":
    import unittest
    from unittest.mock import MagicMock
    import shutil
    from pathlib import Path
    from utils.logger import setup_logger

    setup_logger(default_level=logging.INFO)

    class TestSpecValidationAgent(unittest.TestCase):

        def setUp(self):
            self.test_workspace_path = "./temp_spec_agent_test_ws"
            if Path(self.test_workspace_path).exists():
                shutil.rmtree(self.test_workspace_path)
            
            self.mock_llm_client = MagicMock(spec=LLMClient)
            self.mock_ui = MagicMock(spec=CollaborationUI)
            self.context = GlobalContext(workspace_path=self.test_workspace_path)
            self.agent = SpecValidationAgent(llm_client=self.mock_llm_client, ui_interface=self.mock_ui)
            self.task = TaskNode(
                goal="Create a spec",
                assigned_agent="SpecValidationAgent",
                input_artifact_keys=["clarified_requirements.md"]
            )
            self.context.add_artifact("clarified_requirements.md", "User wants a test API.", "task_clarify")

        def tearDown(self):
            shutil.rmtree(self.test_workspace_path)

        def test_user_approves_specification(self):
            print("\n--- [Test Case 1: User Approves Specification] ---")
            mock_spec_data = {
                "architecture_summary": "REST API with microservices architecture",
                "data_models": [{"name": "User", "description": "User entity", "fields": ["id: UUID", "email: string"]}],
                "technical_decisions": {"database": "PostgreSQL", "framework": "FastAPI"}
            }
            self.mock_llm_client.invoke.return_value = json.dumps(mock_spec_data)
            self.mock_ui.prompt_for_input.return_value = ""  # Empty string means approval

            response = self.agent.execute(self.task.goal, self.context, self.task)

            self.mock_llm_client.invoke.assert_called_once()
            self.mock_ui.prompt_for_input.assert_called_once()
            self.assertTrue(response.success)
            
            generated_spec = self.context.get_artifact("technical_spec.md")
            self.assertIn("REST API with microservices", generated_spec)
            self.assertIn("PostgreSQL", generated_spec)
            logger.info("✅ test_user_approves_specification: PASSED")

        def test_user_provides_feedback(self):
            print("\n--- [Test Case 2: User Provides Feedback] ---")
            spec_data1 = {
                "architecture_summary": "Simple REST API",
                "technical_decisions": {"database": "SQLite"}
            }
            spec_data2 = {
                "architecture_summary": "Scalable REST API with caching and load balancing",
                "technical_decisions": {"database": "PostgreSQL with Redis caching"}
            }
            
            self.mock_llm_client.invoke.side_effect = [
                json.dumps(spec_data1),
                json.dumps(spec_data2)
            ]
            self.mock_ui.prompt_for_input.side_effect = [
                "I need it to be more scalable with caching",  # Feedback
                ""  # Approval
            ]

            response = self.agent.execute(self.task.goal, self.context, self.task)

            self.assertEqual(self.mock_llm_client.invoke.call_count, 2)
            self.assertEqual(self.mock_ui.prompt_for_input.call_count, 2)
            self.assertTrue(response.success)
            
            generated_spec = self.context.get_artifact("technical_spec.md")
            self.assertIn("scalable", generated_spec.lower())
            self.assertIn("caching", generated_spec.lower())
            logger.info("✅ test_user_provides_feedback: PASSED")

    unittest.main(argv=['first-arg-is-ignored'], exit=False)