# fagents/creator.py
"""
CreatorAgent - Foundational agent for all creation tasks.

This agent consolidates the creation capabilities from:
- CoderAgent (code generation)
- TestGenerationAgent (test generation)  
- CLITestGeneratorAgent (CLI test generation)
- DocumentationAgent (documentation generation)
- SpecGeneratorAgent (specification creation)
- CodeManifestAgent (structure planning)

The CreatorAgent is responsible for generating any type of content needed in software development,
from technical specifications to actual code, tests, and documentation.
"""

import logging
from typing import Dict, Any, List
from enum import Enum

from .base import FoundationalAgent
from core.context import GlobalContext
from core.models import AgentResult

# Import all creation tools
from tools.code_generation_tools import (
    generate_code, CodeGenerationContext, CodeQuality, 
    CodeLanguage
)
from tools.test_generation_tools import (
    generate_tests, TestGenerationContext, TestType, TestFramework, TestQuality
)
from tools.documentation_generation_tools import (
    generate_documentation, DocumentationContext, DocumentationType,
    TemplateStyle, DocumentationFormat
)
from tools.specification_generation_tools import (
    generate_specification, SpecificationContext, SpecificationType, SpecificationStyle
)
from tools.manifest_generation_tools import (
    generate_manifest, ManifestContext, ProjectType, ProjectStructure
)

# LLM-based routing (kept for potential future use, but not actively used in main execution path)
from .routing import LLMRouter

logger = logging.getLogger(__name__)

class CreationType(Enum):
    """Types of creation operations the CreatorAgent can perform."""
    CODE = "code"
    TESTS = "tests"
    DOCUMENTATION = "documentation"
    SPECIFICATION = "specification"
    MANIFEST = "manifest"
    FULL_PROJECT = "full_project"

class CreatorAgent(FoundationalAgent):
    """
    Foundational Creator Agent for generating any type of software development content.
    
    This agent can create:
    - Code in multiple programming languages
    - Tests (unit, integration, CLI, API, etc.)
    - Documentation (README, API docs, user guides, etc.)
    - Technical specifications from requirements
    - Project file manifests and structure planning
    - Complete project scaffolds
    """

    def __init__(self, llm_client: Any = None):
        super().__init__(
            name="CreatorAgent",
            description="Foundational agent for generating code, tests, documentation, specifications, and project structures"
        )
        self.llm_client = llm_client
        # Note: Removed LLMRouter - InterAgentRouter now provides creation_type directly

    def get_capabilities(self) -> List[str]:
        """Return list of creation capabilities."""
        return [
            "code_generation",      # Generate code in multiple languages
            "test_generation",      # Generate various types of tests
            "documentation_generation",  # Generate all types of documentation
            "specification_creation",    # Create technical specifications
            "manifest_planning",         # Plan project file structures
            "project_scaffolding"       # Create complete project scaffolds
        ]

    def execute(self, goal: str, inputs: Dict[str, Any], global_context: GlobalContext) -> AgentResult:
        """
        Execute creation tasks based on the goal and inputs.
        
        Args:
            goal: Description of what to create
            inputs: Input data for creation
            global_context: Global execution context
            
        Returns:
            AgentResult with created content
        """
        self.report_progress("Starting creation task", goal)
        
        try:
            # PRIMARY: Trust InterAgentRouter's creation_type decision
            creation_type_str = inputs.get("creation_type")
            if creation_type_str:
                try:
                    creation_type = CreationType(creation_type_str)
                    self.report_progress("Creation type determined by InterAgentRouter", f"{creation_type.value}")
                    logger.info(f"Using InterAgentRouter creation_type: {creation_type.value}")
                except ValueError:
                    logger.warning(f"Invalid creation_type from InterAgentRouter: {creation_type_str}, falling back")
                    creation_type = self._determine_creation_type_fallback(goal, inputs)
            else:
                # FALLBACK: Simple heuristics for backward compatibility (no complex LLM routing)
                creation_type = self._determine_creation_type_fallback(goal, inputs)
                logger.info(f"Creation type determined via fallback heuristics: {creation_type.value}")
            
            if creation_type == CreationType.CODE:
                return self._handle_code_generation(goal, inputs, global_context)
            elif creation_type == CreationType.TESTS:
                return self._handle_test_generation(goal, inputs, global_context)
            elif creation_type == CreationType.DOCUMENTATION:
                return self._handle_documentation_generation(goal, inputs, global_context)
            elif creation_type == CreationType.SPECIFICATION:
                return self._handle_specification_creation(goal, inputs, global_context)
            elif creation_type == CreationType.MANIFEST:
                return self._handle_manifest_planning(goal, inputs, global_context)
            elif creation_type == CreationType.FULL_PROJECT:
                return self._handle_full_project_creation(goal, inputs, global_context)
            else:
                return AgentResult(
                    success=False,
                    message=f"Unknown creation type: {creation_type}"
                )
                
        except Exception as e:
            logger.error(f"Error in CreatorAgent execution: {e}")
            return AgentResult(
                success=False,
                message=f"Creation failed: {str(e)}"
            )

    def _determine_creation_type_fallback(self, goal: str, inputs: Dict[str, Any]) -> CreationType:
        """
        Simple fallback creation type detection for backward compatibility.
        
        Uses basic keyword detection - much simpler than the original complex heuristics.
        Primary detection should come from InterAgentRouter's creation_type flag.
        """
        
        if not goal:
            return CreationType.CODE
        
        goal_lower = goal.lower()
        
        # Very basic keyword detection (simplified from original complex logic)
        if "test" in goal_lower:
            return CreationType.TESTS
        elif any(word in goal_lower for word in ["documentation", "readme", "doc"]):
            return CreationType.DOCUMENTATION
        elif any(word in goal_lower for word in ["specification", "spec"]):
            return CreationType.SPECIFICATION
        elif "project" in goal_lower and any(word in goal_lower for word in ["full", "complete", "scaffold"]):
            return CreationType.FULL_PROJECT
        
        # Simple input-based detection
        if inputs.get("test_type") or inputs.get("test_framework"):
            return CreationType.TESTS
        elif inputs.get("doc_type"):
            return CreationType.DOCUMENTATION
        
        # Default to code generation (most common case)
        logger.info("Fallback creation type detection: defaulting to CODE")
        return CreationType.CODE

    def _handle_code_generation(self, goal: str, inputs: Dict[str, Any], global_context: GlobalContext) -> AgentResult:
        """Handle code generation tasks."""
        self.report_progress("Generating code", f"Goal: {goal}")
        
        # Extract parameters from inputs
        language = inputs.get("language", "Python")
        quality = inputs.get("quality", CodeQuality.FAST.value)
        requirements = inputs.get("code_requirements", inputs.get("requirements", goal))
        technical_spec = inputs.get("technical_spec", requirements)
        
        # Map string values to enums
        try:
            lang_enum = CodeLanguage(language) if isinstance(language, str) else language
        except ValueError:
            lang_enum = CodeLanguage.PYTHON
            
        try:
            quality_enum = CodeQuality(quality) if isinstance(quality, str) else quality
        except ValueError:
            quality_enum = CodeQuality.FAST

        # Get existing code from inputs and workspace
        existing_code = inputs.get("existing_code", {}) or {}
        workspace_context = global_context.get_workspace_code_context()
        existing_code.update(workspace_context)
        
        # Create context for code generation
        context = CodeGenerationContext(
            goal=goal,
            technical_spec=technical_spec,
            language=lang_enum,
            quality_level=quality_enum,
            files_to_generate=inputs.get("files_to_generate", None),
            existing_code=existing_code,
            dependencies=inputs.get("dependencies", None),
            frameworks=inputs.get("frameworks", None)
        )

        # Generate code using the tools
        result = generate_code(context, self.llm_client)
        
        if result.success:
            self.report_intermediate_output("generated_code", result.generated_files)
            
            # Write the generated code to the workspace (ported from CoderAgent)
            written_files = []
            for file_path, code_content in result.generated_files.items():
                if not isinstance(code_content, str) or not code_content.strip():
                    logger.warning(f"Skipping empty or invalid code for file '{file_path}'")
                    continue
                
                global_context.workspace.write_file_content(file_path, code_content, "CreatorAgent_code_generation")
                written_files.append(file_path)
            
            if not written_files:
                return AgentResult(success=False, message="Code generated but no valid files were written.")
            
            return AgentResult(
                success=True,
                message=f"Successfully generated and wrote {len(written_files)} code files",
                outputs={
                    "generated_code": result.generated_files,
                    "artifacts_generated": written_files,
                    "file_structure": result.file_structure,
                    "dependencies": result.dependencies,
                    "quality_metrics": result.quality_metrics
                }
            )
        else:
            return AgentResult(success=False, message=result.error_details or "Code generation failed")

    def _handle_test_generation(self, goal: str, inputs: Dict[str, Any], global_context: GlobalContext) -> AgentResult:
        """Handle test generation tasks."""
        self.report_progress("Generating tests", f"Goal: {goal}")
        
        # Extract parameters from inputs (structured inputs from InterAgentRouter)
        test_type = inputs.get("test_type", TestType.UNIT.value)
        framework = inputs.get("test_framework", TestFramework.PYTEST.value)
        language = inputs.get("language", "Python")
        code_to_test = inputs.get("code_to_test", inputs.get("target_code", ""))
        requirements = inputs.get("test_requirements", inputs.get("requirements", goal))
        
        # Map quality to test quality (defaults to FAST, LLM can upgrade)
        quality = inputs.get("quality", "fast")
        if quality == "production":
            test_quality_enum = TestQuality.PRODUCTION
        elif quality == "decent":
            test_quality_enum = TestQuality.DECENT
        else:  # fast
            test_quality_enum = TestQuality.FAST
        
        # Map string values to enums
        try:
            test_type_enum = TestType(test_type) if isinstance(test_type, str) else test_type
        except ValueError:
            test_type_enum = TestType.UNIT
            
        try:
            framework_enum = TestFramework(framework) if isinstance(framework, str) else framework
        except ValueError:
            framework_enum = TestFramework.PYTEST

        # Create context for test generation (fix parameter mismatch from migration)
        source_files_dict = {}
        if isinstance(code_to_test, dict):
            source_files_dict = code_to_test
        elif isinstance(code_to_test, str) and code_to_test.strip():
            source_files_dict = {"target_code.py": code_to_test}
        elif not code_to_test and global_context.workspace_path:
            # No specific code provided, discover Python files in workspace using shared tools
            from tools.file_system_tools import discover_python_source_files
            source_files_dict = discover_python_source_files(
                working_directory=str(global_context.workspace_path)
            )
        
        context = TestGenerationContext(
            goal=requirements,
            source_files=source_files_dict,
            test_type=test_type_enum,
            test_quality=test_quality_enum,
            framework=framework_enum
        )

        # Generate tests using the tools
        result = generate_tests(context, self.llm_client)
        
        if result.success:
            self.report_intermediate_output("generated_tests", result.generated_tests)
            
            # Write the generated tests to the workspace (ported from CoderAgent)
            written_files = []
            for file_path, test_content in result.generated_tests.items():
                if not isinstance(test_content, str) or not test_content.strip():
                    logger.warning(f"Skipping empty or invalid test file '{file_path}'")
                    continue
                
                global_context.workspace.write_file_content(file_path, test_content, "CreatorAgent_test_generation")
                written_files.append(file_path)
            
            if not written_files:
                return AgentResult(success=False, message="Tests generated but no valid files were written.")
            
            return AgentResult(
                success=True,
                message=f"Successfully generated and wrote {len(written_files)} test files",
                outputs={
                    "generated_tests": result.generated_tests,
                    "artifacts_generated": written_files,
                    "test_structure": result.test_structure,
                    "coverage_mapping": result.coverage_mapping,
                    "test_count": result.test_count
                }
            )
        else:
            return AgentResult(success=False, message=result.error_details or "Test generation failed")

    def _handle_documentation_generation(self, goal: str, inputs: Dict[str, Any], global_context: GlobalContext) -> AgentResult:
        """Handle documentation generation tasks."""
        self.report_progress("Generating documentation", f"Goal: {goal}")
        
        # Extract parameters from inputs
        doc_type = inputs.get("documentation_type", inputs.get("doc_type", DocumentationType.README.value))
        
        # Map quality to documentation style (FAST->MINIMAL, DECENT->STANDARD, PRODUCTION->COMPREHENSIVE)
        quality = inputs.get("quality", "fast")
        if quality == "fast":
            default_style = TemplateStyle.MINIMAL.value
        elif quality == "production":
            default_style = TemplateStyle.COMPREHENSIVE.value
        else:  # decent
            default_style = TemplateStyle.STANDARD.value
        
        style = inputs.get("documentation_style", default_style)
        format_type = inputs.get("documentation_format", DocumentationFormat.MARKDOWN.value)
        content_source = inputs.get("content_source", inputs.get("code_to_document", ""))
        requirements = inputs.get("doc_requirements", goal)
        
        # Map string values to enums
        try:
            doc_type_enum = DocumentationType(doc_type) if isinstance(doc_type, str) else doc_type
        except ValueError:
            doc_type_enum = DocumentationType.README
            
        try:
            style_enum = TemplateStyle(style) if isinstance(style, str) else style
        except ValueError:
            style_enum = TemplateStyle.STANDARD
            
        try:
            format_enum = DocumentationFormat(format_type) if isinstance(format_type, str) else format_type
        except ValueError:
            format_enum = DocumentationFormat.MARKDOWN

        # Create context for documentation generation
        context = DocumentationContext(
            requirements=requirements,
            doc_type=doc_type_enum,
            style=style_enum,
            format=format_enum,
            content_source=content_source,
            project_context=inputs.get("project_context", ""),
            target_audience=inputs.get("target_audience", "developers"),
            files_context=global_context.workspace.list_files() if global_context.workspace else []
        )

        # Generate documentation using the tools
        result = generate_documentation(context, self.llm_client)
        
        if result.success:
            self.report_intermediate_output("generated_documentation", result.generated_docs)
            return AgentResult(
                success=True,
                message="Successfully generated documentation",
                outputs={
                    "generated_documentation": result.generated_docs,
                    "documentation_structure": result.documentation_structure,
                    "quality_metrics": result.quality_metrics
                }
            )
        else:
            return AgentResult(success=False, message=result.error_details or "Documentation generation failed")

    def _handle_specification_creation(self, goal: str, inputs: Dict[str, Any], global_context: GlobalContext) -> AgentResult:
        """Handle technical specification creation tasks."""
        self.report_progress("Creating specification", f"Goal: {goal}")
        
        # Extract parameters from inputs
        requirements = inputs.get("clarified_requirements", inputs.get("requirements", goal))
        spec_type = inputs.get("specification_type", SpecificationType.TECHNICAL_SPEC.value)
        # Map quality to specification style (FAST->MINIMAL, DECENT->STRUCTURED, PRODUCTION->COMPREHENSIVE)
        quality = inputs.get("quality", "fast")
        if quality == "fast":
            default_style = SpecificationStyle.MINIMAL.value
        elif quality == "production":
            default_style = SpecificationStyle.COMPREHENSIVE.value
        else:  # decent
            default_style = SpecificationStyle.STRUCTURED.value
        
        style = inputs.get("specification_style", default_style)
        target_stack = inputs.get("target_stack", "Python/FastAPI")
        domain = inputs.get("domain", "")
        
        # Map string values to enums
        try:
            spec_type_enum = SpecificationType(spec_type) if isinstance(spec_type, str) else spec_type
        except ValueError:
            spec_type_enum = SpecificationType.TECHNICAL_SPEC
            
        try:
            style_enum = SpecificationStyle(style) if isinstance(style, str) else style
        except ValueError:
            style_enum = SpecificationStyle.DETAILED

        # Create context for specification generation
        context = SpecificationContext(
            requirements=requirements,
            spec_type=spec_type_enum,
            style=style_enum,
            target_stack=target_stack,
            existing_files=global_context.workspace.list_files() if global_context.workspace else [],
            domain=domain,
            constraints=inputs.get("constraints", [])
        )

        # Generate specification using the tools
        result = generate_specification(context, self.llm_client)
        
        if result.success:
            self.report_intermediate_output("technical_specification", result.specification_content)
            return AgentResult(
                success=True,
                message=result.message,
                outputs={
                    "technical_spec": result.specification_content,
                    "structured_data": result.structured_data,
                    "data_models": [{"name": m.name, "description": m.description, "fields": m.fields} for m in result.data_models],
                    "api_endpoints": [{"method": e.method, "path": e.path, "description": e.description} for e in result.api_endpoints]
                }
            )
        else:
            return AgentResult(success=False, message=result.message)

    def _handle_manifest_planning(self, goal: str, inputs: Dict[str, Any], global_context: GlobalContext) -> AgentResult:
        """Handle project manifest and structure planning tasks."""
        self.report_progress("Planning project structure", f"Goal: {goal}")
        
        # Extract parameters from inputs
        tech_spec = inputs.get("technical_spec", goal)
        project_type = inputs.get("project_type", ProjectType.WEB_API.value)
        structure_style = inputs.get("structure_style", ProjectStructure.MODULAR.value)
        target_language = inputs.get("target_language", "Python")
        package_name = inputs.get("package_name", "")
        
        # Map string values to enums
        try:
            project_type_enum = ProjectType(project_type) if isinstance(project_type, str) else project_type
        except ValueError:
            project_type_enum = ProjectType.WEB_API
            
        try:
            structure_enum = ProjectStructure(structure_style) if isinstance(structure_style, str) else structure_style
        except ValueError:
            structure_enum = ProjectStructure.MODULAR

        # Create context for manifest generation
        context = ManifestContext(
            technical_spec=tech_spec,
            project_type=project_type_enum,
            structure_style=structure_enum,
            target_language=target_language,
            existing_files=global_context.workspace.list_files() if global_context.workspace else [],
            include_tests=inputs.get("include_tests", True),
            include_docs=inputs.get("include_docs", True),
            include_config=inputs.get("include_config", True),
            package_name=package_name
        )

        # Generate manifest using the tools
        result = generate_manifest(context, self.llm_client)
        
        if result.success:
            self.report_intermediate_output("file_manifest", {"files_to_create": result.files_to_create})
            return AgentResult(
                success=True,
                message=result.message,
                outputs={
                    "file_manifest": {"files_to_create": result.files_to_create},
                    "directory_structure": result.directory_structure,
                    "file_details": [{"path": f.path, "description": f.description, "type": f.file_type} for f in result.file_details]
                }
            )
        else:
            return AgentResult(success=False, message=result.message)

    def _handle_full_project_creation(self, goal: str, inputs: Dict[str, Any], global_context: GlobalContext) -> AgentResult:
        """Handle full project creation by delegating to workflow orchestration tools."""
        self.report_progress("Creating full project", f"Goal: {goal}")
        
        try:
            # Delegate complex multi-step project creation to workflow orchestration
            return self._delegate_to_creation_orchestration(goal, inputs, global_context)
            
        except Exception as e:
            return AgentResult(
                success=False,
                message=f"Failed to create full project: {str(e)}"
            )

    # ====== CREATION WORKFLOW ORCHESTRATION ======
    
    def _delegate_to_creation_orchestration(self, goal: str, inputs: Dict[str, Any], 
                                          global_context: GlobalContext) -> AgentResult:
        """Delegate complex multi-step creation workflows to existing orchestration tools."""
        logger.info(f"Delegating multi-step creation workflow to orchestration tools: {goal[:100]}...")
        
        try:
            # Import orchestration tools
            from tools.workflow_orchestration_tools import (
                orchestrate_workflow, create_orchestration_context
            )
            from core.models import TaskGraph, TaskNode
            import time
            
            # Define creation workflow steps
            creation_steps = self._plan_creation_workflow_steps(goal, inputs)
            
            if not creation_steps:
                return AgentResult(
                    success=False,
                    message="Multi-step creation workflow detected but no steps could be planned",
                    error_details={"goal": goal, "inputs_keys": list(inputs.keys())}
                )
            
            # Convert creation steps to TaskGraph
            task_graph = self._create_creation_task_graph(creation_steps, goal)
            
            # Create orchestration context
            orchestration_context = create_orchestration_context(
                workflow_id=f"creator_workflow_{int(time.time())}",
                orchestration_mode="sequential",  # Creation steps are typically sequential
                max_parallel_tasks=1,  # CreatorAgent workflows are sequential
                error_handling="stop",  # Stop on first failure for creation tasks
                global_context=global_context
            )
            
            # Create mini agent registry for creation execution
            creation_agent_registry = {
                "CreationExecutor": self  # Use CreatorAgent itself for creation execution
            }
            
            # Execute workflow using existing orchestration tools
            orchestration_result = orchestrate_workflow(
                task_graph=task_graph,
                agent_registry=creation_agent_registry,
                context=orchestration_context
            )
            
            # Convert orchestration result to AgentResult
            success = orchestration_result.success
            message = self._create_creation_orchestration_message(orchestration_result, goal)
            
            outputs = {
                "orchestration_type": "delegated_creation_workflow",
                "workflow_id": orchestration_result.workflow_id,
                "total_steps": orchestration_result.total_steps,
                "completed_steps": orchestration_result.completed_steps,
                "failed_steps": orchestration_result.failed_steps,
                "duration_seconds": orchestration_result.total_duration_seconds,
                "final_outputs": orchestration_result.final_outputs,
                "creation_goal": goal,
                "tool_used": "workflow_orchestration_tools"
            }
            
            return AgentResult(
                success=success,
                message=message,
                outputs=outputs
            )
            
        except ImportError as e:
            logger.warning(f"Workflow orchestration tools not available: {e}. Falling back to internal orchestration.")
            # Fallback to internal orchestration if orchestration tools unavailable
            return self._handle_full_project_creation_internal(goal, inputs, global_context)
        except Exception as e:
            logger.error(f"Creation workflow orchestration failed: {e}", exc_info=True)
            return AgentResult(
                success=False,
                message=f"Creation workflow orchestration failed: {e}",
                error_details={
                    "exception": str(e),
                    "goal": goal,
                    "fallback": "Consider using internal creation orchestration"
                }
            )
    
    def _plan_creation_workflow_steps(self, goal: str, inputs: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Plan the creation workflow steps for full project creation."""
        
        steps = []
        
        # Step 1: Create specification if requirements provided
        if "requirements" in inputs and "technical_spec" not in inputs:
            steps.append({
                "step_name": "specification_creation",
                "goal": "Create technical specification from requirements",
                "creation_type": "specification",
                "inputs": {"requirements": inputs["requirements"]}
            })
        
        # Step 2: Create project manifest
        steps.append({
            "step_name": "manifest_planning",
            "goal": "Plan project structure and file manifest",
            "creation_type": "manifest",
            "inputs": {"technical_spec": inputs.get("technical_spec", goal)}
        })
        
        # Step 3: Generate core code files
        steps.append({
            "step_name": "code_generation", 
            "goal": "Generate core application code",
            "creation_type": "code",
            "inputs": {"code_requirements": inputs.get("technical_spec", goal)}
        })
        
        # Step 4: Generate tests
        steps.append({
            "step_name": "test_generation",
            "goal": "Generate tests for core code",
            "creation_type": "tests",
            "inputs": {"test_type": "unit", "test_framework": "pytest"}
        })
        
        return steps
    
    def _create_creation_task_graph(self, creation_steps: List[Dict[str, Any]], goal: str) -> 'TaskGraph':
        """Convert creation steps into a TaskGraph for orchestration."""
        from core.models import TaskGraph, TaskNode
        
        # Create task nodes for each creation step
        nodes = {}
        
        for i, step in enumerate(creation_steps):
            task_id = f"step_{i+1}_{step['step_name']}"
            
            # Create task node
            task_node = TaskNode(
                task_id=task_id,
                goal=step['goal'],
                assigned_agent="CreationExecutor",
                dependencies=[] if i == 0 else [f"step_{i}_{creation_steps[i-1]['step_name']}"]  # Sequential dependency
            )
            
            nodes[task_id] = task_node
        
        return TaskGraph(nodes=nodes)
    
    def _create_creation_orchestration_message(self, orchestration_result, goal: str) -> str:
        """Create message from creation orchestration result."""
        if orchestration_result.success:
            return (f"✅ Multi-step creation workflow completed successfully: {orchestration_result.completed_steps}/"
                   f"{orchestration_result.total_steps} creation steps completed in "
                   f"{orchestration_result.total_duration_seconds:.2f}s for '{goal}'")
        else:
            return (f"⚠️ Multi-step creation workflow completed with issues: {orchestration_result.completed_steps}/"
                   f"{orchestration_result.total_steps} steps completed, "
                   f"{orchestration_result.failed_steps} failed. Errors: {orchestration_result.error_summary}")
    
    def _handle_full_project_creation_internal(self, goal: str, inputs: Dict[str, Any], global_context: GlobalContext) -> AgentResult:
        """Fallback internal orchestration for full project creation (kept for backward compatibility)."""
        logger.info("Using internal creation orchestration as fallback")
        
        try:
            # Step 1: Create specification if requirements provided
            if "requirements" in inputs and "technical_spec" not in inputs:
                self.report_progress("Creating specification from requirements", "Step 1/4")
                spec_result = self._handle_specification_creation(goal, inputs, global_context)
                if not spec_result.success:
                    return spec_result
                inputs["technical_spec"] = spec_result.outputs["technical_spec"]
            
            # Step 2: Create project manifest
            self.report_progress("Planning project structure", "Step 2/4")
            manifest_result = self._handle_manifest_planning(goal, inputs, global_context)
            if not manifest_result.success:
                return manifest_result
            
            # Step 3: Generate core code files
            self.report_progress("Generating core code", "Step 3/4")
            code_inputs = inputs.copy()
            code_inputs["code_requirements"] = inputs.get("technical_spec", goal)
            code_result = self._handle_code_generation("Generate core application code", code_inputs, global_context)
            if not code_result.success:
                return code_result
            
            # Step 4: Generate tests
            self.report_progress("Generating tests", "Step 4/4")
            test_inputs = inputs.copy()
            test_inputs["code_to_test"] = code_result.outputs.get("generated_code", "")
            test_result = self._handle_test_generation("Generate tests for core code", test_inputs, global_context)
            
            # Compile all results
            project_outputs = {
                "technical_specification": inputs.get("technical_spec", ""),
                "file_manifest": manifest_result.outputs["file_manifest"],
                "generated_code": code_result.outputs["generated_code"],
                "generated_tests": test_result.outputs.get("generated_tests", "") if test_result.success else "",
                "directory_structure": manifest_result.outputs["directory_structure"]
            }
            
            return AgentResult(
                success=True,
                message="Successfully created full project scaffold (internal orchestration)",
                outputs=project_outputs
            )
            
        except Exception as e:
            return AgentResult(
                success=False,
                message=f"Internal project creation failed: {str(e)}"
            )

    def required_inputs(self) -> List[str]:
        """Return required inputs based on creation type."""
        # CreatorAgent is flexible and can work with various input combinations
        return []

    def supports_goal(self, goal: str) -> bool:
        """Check if this agent supports the given goal."""
        creation_keywords = [
            "create", "generate", "write", "build", "implement", "develop",
            "test", "documentation", "spec", "manifest", "project"
        ]
        
        # Check for creation-specific patterns
        creation_patterns = [
            "create code", "write code", "generate code", "implement code",
            "build code", "develop code", "code generation"
        ]
        
        goal_lower = goal.lower()
        
        # Exclude debugging and analysis goals
        exclusion_keywords = ["debug", "analyze", "fix", "diagnose", "troubleshoot"]
        if any(exclusion in goal_lower for exclusion in exclusion_keywords):
            return False
        
        # Check for creation keywords or specific code creation patterns
        return (any(keyword in goal_lower for keyword in creation_keywords) or
                any(pattern in goal_lower for pattern in creation_patterns))