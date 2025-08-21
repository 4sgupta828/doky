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
from typing import Dict, Any, List, Optional
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
    generate_tests, TestGenerationContext, TestType, TestFramework
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
            # Determine creation type from goal and inputs
            creation_type = self._determine_creation_type(goal, inputs)
            
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

    def _determine_creation_type(self, goal: str, inputs: Dict[str, Any]) -> CreationType:
        """Determine the type of creation based on goal and inputs."""
        goal_lower = goal.lower()
        
        # Check for specific keywords and input types
        if any(word in goal_lower for word in ["test", "unit test", "integration test", "cli test"]):
            return CreationType.TESTS
        elif any(word in goal_lower for word in ["documentation", "readme", "api doc", "user guide"]):
            return CreationType.DOCUMENTATION
        elif any(word in goal_lower for word in ["specification", "spec", "technical spec", "requirements"]):
            return CreationType.SPECIFICATION
        elif any(word in goal_lower for word in ["manifest", "file structure", "project structure"]):
            return CreationType.MANIFEST
        elif any(word in goal_lower for word in ["full project", "complete project", "scaffold"]):
            return CreationType.FULL_PROJECT
        elif any(word in goal_lower for word in ["code", "function", "class", "module", "implement"]):
            return CreationType.CODE
        
        # Check inputs for type hints
        if "test_type" in inputs or "test_framework" in inputs:
            return CreationType.TESTS
        elif "doc_type" in inputs or "documentation_type" in inputs:
            return CreationType.DOCUMENTATION
        elif "requirements" in inputs and "clarified_requirements" in inputs:
            return CreationType.SPECIFICATION
        elif "technical_spec" in inputs and "file_manifest" not in inputs:
            return CreationType.MANIFEST
        elif "code_requirements" in inputs or "function_signature" in inputs:
            return CreationType.CODE
        
        # Default to code generation
        return CreationType.CODE

    def _handle_code_generation(self, goal: str, inputs: Dict[str, Any], global_context: GlobalContext) -> AgentResult:
        """Handle code generation tasks."""
        self.report_progress("Generating code", f"Goal: {goal}")
        
        # Extract parameters from inputs
        language = inputs.get("language", "Python")
        quality = inputs.get("quality", CodeQuality.DECENT.value)
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
            quality_enum = CodeQuality.DECENT

        # Create context for code generation
        context = CodeGenerationContext(
            goal=goal,
            technical_spec=technical_spec,
            language=lang_enum,
            quality_level=quality_enum,
            files_to_generate=inputs.get("files_to_generate", None),
            existing_code=inputs.get("existing_code", None),
            dependencies=inputs.get("dependencies", None),
            frameworks=inputs.get("frameworks", None)
        )

        # Generate code using the tools
        result = generate_code(context, self.llm_client)
        
        if result.success:
            self.report_intermediate_output("generated_code", result.generated_files)
            return AgentResult(
                success=True,
                message="Successfully generated code",
                outputs={
                    "generated_code": result.generated_files,
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
        
        # Extract parameters from inputs
        test_type = inputs.get("test_type", TestType.UNIT.value)
        framework = inputs.get("test_framework", TestFramework.PYTEST.value)
        language = inputs.get("language", "Python")
        code_to_test = inputs.get("code_to_test", inputs.get("target_code", ""))
        requirements = inputs.get("test_requirements", goal)
        
        # Map string values to enums
        try:
            test_type_enum = TestType(test_type) if isinstance(test_type, str) else test_type
        except ValueError:
            test_type_enum = TestType.UNIT
            
        try:
            framework_enum = TestFramework(framework) if isinstance(framework, str) else framework
        except ValueError:
            framework_enum = TestFramework.PYTEST

        # Create context for test generation
        context = TestGenerationContext(
            test_requirements=requirements,
            test_type=test_type_enum,
            framework=framework_enum,
            language=language,
            code_to_test=code_to_test,
            project_context=inputs.get("project_context", ""),
            files_context=global_context.workspace.list_files() if global_context.workspace else []
        )

        # Generate tests using the tools
        result = generate_tests(context, self.llm_client)
        
        if result.success:
            self.report_intermediate_output("generated_tests", result.generated_tests)
            return AgentResult(
                success=True,
                message="Successfully generated tests",
                outputs={
                    "generated_tests": result.generated_tests,
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
        style = inputs.get("documentation_style", TemplateStyle.STANDARD.value)
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
        style = inputs.get("specification_style", SpecificationStyle.DETAILED.value)
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
        """Handle full project creation by orchestrating multiple creation tasks."""
        self.report_progress("Creating full project", f"Goal: {goal}")
        
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
                message="Successfully created full project scaffold",
                outputs=project_outputs
            )
            
        except Exception as e:
            return AgentResult(
                success=False,
                message=f"Failed to create full project: {str(e)}"
            )

    def required_inputs(self) -> List[str]:
        """Return required inputs based on creation type."""
        # CreatorAgent is flexible and can work with various input combinations
        return []

    def supports_goal(self, goal: str) -> bool:
        """Check if this agent supports the given goal."""
        creation_keywords = [
            "create", "generate", "write", "build", "implement", "develop",
            "code", "test", "documentation", "spec", "manifest", "project"
        ]
        goal_lower = goal.lower()
        return any(keyword in goal_lower for keyword in creation_keywords)