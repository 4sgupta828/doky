# agents/code_analysis.py
import logging
from typing import Dict, Any, List

# Foundational dependencies
from .base import BaseAgent
from core.context import GlobalContext
from core.models import AgentResponse, AgentResult, TaskNode

# Import the comprehensive validation tool
from tools.code_validator import validate_code_comprehensive

# Get a logger instance for this module
logger = logging.getLogger(__name__)


class CodeAnalysisAgent(BaseAgent):
    """
    Analysis Tier: Read-only code analysis and validation.
    
    This agent provides read-only analysis of code without modification.
    
    Responsibilities:
    - Python syntax validation
    - Import resolution checking
    - Static code analysis
    - AST parsing and validation
    - Code compilation verification
    """

    def __init__(self):
        super().__init__(
            name="CodeAnalysisAgent",
            description="Analyzes Python code syntax, imports, and performs static analysis."
        )

    def required_inputs(self) -> List[str]:
        """Required inputs for CodeAnalysisAgent execution."""
        return ["code_files"]

    def optional_inputs(self) -> List[str]:
        """Optional inputs for CodeAnalysisAgent execution."""
        return [
            "python_path",
            "validation_level",
            "check_imports",
            "check_syntax_only",
            "ignore_warnings"
        ]

    def execute_v2(self, goal: str, inputs: Dict[str, Any], global_context: GlobalContext) -> AgentResult:
        """
        NEW INTERFACE: Validate Python code with comprehensive checks using the code_validator tool.
        """
        logger.info(f"CodeAnalysisAgent executing: '{goal}'")
        
        # Validate inputs with graceful handling
        try:
            self.validate_inputs(inputs)
        except Exception as validation_error:
            return self.create_result(
                success=False,
                message=str(validation_error),
                error_details={"validation_error": str(validation_error)}
            )

        # Extract inputs
        code_files = inputs["code_files"]
        python_path = inputs.get("python_path")
        validation_level = inputs.get("validation_level", "standard")  # basic, standard, strict
        check_imports = inputs.get("check_imports", True)
        check_syntax_only = inputs.get("check_syntax_only", False)
        ignore_warnings = inputs.get("ignore_warnings", False)

        try:
            self.report_progress("Starting code validation", f"Validating {len(code_files)} files")

            # Use the centralized validation tool
            validation_results = validate_code_comprehensive(
                code_files=code_files,
                python_path=python_path,
                validation_level=validation_level,
                check_imports=check_imports,
                check_syntax_only=check_syntax_only,
                ignore_warnings=ignore_warnings
            )

            # Create success message
            if validation_results["overall_success"]:
                message = f"Code validation passed: {validation_results['files_passed']}/{len(code_files)} files validated successfully"
            else:
                message = f"Code validation completed with issues: {validation_results['files_passed']}/{len(code_files)} files passed"

            self.report_progress("Code validation complete", message)

            return self.create_result(
                success=validation_results["overall_success"] or validation_level != "strict",
                message=message,
                outputs=validation_results
            )

        except Exception as e:
            error_msg = f"CodeAnalysisAgent execution failed: {e}"
            logger.error(error_msg, exc_info=True)
            return self.create_result(
                success=False,
                message=error_msg,
                error_details={"exception": str(e)}
            )


    # Legacy execute method for backward compatibility
    def execute(self, goal: str, context: GlobalContext, current_task: TaskNode) -> AgentResponse:
        """Legacy execute method for backward compatibility."""
        # Try to get code files from workspace
        code_files = {}
        try:
            workspace_files = context.workspace.list_files()
            for file_path in workspace_files:
                if file_path.endswith('.py'):
                    content = context.workspace.get_file_content(file_path)
                    if content:
                        code_files[file_path] = content
        except:
            pass
        
        if not code_files:
            return AgentResponse(
                success=False,
                message="No Python code files found to validate"
            )
        
        inputs = {"code_files": code_files}
        result = self.execute_v2(goal, inputs, context)
        
        return AgentResponse(
            success=result.success,
            message=result.message,
            artifacts_generated=[]
        )