# agents/environment_analysis.py
import logging
from typing import Dict, Any, List

# Foundational dependencies
from .base import BaseAgent
from core.context import GlobalContext
from core.models import AgentResult
from tools.environment_tools import EnvironmentTools

# Get a logger instance for this module
logger = logging.getLogger(__name__)


class EnvironmentAnalysisAgent(BaseAgent):
    """
    Analysis Tier: Environment state diagnosis ONLY.
    
    This agent provides read-only analysis of the development environment
    using structured inputs and leveraging environment tools.
    
    Responsibilities:
    - Python version and virtual environment analysis
    - System dependency and PATH analysis
    - Environment configuration validation
    - Development tool availability assessment
    
    Does NOT: Create environments, install anything, modify configurations
    """

    def __init__(self):
        super().__init__(
            name="EnvironmentAnalysisAgent",
            description="Analyzes development environment state, Python installations, and tool availability using structured inputs."
        )

    def required_inputs(self) -> List[str]:
        """Required inputs for EnvironmentAnalysisAgent execution."""
        return []  # Can analyze without specific inputs

    def optional_inputs(self) -> List[str]:
        """Optional inputs for EnvironmentAnalysisAgent execution."""
        return [
            "working_directory",
            "check_tools",
            "check_python_packages",
            "analyze_virtual_env",
            "check_system_dependencies",
            "detailed_analysis"
        ]

    def execute_v2(self, goal: str, inputs: Dict[str, Any], global_context: GlobalContext) -> AgentResult:
        """
        Analyze development environment state using environment tools.
        """
        logger.info(f"EnvironmentAnalysisAgent executing: '{goal}'")
        
        # Validate inputs
        try:
            self.validate_inputs(inputs)
        except Exception as validation_error:
            return self.create_result(
                success=False,
                message=str(validation_error),
                error_details={"validation_error": str(validation_error)}
            )

        # Extract inputs
        working_directory = inputs.get("working_directory", str(global_context.workspace_path))
        check_tools = inputs.get("check_tools", True)
        check_python_packages = inputs.get("check_python_packages", True)
        analyze_virtual_env = inputs.get("analyze_virtual_env", True)
        check_system_dependencies = inputs.get("check_system_dependencies", True)
        detailed_analysis = inputs.get("detailed_analysis", False)

        try:
            self.report_progress("Starting environment analysis", f"Working in {working_directory}")

            # Use EnvironmentTools for all analysis
            analysis_results = {
                "system_info": EnvironmentTools.analyze_system_info(),
                "python_info": EnvironmentTools.analyze_python_environment(),
                "virtual_env_info": None,
                "development_tools": None,
                "python_packages": None,
                "system_dependencies": None,
                "environment_variables": EnvironmentTools.analyze_environment_variables(),
                "path_analysis": EnvironmentTools.analyze_system_path()
            }

            # Optional detailed analyses using tools
            if analyze_virtual_env:
                analysis_results["virtual_env_info"] = EnvironmentTools.analyze_virtual_environment(working_directory)

            if check_tools:
                analysis_results["development_tools"] = EnvironmentTools.check_development_tools(detailed_analysis)

            if check_python_packages:
                analysis_results["python_packages"] = EnvironmentTools.analyze_python_packages(working_directory)

            if check_system_dependencies:
                analysis_results["system_dependencies"] = EnvironmentTools.check_system_dependencies()

            # Overall environment health assessment using tools
            health_assessment = EnvironmentTools.assess_environment_health(analysis_results)

            self.report_progress("Environment analysis complete", health_assessment["summary"])

            return self.create_result(
                success=True,
                message=f"Environment analysis complete: {health_assessment['summary']}",
                outputs={
                    "analysis_results": analysis_results,
                    "health_assessment": health_assessment,
                    "working_directory": working_directory
                }
            )

        except Exception as e:
            error_msg = f"EnvironmentAnalysisAgent execution failed: {e}"
            logger.error(error_msg, exc_info=True)
            return self.create_result(
                success=False,
                message=error_msg,
                error_details={"exception": str(e)}
            )