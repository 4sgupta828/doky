# fagents/analyst.py
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

# Foundational base
from .base import FoundationalAgent
from core.context import GlobalContext
from core.models import AgentResult

# Analysis tools - atomic and reusable
from tools.code_validator import validate_python_syntax, validate_code_execution
from tools.environment_tools import EnvironmentTools
from tools.problem_analysis_tools import (
    classify_problem, analyze_errors, assess_severity, 
    analyze_root_causes, recognize_error_patterns, 
    generate_problem_recommendations
)
from tools.quality_analysis_tools import (
    analyze_code_quality, scan_for_security_patterns
)

# Get a logger instance for this module
logger = logging.getLogger(__name__)


class AnalystAgent(FoundationalAgent):
    """
    FOUNDATIONAL AGENT 1: ANALYST
    
    The Intelligence That Understands - Deep comprehension and diagnosis of any artifact or situation.
    
    Core Capability: Advanced analysis and diagnosis of code, environments, problems, and quality.
    
    Powers:
    - Advanced code analysis (AST parsing, dataflow analysis, architectural assessment)
    - Problem diagnosis and root cause analysis from logs, errors, symptoms  
    - Requirements analysis and gap identification
    - Quality assessment and technical debt evaluation
    - Error pattern recognition and failure mode analysis
    - Environment analysis (dependencies, configurations, system state)
    - Performance bottleneck identification
    
    Unique Value: Can understand and diagnose ANY existing artifact or problem state
    """
    
    def __init__(self):
        super().__init__(
            name="AnalystAgent",
            description="Deep comprehension and diagnosis agent that can analyze code, environments, problems, and quality."
        )
    
    def execute(self, goal: str, inputs: Dict[str, Any], global_context: GlobalContext) -> AgentResult:
        """
        Execute comprehensive analysis based on the goal and inputs.
        
        Supports multiple analysis types:
        - code_analysis: Analyze code syntax, imports, and structure
        - environment_analysis: Analyze development environment state
        - problem_analysis: Diagnose problems, errors, and failures
        - quality_analysis: Assess code quality, security, and maintainability
        - comprehensive_analysis: Run all relevant analyses
        """
        self.report_progress("Starting analysis", f"Goal: {goal}")
        
        try:
            # Determine analysis type from goal and inputs
            analysis_type = self._determine_analysis_type(goal, inputs)
            self.report_progress("Analysis type determined", analysis_type)
            
            # Execute the appropriate analysis
            if analysis_type == "code_analysis":
                return self._execute_code_analysis(goal, inputs, global_context)
            elif analysis_type == "environment_analysis":
                return self._execute_environment_analysis(goal, inputs, global_context)
            elif analysis_type == "problem_analysis":
                return self._execute_problem_analysis(goal, inputs, global_context)
            elif analysis_type == "quality_analysis":
                return self._execute_quality_analysis(goal, inputs, global_context)
            elif analysis_type == "comprehensive_analysis":
                return self._execute_comprehensive_analysis(goal, inputs, global_context)
            else:
                # Auto-detect based on available inputs
                return self._execute_auto_analysis(goal, inputs, global_context)
                
        except Exception as e:
            self.report_error(f"Analysis failed: {e}", e)
            return AgentResult(
                success=False,
                message=f"Analysis failed: {e}",
                outputs={},
                error_details={"exception": str(e)}
            )
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Return comprehensive capabilities description for the Strategist."""
        return {
            "name": "AnalystAgent",
            "description": "Deep comprehension and diagnosis of code, environments, problems, and quality",
            "primary_functions": [
                "Code analysis and validation",
                "Environment state diagnosis", 
                "Problem diagnosis and root cause analysis",
                "Quality assessment and security scanning",
                "Error pattern recognition",
                "Performance bottleneck identification"
            ],
            "input_types": [
                "code_files", "error_logs", "stack_trace", "problem_description",
                "environment_context", "quality_requirements", "analysis_scope"
            ],
            "output_types": [
                "validation_results", "environment_assessment", "problem_diagnosis", 
                "quality_report", "security_findings", "root_cause_analysis",
                "recommendations", "severity_assessment"
            ],
            "analysis_modes": [
                "code_analysis", "environment_analysis", "problem_analysis",
                "quality_analysis", "comprehensive_analysis", "auto_analysis"
            ],
            "complexity_handling": "Can analyze simple syntax errors to complex architectural problems"
        }
    
    def _determine_analysis_type(self, goal: str, inputs: Dict[str, Any]) -> str:
        """Determine the type of analysis to perform based on goal and inputs."""
        goal_lower = goal.lower()
        
        # Explicit analysis type requests
        if "code" in goal_lower and any(word in goal_lower for word in ["syntax", "validate", "check", "import"]):
            return "code_analysis"
        elif "environment" in goal_lower or "system" in goal_lower:
            return "environment_analysis"  
        elif any(word in goal_lower for word in ["problem", "error", "debug", "diagnose", "failure", "bug"]):
            return "problem_analysis"
        elif any(word in goal_lower for word in ["quality", "security", "audit", "review"]):
            return "quality_analysis"
        elif "comprehensive" in goal_lower or "full" in goal_lower:
            return "comprehensive_analysis"
            
        # Auto-detect based on inputs
        if "code_files" in inputs:
            if "problem_data" in inputs or "error_logs" in inputs:
                return "problem_analysis"
            else:
                return "code_analysis"
        elif "problem_data" in inputs or "error_logs" in inputs:
            return "problem_analysis"
        elif "environment_context" in inputs:
            return "environment_analysis"
        else:
            return "auto_analysis"
    
    def _execute_code_analysis(self, goal: str, inputs: Dict[str, Any], global_context: GlobalContext) -> AgentResult:
        """Execute code analysis using code validation tools."""
        self.report_progress("Code analysis", "Validating Python code")
        
        # Get code files
        code_files = inputs.get("code_files", {})
        if not code_files and global_context.workspace_path:
            # Auto-discover Python files
            code_files = self._discover_python_files(global_context.workspace_path)
        
        if not code_files:
            return AgentResult(
                success=False,
                message="No code files provided or found for analysis",
                outputs={}
            )
        
        # Extract analysis options
        python_path = inputs.get("python_path")
        validation_level = inputs.get("validation_level", "standard")
        check_imports = inputs.get("check_imports", True)
        check_syntax_only = inputs.get("check_syntax_only", False)
        ignore_warnings = inputs.get("ignore_warnings", False)
        
        # Run code validation using available tools
        syntax_results = validate_python_syntax(code_files)
        
        # Create simplified validation results structure
        validation_results = {
            "overall_success": all(result.result == "passed" for result in syntax_results),
            "files_passed": sum(1 for result in syntax_results if result.result == "passed"),
            "errors": [result for result in syntax_results if result.result == "failed"],
            "syntax_results": syntax_results
        }
        
        success = validation_results.get("overall_success", False)
        files_passed = validation_results.get("files_passed", 0)
        total_files = len(code_files)
        
        message = f"Code analysis complete: {files_passed}/{total_files} files passed validation"
        if not success:
            message += f" ({len(validation_results.get('errors', []))} errors found)"
        
        self.report_progress("Code analysis complete", message)
        
        return AgentResult(
            success=success or validation_level != "strict",
            message=message,
            outputs={
                "analysis_type": "code_analysis",
                "validation_results": validation_results,
                "files_analyzed": total_files,
                "files_passed": files_passed,
                "validation_level": validation_level
            }
        )
    
    def _execute_environment_analysis(self, goal: str, inputs: Dict[str, Any], global_context: GlobalContext) -> AgentResult:
        """Execute environment analysis using environment tools."""
        self.report_progress("Environment analysis", "Analyzing development environment")
        
        # Extract analysis options
        working_directory = inputs.get("working_directory", str(global_context.workspace_path))
        check_tools = inputs.get("check_tools", True)
        check_python_packages = inputs.get("check_python_packages", True)
        analyze_virtual_env = inputs.get("analyze_virtual_env", True)
        check_system_dependencies = inputs.get("check_system_dependencies", True)
        detailed_analysis = inputs.get("detailed_analysis", False)
        
        # Run comprehensive environment analysis using tools
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
        
        # Optional detailed analyses
        if analyze_virtual_env:
            analysis_results["virtual_env_info"] = EnvironmentTools.analyze_virtual_environment(working_directory)
        
        if check_tools:
            analysis_results["development_tools"] = EnvironmentTools.check_development_tools(detailed_analysis)
        
        if check_python_packages:
            analysis_results["python_packages"] = EnvironmentTools.analyze_python_packages(working_directory)
        
        if check_system_dependencies:
            analysis_results["system_dependencies"] = EnvironmentTools.check_system_dependencies()
        
        # Overall environment health assessment
        health_assessment = EnvironmentTools.assess_environment_health(analysis_results)
        
        self.report_progress("Environment analysis complete", health_assessment["summary"])
        
        return AgentResult(
            success=True,
            message=f"Environment analysis complete: {health_assessment['summary']}",
            outputs={
                "analysis_type": "environment_analysis",
                "analysis_results": analysis_results,
                "health_assessment": health_assessment,
                "working_directory": working_directory
            }
        )
    
    def _execute_problem_analysis(self, goal: str, inputs: Dict[str, Any], global_context: GlobalContext) -> AgentResult:
        """Execute problem analysis using problem analysis tools."""
        self.report_progress("Problem analysis", "Analyzing problems and errors")
        
        # Extract problem data
        problem_data = inputs.get("problem_data", goal)
        error_logs = inputs.get("error_logs", [])
        stack_trace = inputs.get("stack_trace")
        context_information = inputs.get("context_information", {})
        previous_errors = inputs.get("previous_errors", [])
        analysis_depth = inputs.get("analysis_depth", "standard")
        
        # Run comprehensive problem analysis
        analysis_results = {}
        
        # Problem classification
        analysis_results["problem_classification"] = classify_problem(problem_data)
        
        # Error analysis
        if error_logs or stack_trace:
            analysis_results["error_analysis"] = analyze_errors(error_logs, stack_trace)
        
        # Severity assessment
        analysis_results["severity_assessment"] = assess_severity(problem_data, error_logs, context_information)
        
        # Root cause analysis
        analysis_results["root_cause_analysis"] = analyze_root_causes(problem_data, stack_trace, context_information)
        
        # Pattern recognition
        if previous_errors:
            analysis_results["pattern_recognition"] = recognize_error_patterns(problem_data, error_logs, previous_errors)
        
        # Generate recommendations
        analysis_results["recommendations"] = generate_problem_recommendations(analysis_results)
        
        # Create problem summary
        problem_class = analysis_results["problem_classification"]
        severity = analysis_results["severity_assessment"]
        primary_category = problem_class.get("primary_category", "unknown")
        overall_severity = severity.get("overall_severity", "low")
        
        summary = f"{primary_category.replace('_', ' ').title()} with {overall_severity} severity"
        
        self.report_progress("Problem analysis complete", summary)
        
        return AgentResult(
            success=True,
            message=f"Problem analysis complete: {summary}",
            outputs={
                "analysis_type": "problem_analysis",
                "analysis_results": analysis_results,
                "problem_summary": {
                    "summary": summary,
                    "category": primary_category,
                    "severity": overall_severity,
                    "complexity": problem_class.get("complexity", "unknown")
                },
                "analysis_depth": analysis_depth
            }
        )
    
    def _execute_quality_analysis(self, goal: str, inputs: Dict[str, Any], global_context: GlobalContext) -> AgentResult:
        """Execute quality analysis using quality analysis tools."""
        self.report_progress("Quality analysis", "Analyzing code quality and security")
        
        # Get code files
        code_files = inputs.get("code_files", {})
        if not code_files and global_context.workspace_path:
            code_files = self._discover_python_files(global_context.workspace_path)
        
        if not code_files:
            return AgentResult(
                success=False,
                message="No code files provided or found for quality analysis",
                outputs={}
            )
        
        # Extract analysis options
        analysis_options = {
            "check_performance": inputs.get("check_performance", True),
            "check_documentation": inputs.get("check_documentation", True),
            "security_scan": inputs.get("security_scan", True)
        }
        
        # Run comprehensive quality analysis
        quality_results = analyze_code_quality(code_files, analysis_options)
        
        # Additional security scanning if requested
        if analysis_options.get("security_scan", True):
            security_issues = []
            for file_path, content in code_files.items():
                file_security_issues = scan_for_security_patterns(content)
                for issue in file_security_issues:
                    issue["file_path"] = file_path
                security_issues.extend(file_security_issues)
            
            quality_results["security_scan_results"] = security_issues
            quality_results["total_issues"] += len(security_issues)
        
        total_issues = quality_results["total_issues"]
        overall_quality = quality_results["overall_quality"]
        
        message = f"Quality analysis complete: {overall_quality} quality with {total_issues} issues found"
        self.report_progress("Quality analysis complete", message)
        
        return AgentResult(
            success=True,
            message=message,
            outputs={
                "analysis_type": "quality_analysis",
                "quality_results": quality_results,
                "files_analyzed": len(code_files),
                "overall_quality": overall_quality,
                "total_issues": total_issues
            }
        )
    
    def _execute_comprehensive_analysis(self, goal: str, inputs: Dict[str, Any], global_context: GlobalContext) -> AgentResult:
        """Execute comprehensive analysis combining multiple analysis types."""
        self.report_progress("Comprehensive analysis", "Running all relevant analyses")
        
        comprehensive_results = {
            "analysis_type": "comprehensive_analysis",
            "analyses_performed": [],
            "overall_assessment": {},
            "recommendations": []
        }
        
        try:
            # Code analysis
            if "code_files" in inputs or global_context.workspace_path:
                code_result = self._execute_code_analysis("Validate code", inputs, global_context)
                comprehensive_results["code_analysis"] = code_result.outputs
                comprehensive_results["analyses_performed"].append("code_analysis")
                
            # Environment analysis
            env_result = self._execute_environment_analysis("Analyze environment", inputs, global_context)
            comprehensive_results["environment_analysis"] = env_result.outputs
            comprehensive_results["analyses_performed"].append("environment_analysis")
            
            # Problem analysis (if problem data available)
            if inputs.get("problem_data") or inputs.get("error_logs"):
                problem_result = self._execute_problem_analysis("Analyze problems", inputs, global_context)
                comprehensive_results["problem_analysis"] = problem_result.outputs
                comprehensive_results["analyses_performed"].append("problem_analysis")
            
            # Quality analysis (if code files available)
            if "code_files" in inputs or global_context.workspace_path:
                quality_result = self._execute_quality_analysis("Analyze quality", inputs, global_context)
                comprehensive_results["quality_analysis"] = quality_result.outputs
                comprehensive_results["analyses_performed"].append("quality_analysis")
            
            # Generate overall assessment
            comprehensive_results["overall_assessment"] = self._generate_overall_assessment(comprehensive_results)
            
            self.report_progress("Comprehensive analysis complete", 
                               f"Completed {len(comprehensive_results['analyses_performed'])} analyses")
            
            return AgentResult(
                success=True,
                message=f"Comprehensive analysis complete: {len(comprehensive_results['analyses_performed'])} analyses performed",
                outputs=comprehensive_results
            )
            
        except Exception as e:
            self.report_error(f"Comprehensive analysis failed: {e}", e)
            return AgentResult(
                success=False,
                message=f"Comprehensive analysis failed: {e}",
                outputs=comprehensive_results
            )
    
    def _execute_auto_analysis(self, goal: str, inputs: Dict[str, Any], global_context: GlobalContext) -> AgentResult:
        """Auto-detect and execute the most appropriate analysis."""
        self.report_progress("Auto analysis", "Detecting analysis type from available data")
        
        # Try to determine what to analyze based on available data
        if global_context.workspace_path:
            # Default to code analysis if workspace is available
            return self._execute_code_analysis(goal, inputs, global_context)
        elif inputs.get("problem_data") or inputs.get("error_logs"):
            return self._execute_problem_analysis(goal, inputs, global_context)
        else:
            # Fall back to environment analysis
            return self._execute_environment_analysis(goal, inputs, global_context)
    
    def _discover_python_files(self, workspace_path: Path) -> Dict[str, str]:
        """Discover and read Python files in the workspace."""
        code_files = {}
        
        try:
            for py_file in Path(workspace_path).rglob("*.py"):
                if py_file.is_file() and not str(py_file).startswith(str(workspace_path / "venv")):
                    try:
                        with open(py_file, 'r', encoding='utf-8') as f:
                            code_files[str(py_file.relative_to(workspace_path))] = f.read()
                    except Exception as e:
                        self.logger.warning(f"Could not read {py_file}: {e}")
                        
        except Exception as e:
            self.logger.warning(f"Error discovering Python files: {e}")
            
        return code_files
    
    def _generate_overall_assessment(self, comprehensive_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate an overall assessment from comprehensive analysis results."""
        assessment = {
            "overall_status": "unknown",
            "critical_issues": [],
            "recommendations": [],
            "summary": ""
        }
        
        critical_issues = []
        
        # Check code analysis results
        if "code_analysis" in comprehensive_results:
            code_analysis = comprehensive_results["code_analysis"]
            if not code_analysis.get("validation_results", {}).get("overall_success", True):
                critical_issues.append("Code validation failed - syntax or import errors present")
        
        # Check quality analysis results
        if "quality_analysis" in comprehensive_results:
            quality_analysis = comprehensive_results["quality_analysis"]
            quality_results = quality_analysis.get("quality_results", {})
            critical_count = quality_results.get("issues_by_severity", {}).get("Critical", 0)
            if critical_count > 0:
                critical_issues.append(f"{critical_count} critical security vulnerabilities found")
        
        # Check problem analysis results
        if "problem_analysis" in comprehensive_results:
            problem_analysis = comprehensive_results["problem_analysis"]
            severity = problem_analysis.get("analysis_results", {}).get("severity_assessment", {})
            if severity.get("overall_severity") in ["critical", "high"]:
                critical_issues.append("High-severity problems detected")
        
        # Determine overall status
        if critical_issues:
            assessment["overall_status"] = "critical_issues_found"
            assessment["critical_issues"] = critical_issues
            assessment["recommendations"] = ["Address critical issues immediately before proceeding"]
        else:
            assessment["overall_status"] = "healthy"
            assessment["recommendations"] = ["Continue with development - no critical issues found"]
        
        # Generate summary
        analyses_count = len(comprehensive_results.get("analyses_performed", []))
        issues_count = len(critical_issues)
        assessment["summary"] = f"Comprehensive analysis of {analyses_count} areas complete"
        if issues_count > 0:
            assessment["summary"] += f" - {issues_count} critical issues require attention"
        
        return assessment