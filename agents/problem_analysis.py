# agents/problem_analysis.py
import logging
import re
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

# Foundational dependencies
from .base import BaseAgent
from core.context import GlobalContext
from core.models import AgentResponse, AgentResult, TaskNode

# Get a logger instance for this module
logger = logging.getLogger(__name__)


class ProblemAnalysisAgent(BaseAgent):
    """
    Analysis Tier: Issue classification and root cause analysis ONLY.
    
    This agent provides read-only analysis of problems and errors.
    
    Responsibilities:
    - Error log analysis and pattern recognition
    - Failure mode classification
    - Root cause hypothesis generation
    - Impact and severity assessment
    
    Does NOT: Apply fixes, modify system, execute commands
    """

    def __init__(self):
        super().__init__(
            name="ProblemAnalysisAgent",
            description="Analyzes problems, errors, and failures to classify issues and identify root causes."
        )

    def required_inputs(self) -> List[str]:
        """Required inputs for ProblemAnalysisAgent execution."""
        return ["problem_data"]

    def optional_inputs(self) -> List[str]:
        """Optional inputs for ProblemAnalysisAgent execution."""
        return [
            "error_logs",
            "stack_trace",
            "context_information",
            "related_files",
            "environment_info",
            "previous_errors",
            "analysis_depth",
            "include_suggestions"
        ]

    def execute_v2(self, goal: str, inputs: Dict[str, Any], global_context: GlobalContext) -> AgentResult:
        """
        NEW INTERFACE: Analyze problems and errors for classification and root cause analysis.
        """
        logger.info(f"ProblemAnalysisAgent executing: '{goal}'")
        
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
        problem_data = inputs["problem_data"]
        error_logs = inputs.get("error_logs", [])
        stack_trace = inputs.get("stack_trace")
        context_information = inputs.get("context_information", {})
        related_files = inputs.get("related_files", [])
        environment_info = inputs.get("environment_info", {})
        previous_errors = inputs.get("previous_errors", [])
        analysis_depth = inputs.get("analysis_depth", "standard")
        include_suggestions = inputs.get("include_suggestions", True)

        try:
            self.report_progress("Starting problem analysis", f"Analyzing: {problem_data[:100]}...")

            analysis_results = {
                "problem_classification": self._classify_problem(problem_data),
                "error_analysis": self._analyze_errors(error_logs, stack_trace),
                "root_cause_analysis": self._analyze_root_causes(
                    problem_data, stack_trace, context_information
                ),
                "severity_assessment": self._assess_severity(
                    problem_data, error_logs, environment_info
                ),
                "impact_analysis": self._analyze_impact(
                    problem_data, related_files, context_information
                ),
                "pattern_recognition": self._recognize_patterns(
                    problem_data, error_logs, previous_errors
                ),
                "context_analysis": self._analyze_context(
                    context_information, environment_info, related_files
                )
            }

            # Advanced analysis for deeper insights
            if analysis_depth in ["detailed", "comprehensive"]:
                analysis_results["detailed_analysis"] = self._perform_detailed_analysis(
                    problem_data, stack_trace, error_logs, context_information
                )

            if analysis_depth == "comprehensive":
                analysis_results["comprehensive_analysis"] = self._perform_comprehensive_analysis(
                    analysis_results, previous_errors
                )

            # Generate recommendations if requested
            if include_suggestions:
                analysis_results["recommendations"] = self._generate_recommendations(
                    analysis_results
                )

            # Overall problem summary
            problem_summary = self._generate_problem_summary(analysis_results)

            self.report_progress("Problem analysis complete", problem_summary["summary"])

            return self.create_result(
                success=True,
                message=f"Problem analysis complete: {problem_summary['summary']}",
                outputs={
                    "analysis_results": analysis_results,
                    "problem_summary": problem_summary,
                    "analysis_depth": analysis_depth
                }
            )

        except Exception as e:
            error_msg = f"ProblemAnalysisAgent execution failed: {e}"
            logger.error(error_msg, exc_info=True)
            return self.create_result(
                success=False,
                message=error_msg,
                error_details={"exception": str(e)}
            )

    def _classify_problem(self, problem_data: str) -> Dict[str, Any]:
        """Classify the type and category of the problem."""
        
        problem_lower = problem_data.lower()
        
        # Define problem categories with keywords
        categories = {
            "syntax_error": ["syntaxerror", "invalid syntax", "unexpected token", "indentation"],
            "import_error": ["importerror", "modulenotfounderror", "no module named", "cannot import"],
            "type_error": ["typeerror", "object has no attribute", "not callable", "unsupported operand"],
            "value_error": ["valueerror", "invalid literal", "could not convert"],
            "key_error": ["keyerror", "key not found"],
            "index_error": ["indexerror", "list index out of range"],
            "file_error": ["filenotfounderror", "no such file", "permission denied", "file not found"],
            "network_error": ["connectionerror", "timeout", "network", "http", "ssl", "certificate"],
            "dependency_error": ["dependency", "version", "requirement", "package", "pip", "conda"],
            "environment_error": ["environment", "path", "python not found", "command not found"],
            "test_failure": ["test failed", "assertion", "assert", "expected", "actual"],
            "runtime_error": ["runtimeerror", "recursion", "memory", "overflow"],
            "configuration_error": ["config", "setting", "option", "parameter"],
            "database_error": ["database", "sql", "connection", "query", "table"],
            "authentication_error": ["authentication", "authorization", "permission", "access denied"],
            "performance_error": ["slow", "performance", "timeout", "hanging", "freeze"]
        }

        # Detect categories
        detected_categories = []
        for category, keywords in categories.items():
            if any(keyword in problem_lower for keyword in keywords):
                detected_categories.append(category)

        # Determine primary category
        primary_category = detected_categories[0] if detected_categories else "unknown"

        # Assess problem complexity
        complexity_indicators = {
            "simple": ["syntax", "typo", "missing", "not found"],
            "moderate": ["import", "dependency", "configuration", "path"],
            "complex": ["recursive", "memory", "performance", "network", "database"],
            "critical": ["security", "data loss", "corruption", "system"]
        }

        complexity = "unknown"
        for level, indicators in complexity_indicators.items():
            if any(indicator in problem_lower for indicator in indicators):
                complexity = level
                break

        return {
            "primary_category": primary_category,
            "all_categories": detected_categories,
            "complexity": complexity,
            "is_error": any(error_word in problem_lower for error_word in ["error", "exception", "failed", "failure"]),
            "is_warning": any(warn_word in problem_lower for warn_word in ["warning", "warn", "deprecated"]),
            "keywords_found": self._extract_key_terms(problem_data)
        }

    def _analyze_errors(self, error_logs: List[str], stack_trace: Optional[str]) -> Dict[str, Any]:
        """Analyze error logs and stack traces."""
        
        if not error_logs and not stack_trace:
            return {"no_errors": True}

        error_analysis = {
            "total_errors": len(error_logs),
            "error_types": {},
            "stack_trace_analysis": None,
            "error_patterns": [],
            "critical_errors": []
        }

        # Analyze individual error logs
        for i, error_log in enumerate(error_logs):
            error_type = self._identify_error_type(error_log)
            if error_type:
                error_analysis["error_types"][error_type] = error_analysis["error_types"].get(error_type, 0) + 1

            # Check for critical errors
            if self._is_critical_error(error_log):
                error_analysis["critical_errors"].append({
                    "index": i,
                    "error": error_log[:200],  # Truncate long errors
                    "type": error_type
                })

        # Analyze stack trace if provided
        if stack_trace:
            error_analysis["stack_trace_analysis"] = self._analyze_stack_trace(stack_trace)

        # Look for error patterns
        if len(error_logs) > 1:
            error_analysis["error_patterns"] = self._find_error_patterns(error_logs)

        return error_analysis

    def _analyze_root_causes(self, problem_data: str, stack_trace: Optional[str], 
                           context_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze potential root causes of the problem."""
        
        root_causes = {
            "likely_causes": [],
            "possible_causes": [],
            "evidence": {},
            "confidence_scores": {}
        }

        # Analyze based on problem type
        problem_classification = self._classify_problem(problem_data)
        primary_category = problem_classification["primary_category"]

        # Category-specific root cause analysis
        if primary_category == "syntax_error":
            causes = self._analyze_syntax_error_causes(problem_data, stack_trace)
        elif primary_category == "import_error":
            causes = self._analyze_import_error_causes(problem_data, context_info)
        elif primary_category == "dependency_error":
            causes = self._analyze_dependency_error_causes(problem_data, context_info)
        elif primary_category == "environment_error":
            causes = self._analyze_environment_error_causes(problem_data, context_info)
        elif primary_category == "file_error":
            causes = self._analyze_file_error_causes(problem_data, stack_trace)
        else:
            causes = self._analyze_generic_error_causes(problem_data, stack_trace)

        root_causes.update(causes)

        # Cross-reference with context information
        if context_info:
            context_causes = self._analyze_context_based_causes(context_info, problem_data)
            root_causes["context_based_causes"] = context_causes

        return root_causes

    def _assess_severity(self, problem_data: str, error_logs: List[str], 
                        environment_info: Dict[str, Any]) -> Dict[str, Any]:
        """Assess the severity and impact of the problem."""
        
        severity_factors = {
            "critical": 0,
            "high": 0,
            "medium": 0,
            "low": 0
        }

        problem_lower = problem_data.lower()

        # Critical severity indicators
        critical_indicators = [
            "system crash", "data loss", "security", "corruption", 
            "memory error", "segmentation fault", "kernel panic"
        ]
        if any(indicator in problem_lower for indicator in critical_indicators):
            severity_factors["critical"] += 1

        # High severity indicators
        high_indicators = [
            "cannot start", "complete failure", "all tests failed",
            "import error", "module not found", "dependency"
        ]
        if any(indicator in problem_lower for indicator in high_indicators):
            severity_factors["high"] += 1

        # Medium severity indicators
        medium_indicators = [
            "warning", "deprecated", "some tests failed", "performance"
        ]
        if any(indicator in problem_lower for indicator in medium_indicators):
            severity_factors["medium"] += 1

        # Low severity indicators
        low_indicators = [
            "style", "formatting", "minor", "suggestion", "typo"
        ]
        if any(indicator in problem_lower for indicator in low_indicators):
            severity_factors["low"] += 1

        # Determine overall severity
        if severity_factors["critical"] > 0:
            overall_severity = "critical"
        elif severity_factors["high"] > 0:
            overall_severity = "high"
        elif severity_factors["medium"] > 0:
            overall_severity = "medium"
        else:
            overall_severity = "low"

        return {
            "overall_severity": overall_severity,
            "severity_factors": severity_factors,
            "impact_assessment": self._assess_impact_level(overall_severity),
            "urgency": self._determine_urgency(overall_severity, error_logs)
        }

    def _analyze_impact(self, problem_data: str, related_files: List[str], 
                       context_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the impact scope of the problem."""
        
        impact_analysis = {
            "scope": "unknown",
            "affected_components": [],
            "affected_files": related_files,
            "downstream_effects": [],
            "user_impact": "unknown"
        }

        problem_lower = problem_data.lower()

        # Determine scope
        if any(scope_indicator in problem_lower for scope_indicator in ["system", "global", "all"]):
            impact_analysis["scope"] = "system-wide"
        elif any(scope_indicator in problem_lower for scope_indicator in ["module", "package", "library"]):
            impact_analysis["scope"] = "module-level"
        elif any(scope_indicator in problem_lower for scope_indicator in ["function", "method", "class"]):
            impact_analysis["scope"] = "function-level"
        else:
            impact_analysis["scope"] = "localized"

        # Identify affected components
        components = self._identify_affected_components(problem_data, related_files)
        impact_analysis["affected_components"] = components

        # Assess user impact
        if any(user_indicator in problem_lower for user_indicator in ["crash", "fail", "error", "broken"]):
            impact_analysis["user_impact"] = "high"
        elif any(user_indicator in problem_lower for user_indicator in ["slow", "warning", "deprecated"]):
            impact_analysis["user_impact"] = "medium"
        else:
            impact_analysis["user_impact"] = "low"

        return impact_analysis

    def _recognize_patterns(self, problem_data: str, error_logs: List[str], 
                          previous_errors: List[str]) -> Dict[str, Any]:
        """Recognize patterns in errors and problems."""
        
        patterns = {
            "recurring_patterns": [],
            "similar_previous_errors": [],
            "common_themes": [],
            "frequency_analysis": {}
        }

        all_errors = error_logs + previous_errors + [problem_data]

        # Look for recurring error messages
        error_counts = {}
        for error in all_errors:
            # Normalize error message
            normalized = re.sub(r'[0-9]+', 'NUM', error.lower())
            normalized = re.sub(r'[\'"][^\'"]++[\'"]', 'STRING', normalized)
            
            error_counts[normalized] = error_counts.get(normalized, 0) + 1

        recurring = {k: v for k, v in error_counts.items() if v > 1}
        patterns["recurring_patterns"] = list(recurring.keys())
        patterns["frequency_analysis"] = recurring

        # Find similar previous errors
        current_keywords = set(self._extract_key_terms(problem_data))
        for prev_error in previous_errors:
            prev_keywords = set(self._extract_key_terms(prev_error))
            similarity = len(current_keywords.intersection(prev_keywords))
            if similarity > 2:  # Threshold for similarity
                patterns["similar_previous_errors"].append({
                    "error": prev_error[:100],
                    "similarity_score": similarity,
                    "common_keywords": list(current_keywords.intersection(prev_keywords))
                })

        # Identify common themes
        all_keywords = []
        for error in all_errors:
            all_keywords.extend(self._extract_key_terms(error))
        
        keyword_counts = {}
        for keyword in all_keywords:
            keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1

        common_themes = [k for k, v in keyword_counts.items() if v >= 2]
        patterns["common_themes"] = common_themes

        return patterns

    def _analyze_context(self, context_info: Dict[str, Any], environment_info: Dict[str, Any], 
                        related_files: List[str]) -> Dict[str, Any]:
        """Analyze contextual information that might be relevant to the problem."""
        
        context_analysis = {
            "environment_factors": [],
            "file_factors": [],
            "timing_factors": [],
            "configuration_factors": []
        }

        # Analyze environment context
        if environment_info:
            if "python_version" in environment_info:
                context_analysis["environment_factors"].append(
                    f"Python version: {environment_info['python_version']}"
                )
            
            if "virtual_env" in environment_info:
                context_analysis["environment_factors"].append(
                    f"Virtual environment: {environment_info['virtual_env']}"
                )

        # Analyze file context
        if related_files:
            file_extensions = set(Path(f).suffix for f in related_files)
            context_analysis["file_factors"].append(f"File types involved: {list(file_extensions)}")
            
            if len(related_files) > 10:
                context_analysis["file_factors"].append("Large number of files involved")

        # Analyze other context information
        for key, value in context_info.items():
            if key not in ["python_version", "virtual_env"]:
                context_analysis["configuration_factors"].append(f"{key}: {str(value)[:100]}")

        return context_analysis

    def _perform_detailed_analysis(self, problem_data: str, stack_trace: Optional[str],
                                  error_logs: List[str], context_info: Dict[str, Any]) -> Dict[str, Any]:
        """Perform detailed analysis for complex problems."""
        
        detailed = {
            "complexity_analysis": self._analyze_problem_complexity(problem_data, stack_trace),
            "dependency_analysis": self._analyze_dependencies_involved(problem_data, context_info),
            "timing_analysis": self._analyze_timing_factors(error_logs),
            "resource_analysis": self._analyze_resource_factors(problem_data, error_logs)
        }

        return detailed

    def _perform_comprehensive_analysis(self, analysis_results: Dict[str, Any], 
                                      previous_errors: List[str]) -> Dict[str, Any]:
        """Perform comprehensive analysis including historical patterns."""
        
        comprehensive = {
            "trend_analysis": self._analyze_error_trends(previous_errors),
            "risk_assessment": self._assess_risks(analysis_results),
            "prevention_analysis": self._analyze_prevention_opportunities(analysis_results),
            "learning_opportunities": self._identify_learning_opportunities(analysis_results)
        }

        return comprehensive

    def _generate_recommendations(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate recommendations based on the analysis."""
        
        recommendations = {
            "immediate_actions": [],
            "short_term_actions": [],
            "long_term_actions": [],
            "preventive_measures": []
        }

        # Extract information from analysis
        problem_class = analysis_results.get("problem_classification", {})
        severity = analysis_results.get("severity_assessment", {})
        root_causes = analysis_results.get("root_cause_analysis", {})

        # Generate category-specific recommendations
        primary_category = problem_class.get("primary_category", "unknown")
        
        if primary_category == "import_error":
            recommendations["immediate_actions"].append("Check package installation and import paths")
            recommendations["short_term_actions"].append("Review project dependencies")
            
        elif primary_category == "syntax_error":
            recommendations["immediate_actions"].append("Review code syntax and indentation")
            recommendations["preventive_measures"].append("Use code linters and formatters")

        elif primary_category == "dependency_error":
            recommendations["immediate_actions"].append("Update or reinstall dependencies")
            recommendations["long_term_actions"].append("Implement dependency version management")

        # Add severity-based recommendations
        overall_severity = severity.get("overall_severity", "low")
        
        if overall_severity in ["critical", "high"]:
            recommendations["immediate_actions"].append("Address this issue as highest priority")
            
        if overall_severity == "critical":
            recommendations["immediate_actions"].append("Consider rollback if in production")

        # Add generic recommendations
        recommendations["preventive_measures"].extend([
            "Implement automated testing",
            "Set up error monitoring and alerting",
            "Regular code reviews and quality checks"
        ])

        return recommendations

    def _generate_problem_summary(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a comprehensive summary of the problem analysis."""
        
        problem_class = analysis_results.get("problem_classification", {})
        severity = analysis_results.get("severity_assessment", {})
        root_causes = analysis_results.get("root_cause_analysis", {})
        
        primary_category = problem_class.get("primary_category", "unknown")
        overall_severity = severity.get("overall_severity", "low")
        likely_causes = root_causes.get("likely_causes", [])

        summary_text = f"{primary_category.replace('_', ' ').title()} with {overall_severity} severity"
        
        if likely_causes:
            summary_text += f", likely caused by {likely_causes[0]}"

        return {
            "summary": summary_text,
            "category": primary_category,
            "severity": overall_severity,
            "complexity": problem_class.get("complexity", "unknown"),
            "actionable": len(analysis_results.get("recommendations", {}).get("immediate_actions", [])) > 0
        }

    # Helper methods for specific analysis types
    def _identify_error_type(self, error_log: str) -> Optional[str]:
        """Identify the type of error from log."""
        error_patterns = {
            "SyntaxError": r"SyntaxError",
            "ImportError": r"ImportError|ModuleNotFoundError",
            "TypeError": r"TypeError",
            "ValueError": r"ValueError",
            "KeyError": r"KeyError",
            "IndexError": r"IndexError",
            "FileNotFoundError": r"FileNotFoundError",
            "PermissionError": r"PermissionError",
            "ConnectionError": r"ConnectionError",
            "TimeoutError": r"TimeoutError"
        }
        
        for error_type, pattern in error_patterns.items():
            if re.search(pattern, error_log, re.IGNORECASE):
                return error_type
        
        return None

    def _is_critical_error(self, error_log: str) -> bool:
        """Check if an error is critical."""
        critical_patterns = [
            "segmentation fault", "memory error", "system error",
            "fatal", "critical", "severe", "crash"
        ]
        return any(pattern in error_log.lower() for pattern in critical_patterns)

    def _analyze_stack_trace(self, stack_trace: str) -> Dict[str, Any]:
        """Analyze stack trace for insights."""
        lines = stack_trace.split('\n')
        
        return {
            "total_frames": len([line for line in lines if 'File "' in line]),
            "error_location": self._extract_error_location(stack_trace),
            "call_chain": self._extract_call_chain(lines),
            "involved_modules": self._extract_modules_from_stack(lines)
        }

    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from text for analysis."""
        # Simple keyword extraction
        words = re.findall(r'\w+', text.lower())
        
        # Filter out common words and keep technical terms
        technical_terms = []
        for word in words:
            if len(word) > 3 and word not in ['error', 'with', 'from', 'file', 'line', 'code']:
                technical_terms.append(word)
        
        return list(set(technical_terms))[:20]  # Limit to 20 unique terms

    # Additional helper methods would go here for specific analysis types
    def _analyze_syntax_error_causes(self, problem_data: str, stack_trace: Optional[str]) -> Dict[str, Any]:
        """Analyze causes specific to syntax errors."""
        return {
            "likely_causes": ["Missing parentheses or brackets", "Incorrect indentation"],
            "possible_causes": ["Typos in keywords", "Mixed tabs and spaces"],
            "confidence_scores": {"indentation": 0.8, "brackets": 0.7}
        }

    def _analyze_import_error_causes(self, problem_data: str, context_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze causes specific to import errors."""
        return {
            "likely_causes": ["Package not installed", "Incorrect module path"],
            "possible_causes": ["Virtual environment not activated", "PYTHONPATH issues"],
            "confidence_scores": {"package_missing": 0.9, "path_issue": 0.6}
        }

    # Placeholder implementations for other specific analysis methods
    def _analyze_dependency_error_causes(self, problem_data: str, context_info: Dict[str, Any]) -> Dict[str, Any]:
        return {"likely_causes": ["Version conflicts", "Missing dependencies"]}

    def _analyze_environment_error_causes(self, problem_data: str, context_info: Dict[str, Any]) -> Dict[str, Any]:
        return {"likely_causes": ["Path configuration", "Environment variables"]}

    def _analyze_file_error_causes(self, problem_data: str, stack_trace: Optional[str]) -> Dict[str, Any]:
        return {"likely_causes": ["File not found", "Permission denied"]}

    def _analyze_generic_error_causes(self, problem_data: str, stack_trace: Optional[str]) -> Dict[str, Any]:
        return {"likely_causes": ["Logic error", "Runtime condition"]}

    def _analyze_context_based_causes(self, context_info: Dict[str, Any], problem_data: str) -> List[str]:
        return ["Context-based analysis not implemented"]

    def _assess_impact_level(self, severity: str) -> str:
        impact_map = {
            "critical": "system-wide impact",
            "high": "significant impact", 
            "medium": "moderate impact",
            "low": "minimal impact"
        }
        return impact_map.get(severity, "unknown impact")

    def _determine_urgency(self, severity: str, error_logs: List[str]) -> str:
        if severity == "critical":
            return "immediate"
        elif severity == "high":
            return "urgent"
        elif severity == "medium":
            return "moderate"
        else:
            return "low"

    def _identify_affected_components(self, problem_data: str, related_files: List[str]) -> List[str]:
        components = []
        if related_files:
            components.extend([Path(f).stem for f in related_files[:5]])  # Limit to 5
        return components

    def _find_error_patterns(self, error_logs: List[str]) -> List[str]:
        return ["Pattern analysis not fully implemented"]

    def _analyze_problem_complexity(self, problem_data: str, stack_trace: Optional[str]) -> Dict[str, Any]:
        return {"complexity_level": "moderate", "factors": ["Multiple components involved"]}

    def _analyze_dependencies_involved(self, problem_data: str, context_info: Dict[str, Any]) -> Dict[str, Any]:
        return {"dependencies": [], "conflicts": []}

    def _analyze_timing_factors(self, error_logs: List[str]) -> Dict[str, Any]:
        return {"timing_issues": False}

    def _analyze_resource_factors(self, problem_data: str, error_logs: List[str]) -> Dict[str, Any]:
        return {"memory_issues": False, "disk_issues": False}

    def _analyze_error_trends(self, previous_errors: List[str]) -> Dict[str, Any]:
        return {"trend": "stable", "frequency": "low"}

    def _assess_risks(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        return {"risk_level": "low", "risk_factors": []}

    def _analyze_prevention_opportunities(self, analysis_results: Dict[str, Any]) -> List[str]:
        return ["Implement better error handling", "Add validation checks"]

    def _identify_learning_opportunities(self, analysis_results: Dict[str, Any]) -> List[str]:
        return ["Code review practices", "Testing strategies"]

    def _extract_error_location(self, stack_trace: str) -> Optional[Dict[str, Any]]:
        # Extract file and line number from stack trace
        match = re.search(r'File "([^"]+)", line (\d+)', stack_trace)
        if match:
            return {"file": match.group(1), "line": int(match.group(2))}
        return None

    def _extract_call_chain(self, stack_lines: List[str]) -> List[str]:
        # Extract function call chain from stack trace
        calls = []
        for line in stack_lines:
            if line.strip().startswith('in '):
                func_match = re.search(r'in (\w+)', line)
                if func_match:
                    calls.append(func_match.group(1))
        return calls

    def _extract_modules_from_stack(self, stack_lines: List[str]) -> List[str]:
        # Extract module names from stack trace
        modules = set()
        for line in stack_lines:
            if 'File "' in line:
                file_match = re.search(r'File "([^"]+)"', line)
                if file_match:
                    file_path = file_match.group(1)
                    if '.py' in file_path:
                        module = Path(file_path).stem
                        modules.add(module)
        return list(modules)
