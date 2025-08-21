# tools/analysis/problem_analysis_tools.py
import logging
import re
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

# Get a logger instance for this module
logger = logging.getLogger(__name__)


def classify_problem(problem_data: str) -> Dict[str, Any]:
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
        "keywords_found": extract_key_terms(problem_data)
    }


def analyze_errors(error_logs: List[str], stack_trace: Optional[str] = None) -> Dict[str, Any]:
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
        error_type = identify_error_type(error_log)
        if error_type:
            error_analysis["error_types"][error_type] = error_analysis["error_types"].get(error_type, 0) + 1

        # Check for critical errors
        if is_critical_error(error_log):
            error_analysis["critical_errors"].append({
                "index": i,
                "error": error_log[:200],  # Truncate long errors
                "type": error_type
            })

    # Analyze stack trace if provided
    if stack_trace:
        error_analysis["stack_trace_analysis"] = analyze_stack_trace(stack_trace)

    # Look for error patterns
    if len(error_logs) > 1:
        error_analysis["error_patterns"] = find_error_patterns(error_logs)

    return error_analysis


def assess_severity(problem_data: str, error_logs: List[str] = None, 
                   environment_info: Dict[str, Any] = None) -> Dict[str, Any]:
    """Assess the severity and impact of the problem."""
    
    error_logs = error_logs or []
    environment_info = environment_info or {}
    
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
        "impact_assessment": assess_impact_level(overall_severity),
        "urgency": determine_urgency(overall_severity, error_logs)
    }


def analyze_root_causes(problem_data: str, stack_trace: Optional[str] = None, 
                       context_info: Dict[str, Any] = None) -> Dict[str, Any]:
    """Analyze potential root causes of the problem."""
    
    context_info = context_info or {}
    
    root_causes = {
        "likely_causes": [],
        "possible_causes": [],
        "evidence": {},
        "confidence_scores": {}
    }

    # Analyze based on problem type
    problem_classification = classify_problem(problem_data)
    primary_category = problem_classification["primary_category"]

    # Category-specific root cause analysis
    if primary_category == "syntax_error":
        causes = analyze_syntax_error_causes(problem_data, stack_trace)
    elif primary_category == "import_error":
        causes = analyze_import_error_causes(problem_data, context_info)
    elif primary_category == "dependency_error":
        causes = analyze_dependency_error_causes(problem_data, context_info)
    elif primary_category == "environment_error":
        causes = analyze_environment_error_causes(problem_data, context_info)
    elif primary_category == "file_error":
        causes = analyze_file_error_causes(problem_data, stack_trace)
    else:
        causes = analyze_generic_error_causes(problem_data, stack_trace)

    root_causes.update(causes)

    # Cross-reference with context information
    if context_info:
        context_causes = analyze_context_based_causes(context_info, problem_data)
        root_causes["context_based_causes"] = context_causes

    return root_causes


def recognize_error_patterns(problem_data: str, error_logs: List[str] = None, 
                           previous_errors: List[str] = None) -> Dict[str, Any]:
    """Recognize patterns in errors and problems."""
    
    error_logs = error_logs or []
    previous_errors = previous_errors or []
    
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
    current_keywords = set(extract_key_terms(problem_data))
    for prev_error in previous_errors:
        prev_keywords = set(extract_key_terms(prev_error))
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
        all_keywords.extend(extract_key_terms(error))
    
    keyword_counts = {}
    for keyword in all_keywords:
        keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1

    common_themes = [k for k, v in keyword_counts.items() if v >= 2]
    patterns["common_themes"] = common_themes

    return patterns


def generate_problem_recommendations(analysis_results: Dict[str, Any]) -> Dict[str, Any]:
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


# Helper functions
def identify_error_type(error_log: str) -> Optional[str]:
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


def is_critical_error(error_log: str) -> bool:
    """Check if an error is critical."""
    critical_patterns = [
        "segmentation fault", "memory error", "system error",
        "fatal", "critical", "severe", "crash"
    ]
    return any(pattern in error_log.lower() for pattern in critical_patterns)


def analyze_stack_trace(stack_trace: str) -> Dict[str, Any]:
    """Analyze stack trace for insights."""
    lines = stack_trace.split('\n')
    
    return {
        "total_frames": len([line for line in lines if 'File "' in line]),
        "error_location": extract_error_location(stack_trace),
        "call_chain": extract_call_chain(lines),
        "involved_modules": extract_modules_from_stack(lines)
    }


def extract_key_terms(text: str) -> List[str]:
    """Extract key terms from text for analysis."""
    # Simple keyword extraction
    words = re.findall(r'\w+', text.lower())
    
    # Filter out common words and keep technical terms
    technical_terms = []
    for word in words:
        if len(word) > 3 and word not in ['error', 'with', 'from', 'file', 'line', 'code']:
            technical_terms.append(word)
    
    return list(set(technical_terms))[:20]  # Limit to 20 unique terms


def assess_impact_level(severity: str) -> str:
    """Assess impact level based on severity."""
    impact_map = {
        "critical": "system-wide impact",
        "high": "significant impact", 
        "medium": "moderate impact",
        "low": "minimal impact"
    }
    return impact_map.get(severity, "unknown impact")


def determine_urgency(severity: str, error_logs: List[str]) -> str:
    """Determine urgency based on severity and context."""
    if severity == "critical":
        return "immediate"
    elif severity == "high":
        return "urgent"
    elif severity == "medium":
        return "moderate"
    else:
        return "low"


# Specific analysis functions
def analyze_syntax_error_causes(problem_data: str, stack_trace: Optional[str]) -> Dict[str, Any]:
    """Analyze causes specific to syntax errors."""
    return {
        "likely_causes": ["Missing parentheses or brackets", "Incorrect indentation"],
        "possible_causes": ["Typos in keywords", "Mixed tabs and spaces"],
        "confidence_scores": {"indentation": 0.8, "brackets": 0.7}
    }


def analyze_import_error_causes(problem_data: str, context_info: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze causes specific to import errors."""
    return {
        "likely_causes": ["Package not installed", "Incorrect module path"],
        "possible_causes": ["Virtual environment not activated", "PYTHONPATH issues"],
        "confidence_scores": {"package_missing": 0.9, "path_issue": 0.6}
    }


def analyze_dependency_error_causes(problem_data: str, context_info: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze causes specific to dependency errors."""
    return {"likely_causes": ["Version conflicts", "Missing dependencies"]}


def analyze_environment_error_causes(problem_data: str, context_info: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze causes specific to environment errors."""
    return {"likely_causes": ["Path configuration", "Environment variables"]}


def analyze_file_error_causes(problem_data: str, stack_trace: Optional[str]) -> Dict[str, Any]:
    """Analyze causes specific to file errors."""
    return {"likely_causes": ["File not found", "Permission denied"]}


def analyze_generic_error_causes(problem_data: str, stack_trace: Optional[str]) -> Dict[str, Any]:
    """Analyze generic error causes."""
    return {"likely_causes": ["Logic error", "Runtime condition"]}


def analyze_context_based_causes(context_info: Dict[str, Any], problem_data: str) -> List[str]:
    """Analyze causes based on context information."""
    return ["Context-based analysis not fully implemented"]


def find_error_patterns(error_logs: List[str]) -> List[str]:
    """Find patterns in error logs."""
    return ["Pattern analysis not fully implemented"]


def extract_error_location(stack_trace: str) -> Optional[Dict[str, Any]]:
    """Extract file and line number from stack trace."""
    match = re.search(r'File "([^"]+)", line (\d+)', stack_trace)
    if match:
        return {"file": match.group(1), "line": int(match.group(2))}
    return None


def extract_call_chain(stack_lines: List[str]) -> List[str]:
    """Extract function call chain from stack trace."""
    calls = []
    for line in stack_lines:
        if line.strip().startswith('in '):
            func_match = re.search(r'in (\w+)', line)
            if func_match:
                calls.append(func_match.group(1))
    return calls


def extract_modules_from_stack(stack_lines: List[str]) -> List[str]:
    """Extract module names from stack trace."""
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