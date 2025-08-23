# tools/debugging_tools.py
"""
Atomic debugging tools for bug reproduction, hypothesis formation, evidence gathering, 
and fix validation. Extracted from DebuggingAgent to provide reusable debugging capabilities.
"""

import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)


class FixStrategy(Enum):
    """Types of fix strategies that can be applied."""
    SURGICAL = "surgical"  # Small, targeted changes
    DESIGN_CHANGE = "design_change"  # Larger refactoring needed


class FailureCategory(Enum):
    """Categories of fix failures for reflection."""
    SAME_ERROR = "same_error"
    NEW_ERROR = "new_error"
    REGRESSION = "regression"  
    INCOMPLETE_FIX = "incomplete_fix"
    ENVIRONMENT_ISSUE = "environment_issue"


@dataclass
class DebuggingState:
    """State object to track debugging session progress."""
    session_id: str = field(default_factory=lambda: f"debug_{uuid.uuid4().hex[:8]}")
    problem_description: str = ""
    hypotheses_tested: List[str] = field(default_factory=list)
    fixes_attempted: List[str] = field(default_factory=list)
    validation_history: List[Dict] = field(default_factory=list)
    reflection_history: List[Dict] = field(default_factory=list)
    confidence: float = 1.0
    last_error_report: Optional[Dict] = None
    start_timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


def discover_log_configurations(project_path: str = ".") -> List[str]:
    """
    Discover logging configurations within the codebase.
    Returns a list of potential log file paths.
    """
    import subprocess
    
    discovered_paths = []
    
    try:
        # Grep for logging setup in Python files to find explicit filenames
        grep_command = ["grep", "-r", "logging.basicConfig", project_path, "--include", "*.py"]
        result = subprocess.run(
            grep_command, 
            capture_output=True, 
            text=True, 
            timeout=10
        )
        
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                match = re.search(r"filename=['\"]([^'\"]+)['\"]", line)
                if match:
                    discovered_paths.append(match.group(1))
    
    except (subprocess.TimeoutExpired, subprocess.SubprocessError) as e:
        logger.warning(f"Log discovery failed: {e}")
    
    # Clean up and return unique paths
    return sorted(list(set(p for p in discovered_paths if p)))


def scan_for_relevant_logs(
    problem_description: str, 
    failure_timestamp: str = None,
    project_path: str = "."
) -> Dict[str, Any]:
    """
    Scan for relevant error logs based on the problem description.
    Returns structured log scan results.
    """
    import subprocess
    from pathlib import Path
    
    if not failure_timestamp:
        failure_timestamp = datetime.utcnow().isoformat()
    
    # Discover configured log files
    configured_log_files = discover_log_configurations(project_path)
    
    # Define search patterns
    patterns = [
        "Traceback (most recent call last):",
        "CRITICAL", "FATAL", "ERROR", "Exception",
        "ValueError", "TypeError", "KeyError", "IndexError"
    ]
    
    # Add keywords from problem description
    patterns.extend(problem_description.split()[:5])
    search_pattern = '|'.join(set(patterns))
    
    results = {
        "configured_logs_found": configured_log_files,
        "patterns_used": patterns,
        "log_entries": [],
        "errors": []
    }
    
    scan_commands = []
    
    # Search configured log files first
    if configured_log_files:
        for log_file in configured_log_files:
            if Path(log_file).exists():
                scan_commands.append([
                    "awk", f'$0 >= "{failure_timestamp}"', log_file
                ])
    
    # Search for .log files in project
    try:
        find_cmd = ["find", project_path, "-name", "*.log", "-type", "f"]
        find_result = subprocess.run(find_cmd, capture_output=True, text=True, timeout=30)
        
        if find_result.returncode == 0:
            log_files = find_result.stdout.strip().split('\n')
            for log_file in log_files:
                if log_file and Path(log_file).exists():
                    scan_commands.append(["grep", "-iE", search_pattern, log_file])
    
    except (subprocess.TimeoutExpired, subprocess.SubprocessError) as e:
        results["errors"].append(f"Log file discovery failed: {e}")
    
    # Execute scan commands
    for cmd in scan_commands[:10]:  # Limit to first 10 commands
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0 and result.stdout.strip():
                results["log_entries"].extend(result.stdout.strip().split('\n')[-50:])  # Last 50 lines
        except (subprocess.TimeoutExpired, subprocess.SubprocessError) as e:
            results["errors"].append(f"Log scan command failed {' '.join(cmd)}: {e}")
    
    return results


def classify_problem_type(problem_description: str, error_logs: List[str] = None) -> Dict[str, Any]:
    """
    Classify the type of problem for debugging strategy selection.
    """
    problem_lower = problem_description.lower()
    
    categories = {
        "syntax_error": ["syntaxerror", "invalid syntax", "unexpected token", "indentation"],
        "import_error": ["importerror", "modulenotfounderror", "no module named", "cannot import"],
        "type_error": ["typeerror", "object has no attribute", "not callable", "unsupported operand"],
        "value_error": ["valueerror", "invalid literal", "could not convert"],
        "runtime_error": ["runtimeerror", "recursion", "memory", "overflow"],
        "test_failure": ["test failed", "assertion", "assert", "expected", "actual"],
        "network_error": ["connectionerror", "timeout", "network", "http"],
        "file_error": ["filenotfounderror", "no such file", "permission denied"],
        "configuration_error": ["config", "setting", "option", "parameter"]
    }
    
    detected_categories = []
    for category, keywords in categories.items():
        if any(keyword in problem_lower for keyword in keywords):
            detected_categories.append(category)
    
    # Analyze error logs if provided
    log_indicators = []
    if error_logs:
        for log_line in error_logs:
            log_lower = log_line.lower()
            for category, keywords in categories.items():
                if any(keyword in log_lower for keyword in keywords):
                    log_indicators.append(category)
    
    return {
        "primary_category": detected_categories[0] if detected_categories else "unknown",
        "all_categories": detected_categories,
        "log_indicators": list(set(log_indicators)),
        "complexity": "simple" if len(detected_categories) <= 1 else "complex",
        "requires_environment_check": any(cat in detected_categories for cat in ["import_error", "configuration_error"])
    }


def generate_debugging_hypothesis(
    problem_description: str,
    evidence: Dict[str, Any],
    debugging_state: DebuggingState,
    llm_client: Any = None
) -> Dict[str, Any]:
    """
    Generate a debugging hypothesis based on evidence and previous attempts.
    Uses the original DebuggingAgent analysis prompt.
    """
    if not llm_client:
        # Fallback to rule-based hypothesis generation
        return generate_rule_based_hypothesis(problem_description, evidence, debugging_state)
    
    # Build the original analysis prompt from DebuggingAgent
    last_reflection = debugging_state.reflection_history[-1] if debugging_state.reflection_history else None
    
    prompt = f"""
    **Persona**: You are an expert staff engineer and researcher, specializing in root cause analysis of complex software failures.

    **Overall Goal**: Your primary objective is to analyze the provided context and evidence to form a new, actionable hypothesis that will lead to a successful bug fix.

    **Context**:
    - **Problem Statement**: {debugging_state.problem_description}
    - **Debugging History**:
      - Hypotheses Already Tested and Failed: {json.dumps(debugging_state.hypotheses_tested)}
      - Fix Strategies Already Attempted: {json.dumps(debugging_state.fixes_attempted)}
      - Last Known Error Report: {json.dumps(debugging_state.last_error_report, default=str)}
      - Reflection on Last Failure: {json.dumps(last_reflection, default=str)}
    - **Current Evidence**:
      {json.dumps(evidence, indent=2, default=str)}

    **Your Task**:
    1.  **Synthesize All Information**: Carefully consider the original problem, the history of failed attempts, and the latest evidence.
    2.  **Formulate a New Hypothesis**: Based on your synthesis, generate a NEW and UNTRIED hypothesis. Do not repeat a hypothesis that has already been tested.
    3.  **Determine a Solution Strategy**: Decide if the fix requires a small, targeted change ('SURGICAL') or a larger refactoring ('DESIGN_CHANGE').

    **Constraints**:
    - **Do Not Repeat Past Mistakes**: The `Debugging History` section shows what has already failed. Your new hypothesis must be different.
    - **Be Actionable**: Your recommended strategy must be a clear, concrete action that another agent can take.

    **Output Format**: You MUST return a single, valid JSON object with the following structure:
    {{
        "root_cause_analysis": "A detailed explanation of the bug, incorporating insights from the reflection on the last failure.",
        "primary_hypothesis": "A NEW, UNTRIED hypothesis about the most likely cause of the failure.",
        "solution_type": "SURGICAL|DESIGN_CHANGE",
        "recommended_strategy": "A NEW, UNTRIED, and specific approach to fix the issue based on your new hypothesis."
    }}
    """
    
    try:
        response = llm_client.invoke(prompt)
        hypothesis = json.loads(response)
        
        # Validate required fields
        required_fields = ["root_cause_analysis", "primary_hypothesis", "solution_type", "recommended_strategy"]
        if all(field in hypothesis for field in required_fields):
            return hypothesis
        else:
            logger.warning("LLM hypothesis missing required fields, falling back to rule-based")
            return generate_rule_based_hypothesis(problem_description, evidence, debugging_state)
            
    except Exception as e:
        logger.error(f"LLM hypothesis generation failed: {e}")
        return generate_rule_based_hypothesis(problem_description, evidence, debugging_state)


def generate_rule_based_hypothesis(
    problem_description: str,
    evidence: Dict[str, Any], 
    debugging_state: DebuggingState
) -> Dict[str, Any]:
    """
    Generate debugging hypothesis using rule-based approach when LLM is unavailable.
    """
    problem_classification = classify_problem_type(
        problem_description, 
        evidence.get("log_scan_results", {}).get("log_entries", [])
    )
    
    primary_category = problem_classification["primary_category"]
    
    # Rule-based hypothesis generation
    hypothesis_templates = {
        "syntax_error": {
            "hypothesis": "Syntax error in Python code preventing execution",
            "strategy": "Fix syntax issues identified in error messages",
            "solution_type": "SURGICAL"
        },
        "import_error": {
            "hypothesis": "Missing dependencies or incorrect import paths",
            "strategy": "Install missing packages or fix import statements", 
            "solution_type": "SURGICAL"
        },
        "type_error": {
            "hypothesis": "Incorrect data types being passed to functions",
            "strategy": "Add type checking and fix type mismatches",
            "solution_type": "SURGICAL"
        },
        "runtime_error": {
            "hypothesis": "Logic error causing runtime failure",
            "strategy": "Review business logic and add defensive programming",
            "solution_type": "DESIGN_CHANGE"
        },
        "unknown": {
            "hypothesis": "Complex issue requiring investigation",
            "strategy": "Systematic debugging through instrumentation",
            "solution_type": "DESIGN_CHANGE"
        }
    }
    
    template = hypothesis_templates.get(primary_category, hypothesis_templates["unknown"])
    
    return {
        "root_cause_analysis": f"{primary_category.replace('_', ' ').title()}: {problem_description}",
        "primary_hypothesis": template["hypothesis"],
        "solution_type": template["solution_type"],
        "recommended_strategy": template["strategy"],
        "confidence": max(0.3, 0.9 - len(debugging_state.hypotheses_tested) * 0.1)
    }


def perform_failure_reflection(
    old_error: Optional[Dict],
    new_error: Optional[Dict],
    fix_attempted: str,
    llm_client: Any = None
) -> Dict[str, Any]:
    """
    Perform post-mortem analysis on a failed fix attempt.
    Uses the original DebuggingAgent reflection prompt.
    """
    if not llm_client:
        return perform_rule_based_reflection(old_error, new_error, fix_attempted)
    
    # Original reflection prompt from DebuggingAgent
    prompt = f"""
    You are a senior software engineer performing a post-mortem on a failed bug fix attempt.
    Analyze the original error, the attempted fix, and the new error to guide the next debugging step.

    **Original Error Report:**
    {json.dumps(old_error, indent=2, default=str)}

    **Attempted Fix Strategy:**
    "{fix_attempted}"

    **New Error Report (after applying the fix):**
    {json.dumps(new_error, indent=2, default=str)}

    Analyze the situation and return a JSON object with this structure:
    {{
        "failure_category": "SAME_ERROR|NEW_ERROR|REGRESSION|INCOMPLETE_FIX|ENVIRONMENT_ISSUE",
        "analysis": "A brief explanation of why the fix likely failed.",
        "next_strategy_hint": "A high-level suggestion for what to try in the next iteration."
    }}
    """
    
    try:
        response = llm_client.invoke(prompt)
        return json.loads(response)
    except Exception as e:
        logger.error(f"LLM reflection failed: {e}")
        return perform_rule_based_reflection(old_error, new_error, fix_attempted)


def perform_rule_based_reflection(
    old_error: Optional[Dict],
    new_error: Optional[Dict], 
    fix_attempted: str
) -> Dict[str, Any]:
    """
    Rule-based reflection when LLM is unavailable.
    """
    if not new_error:
        return {
            "failure_category": "INCOMPLETE_FIX",
            "analysis": "Fix was applied but validation failed to run",
            "next_strategy_hint": "Check test runner and validation process"
        }
    
    old_error_str = json.dumps(old_error, default=str) if old_error else ""
    new_error_str = json.dumps(new_error, default=str)
    
    if old_error_str and old_error_str == new_error_str:
        category = "SAME_ERROR"
        analysis = "Same error persists after fix attempt"
    elif "test" in fix_attempted.lower() and "fail" in new_error_str.lower():
        category = "REGRESSION"
        analysis = "Fix may have introduced new test failures"
    else:
        category = "NEW_ERROR"
        analysis = "Different error appeared after fix attempt"
    
    return {
        "failure_category": category,
        "analysis": analysis,
        "next_strategy_hint": "Try a different approach or gather more evidence"
    }


def update_debugging_confidence(
    state: DebuggingState,
    validation_passed: bool,
    failure_category: str = None
) -> float:
    """
    Update debugging confidence based on results.
    """
    if validation_passed:
        state.confidence = 1.0
        return state.confidence
    
    # Confidence degradation based on failure type
    degradation_factors = {
        "SAME_ERROR": 0.6,
        "REGRESSION": 0.5,
        "NEW_ERROR": 0.8,
        "INCOMPLETE_FIX": 0.7,
        "ENVIRONMENT_ISSUE": 0.9
    }
    
    factor = degradation_factors.get(failure_category, 0.7)
    state.confidence *= factor
    state.confidence = max(0.1, state.confidence)  # Minimum confidence
    
    return state.confidence


def generate_instrumentation_plan(
    hypothesis: Dict[str, Any],
    code_context: Dict[str, str],
    llm_client: Any = None
) -> Dict[str, Dict[str, str]]:
    """
    Generate a plan for code instrumentation to gather debug information.
    Uses the original debugging_strategies instrumentation prompt.
    """
    if not llm_client:
        return generate_simple_instrumentation_plan(hypothesis, code_context)
    
    # Original instrumentation prompt from debugging_strategies.py
    prompt = f"""
    Based on this hypothesis and code, where should I add debug print statements to get more information?
    Hypothesis: {json.dumps(hypothesis)}
    Code: {json.dumps(code_context)}

    Return a JSON object where keys are file paths and values are dictionaries mapping line numbers to the Python print statements to insert.
    Example:
    {{
      "src/main.py": {{
        "42": "print(f'DEBUG: user_id={{user.id}}, status={{user.status}}')"
      }}
    }}
    """
    
    try:
        response = llm_client.invoke(prompt)
        instrumentation_plan = json.loads(response)
        
        if isinstance(instrumentation_plan, dict):
            return instrumentation_plan
        else:
            return generate_simple_instrumentation_plan(hypothesis, code_context)
            
    except Exception as e:
        logger.error(f"LLM instrumentation planning failed: {e}")
        return generate_simple_instrumentation_plan(hypothesis, code_context)


def generate_simple_instrumentation_plan(
    hypothesis: Dict[str, Any],
    code_context: Dict[str, str]
) -> Dict[str, Dict[str, str]]:
    """
    Generate simple instrumentation plan without LLM.
    """
    plan = {}
    
    # Add basic debug prints to key locations in each file
    for file_path, content in code_context.items():
        if not content.strip():
            continue
            
        lines = content.split('\n')
        file_plan = {}
        
        # Look for function definitions and add debug prints
        for i, line in enumerate(lines):
            if line.strip().startswith('def ') and ':' in line:
                func_name = line.strip().split('def ')[1].split('(')[0]
                file_plan[str(i + 2)] = f"print(f'DEBUG: Entering {func_name}()')"
        
        if file_plan:
            plan[file_path] = file_plan
    
    return plan


def validate_debugging_inputs(inputs: Dict[str, Any]) -> None:
    """
    Validate inputs for debugging operations.
    """
    # Check for problem_description (should be provided by InterAgentRouter)
    problem_description = inputs.get("problem_description", "")
    
    if not problem_description:
        raise ValueError(f"Missing required debugging inputs: ['problem_description'] - available keys: {list(inputs.keys())}")
        
    if not isinstance(problem_description, str):
        raise ValueError("problem_description must be a string")


def create_debugging_session(problem_description: str, **kwargs) -> DebuggingState:
    """
    Create a new debugging session with initial state.
    """
    return DebuggingState(
        problem_description=problem_description,
        last_error_report=kwargs.get("failed_test_report"),
        start_timestamp=datetime.utcnow().isoformat()
    )