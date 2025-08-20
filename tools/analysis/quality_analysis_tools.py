# tools/analysis/quality_analysis_tools.py
import ast
import re
import logging
from typing import Dict, Any, List, Optional, Set, Union
from pathlib import Path

# Get a logger instance for this module
logger = logging.getLogger(__name__)


def analyze_code_quality(code_files: Dict[str, str], analysis_options: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Comprehensive code quality analysis.
    
    Args:
        code_files: Dictionary mapping file paths to their content
        analysis_options: Configuration options for analysis
        
    Returns:
        Dictionary containing quality analysis results
    """
    analysis_options = analysis_options or {}
    
    quality_results = {
        "overall_quality": "unknown",
        "total_files": len(code_files),
        "total_issues": 0,
        "issues_by_category": {},
        "issues_by_severity": {},
        "detailed_issues": [],
        "metrics": {},
        "recommendations": []
    }
    
    all_issues = []
    
    for file_path, content in code_files.items():
        try:
            # Analyze individual file
            file_issues = analyze_file_quality(file_path, content, analysis_options)
            all_issues.extend(file_issues)
            
        except Exception as e:
            logger.warning(f"Failed to analyze {file_path}: {e}")
            continue
    
    # Categorize and summarize issues
    quality_results["detailed_issues"] = all_issues
    quality_results["total_issues"] = len(all_issues)
    
    # Group by category
    for issue in all_issues:
        category = issue.get("category", "unknown")
        quality_results["issues_by_category"][category] = quality_results["issues_by_category"].get(category, 0) + 1
        
        severity = issue.get("severity", "unknown")
        quality_results["issues_by_severity"][severity] = quality_results["issues_by_severity"].get(severity, 0) + 1
    
    # Calculate overall quality
    quality_results["overall_quality"] = calculate_overall_quality(quality_results)
    
    # Generate metrics
    quality_results["metrics"] = calculate_quality_metrics(code_files, all_issues)
    
    # Generate recommendations
    quality_results["recommendations"] = generate_quality_recommendations(quality_results)
    
    return quality_results


def analyze_file_quality(file_path: str, content: str, options: Dict[str, Any] = None) -> List[Dict[str, Any]]:
    """
    Analyze quality issues in a single file.
    
    Args:
        file_path: Path to the file
        content: File content
        options: Analysis options
        
    Returns:
        List of quality issues found in the file
    """
    options = options or {}
    issues = []
    
    try:
        # Parse AST for analysis
        tree = ast.parse(content, filename=file_path)
        
        # Run different quality checks
        issues.extend(check_security_issues(file_path, content, tree))
        issues.extend(check_maintainability_issues(file_path, content, tree))
        issues.extend(check_best_practices(file_path, content, tree))
        
        if options.get("check_performance", True):
            issues.extend(check_performance_issues(file_path, content, tree))
            
        if options.get("check_documentation", True):
            issues.extend(check_documentation_issues(file_path, content, tree))
            
    except SyntaxError as e:
        issues.append({
            "file_path": file_path,
            "line_number": e.lineno,
            "severity": "High",
            "category": "Syntax",
            "description": f"Syntax error: {e.msg}",
            "code_snippet": getattr(e, 'text', '').strip()
        })
        
    except Exception as e:
        logger.warning(f"Error analyzing {file_path}: {e}")
        
    return issues


def check_security_issues(file_path: str, content: str, tree: ast.AST) -> List[Dict[str, Any]]:
    """Check for security vulnerabilities."""
    issues = []
    lines = content.split('\n')
    
    # Check for hardcoded secrets
    secret_patterns = [
        (r'password\s*=\s*["\'][^"\']+["\']', "Hardcoded password found"),
        (r'api_key\s*=\s*["\'][^"\']+["\']', "Hardcoded API key found"),
        (r'secret\s*=\s*["\'][^"\']+["\']', "Hardcoded secret found"),
        (r'token\s*=\s*["\'][^"\']+["\']', "Hardcoded token found"),
        (r'key\s*=\s*["\'][a-zA-Z0-9]{20,}["\']', "Potential hardcoded key found")
    ]
    
    for i, line in enumerate(lines, 1):
        for pattern, message in secret_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                issues.append({
                    "file_path": file_path,
                    "line_number": i,
                    "severity": "Critical",
                    "category": "Security",
                    "description": f"{message}. Use environment variables or secure storage instead.",
                    "code_snippet": line.strip()
                })
    
    # Check for dangerous function usage
    dangerous_functions = {
        "eval": "Use of eval() is dangerous - it can execute arbitrary code",
        "exec": "Use of exec() is dangerous - it can execute arbitrary code", 
        "os.system": "Use of os.system() is dangerous - use subprocess instead",
        "subprocess.call": "Consider using subprocess.run() with shell=False",
    }
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func_name = get_function_name(node)
            if func_name in dangerous_functions:
                issues.append({
                    "file_path": file_path,
                    "line_number": node.lineno,
                    "severity": "High",
                    "category": "Security", 
                    "description": dangerous_functions[func_name],
                    "code_snippet": lines[node.lineno - 1].strip() if node.lineno <= len(lines) else ""
                })
    
    # Check for SQL injection risks
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr in ['execute', 'executemany'] and any(
                isinstance(arg, ast.BinOp) and isinstance(arg.op, ast.Mod) 
                for arg in node.args
            ):
                issues.append({
                    "file_path": file_path,
                    "line_number": node.lineno,
                    "severity": "High",
                    "category": "Security",
                    "description": "Potential SQL injection risk. Use parameterized queries instead of string formatting.",
                    "code_snippet": lines[node.lineno - 1].strip() if node.lineno <= len(lines) else ""
                })
    
    return issues


def check_maintainability_issues(file_path: str, content: str, tree: ast.AST) -> List[Dict[str, Any]]:
    """Check for maintainability issues."""
    issues = []
    lines = content.split('\n')
    
    # Check for complex functions (high cyclomatic complexity)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            complexity = calculate_cyclomatic_complexity(node)
            if complexity > 10:
                issues.append({
                    "file_path": file_path,
                    "line_number": node.lineno,
                    "severity": "Medium",
                    "category": "Maintainability",
                    "description": f"Function '{node.name}' has high cyclomatic complexity ({complexity}). Consider breaking it down.",
                    "code_snippet": f"def {node.name}(...):"
                })
    
    # Check for magic numbers
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            if node.value not in [0, 1, -1] and abs(node.value) > 1:
                issues.append({
                    "file_path": file_path,
                    "line_number": node.lineno,
                    "severity": "Low",
                    "category": "Maintainability",
                    "description": f"Magic number {node.value} found. Consider using a named constant.",
                    "code_snippet": lines[node.lineno - 1].strip() if node.lineno <= len(lines) else ""
                })
    
    # Check for long functions
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            func_length = len([n for n in ast.walk(node) if isinstance(n, ast.stmt)])
            if func_length > 50:
                issues.append({
                    "file_path": file_path,
                    "line_number": node.lineno,
                    "severity": "Medium",
                    "category": "Maintainability",
                    "description": f"Function '{node.name}' is very long ({func_length} statements). Consider breaking it down.",
                    "code_snippet": f"def {node.name}(...):"
                })
    
    # Check for too many parameters
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            param_count = len(node.args.args)
            if param_count > 7:
                issues.append({
                    "file_path": file_path,
                    "line_number": node.lineno,
                    "severity": "Medium",
                    "category": "Maintainability",
                    "description": f"Function '{node.name}' has too many parameters ({param_count}). Consider using a configuration object.",
                    "code_snippet": f"def {node.name}(...):"
                })
    
    return issues


def check_best_practices(file_path: str, content: str, tree: ast.AST) -> List[Dict[str, Any]]:
    """Check for best practice violations."""
    issues = []
    lines = content.split('\n')
    
    # Check for missing docstrings
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            if not ast.get_docstring(node):
                node_type = "Function" if isinstance(node, ast.FunctionDef) else "Class"
                issues.append({
                    "file_path": file_path,
                    "line_number": node.lineno,
                    "severity": "Low",
                    "category": "Best Practice",
                    "description": f"{node_type} '{node.name}' is missing a docstring.",
                    "code_snippet": f"{'def' if isinstance(node, ast.FunctionDef) else 'class'} {node.name}(...):"
                })
    
    # Check for non-descriptive variable names
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            if len(node.id) == 1 and node.id not in ['i', 'j', 'k', '_']:
                issues.append({
                    "file_path": file_path,
                    "line_number": node.lineno,
                    "severity": "Low", 
                    "category": "Best Practice",
                    "description": f"Variable name '{node.id}' is not descriptive. Use meaningful names.",
                    "code_snippet": lines[node.lineno - 1].strip() if node.lineno <= len(lines) else ""
                })
    
    # Check for bare except clauses
    for node in ast.walk(tree):
        if isinstance(node, ast.ExceptHandler) and node.type is None:
            issues.append({
                "file_path": file_path,
                "line_number": node.lineno,
                "severity": "Medium",
                "category": "Best Practice",
                "description": "Bare except clause found. Catch specific exceptions instead.",
                "code_snippet": "except:"
            })
    
    # Check for unused imports
    imported_names = set()
    used_names = set()
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_names.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imported_names.add(alias.asname or alias.name)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            used_names.add(node.id)
    
    unused_imports = imported_names - used_names - {'__name__', '__file__', '__doc__'}
    for unused in unused_imports:
        issues.append({
            "file_path": file_path,
            "line_number": 1,  # Import line number would need more complex tracking
            "severity": "Low",
            "category": "Best Practice", 
            "description": f"Unused import '{unused}' found.",
            "code_snippet": f"import {unused}"
        })
    
    return issues


def check_performance_issues(file_path: str, content: str, tree: ast.AST) -> List[Dict[str, Any]]:
    """Check for potential performance issues."""
    issues = []
    lines = content.split('\n')
    
    # Check for inefficient list operations in loops
    for node in ast.walk(tree):
        if isinstance(node, (ast.For, ast.While)):
            for child in ast.walk(node):
                if (isinstance(child, ast.Call) and 
                    isinstance(child.func, ast.Attribute) and
                    child.func.attr == 'append' and
                    isinstance(child.func.value, ast.Name)):
                    
                    # This is a simple heuristic - in practice you'd want more sophisticated analysis
                    issues.append({
                        "file_path": file_path,
                        "line_number": child.lineno,
                        "severity": "Low",
                        "category": "Performance",
                        "description": "Consider using list comprehension instead of append() in loop for better performance.",
                        "code_snippet": lines[child.lineno - 1].strip() if child.lineno <= len(lines) else ""
                    })
    
    return issues


def check_documentation_issues(file_path: str, content: str, tree: ast.AST) -> List[Dict[str, Any]]:
    """Check for documentation-related issues."""
    issues = []
    
    # Check for complex functions without detailed docstrings
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            docstring = ast.get_docstring(node)
            if docstring:
                param_count = len(node.args.args)
                if param_count > 3 and len(docstring.split()) < 10:
                    issues.append({
                        "file_path": file_path,
                        "line_number": node.lineno,
                        "severity": "Low",
                        "category": "Documentation",
                        "description": f"Function '{node.name}' has {param_count} parameters but minimal documentation.",
                        "code_snippet": f"def {node.name}(...):"
                    })
    
    return issues


def calculate_cyclomatic_complexity(node: ast.FunctionDef) -> int:
    """Calculate cyclomatic complexity for a function."""
    complexity = 1  # Base complexity
    
    for child in ast.walk(node):
        if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
            complexity += 1
        elif isinstance(child, ast.ExceptHandler):
            complexity += 1
        elif isinstance(child, ast.BoolOp):
            complexity += len(child.values) - 1
        elif isinstance(child, ast.ListComp):
            complexity += sum(1 for gen in child.generators for _ in gen.ifs)
            
    return complexity


def get_function_name(node: ast.Call) -> str:
    """Extract function name from a call node."""
    if isinstance(node.func, ast.Name):
        return node.func.id
    elif isinstance(node.func, ast.Attribute):
        if isinstance(node.func.value, ast.Name):
            return f"{node.func.value.id}.{node.func.attr}"
        return node.func.attr
    return ""


def calculate_overall_quality(quality_results: Dict[str, Any]) -> str:
    """Calculate overall code quality rating."""
    total_issues = quality_results["total_issues"]
    total_files = quality_results["total_files"]
    
    if total_files == 0:
        return "unknown"
    
    issues_per_file = total_issues / total_files
    critical_count = quality_results["issues_by_severity"].get("Critical", 0)
    high_count = quality_results["issues_by_severity"].get("High", 0)
    
    if critical_count > 0 or high_count > 3:
        return "poor"
    elif issues_per_file > 5:
        return "below_average"
    elif issues_per_file > 2:
        return "average"
    elif issues_per_file > 0.5:
        return "good"
    else:
        return "excellent"


def calculate_quality_metrics(code_files: Dict[str, str], issues: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calculate quality metrics."""
    total_lines = sum(len(content.split('\n')) for content in code_files.values())
    
    return {
        "total_lines_of_code": total_lines,
        "issues_per_file": len(issues) / len(code_files) if code_files else 0,
        "issues_per_1000_lines": (len(issues) / total_lines * 1000) if total_lines > 0 else 0,
        "security_issues": len([i for i in issues if i.get("category") == "Security"]),
        "maintainability_issues": len([i for i in issues if i.get("category") == "Maintainability"]),
        "best_practice_violations": len([i for i in issues if i.get("category") == "Best Practice"])
    }


def generate_quality_recommendations(quality_results: Dict[str, Any]) -> List[str]:
    """Generate quality improvement recommendations."""
    recommendations = []
    
    issues_by_category = quality_results["issues_by_category"]
    issues_by_severity = quality_results["issues_by_severity"]
    
    if issues_by_severity.get("Critical", 0) > 0:
        recommendations.append("Address critical security vulnerabilities immediately")
        
    if issues_by_severity.get("High", 0) > 0:
        recommendations.append("Fix high-severity issues before deployment")
        
    if issues_by_category.get("Security", 0) > 0:
        recommendations.append("Implement security code review process")
        recommendations.append("Use static security analysis tools")
        
    if issues_by_category.get("Maintainability", 0) > 5:
        recommendations.append("Refactor complex functions to improve maintainability")
        recommendations.append("Establish coding standards and complexity limits")
        
    if issues_by_category.get("Best Practice", 0) > 5:
        recommendations.append("Improve code documentation and naming conventions")
        recommendations.append("Set up automated linting and formatting tools")
        
    if not recommendations:
        recommendations.append("Code quality is good - continue with current practices")
        
    return recommendations


def scan_for_security_patterns(content: str) -> List[Dict[str, Any]]:
    """Scan content for security-related patterns."""
    security_issues = []
    lines = content.split('\n')
    
    # Additional security patterns
    patterns = {
        r'subprocess\.call.*shell\s*=\s*True': "Shell injection risk - avoid shell=True",
        r'pickle\.loads?\([^)]*\)': "Pickle deserialization can be dangerous",
        r'yaml\.load\([^,)]*\)': "Use yaml.safe_load() instead of yaml.load()",
        r'random\.random\(\).*password|random\.random\(\).*token': "Don't use random.random() for security purposes"
    }
    
    for i, line in enumerate(lines, 1):
        for pattern, message in patterns.items():
            if re.search(pattern, line, re.IGNORECASE):
                security_issues.append({
                    "line_number": i,
                    "severity": "High",
                    "category": "Security",
                    "description": message,
                    "code_snippet": line.strip()
                })
                
    return security_issues