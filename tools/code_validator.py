# tools/code_validator.py
import ast
import logging
import sys
import os
import importlib.util
from pathlib import Path
from typing import Dict, Any, List, Optional

# Get a logger instance for this module
logger = logging.getLogger(__name__)


def validate_syntax(code_files: Dict[str, str]) -> Dict[str, Any]:
    """
    Validate Python syntax for all code files.
    
    Args:
        code_files: Dictionary mapping file paths to their content
        
    Returns:
        Dictionary containing validation results with success status, details, errors, and passed files
    """
    details = []
    errors = []
    passed_files = []
    
    logger.info(f"Validating syntax for {len(code_files)} files")
    
    for file_path, content in code_files.items():
        try:
            # Attempt to compile the code
            compile(content, file_path, 'exec')
            details.append(f"✓ Syntax valid: {file_path}")
            passed_files.append(file_path)
            
        except SyntaxError as e:
            error_msg = f"Syntax error in {file_path} line {e.lineno}: {e.msg}"
            if hasattr(e, 'text') and e.text:
                error_msg += f" | Code: {e.text.strip()}"
            errors.append(error_msg)
            details.append(f"✗ {error_msg}")
            logger.error(error_msg)
            
        except Exception as e:
            error_msg = f"Compilation error in {file_path}: {e}"
            errors.append(error_msg)
            details.append(f"✗ {error_msg}")
            logger.error(error_msg)
    
    success = len(errors) == 0
    
    if success:
        details.append(f"✓ All {len(code_files)} files have valid Python syntax")
    else:
        details.append(f"✗ {len(errors)} files have syntax errors")
    
    return {
        "success": success,
        "details": details,
        "errors": errors,
        "passed_files": passed_files
    }


def validate_imports(code_files: Dict[str, str], python_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Validate that all imports in the code files can be resolved.
    
    Args:
        code_files: Dictionary mapping file paths to their content
        python_path: Optional additional path to add to sys.path for import resolution
        
    Returns:
        Dictionary containing validation results with success status, details, errors, and passed files
    """
    details = []
    errors = []
    passed_files = []
    
    logger.info(f"Validating imports for {len(code_files)} files")
    
    # Temporarily modify Python path if provided
    original_path = sys.path[:]
    if python_path and os.path.exists(python_path):
        sys.path.insert(0, python_path)
    
    try:
        for file_path, content in code_files.items():
            file_errors = []
            
            try:
                # Parse the AST to extract import statements
                tree = ast.parse(content, filename=file_path)
                imports = extract_imports_from_ast(tree)
                
                # Test each import
                for import_info in imports:
                    try:
                        if import_info["type"] == "import":
                            # Test "import module"
                            spec = importlib.util.find_spec(import_info["module"])
                            if spec is None:
                                file_errors.append(f"Cannot resolve import: {import_info['module']}")
                                
                        elif import_info["type"] == "from_import":
                            # Test "from module import item"
                            spec = importlib.util.find_spec(import_info["module"])
                            if spec is None:
                                file_errors.append(f"Cannot resolve module: {import_info['module']}")
                                
                    except (ImportError, ModuleNotFoundError) as e:
                        file_errors.append(f"Import error: {e}")
                    except Exception as e:
                        file_errors.append(f"Import validation error: {e}")
                
                if not file_errors:
                    details.append(f"✓ All imports valid: {file_path}")
                    passed_files.append(file_path)
                else:
                    for error in file_errors:
                        details.append(f"✗ {file_path}: {error}")
                    errors.extend(file_errors)
                    
            except SyntaxError as e:
                error_msg = f"Cannot parse {file_path} for import validation: {e}"
                details.append(f"✗ {error_msg}")
                errors.append(error_msg)
                
    finally:
        # Restore original Python path
        sys.path[:] = original_path
    
    success = len(errors) == 0
    
    if success:
        details.append(f"✓ All imports in {len(code_files)} files are valid")
    else:
        details.append(f"✗ {len(errors)} import issues found")
    
    return {
        "success": success,
        "details": details,
        "errors": errors,
        "passed_files": passed_files
    }


def validate_ast_structure(code_files: Dict[str, str]) -> Dict[str, Any]:
    """
    Perform AST-based structural validation of code files.
    
    Args:
        code_files: Dictionary mapping file paths to their content
        
    Returns:
        Dictionary containing validation results with success status, details, errors, and passed files
    """
    details = []
    errors = []
    passed_files = []
    
    logger.info(f"Analyzing AST structure for {len(code_files)} files")
    
    for file_path, content in code_files.items():
        file_errors = []
        
        try:
            tree = ast.parse(content, filename=file_path)
            
            # Check for common structural issues
            issues = analyze_ast_issues(tree, file_path)
            
            if not issues:
                details.append(f"✓ AST structure valid: {file_path}")
                passed_files.append(file_path)
            else:
                for issue in issues:
                    details.append(f"✗ {file_path}: {issue}")
                    file_errors.append(issue)
                
            errors.extend(file_errors)
                
        except SyntaxError as e:
            error_msg = f"AST parsing failed for {file_path}: {e}"
            details.append(f"✗ {error_msg}")
            errors.append(error_msg)
        except Exception as e:
            error_msg = f"AST analysis error for {file_path}: {e}"
            details.append(f"✗ {error_msg}")
            errors.append(error_msg)
    
    success = len(errors) == 0
    
    if success:
        details.append(f"✓ All {len(code_files)} files have valid AST structure")
    else:
        details.append(f"✗ {len(errors)} structural issues found")
    
    return {
        "success": success,
        "details": details,
        "errors": errors,
        "passed_files": passed_files
    }


def extract_imports_from_ast(tree: ast.AST) -> List[Dict[str, Any]]:
    """
    Extract import statements from AST.
    
    Args:
        tree: Parsed AST tree
        
    Returns:
        List of dictionaries containing import information
    """
    imports = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append({
                    "type": "import",
                    "module": alias.name,
                    "alias": alias.asname
                })
        elif isinstance(node, ast.ImportFrom):
            if node.module:  # Skip relative imports without module
                imports.append({
                    "type": "from_import",
                    "module": node.module,
                    "names": [alias.name for alias in node.names]
                })
    
    return imports


def analyze_ast_issues(tree: ast.AST, file_path: str) -> List[str]:
    """
    Analyze AST for common structural issues.
    
    Args:
        tree: Parsed AST tree
        file_path: Path to the file being analyzed
        
    Returns:
        List of issue descriptions found in the code
    """
    issues = []
    
    # Check for unused variables (simple check)
    assigned_vars = set()
    used_vars = set()
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Store):
                assigned_vars.add(node.id)
            elif isinstance(node.ctx, ast.Load):
                used_vars.add(node.id)
    
    # Find unused variables (excluding common patterns)
    unused_vars = assigned_vars - used_vars
    common_unused = {"_", "__name__", "__file__", "__doc__"}
    unused_vars = unused_vars - common_unused
    
    if unused_vars:
        issues.append(f"Potentially unused variables: {', '.join(sorted(unused_vars))}")
    
    # Check for empty functions/classes
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
            issues.append(f"Empty function: {node.name}")
        elif isinstance(node, ast.ClassDef) and len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
            issues.append(f"Empty class: {node.name}")
    
    return issues


def file_passed_validation(file_path: str, validation_results: Dict[str, Any]) -> bool:
    """
    Check if a file passed all validation steps.
    
    Args:
        file_path: Path to the file to check
        validation_results: Complete validation results dictionary
        
    Returns:
        True if the file passed all enabled validation steps
    """
    syntax_passed = file_path in validation_results.get("syntax_validation", {}).get("passed_files", [])
    import_passed = file_path in validation_results.get("import_validation", {}).get("passed_files", [])
    ast_passed = file_path in validation_results.get("ast_validation", {}).get("passed_files", [])
    
    # File passes if it passes syntax and doesn't fail other enabled checks
    import_check_enabled = validation_results.get("import_validation", {}).get("success") is not None
    ast_check_enabled = validation_results.get("ast_validation", {}).get("success") is not None
    
    return (syntax_passed and 
            (not import_check_enabled or import_passed) and 
            (not ast_check_enabled or ast_passed))


def validate_code_comprehensive(
    code_files: Dict[str, str],
    python_path: Optional[str] = None,
    validation_level: str = "standard",
    check_imports: bool = True,
    check_syntax_only: bool = False,
    ignore_warnings: bool = False
) -> Dict[str, Any]:
    """
    Main comprehensive code validation function.
    
    Args:
        code_files: Dictionary mapping file paths to their content
        python_path: Optional additional path to add to sys.path for import resolution
        validation_level: Validation strictness level ('basic', 'standard', 'strict')
        check_imports: Whether to perform import validation
        check_syntax_only: Whether to only check syntax and skip other validations
        ignore_warnings: Whether to ignore warning-level issues
        
    Returns:
        Complete validation results dictionary with all validation steps
    """
    logger.info(f"Starting comprehensive code validation for {len(code_files)} files")
    
    validation_results = {
        "syntax_validation": {"success": None, "details": []},
        "import_validation": {"success": None, "details": []},
        "ast_validation": {"success": None, "details": []},
        "overall_success": False,
        "files_processed": len(code_files),
        "files_passed": 0,
        "files_failed": 0,
        "warnings": [],
        "errors": []
    }
    
    try:
        # Step 1: Syntax validation (always performed)
        syntax_result = validate_syntax(code_files)
        validation_results["syntax_validation"] = syntax_result
        
        if not syntax_result["success"]:
            validation_results["errors"].extend(syntax_result["errors"])
            
            # Early return if syntax fails
            validation_results["files_failed"] = len(code_files)
            validation_results["overall_success"] = False
            
            return validation_results
        
        # Step 2: Import validation (if requested and syntax is valid)
        if check_imports and not check_syntax_only:
            import_result = validate_imports(code_files, python_path)
            validation_results["import_validation"] = import_result
            
            if not import_result["success"] and validation_level == "strict":
                validation_results["errors"].extend(import_result["errors"])
                
                # Early return on strict mode import failure
                validation_results["files_passed"] = len(syntax_result["passed_files"])
                validation_results["files_failed"] = len(code_files) - validation_results["files_passed"]
                validation_results["overall_success"] = False
                
                return validation_results
            elif not import_result["success"]:
                validation_results["warnings"].extend(import_result["errors"])
        
        # Step 3: AST validation (advanced structural validation)
        if validation_level in ["standard", "strict"] and not check_syntax_only:
            ast_result = validate_ast_structure(code_files)
            validation_results["ast_validation"] = ast_result
            
            if not ast_result["success"] and validation_level == "strict":
                validation_results["errors"].extend(ast_result["errors"])
                
                # Early return on strict mode AST failure
                validation_results["files_passed"] = len([f for f in code_files if file_passed_validation(f, validation_results)])
                validation_results["files_failed"] = len(code_files) - validation_results["files_passed"]
                validation_results["overall_success"] = False
                
                return validation_results
            elif not ast_result["success"]:
                validation_results["warnings"].extend(ast_result["errors"])
        
        # Calculate final results
        validation_results["files_passed"] = len([f for f in code_files if file_passed_validation(f, validation_results)])
        validation_results["files_failed"] = len(code_files) - validation_results["files_passed"]
        validation_results["overall_success"] = validation_results["files_failed"] == 0
        
        logger.info(f"Code validation complete: {validation_results['files_passed']}/{len(code_files)} files passed")
        
        return validation_results
    
    except Exception as e:
        logger.error(f"Code validation failed with exception: {e}", exc_info=True)
        validation_results["errors"].append(f"Validation exception: {e}")
        validation_results["overall_success"] = False
        return validation_results