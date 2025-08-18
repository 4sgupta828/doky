# agents/code_modifier.py
import ast
import logging
import sys
import os
import importlib.util
from pathlib import Path
from typing import Dict, Any, List, Tuple

# Foundational dependencies
from .base import BaseAgent
from core.context import GlobalContext
from core.models import AgentResponse, AgentResult, TaskNode

# Get a logger instance for this module
logger = logging.getLogger(__name__)


class CodeModifierAgent(BaseAgent):
    """
    Specialized Tier: Write-only code modification operations.
    
    This agent handles write-only code modification operations.
    
    Responsibilities:
    - Code file creation and modification
    - Refactoring operations
    - Code formatting and style fixes
    - Import statement management
    - Code structure modifications
    """

    def __init__(self):
        super().__init__(
            name="CodeModifierAgent",
            description="Modifies Python code files, handles refactoring, and code transformations."
        )

    def required_inputs(self) -> List[str]:
        """Required inputs for CodeValidatorAgent execution."""
        return ["code_files"]

    def optional_inputs(self) -> List[str]:
        """Optional inputs for CodeValidatorAgent execution."""
        return [
            "python_path",
            "validation_level",
            "check_imports",
            "check_syntax_only",
            "ignore_warnings"
        ]

    def execute_v2(self, goal: str, inputs: Dict[str, Any], global_context: GlobalContext) -> AgentResult:
        """
        NEW INTERFACE: Validate Python code with comprehensive checks.
        """
        logger.info(f"CodeValidatorAgent executing: '{goal}'")
        
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

            validation_results = {
                "syntax_validation": {"success": False, "details": []},
                "import_validation": {"success": False, "details": []},
                "ast_validation": {"success": False, "details": []},
                "overall_success": False,
                "files_processed": len(code_files),
                "files_passed": 0,
                "files_failed": 0,
                "warnings": [],
                "errors": []
            }

            # Step 1: Syntax validation (always performed)
            syntax_result = self._validate_syntax(code_files)
            validation_results["syntax_validation"] = syntax_result
            
            if not syntax_result["success"]:
                validation_results["errors"].extend(syntax_result["errors"])
                
                return self.create_result(
                    success=False,
                    message=f"Syntax validation failed: {len(syntax_result['errors'])} syntax errors found",
                    outputs=validation_results
                )

            # Step 2: Import validation (if requested and syntax is valid)
            if check_imports and not check_syntax_only:
                import_result = self._validate_imports(code_files, python_path)
                validation_results["import_validation"] = import_result
                
                if not import_result["success"] and validation_level == "strict":
                    validation_results["errors"].extend(import_result["errors"])
                    
                    return self.create_result(
                        success=False,
                        message=f"Import validation failed: {len(import_result['errors'])} import errors found",
                        outputs=validation_results
                    )
                elif not import_result["success"]:
                    validation_results["warnings"].extend(import_result["errors"])

            # Step 3: AST validation (advanced structural validation)
            if validation_level in ["standard", "strict"] and not check_syntax_only:
                ast_result = self._validate_ast_structure(code_files)
                validation_results["ast_validation"] = ast_result
                
                if not ast_result["success"] and validation_level == "strict":
                    validation_results["errors"].extend(ast_result["errors"])
                    
                    return self.create_result(
                        success=False,
                        message=f"AST validation failed: {len(ast_result['errors'])} structural issues found",
                        outputs=validation_results
                    )
                elif not ast_result["success"]:
                    validation_results["warnings"].extend(ast_result["errors"])

            # Calculate final results
            validation_results["files_passed"] = len([f for f in code_files if self._file_passed_validation(f, validation_results)])
            validation_results["files_failed"] = len(code_files) - validation_results["files_passed"]
            validation_results["overall_success"] = validation_results["files_failed"] == 0

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
            error_msg = f"CodeValidatorAgent execution failed: {e}"
            logger.error(error_msg, exc_info=True)
            return self.create_result(
                success=False,
                message=error_msg,
                error_details={"exception": str(e)}
            )

    def _validate_syntax(self, code_files: Dict[str, str]) -> Dict[str, Any]:
        """Validate Python syntax for all code files."""
        details = []
        errors = []
        passed_files = []
        
        self.report_progress("Checking syntax", f"Validating syntax for {len(code_files)} files")
        
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

    def _validate_imports(self, code_files: Dict[str, str], python_path: str = None) -> Dict[str, Any]:
        """Validate that all imports in the code files can be resolved."""
        details = []
        errors = []
        passed_files = []
        
        self.report_progress("Checking imports", f"Validating imports for {len(code_files)} files")
        
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
                    imports = self._extract_imports_from_ast(tree)
                    
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

    def _validate_ast_structure(self, code_files: Dict[str, str]) -> Dict[str, Any]:
        """Perform AST-based structural validation of code files."""
        details = []
        errors = []
        passed_files = []
        
        self.report_progress("Checking AST structure", f"Analyzing structure for {len(code_files)} files")
        
        for file_path, content in code_files.items():
            file_errors = []
            
            try:
                tree = ast.parse(content, filename=file_path)
                
                # Check for common structural issues
                issues = self._analyze_ast_issues(tree, file_path)
                
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

    def _extract_imports_from_ast(self, tree: ast.AST) -> List[Dict[str, str]]:
        """Extract import statements from AST."""
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

    def _analyze_ast_issues(self, tree: ast.AST, file_path: str) -> List[str]:
        """Analyze AST for common structural issues."""
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

    def _file_passed_validation(self, file_path: str, validation_results: Dict[str, Any]) -> bool:
        """Check if a file passed all validation steps."""
        syntax_passed = file_path in validation_results["syntax_validation"].get("passed_files", [])
        import_passed = file_path in validation_results["import_validation"].get("passed_files", [])
        ast_passed = file_path in validation_results["ast_validation"].get("passed_files", [])
        
        # File passes if it passes syntax and doesn't fail other enabled checks
        return syntax_passed and (not validation_results["import_validation"]["success"] or import_passed) and (not validation_results["ast_validation"]["success"] or ast_passed)

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