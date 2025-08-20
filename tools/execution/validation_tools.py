# tools/execution/validation_tools.py
"""
Validation tools for code execution validation, syntax checking, and functionality testing.
Extracted from ExecutionValidatorAgent to provide atomic, reusable validation capabilities.
"""

import ast
import importlib.util
import logging
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

class ValidationType(Enum):
    """Types of validation operations."""
    SYNTAX_CHECK = "syntax_check"
    IMPORT_CHECK = "import_check"
    EXECUTION_CHECK = "execution_check"
    TEST_EXECUTION = "test_execution"
    LINTING = "linting"
    TYPE_CHECK = "type_check"

class ValidationResult(Enum):
    """Validation results."""
    PASSED = "passed"
    FAILED = "failed"
    WARNING = "warning"
    SKIPPED = "skipped"

@dataclass
class ValidationContext:
    """Context for validation operations."""
    code_files: Dict[str, str]
    working_directory: str = "."
    python_executable: Optional[str] = None
    timeout_seconds: int = 30
    test_script: Optional[str] = None
    validation_types: List[ValidationType] = None
    strict_mode: bool = False
    
    def __post_init__(self):
        if self.validation_types is None:
            self.validation_types = [
                ValidationType.SYNTAX_CHECK,
                ValidationType.IMPORT_CHECK
            ]

@dataclass
class ValidationReport:
    """Detailed validation report for a single file."""
    file_path: str
    validation_type: ValidationType
    result: ValidationResult
    message: str
    details: List[str] = None
    line_number: Optional[int] = None
    
    def __post_init__(self):
        if self.details is None:
            self.details = []

@dataclass
class ValidationSummary:
    """Summary of all validation operations."""
    overall_success: bool
    total_files: int
    passed_validations: int
    failed_validations: int
    warnings: int
    execution_time: float
    reports: List[ValidationReport] = None
    summary_message: str = ""
    
    def __post_init__(self):
        if self.reports is None:
            self.reports = []

def validate_python_syntax(code_files: Dict[str, str]) -> List[ValidationReport]:
    """Validate Python syntax for all provided code files."""
    reports = []
    
    logger.info(f"Validating syntax for {len(code_files)} files")
    
    for file_path, content in code_files.items():
        try:
            # Attempt to parse the AST
            ast.parse(content)
            
            reports.append(ValidationReport(
                file_path=file_path,
                validation_type=ValidationType.SYNTAX_CHECK,
                result=ValidationResult.PASSED,
                message=f"Syntax validation passed",
                details=[f"Valid Python syntax: {file_path}"]
            ))
            
        except SyntaxError as e:
            reports.append(ValidationReport(
                file_path=file_path,
                validation_type=ValidationType.SYNTAX_CHECK,
                result=ValidationResult.FAILED,
                message=f"Syntax error: {e.msg}",
                details=[f"Syntax error in {file_path} at line {e.lineno}: {e.msg}"],
                line_number=e.lineno
            ))
            
        except Exception as e:
            reports.append(ValidationReport(
                file_path=file_path,
                validation_type=ValidationType.SYNTAX_CHECK,
                result=ValidationResult.FAILED,
                message=f"Compilation error: {str(e)}",
                details=[f"Compilation error in {file_path}: {str(e)}"]
            ))
    
    return reports

def validate_imports(
    code_files: Dict[str, str], 
    working_directory: str = ".",
    python_path: Optional[List[str]] = None
) -> List[ValidationReport]:
    """Validate that all imports in the code can be resolved."""
    reports = []
    
    logger.info(f"Validating imports for {len(code_files)} files")
    
    # Setup environment
    original_cwd = os.getcwd()
    original_path = sys.path.copy()
    
    try:
        os.chdir(working_directory)
        
        # Add working directory and any additional paths
        if working_directory not in sys.path:
            sys.path.insert(0, working_directory)
        
        if python_path:
            for path in python_path:
                if path not in sys.path:
                    sys.path.insert(0, path)
        
        for file_path, content in code_files.items():
            try:
                # Extract imports using AST
                imports_info = _extract_imports_from_content(content)
                
                failed_imports = []
                successful_imports = []
                
                for import_info in imports_info:
                    try:
                        if import_info["type"] == "import":
                            # import module
                            importlib.import_module(import_info["module"])
                            successful_imports.append(import_info["module"])
                        elif import_info["type"] == "from":
                            # from module import name
                            module = importlib.import_module(import_info["module"])
                            if hasattr(module, import_info["name"]):
                                successful_imports.append(f"{import_info['module']}.{import_info['name']}")
                            else:
                                failed_imports.append(f"{import_info['module']}.{import_info['name']} (attribute not found)")
                        
                    except ImportError as e:
                        failed_imports.append(f"{import_info['module']} ({str(e)})")
                    except Exception as e:
                        failed_imports.append(f"{import_info['module']} (error: {str(e)})")
                
                if failed_imports:
                    reports.append(ValidationReport(
                        file_path=file_path,
                        validation_type=ValidationType.IMPORT_CHECK,
                        result=ValidationResult.FAILED,
                        message=f"Import validation failed: {len(failed_imports)} imports failed",
                        details=failed_imports + [f"Successful imports: {', '.join(successful_imports)}"]
                    ))
                else:
                    reports.append(ValidationReport(
                        file_path=file_path,
                        validation_type=ValidationType.IMPORT_CHECK,
                        result=ValidationResult.PASSED,
                        message=f"All imports resolved successfully",
                        details=[f"Successfully resolved {len(successful_imports)} imports"]
                    ))
                    
            except Exception as e:
                reports.append(ValidationReport(
                    file_path=file_path,
                    validation_type=ValidationType.IMPORT_CHECK,
                    result=ValidationResult.FAILED,
                    message=f"Import validation error: {str(e)}",
                    details=[f"Error analyzing imports in {file_path}: {str(e)}"]
                ))
    
    finally:
        # Restore environment
        os.chdir(original_cwd)
        sys.path = original_path
    
    return reports

def validate_code_execution(
    code_files: Dict[str, str],
    working_directory: str = ".",
    timeout_seconds: int = 30,
    python_executable: Optional[str] = None
) -> List[ValidationReport]:
    """Validate that code can be executed without runtime errors."""
    reports = []
    
    if python_executable is None:
        python_executable = sys.executable
    
    logger.info(f"Validating execution for {len(code_files)} files")
    
    for file_path, content in code_files.items():
        try:
            # Create a temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
                temp_file.write(content)
                temp_file_path = temp_file.name
            
            try:
                # Execute the file
                result = subprocess.run(
                    [python_executable, temp_file_path],
                    capture_output=True,
                    text=True,
                    timeout=timeout_seconds,
                    cwd=working_directory
                )
                
                if result.returncode == 0:
                    reports.append(ValidationReport(
                        file_path=file_path,
                        validation_type=ValidationType.EXECUTION_CHECK,
                        result=ValidationResult.PASSED,
                        message="Code executed successfully",
                        details=[
                            f"Exit code: 0",
                            f"Output: {result.stdout.strip()}" if result.stdout.strip() else "No output"
                        ]
                    ))
                else:
                    reports.append(ValidationReport(
                        file_path=file_path,
                        validation_type=ValidationType.EXECUTION_CHECK,
                        result=ValidationResult.FAILED,
                        message=f"Code execution failed (exit code {result.returncode})",
                        details=[
                            f"Exit code: {result.returncode}",
                            f"Error output: {result.stderr.strip()}" if result.stderr.strip() else "No error output",
                            f"Standard output: {result.stdout.strip()}" if result.stdout.strip() else "No standard output"
                        ]
                    ))
            finally:
                # Clean up temp file
                try:
                    os.unlink(temp_file_path)
                except Exception:
                    pass
                    
        except subprocess.TimeoutExpired:
            reports.append(ValidationReport(
                file_path=file_path,
                validation_type=ValidationType.EXECUTION_CHECK,
                result=ValidationResult.FAILED,
                message=f"Code execution timed out after {timeout_seconds} seconds",
                details=[f"Execution timeout: {timeout_seconds}s"]
            ))
            
        except Exception as e:
            reports.append(ValidationReport(
                file_path=file_path,
                validation_type=ValidationType.EXECUTION_CHECK,
                result=ValidationResult.FAILED,
                message=f"Execution validation error: {str(e)}",
                details=[f"Error executing {file_path}: {str(e)}"]
            ))
    
    return reports

def validate_with_tests(
    test_script_path: str,
    working_directory: str = ".",
    timeout_seconds: int = 60,
    python_executable: Optional[str] = None
) -> ValidationReport:
    """Validate code by running a test script."""
    if python_executable is None:
        python_executable = sys.executable
    
    logger.info(f"Running test script: {test_script_path}")
    
    if not os.path.exists(test_script_path):
        return ValidationReport(
            file_path=test_script_path,
            validation_type=ValidationType.TEST_EXECUTION,
            result=ValidationResult.FAILED,
            message=f"Test script not found: {test_script_path}",
            details=[f"File does not exist: {test_script_path}"]
        )
    
    try:
        result = subprocess.run(
            [python_executable, test_script_path],
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            cwd=working_directory
        )
        
        if result.returncode == 0:
            return ValidationReport(
                file_path=test_script_path,
                validation_type=ValidationType.TEST_EXECUTION,
                result=ValidationResult.PASSED,
                message="Test script executed successfully",
                details=[
                    f"Exit code: 0",
                    f"Test output: {result.stdout.strip()}" if result.stdout.strip() else "No output"
                ]
            )
        else:
            return ValidationReport(
                file_path=test_script_path,
                validation_type=ValidationType.TEST_EXECUTION,
                result=ValidationResult.FAILED,
                message=f"Test script failed (exit code {result.returncode})",
                details=[
                    f"Exit code: {result.returncode}",
                    f"Error output: {result.stderr.strip()}" if result.stderr.strip() else "No error output",
                    f"Standard output: {result.stdout.strip()}" if result.stdout.strip() else "No standard output"
                ]
            )
            
    except subprocess.TimeoutExpired:
        return ValidationReport(
            file_path=test_script_path,
            validation_type=ValidationType.TEST_EXECUTION,
            result=ValidationResult.FAILED,
            message=f"Test script timed out after {timeout_seconds} seconds",
            details=[f"Test timeout: {timeout_seconds}s"]
        )
        
    except Exception as e:
        return ValidationReport(
            file_path=test_script_path,
            validation_type=ValidationType.TEST_EXECUTION,
            result=ValidationResult.FAILED,
            message=f"Test execution error: {str(e)}",
            details=[f"Error running test script: {str(e)}"]
        )

def run_linting(
    code_files: Dict[str, str],
    working_directory: str = ".",
    linter: str = "flake8",
    python_executable: Optional[str] = None
) -> List[ValidationReport]:
    """Run linting tools on code files."""
    reports = []
    
    if python_executable is None:
        python_executable = sys.executable
    
    logger.info(f"Running {linter} on {len(code_files)} files")
    
    for file_path, content in code_files.items():
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
                temp_file.write(content)
                temp_file_path = temp_file.name
            
            try:
                # Run linter
                if linter == "flake8":
                    cmd = [python_executable, "-m", "flake8", temp_file_path]
                elif linter == "pylint":
                    cmd = [python_executable, "-m", "pylint", temp_file_path]
                elif linter == "pycodestyle":
                    cmd = [python_executable, "-m", "pycodestyle", temp_file_path]
                else:
                    reports.append(ValidationReport(
                        file_path=file_path,
                        validation_type=ValidationType.LINTING,
                        result=ValidationResult.SKIPPED,
                        message=f"Unsupported linter: {linter}",
                        details=[f"Linter '{linter}' not supported"]
                    ))
                    continue
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    cwd=working_directory
                )
                
                if result.returncode == 0:
                    reports.append(ValidationReport(
                        file_path=file_path,
                        validation_type=ValidationType.LINTING,
                        result=ValidationResult.PASSED,
                        message=f"Linting passed ({linter})",
                        details=[f"No linting issues found"]
                    ))
                else:
                    # Parse linting output
                    issues = result.stdout.strip().split('\n') if result.stdout.strip() else []
                    reports.append(ValidationReport(
                        file_path=file_path,
                        validation_type=ValidationType.LINTING,
                        result=ValidationResult.WARNING,  # Linting issues are warnings by default
                        message=f"Linting issues found ({linter}): {len(issues)} issues",
                        details=issues
                    ))
                    
            finally:
                # Clean up temp file
                try:
                    os.unlink(temp_file_path)
                except Exception:
                    pass
                    
        except subprocess.TimeoutExpired:
            reports.append(ValidationReport(
                file_path=file_path,
                validation_type=ValidationType.LINTING,
                result=ValidationResult.FAILED,
                message=f"Linting timed out",
                details=[f"Linter {linter} timed out after 30 seconds"]
            ))
            
        except Exception as e:
            reports.append(ValidationReport(
                file_path=file_path,
                validation_type=ValidationType.LINTING,
                result=ValidationResult.FAILED,
                message=f"Linting error: {str(e)}",
                details=[f"Error running {linter}: {str(e)}"]
            ))
    
    return reports

def comprehensive_validation(context: ValidationContext) -> ValidationSummary:
    """Run comprehensive validation on code files."""
    import time
    
    start_time = time.time()
    all_reports = []
    
    # Run requested validation types
    for validation_type in context.validation_types:
        if validation_type == ValidationType.SYNTAX_CHECK:
            reports = validate_python_syntax(context.code_files)
            all_reports.extend(reports)
            
        elif validation_type == ValidationType.IMPORT_CHECK:
            reports = validate_imports(context.code_files, context.working_directory)
            all_reports.extend(reports)
            
        elif validation_type == ValidationType.EXECUTION_CHECK:
            reports = validate_code_execution(
                context.code_files,
                context.working_directory,
                context.timeout_seconds,
                context.python_executable
            )
            all_reports.extend(reports)
            
        elif validation_type == ValidationType.TEST_EXECUTION and context.test_script:
            report = validate_with_tests(
                context.test_script,
                context.working_directory,
                context.timeout_seconds,
                context.python_executable
            )
            all_reports.append(report)
            
        elif validation_type == ValidationType.LINTING:
            reports = run_linting(
                context.code_files,
                context.working_directory,
                "flake8",  # Default linter
                context.python_executable
            )
            all_reports.extend(reports)
    
    # Analyze results
    passed_count = sum(1 for r in all_reports if r.result == ValidationResult.PASSED)
    failed_count = sum(1 for r in all_reports if r.result == ValidationResult.FAILED)
    warning_count = sum(1 for r in all_reports if r.result == ValidationResult.WARNING)
    
    # Determine overall success
    if context.strict_mode:
        overall_success = failed_count == 0 and warning_count == 0
    else:
        overall_success = failed_count == 0
    
    execution_time = time.time() - start_time
    
    # Create summary message
    if overall_success:
        summary_message = f"Validation PASSED - {passed_count}/{len(all_reports)} checks successful"
    else:
        summary_message = f"Validation FAILED - {failed_count} failures, {warning_count} warnings"
    
    return ValidationSummary(
        overall_success=overall_success,
        total_files=len(context.code_files),
        passed_validations=passed_count,
        failed_validations=failed_count,
        warnings=warning_count,
        execution_time=execution_time,
        reports=all_reports,
        summary_message=summary_message
    )

def _extract_imports_from_content(content: str) -> List[Dict[str, str]]:
    """Extract import information from code content using AST."""
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return []  # Can't parse, return empty
    
    imports = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append({
                    "type": "import",
                    "module": alias.name,
                    "name": alias.name,
                    "as_name": alias.asname
                })
                
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                imports.append({
                    "type": "from",
                    "module": module,
                    "name": alias.name,
                    "as_name": alias.asname
                })
    
    return imports

def validate_file_exists(file_path: str, working_directory: str = ".") -> ValidationReport:
    """Validate that a specific file exists."""
    full_path = Path(working_directory) / file_path
    
    if full_path.exists():
        return ValidationReport(
            file_path=file_path,
            validation_type=ValidationType.EXECUTION_CHECK,
            result=ValidationResult.PASSED,
            message="File exists",
            details=[f"File found at: {full_path.absolute()}"]
        )
    else:
        return ValidationReport(
            file_path=file_path,
            validation_type=ValidationType.EXECUTION_CHECK,
            result=ValidationResult.FAILED,
            message="File not found",
            details=[f"File not found: {full_path.absolute()}"]
        )

def get_validation_summary_text(summary: ValidationSummary) -> str:
    """Generate a human-readable summary of validation results."""
    lines = [
        f"Validation Summary:",
        f"  Total files: {summary.total_files}",
        f"  Passed: {summary.passed_validations}",
        f"  Failed: {summary.failed_validations}",
        f"  Warnings: {summary.warnings}",
        f"  Execution time: {summary.execution_time:.2f}s",
        f"  Overall result: {'✓ PASSED' if summary.overall_success else '✗ FAILED'}",
        ""
    ]
    
    if summary.reports:
        lines.append("Detailed Results:")
        for report in summary.reports:
            status = "✓" if report.result == ValidationResult.PASSED else "✗" if report.result == ValidationResult.FAILED else "⚠"
            lines.append(f"  {status} {report.file_path}: {report.message}")
            
            if report.details:
                for detail in report.details[:3]:  # Show first 3 details
                    lines.append(f"    - {detail}")
                if len(report.details) > 3:
                    lines.append(f"    ... and {len(report.details) - 3} more")
        
    return "\n".join(lines)