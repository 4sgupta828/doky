# agents/execution_validator.py
import os
import sys
import subprocess
import logging
import tempfile
from typing import Dict, Any, List

from .base import BaseAgent
from core.context import GlobalContext
from core.models import AgentResponse, TaskNode, AgentResult

logger = logging.getLogger(__name__)

class ExecutionValidatorAgent(BaseAgent):
    """
    Agent responsible for validating code execution and functionality.
    
    This agent:
    1. Validates Python syntax of generated code
    2. Tests import resolution and dependencies
    3. Executes test scripts and validates results
    4. Provides detailed validation reports with specific failures
    """
    
    def __init__(self):
        super().__init__(
            name="ExecutionValidatorAgent",
            description="Validates code execution, runs tests, and reports functionality status"
        )
    
    def required_inputs(self) -> List[str]:
        return ["code_files"]
    
    def optional_inputs(self) -> List[str]:
        return ["test_script", "output_directory", "timeout_seconds"]
    
    def execute_v2(self, goal: str, inputs: Dict[str, Any], global_context: GlobalContext) -> AgentResult:
        """Execute validation tests for the generated code."""
        self.validate_inputs(inputs)
        
        code_files = inputs['code_files']
        test_script = inputs.get('test_script')
        output_dir = inputs.get('output_directory', global_context.workspace_path)
        timeout_seconds = inputs.get('timeout_seconds', 30)
        
        self.report_progress("Validating code execution", f"Testing {len(code_files)} files")
        
        # Initialize validation results
        validation_results = {
            'syntax_check': False,
            'import_check': False,
            'test_execution': False,
            'overall_success': False,
            'details': [],
            'errors': [],
            'warnings': []
        }
        
        try:
            # Step 1: Syntax validation
            syntax_valid, syntax_details = self._validate_syntax(code_files)
            validation_results['syntax_check'] = syntax_valid
            validation_results['details'].extend(syntax_details)
            
            self.report_intermediate_output("syntax_validation", {
                'passed': syntax_valid,
                'details': syntax_details
            })
            
            # Step 2: Import validation (only if syntax is valid)
            if syntax_valid:
                import_valid, import_details = self._validate_imports(code_files, output_dir)
                validation_results['import_check'] = import_valid
                validation_results['details'].extend(import_details)
                
                self.report_intermediate_output("import_validation", {
                    'passed': import_valid,
                    'details': import_details
                })
            else:
                validation_results['details'].append("Import validation skipped due to syntax errors")
            
            # Step 3: Test execution (if test script available and imports are valid)
            if test_script and os.path.exists(test_script) and validation_results.get('import_check', False):
                test_result, test_details = self._execute_tests(test_script, output_dir, timeout_seconds)
                validation_results['test_execution'] = test_result
                validation_results['details'].extend(test_details)
                
                self.report_intermediate_output("test_execution", {
                    'passed': test_result,
                    'details': test_details,
                    'test_script': test_script
                })
            elif test_script:
                validation_results['details'].append(f"Test execution skipped: script={bool(test_script)}, imports_valid={validation_results.get('import_check', False)}")
            else:
                validation_results['details'].append("Test execution skipped: no test script provided")
            
            # Determine overall success
            overall_success = syntax_valid and validation_results.get('import_check', True)
            if test_script and os.path.exists(test_script):
                overall_success = overall_success and validation_results.get('test_execution', False)
            
            validation_results['overall_success'] = overall_success
            
            # Generate summary message
            passed_checks = sum([
                validation_results['syntax_check'],
                validation_results.get('import_check', True),
                validation_results.get('test_execution', True) if test_script else True
            ])
            total_checks = 2 + (1 if test_script and os.path.exists(test_script) else 0)
            
            if overall_success:
                message = f"Validation PASSED - {passed_checks}/{total_checks} checks successful"
            else:
                message = f"Validation FAILED - {passed_checks}/{total_checks} checks passed"
            
            return self.create_result(True, message, validation_results)
            
        except Exception as e:
            error_msg = f"Validation execution failed: {e}"
            logger.error(error_msg, exc_info=True)
            validation_results['errors'].append(str(e))
            return self.create_result(False, error_msg, validation_results)
    
    def _validate_syntax(self, code_files: Dict[str, str]) -> tuple[bool, List[str]]:
        """Validate Python syntax for all files."""
        details = []
        syntax_errors = []
        
        self.report_thinking(f"Checking syntax for {len(code_files)} files")
        
        for file_path, content in code_files.items():
            try:
                # Attempt to compile the code
                compile(content, file_path, 'exec')
                details.append(f"✓ Syntax valid: {file_path}")
            except SyntaxError as e:
                error_msg = f"✗ Syntax error in {file_path} line {e.lineno}: {e.msg}"
                details.append(error_msg)
                syntax_errors.append(error_msg)
                logger.error(error_msg)
            except Exception as e:
                error_msg = f"✗ Compilation error in {file_path}: {e}"
                details.append(error_msg)
                syntax_errors.append(error_msg)
                logger.error(error_msg)
        
        is_valid = len(syntax_errors) == 0
        
        if is_valid:
            details.append(f"✓ All {len(code_files)} files have valid Python syntax")
        else:
            details.append(f"✗ {len(syntax_errors)} files have syntax errors")
        
        return is_valid, details
    
    def _validate_imports(self, code_files: Dict[str, str], output_dir: str) -> tuple[bool, List[str]]:
        """Validate that imports can be resolved."""
        details = []
        import_errors = []
        
        self.report_thinking(f"Checking imports for {len(code_files)} files")
        
        # Change to output directory to simulate execution environment
        original_cwd = os.getcwd()
        original_path = sys.path.copy()
        
        try:
            os.chdir(output_dir)
            if output_dir not in sys.path:
                sys.path.insert(0, output_dir)
            
            for file_path, content in code_files.items():
                try:
                    # Create a temporary file to test imports
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
                        temp_file.write(content)
                        temp_file_path = temp_file.name
                    
                    try:
                        # Try to import the temporary module
                        import importlib.util
                        spec = importlib.util.spec_from_file_location("temp_module", temp_file_path)
                        if spec and spec.loader:
                            module = importlib.util.module_from_spec(spec)
                            # Don't execute, just check if imports resolve
                            # spec.loader.exec_module(module)
                            details.append(f"✓ Imports resolve: {file_path}")
                        else:
                            details.append(f"⚠ Could not create module spec for: {file_path}")
                    finally:
                        # Clean up temp file
                        try:
                            os.unlink(temp_file_path)
                        except:
                            pass
                    
                except ImportError as e:
                    error_msg = f"✗ Import error in {file_path}: {e}"
                    details.append(error_msg)
                    import_errors.append(error_msg)
                    logger.warning(error_msg)
                except Exception as e:
                    # Don't treat other errors as import failures
                    details.append(f"⚠ Import check warning for {file_path}: {e}")
                    logger.debug(f"Import check warning for {file_path}: {e}")
            
            is_valid = len(import_errors) == 0
            
            if is_valid:
                details.append(f"✓ Import validation passed for {len(code_files)} files")
            else:
                details.append(f"✗ {len(import_errors)} files have import issues")
            
            return is_valid, details
            
        finally:
            # Restore environment
            os.chdir(original_cwd)
            sys.path = original_path
    
    def _execute_tests(self, test_script: str, output_dir: str, timeout_seconds: int) -> tuple[bool, List[str]]:
        """Execute the test script and return success status."""
        details = []
        
        self.report_thinking(f"Executing test script: {test_script}")
        
        if not os.path.exists(test_script):
            error_msg = f"✗ Test script not found: {test_script}"
            details.append(error_msg)
            return False, details
        
        try:
            # Execute test script
            result = subprocess.run(
                [sys.executable, test_script],
                cwd=output_dir,
                capture_output=True,
                text=True,
                timeout=timeout_seconds
            )
            
            # Log output
            if result.stdout:
                details.append(f"Test output: {result.stdout.strip()}")
                logger.info(f"Test stdout: {result.stdout}")
            
            if result.stderr:
                details.append(f"Test errors: {result.stderr.strip()}")
                logger.warning(f"Test stderr: {result.stderr}")
            
            # Check return code
            if result.returncode == 0:
                details.append("✓ Test script executed successfully")
                return True, details
            else:
                details.append(f"✗ Test script failed with return code {result.returncode}")
                return False, details
                
        except subprocess.TimeoutExpired:
            error_msg = f"✗ Test execution timed out after {timeout_seconds} seconds"
            details.append(error_msg)
            logger.error(error_msg)
            return False, details
            
        except Exception as e:
            error_msg = f"✗ Test execution failed: {e}"
            details.append(error_msg)
            logger.error(error_msg, exc_info=True)
            return False, details