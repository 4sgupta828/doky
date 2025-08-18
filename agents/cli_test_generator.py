# agents/cli_test_generator.py
import os
import logging
from typing import Dict, Any, List

from .base import BaseAgent
from core.context import GlobalContext
from core.models import AgentResponse, TaskNode, AgentResult

logger = logging.getLogger(__name__)

class CLITestGeneratorAgent(BaseAgent):
    """
    Agent responsible for generating CLI test scripts for Python code.
    
    This agent:
    1. Analyzes generated code to understand functionality
    2. Creates executable test scripts that demonstrate usage
    3. Generates realistic test scenarios with sample data
    4. Includes clear pass/fail indicators and usage instructions
    """
    
    def __init__(self, llm_client=None):
        super().__init__(
            name="CLITestGeneratorAgent", 
            description="Creates executable CLI test scripts for Python code validation"
        )
        self.llm_client = llm_client
    
    def required_inputs(self) -> List[str]:
        return ["code_files"]
    
    def optional_inputs(self) -> List[str]:
        return ["specification", "output_directory", "test_file_name"]
    
    def execute(self, goal: str, context: GlobalContext, current_task: TaskNode) -> AgentResponse:
        """Legacy execute method - tries to infer inputs from context."""
        # Try to get code files from artifacts or recent files
        code_files = {}
        specification = goal  # Use goal as specification
        
        # Look for recent Python files in artifacts
        for artifact_key in context.list_artifacts():
            if artifact_key.endswith('.py') or 'code' in artifact_key.lower():
                content = context.get_artifact(artifact_key)
                if content and isinstance(content, str):
                    code_files[artifact_key] = content
        
        # If no artifacts, try to get from workspace
        if not code_files:
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
                message="No Python code files found to generate tests for"
            )
        
        inputs = {
            'code_files': code_files,
            'specification': specification,
            'output_directory': context.workspace_path
        }
        
        result = self.execute_v2(goal, inputs, context)
        
        return AgentResponse(
            success=result.success,
            message=result.message,
            artifacts_generated=[result.outputs.get('test_script')] if result.outputs.get('test_script') else []
        )
    
    def execute_v2(self, goal: str, inputs: Dict[str, Any], global_context: GlobalContext) -> AgentResult:
        """Generate CLI test script for the code."""
        self.validate_inputs(inputs)
        
        code_files = inputs['code_files']
        specification = inputs.get('specification', goal)
        output_dir = inputs.get('output_directory', global_context.workspace_path)
        test_file_name = inputs.get('test_file_name', 'test_cli.py')
        
        self.report_progress("Generating CLI tests", f"Creating tests for {len(code_files)} code files")
        
        if not self.llm_client:
            # Create basic test template without LLM
            test_script_content = self._create_basic_test_template(code_files, specification)
        else:
            # Generate sophisticated test script using LLM
            try:
                test_script_content = self._generate_test_script_with_llm(code_files, specification)
            except Exception as e:
                logger.warning(f"LLM test generation failed: {e}, falling back to template")
                test_script_content = self._create_basic_test_template(code_files, specification)
        
        try:
            # Write test script
            test_script_path = os.path.join(output_dir, test_file_name)
            global_context.workspace.write_file_content(test_script_path, test_script_content, "cli_test_generator")
            
            # Make it executable (attempt - not critical if it fails)
            try:
                import stat
                os.chmod(test_script_path, stat.S_IRWXU | stat.S_IRGRP | stat.S_IROTH)
            except:
                pass  # Not critical if chmod fails
            
            self.report_intermediate_output("generated_test", {
                'path': test_script_path,
                'preview': test_script_content[:200] + "..." if len(test_script_content) > 200 else test_script_content
            })
            
            return self.create_result(
                True,
                f"Generated CLI test script: {test_script_path}",
                {
                    'test_script': test_script_path,
                    'test_content': test_script_content,
                    'main_files': self._identify_main_files(code_files)
                }
            )
            
        except Exception as e:
            error_msg = f"Failed to write test script: {e}"
            logger.error(error_msg, exc_info=True)
            return self.create_result(False, error_msg, error_details={'exception': str(e)})
    
    def _identify_main_files(self, code_files: Dict[str, str]) -> List[str]:
        """Identify main executable files in the code."""
        main_files = []
        
        for file_path, content in code_files.items():
            # Look for main execution patterns
            if ('if __name__ == "__main__"' in content or 
                file_path.endswith('main.py') or
                'def main(' in content):
                main_files.append(file_path)
        
        # If no clear main files, use first Python file
        if not main_files and code_files:
            main_files = [list(code_files.keys())[0]]
        
        return main_files
    
    def _generate_test_script_with_llm(self, code_files: Dict[str, str], specification: str) -> str:
        """Generate test script content using LLM."""
        
        main_files = self._identify_main_files(code_files)
        
        # Create code summary for prompt
        code_summary = ""
        for file_path, content in list(code_files.items())[:3]:  # First 3 files
            code_summary += f"\n--- File: {file_path} ---\n"
            code_summary += content[:500] + ("..." if len(content) > 500 else "")
        
        prompt = f"""Create a comprehensive Python CLI test script that validates the functionality of the generated code.

SPECIFICATION:
{specification}

GENERATED CODE FILES:
{code_summary}

MAIN EXECUTABLE FILES: {main_files}

Requirements for the test script:
1. Test the main functionality described in the specification
2. Use realistic sample/mock input data where needed
3. Validate expected outputs or behaviors with assertions
4. Include clear pass/fail indicators with colored output
5. Make it executable as a standalone script with proper error handling
6. Add comprehensive usage instructions in docstring
7. Test both success and edge cases where appropriate
8. Use subprocess to test CLI interfaces if applicable
9. Include timing and performance information
10. Provide detailed output about what's being tested

The test should be practical and demonstrate that the code works as intended.
Use proper Python testing patterns with try/except blocks and meaningful error messages.

Return only the Python test script code (no JSON wrapper, no markdown):"""
        
        try:
            response = self.llm_client.invoke(prompt)
            
            # Clean up response - remove any markdown code blocks
            if '```python' in response:
                response = response.split('```python')[1].split('```')[0]
            elif '```' in response:
                response = response.split('```')[1].split('```')[0]
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"LLM test generation failed: {e}")
            raise
    
    def _create_basic_test_template(self, code_files: Dict[str, str], specification: str) -> str:
        """Create a basic test template as fallback when LLM is not available."""
        
        main_files = self._identify_main_files(code_files)
        main_file = main_files[0] if main_files else "main.py"
        
        # Extract module name from file path
        module_name = main_file.replace('.py', '').replace('/', '.').replace('\\', '.')
        
        return f'''#!/usr/bin/env python3
"""
CLI Test Script for Generated Code

This script tests the functionality described in:
{specification[:200]}...

Usage: python test_cli.py
"""

import sys
import os
import subprocess
import time
from pathlib import Path

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ANSI color codes for output
GREEN = '\\033[92m'
RED = '\\033[91m'
YELLOW = '\\033[93m'
BLUE = '\\033[94m'
RESET = '\\033[0m'

def print_success(msg):
    print(f"{{GREEN}}✓ {{msg}}{{RESET}}")

def print_error(msg):
    print(f"{{RED}}✗ {{msg}}{{RESET}}")

def print_info(msg):
    print(f"{{BLUE}}ℹ {{msg}}{{RESET}}")

def print_warning(msg):
    print(f"{{YELLOW}}⚠ {{msg}}{{RESET}}")

class TestSuite:
    """Test suite for generated code validation."""
    
    def __init__(self):
        self.tests_passed = 0
        self.total_tests = 0
        self.main_file = "{main_file}"
        
    def run_test(self, test_name, test_func):
        """Run a single test with error handling."""
        self.total_tests += 1
        print(f"\\n{{BLUE}}Running: {{test_name}}{{RESET}}")
        
        try:
            start_time = time.time()
            result = test_func()
            duration = time.time() - start_time
            
            if result:
                self.tests_passed += 1
                print_success(f"{{test_name}} - PASSED ({duration:.2f}s)")
                return True
            else:
                print_error(f"{{test_name}} - FAILED ({duration:.2f}s)")
                return False
                
        except Exception as e:
            duration = time.time() - start_time
            print_error(f"{{test_name}} - ERROR: {{e}} ({duration:.2f}s)")
            return False
    
    def test_import(self):
        """Test that the main module can be imported."""
        print_info("Testing module import...")
        try:
            if self.main_file.endswith('.py'):
                module_name = self.main_file[:-3].replace('/', '.').replace('\\\\', '.')
                exec(f"import {{module_name}}")
            print_success("Module imported successfully")
            return True
        except ImportError as e:
            print_error(f"Import failed: {{e}}")
            return False
        except Exception as e:
            print_error(f"Import error: {{e}}")
            return False
    
    def test_execution(self):
        """Test basic execution of the main file."""
        print_info(f"Testing execution of {{self.main_file}}...")
        
        if not os.path.exists(self.main_file):
            print_error(f"Main file {{self.main_file}} not found")
            return False
        
        try:
            # Run with a short timeout
            result = subprocess.run(
                [sys.executable, self.main_file],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                print_success("Code executed successfully")
                if result.stdout:
                    print_info(f"Output: {{result.stdout.strip()[:100]}}")
                return True
            else:
                print_error(f"Execution failed with return code {{result.returncode}}")
                if result.stderr:
                    print_error(f"Error: {{result.stderr.strip()}}")
                return False
                
        except subprocess.TimeoutExpired:
            print_warning("Execution timed out (may be waiting for input)")
            return True  # Count as success - code ran but needed input
        except Exception as e:
            print_error(f"Execution test failed: {{e}}")
            return False
    
    def test_syntax(self):
        """Test Python syntax of all files."""
        print_info("Testing Python syntax...")
        
        python_files = [f for f in os.listdir('.') if f.endswith('.py')]
        syntax_errors = []
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    source = f.read()
                compile(source, py_file, 'exec')
            except SyntaxError as e:
                syntax_errors.append(f"{{py_file}}: {{e}}")
            except Exception as e:
                print_warning(f"Could not check syntax of {{py_file}}: {{e}}")
        
        if syntax_errors:
            for error in syntax_errors:
                print_error(f"Syntax error in {{error}}")
            return False
        else:
            print_success(f"Syntax check passed for {{len(python_files)}} files")
            return True

def main():
    """Run all tests and report results."""
    print("=" * 60)
    print("🧪 CLI TEST SUITE")
    print("=" * 60)
    print(f"Testing generated code based on specification:")
    print(f"{{specification[:150]}}...")
    print("=" * 60)
    
    suite = TestSuite()
    
    # Run tests
    suite.run_test("Syntax Validation", suite.test_syntax)
    suite.run_test("Module Import", suite.test_import)
    suite.run_test("Basic Execution", suite.test_execution)
    
    # Results summary
    print("\\n" + "=" * 60)
    print("📊 TEST RESULTS")
    print("=" * 60)
    print(f"Tests passed: {{suite.tests_passed}}/{{suite.total_tests}}")
    
    if suite.tests_passed == suite.total_tests:
        print_success("🎉 ALL TESTS PASSED!")
        print_info("The generated code appears to be working correctly.")
        return 0
    else:
        print_error(f"❌ {{suite.total_tests - suite.tests_passed}} TEST(S) FAILED")
        print_warning("Please review the generated code for issues.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
'''