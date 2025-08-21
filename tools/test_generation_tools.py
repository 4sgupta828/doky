# tools/creation/test_generation_tools.py
import logging
import re
from typing import Dict, Any, List, Optional
from enum import Enum
from dataclasses import dataclass
from pathlib import Path

# Get a logger instance for this module
logger = logging.getLogger(__name__)


class TestType(Enum):
    """Types of tests that can be generated."""
    UNIT = "unit"
    INTEGRATION = "integration"
    CLI = "cli"
    API = "api"
    PERFORMANCE = "performance"
    SECURITY = "security"


class TestQuality(Enum):
    """Test quality levels for different use cases."""
    FAST = "fast"
    DECENT = "decent"
    PRODUCTION = "production"


class TestFramework(Enum):
    """Supported test frameworks."""
    UNITTEST = "unittest"
    PYTEST = "pytest"
    JEST = "jest"
    MOCHA = "mocha"
    JUNIT = "junit"
    GO_TEST = "go_test"


@dataclass
class TestGenerationContext:
    """Context for test generation operations."""
    goal: str
    source_files: Dict[str, str]
    test_type: TestType = TestType.UNIT
    test_quality: TestQuality = TestQuality.DECENT
    framework: TestFramework = TestFramework.PYTEST
    output_directory: str = "tests"
    specification: Optional[str] = None
    coverage_requirements: Dict[str, Any] = None
    custom_patterns: List[str] = None


@dataclass
class TestGenerationResult:
    """Result of test generation operation."""
    success: bool
    generated_tests: Dict[str, str]
    test_structure: Dict[str, Any]
    coverage_mapping: Dict[str, List[str]]
    error_details: Optional[str] = None
    test_count: int = 0


def generate_tests(context: TestGenerationContext, llm_client=None) -> TestGenerationResult:
    """
    Generate comprehensive tests based on source code and context.
    
    Args:
        context: Test generation context with source files and configuration
        llm_client: LLM client for intelligent test generation (optional)
        
    Returns:
        TestGenerationResult with generated tests and metadata
    """
    logger.info(f"Generating {context.test_type.value} tests for {len(context.source_files)} files")
    
    try:
        # Generate test files
        if llm_client:
            generated_tests = _generate_tests_with_llm(context, llm_client)
        else:
            generated_tests = _generate_tests_fallback(context)
        
        # Analyze test structure
        test_structure = _analyze_test_structure(generated_tests)
        coverage_mapping = _create_coverage_mapping(context.source_files, generated_tests)
        test_count = _count_test_cases(generated_tests)
        
        return TestGenerationResult(
            success=True,
            generated_tests=generated_tests,
            test_structure=test_structure,
            coverage_mapping=coverage_mapping,
            test_count=test_count
        )
        
    except Exception as e:
        logger.error(f"Test generation failed: {e}")
        return TestGenerationResult(
            success=False,
            generated_tests={},
            test_structure={},
            coverage_mapping={},
            error_details=str(e)
        )


def _generate_tests_with_llm(context: TestGenerationContext, llm_client) -> Dict[str, str]:
    """Generate tests using LLM client."""
    generated_tests = {}
    
    for source_file, source_code in context.source_files.items():
        test_file_path = _get_test_file_path(source_file, context)
        
        # Build prompt for this specific source file
        prompt = _build_test_generation_prompt(source_file, source_code, context)
        
        try:
            response = llm_client.invoke(prompt)
            test_code = _extract_test_code_from_response(response)
            generated_tests[test_file_path] = test_code
            
        except Exception as e:
            logger.warning(f"LLM test generation failed for {source_file}: {e}")
            # Fallback to template-based generation
            test_code = _generate_test_fallback_single(source_file, source_code, context)
            generated_tests[test_file_path] = test_code
    
    return generated_tests


def _generate_tests_fallback(context: TestGenerationContext) -> Dict[str, str]:
    """Generate tests using template-based approach without LLM."""
    generated_tests = {}
    
    for source_file, source_code in context.source_files.items():
        test_file_path = _get_test_file_path(source_file, context)
        test_code = _generate_test_fallback_single(source_file, source_code, context)
        generated_tests[test_file_path] = test_code
    
    return generated_tests


def _generate_test_fallback_single(source_file: str, source_code: str, context: TestGenerationContext) -> str:
    """Generate test for a single source file using templates."""
    
    if context.framework == TestFramework.PYTEST:
        return _generate_pytest_fallback(source_file, source_code, context)
    elif context.framework == TestFramework.UNITTEST:
        return _generate_unittest_fallback(source_file, source_code, context)
    elif context.framework == TestFramework.JEST:
        return _generate_jest_fallback(source_file, source_code, context)
    else:
        return _generate_generic_test_fallback(source_file, source_code, context)


def _generate_pytest_fallback(source_file: str, source_code: str, context: TestGenerationContext) -> str:
    """Generate pytest-style tests."""
    file_name = Path(source_file).stem
    module_name = file_name.replace('-', '_').replace(' ', '_')
    
    # Extract functions and classes from source code
    functions = _extract_python_functions(source_code)
    classes = _extract_python_classes(source_code)
    
    test_code = f'''"""
Test module for {source_file}

Generated tests for: {context.goal}
"""

import pytest
import sys
from pathlib import Path

# Add source directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

'''
    
    # Add imports
    if functions or classes:
        if '/' in source_file:
            import_path = source_file.replace('/', '.').replace('.py', '')
        else:
            import_path = module_name
        
        test_code += f"from {import_path} import "
        
        imports = []
        if classes:
            imports.extend(classes)
        if functions:
            imports.extend(functions)
        
        test_code += ", ".join(imports[:5])  # Limit imports
        test_code += "\n\n"
    
    # Generate test fixtures if production quality
    if context.test_quality == TestQuality.PRODUCTION:
        test_code += '''
@pytest.fixture
def sample_data():
    """Provide sample data for tests."""
    return {
        "test_string": "hello world",
        "test_number": 42,
        "test_list": [1, 2, 3],
        "test_dict": {"key": "value"}
    }


@pytest.fixture
def mock_environment(monkeypatch):
    """Mock environment variables for tests."""
    monkeypatch.setenv("TEST_MODE", "true")
    yield
    monkeypatch.delenv("TEST_MODE", raising=False)

'''
    
    # Generate tests for each function
    for func_name in functions[:5]:  # Limit number of functions
        test_code += f'''
def test_{func_name}_basic():
    """Test basic functionality of {func_name}."""
    # TODO: Add specific test cases for {func_name}
    result = {func_name}()
    assert result is not None


def test_{func_name}_edge_cases():
    """Test edge cases for {func_name}."""
    # TODO: Add edge case tests for {func_name}
    with pytest.raises(Exception):
        {func_name}(None)

'''
    
    # Generate tests for each class
    for class_name in classes[:3]:  # Limit number of classes
        test_code += f'''
class Test{class_name}:
    """Test cases for {class_name} class."""
    
    def test_{class_name.lower()}_initialization(self):
        """Test {class_name} initialization."""
        instance = {class_name}()
        assert instance is not None
        
    def test_{class_name.lower()}_methods(self):
        """Test {class_name} methods."""
        instance = {class_name}()
        # TODO: Add method tests for {class_name}
        assert hasattr(instance, '__dict__')

'''
    
    # Add integration tests for production quality
    if context.test_quality == TestQuality.PRODUCTION:
        test_code += f'''
@pytest.mark.integration
def test_{module_name}_integration():
    """Integration test for {module_name} module."""
    # TODO: Add integration test cases
    pass


@pytest.mark.performance
def test_{module_name}_performance():
    """Performance test for {module_name} module."""
    import time
    start_time = time.time()
    
    # TODO: Add performance test logic
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Performance assertion (adjust threshold as needed)
    assert execution_time < 1.0, f"Execution took too long: {{execution_time:.2f}}s"

'''
    
    return test_code


def _generate_unittest_fallback(source_file: str, source_code: str, context: TestGenerationContext) -> str:
    """Generate unittest-style tests."""
    file_name = Path(source_file).stem
    module_name = file_name.replace('-', '_').replace(' ', '_')
    
    functions = _extract_python_functions(source_code)
    classes = _extract_python_classes(source_code)
    
    test_code = f'''"""
Test module for {source_file}

Generated tests using unittest framework.
"""

import unittest
import sys
from pathlib import Path

# Add source directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

'''
    
    # Add imports
    if functions or classes:
        test_code += f"from {module_name} import "
        imports = (classes + functions)[:5]
        test_code += ", ".join(imports)
        test_code += "\n\n"
    
    # Generate test class
    test_code += f'''
class Test{module_name.title()}(unittest.TestCase):
    """Test cases for {module_name} module."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_data = {{
            "sample_string": "test",
            "sample_number": 123,
            "sample_list": [1, 2, 3]
        }}
    
    def tearDown(self):
        """Clean up after tests."""
        pass

'''
    
    # Generate test methods for functions
    for func_name in functions[:5]:
        test_code += f'''
    def test_{func_name}(self):
        """Test {func_name} function."""
        # TODO: Implement test for {func_name}
        result = {func_name}()
        self.assertIsNotNone(result)
        
    def test_{func_name}_error_handling(self):
        """Test error handling in {func_name}."""
        # TODO: Test error conditions
        with self.assertRaises(Exception):
            {func_name}(None)

'''
    
    # Generate test methods for classes
    for class_name in classes[:3]:
        test_code += f'''
    def test_{class_name.lower()}_creation(self):
        """Test {class_name} instantiation."""
        instance = {class_name}()
        self.assertIsInstance(instance, {class_name})

'''
    
    test_code += '''

if __name__ == "__main__":
    unittest.main()
'''
    
    return test_code


def _generate_jest_fallback(source_file: str, source_code: str, context: TestGenerationContext) -> str:
    """Generate Jest-style JavaScript tests."""
    file_name = Path(source_file).stem
    
    functions = _extract_javascript_functions(source_code)
    classes = _extract_javascript_classes(source_code)
    
    test_code = f'''/**
 * Test module for {source_file}
 * 
 * Generated tests for: {context.goal}
 */

const {{ {", ".join((functions + classes)[:5])} }} = require('../{source_file.replace(".js", "")}');

describe('{file_name}', () => {{
    
    beforeEach(() => {{
        // Set up test fixtures
        jest.clearAllMocks();
    }});
    
    afterEach(() => {{
        // Clean up after tests
    }});

'''
    
    # Generate tests for functions
    for func_name in functions[:5]:
        test_code += f'''
    describe('{func_name}', () => {{
        
        it('should handle basic functionality', () => {{
            // TODO: Implement test for {func_name}
            const result = {func_name}();
            expect(result).toBeDefined();
        }});
        
        it('should handle error cases', () => {{
            // TODO: Test error conditions
            expect(() => {func_name}(null)).toThrow();
        }});
        
    }});

'''
    
    # Generate tests for classes
    for class_name in classes[:3]:
        test_code += f'''
    describe('{class_name}', () => {{
        
        it('should create instance correctly', () => {{
            const instance = new {class_name}();
            expect(instance).toBeInstanceOf({class_name});
        }});
        
        it('should have required methods', () => {{
            const instance = new {class_name}();
            // TODO: Test class methods
            expect(typeof instance).toBe('object');
        }});
        
    }});

'''
    
    test_code += '''
});
'''
    
    return test_code


def _generate_generic_test_fallback(source_file: str, source_code: str, context: TestGenerationContext) -> str:
    """Generate generic test structure."""
    return f'''"""
Generic test template for {source_file}

Test Type: {context.test_type.value}
Quality Level: {context.test_quality.value}
Framework: {context.framework.value}

Generated for: {context.goal}
"""

# TODO: Implement tests specific to {context.framework.value} framework
# TODO: Add test cases based on source code analysis

def test_placeholder():
    """Placeholder test - replace with actual tests."""
    assert True, "Placeholder test should pass"

'''


def _extract_python_functions(source_code: str) -> List[str]:
    """Extract function names from Python source code."""
    functions = []
    lines = source_code.split('\n')
    
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('def ') and not stripped.startswith('def _'):
            # Extract function name
            match = re.match(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', stripped)
            if match:
                func_name = match.group(1)
                if func_name not in ['__init__', '__str__', '__repr__']:
                    functions.append(func_name)
    
    return functions


def _extract_python_classes(source_code: str) -> List[str]:
    """Extract class names from Python source code."""
    classes = []
    lines = source_code.split('\n')
    
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('class '):
            # Extract class name
            match = re.match(r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)', stripped)
            if match:
                classes.append(match.group(1))
    
    return classes


def _extract_javascript_functions(source_code: str) -> List[str]:
    """Extract function names from JavaScript source code."""
    functions = []
    
    # Regular function declarations
    func_matches = re.findall(r'function\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*\(', source_code)
    functions.extend(func_matches)
    
    # Arrow functions assigned to variables
    arrow_matches = re.findall(r'const\s+([a-zA-Z_$][a-zA-Z0-9_$]*)\s*=\s*\(', source_code)
    functions.extend(arrow_matches)
    
    return functions


def _extract_javascript_classes(source_code: str) -> List[str]:
    """Extract class names from JavaScript source code."""
    classes = []
    
    class_matches = re.findall(r'class\s+([a-zA-Z_$][a-zA-Z0-9_$]*)', source_code)
    classes.extend(class_matches)
    
    return classes


def _get_test_file_path(source_file: str, context: TestGenerationContext) -> str:
    """Generate appropriate test file path."""
    source_path = Path(source_file)
    
    if context.framework in [TestFramework.PYTEST, TestFramework.UNITTEST]:
        # Python test naming conventions
        test_name = f"test_{source_path.stem}.py"
    elif context.framework in [TestFramework.JEST, TestFramework.MOCHA]:
        # JavaScript test naming conventions
        test_name = f"{source_path.stem}.test.js"
    elif context.framework == TestFramework.JUNIT:
        # Java test naming conventions
        class_name = source_path.stem.title().replace('_', '')
        test_name = f"{class_name}Test.java"
    else:
        # Generic naming
        test_name = f"test_{source_path.stem}.py"
    
    return str(Path(context.output_directory) / test_name)


def _build_test_generation_prompt(source_file: str, source_code: str, context: TestGenerationContext) -> str:
    """Build LLM prompt for intelligent test generation."""
    
    framework_instructions = {
        TestFramework.PYTEST: "Use pytest framework with fixtures and parametrized tests",
        TestFramework.UNITTEST: "Use unittest framework with setUp/tearDown methods",
        TestFramework.JEST: "Use Jest framework with describe/it blocks",
        TestFramework.MOCHA: "Use Mocha framework with describe/it blocks"
    }
    
    quality_instructions = {
        TestQuality.FAST: "Generate basic test cases focusing on core functionality",
        TestQuality.DECENT: "Generate comprehensive test cases with edge cases and error handling",
        TestQuality.PRODUCTION: "Generate extensive test suite with fixtures, mocks, and integration tests"
    }
    
    return f"""
You are an expert test engineer. Generate comprehensive {context.test_type.value} tests for the following source code.

**Source File:** {source_file}

**Source Code:**
```
{source_code}
```

**Test Requirements:**
- Framework: {context.framework.value}
- Test Type: {context.test_type.value}
- Quality Level: {context.test_quality.value}

**Framework Instructions:**
{framework_instructions.get(context.framework, "Use appropriate testing patterns")}

**Quality Instructions:**
{quality_instructions.get(context.test_quality, "Generate standard test coverage")}

**Additional Context:**
{context.specification or "No additional specification provided"}

**Output Requirements:**
Generate complete test code that:
1. Tests all public functions and methods
2. Includes appropriate test fixtures and setup
3. Tests both success and error cases
4. Follows {context.framework.value} best practices
5. Is ready to run without modification

Return only the complete test code, no additional explanation.
"""


def _extract_test_code_from_response(response: str) -> str:
    """Extract clean test code from LLM response."""
    # Remove markdown code blocks if present
    if '```' in response:
        # Find code between triple backticks
        code_match = re.search(r'```(?:python|javascript|js)?\n?(.*?)\n?```', response, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
    
    # Return response as-is if no code blocks found
    return response.strip()


def _analyze_test_structure(generated_tests: Dict[str, str]) -> Dict[str, Any]:
    """Analyze the structure of generated tests."""
    structure = {
        "total_test_files": len(generated_tests),
        "framework_distribution": {},
        "test_types": {},
        "total_test_cases": 0,
        "average_tests_per_file": 0
    }
    
    total_test_cases = 0
    
    for test_file, test_code in generated_tests.items():
        # Count test cases (simple heuristic)
        test_cases = len(re.findall(r'def test_|it\(|test\(', test_code))
        total_test_cases += test_cases
        
        # Identify framework
        if 'pytest' in test_code or '@pytest' in test_code:
            structure["framework_distribution"]["pytest"] = structure["framework_distribution"].get("pytest", 0) + 1
        elif 'unittest' in test_code or 'TestCase' in test_code:
            structure["framework_distribution"]["unittest"] = structure["framework_distribution"].get("unittest", 0) + 1
        elif 'describe(' in test_code or 'it(' in test_code:
            structure["framework_distribution"]["jest/mocha"] = structure["framework_distribution"].get("jest/mocha", 0) + 1
    
    structure["total_test_cases"] = total_test_cases
    structure["average_tests_per_file"] = total_test_cases / len(generated_tests) if generated_tests else 0
    
    return structure


def _create_coverage_mapping(source_files: Dict[str, str], generated_tests: Dict[str, str]) -> Dict[str, List[str]]:
    """Create mapping of source files to their test files."""
    coverage_mapping = {}
    
    for source_file in source_files.keys():
        source_stem = Path(source_file).stem
        related_tests = []
        
        for test_file in generated_tests.keys():
            if source_stem in test_file or source_file.replace('.py', '') in test_file:
                related_tests.append(test_file)
        
        coverage_mapping[source_file] = related_tests
    
    return coverage_mapping


def _count_test_cases(generated_tests: Dict[str, str]) -> int:
    """Count total number of test cases across all test files."""
    total_count = 0
    
    for test_code in generated_tests.values():
        # Count different test patterns
        total_count += len(re.findall(r'def test_', test_code))  # Python
        total_count += len(re.findall(r'it\s*\(', test_code))   # JavaScript
        total_count += len(re.findall(r'@Test', test_code))     # Java
    
    return total_count


def generate_cli_tests(source_files: Dict[str, str], context: TestGenerationContext) -> Dict[str, str]:
    """Generate CLI-specific test scripts."""
    cli_tests = {}
    
    main_files = [f for f in source_files.keys() if 'main' in f.lower() or 'cli' in f.lower()]
    
    if not main_files:
        main_files = list(source_files.keys())[:1]  # Use first file as fallback
    
    for source_file in main_files:
        test_file_path = f"{context.output_directory}/test_cli_{Path(source_file).stem}.py"
        
        cli_test_code = f'''#!/usr/bin/env python3
"""
CLI test script for {source_file}

This script tests the command-line interface functionality.
"""

import subprocess
import sys
import os
import tempfile
from pathlib import Path

# Test configuration
SCRIPT_PATH = Path(__file__).parent.parent / "{source_file}"
TEST_TIMEOUT = 30  # seconds


def run_cli_command(args, input_data=None, expect_success=True):
    """
    Run CLI command and return result.
    
    Args:
        args: Command line arguments
        input_data: Input data to pass to stdin
        expect_success: Whether command should succeed
        
    Returns:
        Tuple of (stdout, stderr, returncode)
    """
    cmd = [sys.executable, str(SCRIPT_PATH)] + args
    
    try:
        result = subprocess.run(
            cmd,
            input=input_data,
            capture_output=True,
            text=True,
            timeout=TEST_TIMEOUT
        )
        
        if expect_success and result.returncode != 0:
            print(f"Command failed: {{' '.join(cmd)}}")
            print(f"STDOUT: {{result.stdout}}")
            print(f"STDERR: {{result.stderr}}")
            
        return result.stdout, result.stderr, result.returncode
        
    except subprocess.TimeoutExpired:
        print(f"Command timed out: {{' '.join(cmd)}}")
        return "", "Command timed out", -1


def test_help_option():
    """Test --help option."""
    print("Testing --help option...")
    stdout, stderr, returncode = run_cli_command(["--help"])
    
    assert returncode == 0, "Help command should succeed"
    assert len(stdout) > 0, "Help should produce output"
    print("✓ Help option works")


def test_version_option():
    """Test --version option if available."""
    print("Testing --version option...")
    stdout, stderr, returncode = run_cli_command(["--version"], expect_success=False)
    
    # Version might not be implemented, so just check it doesn't crash badly
    if returncode == 0:
        print("✓ Version option works")
    else:
        print("- Version option not available or failed")


def test_basic_execution():
    """Test basic execution without arguments."""
    print("Testing basic execution...")
    stdout, stderr, returncode = run_cli_command([])
    
    # Basic execution might succeed or fail depending on requirements
    print(f"Basic execution returned: {{returncode}}")
    print("✓ Basic execution test completed")


def test_invalid_arguments():
    """Test handling of invalid arguments."""
    print("Testing invalid arguments...")
    stdout, stderr, returncode = run_cli_command(["--invalid-option"], expect_success=False)
    
    assert returncode != 0, "Invalid arguments should fail"
    print("✓ Invalid arguments handled correctly")


def test_with_sample_input():
    """Test with sample input data."""
    print("Testing with sample input...")
    
    # Create temporary test file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
        temp_file.write("Sample test data\\n")
        temp_file.write("Line 2\\n")
        temp_path = temp_file.name
    
    try:
        # Test with file input if the script supports it
        stdout, stderr, returncode = run_cli_command([temp_path], expect_success=False)
        print(f"Sample input test returned: {{returncode}}")
        print("✓ Sample input test completed")
        
    finally:
        # Clean up temporary file
        os.unlink(temp_path)


def main():
    """Run all CLI tests."""
    print(f"Running CLI tests for {{SCRIPT_PATH}}")
    print("=" * 50)
    
    tests = [
        test_help_option,
        test_version_option,
        test_basic_execution,
        test_invalid_arguments,
        test_with_sample_input
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            print(f"✗ {{test_func.__name__}} failed: {{e}}")
            failed += 1
        print()
    
    print("=" * 50)
    print(f"Tests completed: {{passed}} passed, {{failed}} failed")
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
'''
        
        cli_tests[test_file_path] = cli_test_code
    
    return cli_tests