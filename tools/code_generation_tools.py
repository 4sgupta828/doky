# tools/creation/code_generation_tools.py
import json
import logging
from typing import Dict, Any, List, Optional
from enum import Enum
from dataclasses import dataclass
from pathlib import Path

# Get a logger instance for this module
logger = logging.getLogger(__name__)


class CodeQuality(Enum):
    """Defines different code quality levels for speed vs quality trade-offs."""
    FAST = "fast"
    DECENT = "decent" 
    PRODUCTION = "production"


class CodeLanguage(Enum):
    """Supported programming languages."""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TYPESCRIPT = "typescript"
    JAVA = "java"
    GO = "go"
    RUST = "rust"


@dataclass
class CodeGenerationContext:
    """Context for code generation operations."""
    goal: str
    technical_spec: str
    language: CodeLanguage = CodeLanguage.PYTHON
    quality_level: CodeQuality = CodeQuality.DECENT
    files_to_generate: List[str] = None
    existing_code: Dict[str, str] = None
    dependencies: List[str] = None
    frameworks: List[str] = None


@dataclass
class CodeGenerationResult:
    """Result of code generation operation."""
    success: bool
    generated_files: Dict[str, str]
    file_structure: Dict[str, Any]
    dependencies: List[str]
    error_details: Optional[str] = None
    quality_metrics: Dict[str, Any] = None


def generate_code(context: CodeGenerationContext, llm_client=None) -> CodeGenerationResult:
    """
    Generate code based on specifications and context.
    
    Args:
        context: Code generation context with specifications
        llm_client: LLM client for code generation (optional for fallback)
        
    Returns:
        CodeGenerationResult with generated code and metadata
    """
    logger.info(f"Generating {context.language.value} code: {context.goal}")
    
    try:
        # Generate code files
        if llm_client:
            generated_files = _generate_code_with_llm(context, llm_client)
        else:
            generated_files = _generate_code_fallback(context)
        
        # Analyze generated code
        file_structure = _analyze_file_structure(generated_files)
        dependencies = _extract_dependencies(generated_files, context.language)
        quality_metrics = _calculate_quality_metrics(generated_files, context.quality_level)
        
        return CodeGenerationResult(
            success=True,
            generated_files=generated_files,
            file_structure=file_structure,
            dependencies=dependencies,
            quality_metrics=quality_metrics
        )
        
    except Exception as e:
        logger.error(f"Code generation failed: {e}")
        return CodeGenerationResult(
            success=False,
            generated_files={},
            file_structure={},
            dependencies=[],
            error_details=str(e)
        )


def _generate_code_with_llm(context: CodeGenerationContext, llm_client) -> Dict[str, str]:
    """Generate code using LLM client."""
    prompt = build_code_generation_prompt(context)
    
    try:
        response = llm_client.invoke(prompt)
        generated_code = json.loads(response)
        
        if not isinstance(generated_code, dict):
            raise ValueError("LLM response is not a valid code dictionary")
            
        return generated_code
        
    except (json.JSONDecodeError, ValueError) as e:
        logger.error(f"Failed to parse LLM response: {e}")
        raise


def _generate_code_fallback(context: CodeGenerationContext) -> Dict[str, str]:
    """Generate basic code structure without LLM."""
    generated_files = {}
    
    # Determine files to generate
    files_to_generate = context.files_to_generate or _infer_files_from_spec(context)
    
    for file_path in files_to_generate:
        # Generate basic code structure
        if context.language == CodeLanguage.PYTHON:
            content = _generate_python_fallback(file_path, context)
        elif context.language == CodeLanguage.JAVASCRIPT:
            content = _generate_javascript_fallback(file_path, context)
        else:
            content = _generate_generic_fallback(file_path, context)
            
        generated_files[file_path] = content
    
    return generated_files


def _generate_python_fallback(file_path: str, context: CodeGenerationContext) -> str:
    """Generate basic Python code structure."""
    file_name = Path(file_path).stem
    
    # Basic Python templates
    if file_path.endswith('main.py'):
        return f'''#!/usr/bin/env python3
"""
{context.goal}

Generated code based on: {context.technical_spec[:100]}...
"""

import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for the application."""
    logger.info("Starting application")
    
    # TODO: Implement main functionality
    print("Hello, World!")
    
    logger.info("Application completed")


if __name__ == "__main__":
    main()
'''
    
    elif 'test' in file_path.lower():
        return f'''"""
Test module for {context.goal}
"""

import unittest
import logging

# Configure test logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


class Test{file_name.title().replace('_', '')}(unittest.TestCase):
    """Test cases for {file_name} functionality."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        logger.debug("Setting up test fixtures")
        
    def tearDown(self):
        """Clean up after each test method."""
        logger.debug("Cleaning up test fixtures")
        
    def test_basic_functionality(self):
        """Test basic functionality."""
        # TODO: Implement actual test cases
        self.assertTrue(True, "Basic test case")
        
    def test_error_handling(self):
        """Test error handling."""
        # TODO: Implement error handling tests
        self.assertIsNotNone(None, "Error handling test")


if __name__ == "__main__":
    unittest.main()
'''
    
    else:
        return f'''"""
{file_name.title().replace('_', ' ')} module

{context.technical_spec[:200]}...
"""

import logging
from typing import Dict, Any, List, Optional

# Configure module logging
logger = logging.getLogger(__name__)


class {file_name.title().replace('_', '')}:
    """
    Main class for {file_name} functionality.
    
    This class implements the core functionality for {context.goal}.
    """
    
    def __init__(self):
        """Initialize the {file_name} instance."""
        logger.debug(f"Initializing {{self.__class__.__name__}}")
        
    def process(self, data: Any) -> Any:
        """
        Process the input data.
        
        Args:
            data: Input data to process
            
        Returns:
            Processed data
        """
        logger.info("Processing data")
        
        # TODO: Implement actual processing logic
        return data
        
    def validate(self, data: Any) -> bool:
        """
        Validate the input data.
        
        Args:
            data: Data to validate
            
        Returns:
            True if data is valid, False otherwise
        """
        logger.debug("Validating data")
        
        # TODO: Implement validation logic
        return data is not None


def helper_function(param: str) -> str:
    """
    Helper function for common operations.
    
    Args:
        param: Input parameter
        
    Returns:
        Processed parameter
    """
    logger.debug(f"Processing parameter: {{param}}")
    
    # TODO: Implement helper functionality
    return param.upper() if param else ""
'''


def _generate_javascript_fallback(file_path: str, context: CodeGenerationContext) -> str:
    """Generate basic JavaScript code structure."""
    file_name = Path(file_path).stem
    
    if 'test' in file_path.lower():
        return f'''/**
 * Test module for {context.goal}
 */

const {{ expect }} = require('chai');

describe('{file_name}', function() {{
    
    beforeEach(function() {{
        // Set up test fixtures
        console.log('Setting up test fixtures');
    }});
    
    afterEach(function() {{
        // Clean up after tests
        console.log('Cleaning up test fixtures');
    }});
    
    it('should handle basic functionality', function() {{
        // TODO: Implement actual test cases
        expect(true).to.be.true;
    }});
    
    it('should handle error cases', function() {{
        // TODO: Implement error handling tests
        expect(false).to.be.false;
    }});
    
}});
'''
    
    else:
        return f'''/**
 * {file_name.title().replace('_', ' ')} module
 * 
 * {context.technical_spec[:200]}...
 */

/**
 * Main class for {file_name} functionality
 */
class {file_name.title().replace('_', '')} {{
    
    /**
     * Initialize the {file_name} instance
     */
    constructor() {{
        console.log(`Initializing ${{this.constructor.name}}`);
    }}
    
    /**
     * Process the input data
     * @param {{any}} data - Input data to process
     * @returns {{any}} Processed data
     */
    process(data) {{
        console.log('Processing data');
        
        // TODO: Implement actual processing logic
        return data;
    }}
    
    /**
     * Validate the input data
     * @param {{any}} data - Data to validate
     * @returns {{boolean}} True if data is valid, false otherwise
     */
    validate(data) {{
        console.log('Validating data');
        
        // TODO: Implement validation logic
        return data !== null && data !== undefined;
    }}
}}

/**
 * Helper function for common operations
 * @param {{string}} param - Input parameter
 * @returns {{string}} Processed parameter
 */
function helperFunction(param) {{
    console.log(`Processing parameter: ${{param}}`);
    
    // TODO: Implement helper functionality
    return param ? param.toUpperCase() : '';
}}

module.exports = {{
    {file_name.title().replace('_', '')},
    helperFunction
}};
'''


def _generate_generic_fallback(file_path: str, context: CodeGenerationContext) -> str:
    """Generate generic code structure."""
    return f'''/*
 * {Path(file_path).stem} - {context.goal}
 * 
 * Generated code based on: {context.technical_spec[:200]}...
 */

// TODO: Implement functionality based on technical specification
// Language: {context.language.value}
// Quality Level: {context.quality_level.value}

function main() {{
    // Main entry point
    console.log("Hello, World!");
}}

main();
'''


def _infer_files_from_spec(context: CodeGenerationContext) -> List[str]:
    """Infer files to generate from specification."""
    spec_lower = context.technical_spec.lower()
    goal_lower = context.goal.lower()
    
    files = []
    
    # Main application file
    if context.language == CodeLanguage.PYTHON:
        files.append("main.py")
        
        # Add common Python files based on keywords
        if any(keyword in spec_lower for keyword in ['api', 'web', 'flask', 'fastapi']):
            files.append("app.py")
        if any(keyword in spec_lower for keyword in ['util', 'helper', 'common']):
            files.append("utils.py")
        if any(keyword in spec_lower for keyword in ['config', 'setting']):
            files.append("config.py")
        if any(keyword in spec_lower for keyword in ['model', 'data', 'class']):
            files.append("models.py")
    
    elif context.language == CodeLanguage.JAVASCRIPT:
        files.append("index.js")
        
        if any(keyword in spec_lower for keyword in ['util', 'helper', 'common']):
            files.append("utils.js")
        if any(keyword in spec_lower for keyword in ['config', 'setting']):
            files.append("config.js")
            
    else:
        # Generic fallback
        files.append("main.py")
    
    # Add test files if testing is mentioned
    if any(keyword in spec_lower for keyword in ['test', 'testing', 'unit test']):
        if context.language == CodeLanguage.PYTHON:
            files.append("test_main.py")
        else:
            files.append("test.js")
    
    return files


def build_code_generation_prompt(context: CodeGenerationContext) -> str:
    """Build prompt for LLM code generation."""
    quality_config = get_quality_instructions(context.quality_level)
    
    # Handle existing code
    existing_code_str = ""
    if context.existing_code:
        existing_code_str = "\n".join(
            f"--- File: {path} ---\n```{context.language.value}\n{content}\n```"
            for path, content in context.existing_code.items()
        )
    
    # Handle files to generate
    if context.files_to_generate:
        files_section = f"**Files to Generate/Modify:**\n- {chr(10).join(context.files_to_generate)}"
    else:
        files_section = "**Files to Generate/Modify:**\nYou must determine the appropriate file paths based on the specification."
    
    # Handle dependencies and frameworks
    tech_context = ""
    if context.dependencies:
        tech_context += f"**Required Dependencies:** {', '.join(context.dependencies)}\n"
    if context.frameworks:
        tech_context += f"**Frameworks to Use:** {', '.join(context.frameworks)}\n"
    
    quality_instructions = "\n        ".join([f"- {inst}" for inst in quality_config["instructions"]])
    
    # Use original CoderAgent prompt format - preserved exactly
    return f"""
        You are an expert software developer. Your task is to write {quality_config["description"]} based on the provided technical specification.
        
        **Code Quality Level: {context.quality_level.value.upper()}**
        
        **Technical Specification / Goal:**
        ---
        {context.technical_spec}
        ---
        {files_section}

        **Existing Code for Context (if any):**
        ---
        {existing_code_str if context.existing_code else "No existing code provided. You are writing these files from scratch."}
        ---

        **Code Reuse Guidelines:**
        If existing code is provided above, prioritize reusing it where appropriate:
        - Import and build upon existing functionality rather than duplicating it
        - Extend or compose with existing code when it fits your requirements
        - Only reimplement if the existing approach doesn't meet the new requirements

        **Quality-Specific Instructions:**
        {quality_instructions}
        
        **Final Output Requirement:**
        Your output MUST be a single, valid JSON object where keys are the file paths and values are the complete code content as a string.

        **JSON Output Format Example:**
        {{
            "src/main.py": "def main():\\n    print('Hello')",
            "src/utils.py": "def helper():\\n    pass"
        }}

        Now, generate the code files.
        """


def get_quality_instructions(quality: CodeQuality) -> Dict[str, Any]:
    """Get quality-specific instructions for code generation."""
    quality_configs = {
        CodeQuality.FAST: {
            "description": "working code quickly",
            "instructions": [
                "Focus on getting working code fast - don't over-engineer.",
                "Minimal comments and basic error handling.",
                "Simple, straightforward implementations."
            ]
        },
        CodeQuality.DECENT: {
            "description": "clean, well-structured code",
            "instructions": [
                "Write clean, readable code with reasonable comments.",
                "Include basic error handling and validation.",
                "Use proper naming conventions and structure.",
                "Add docstrings for functions and classes."
            ]
        },
        CodeQuality.PRODUCTION: {
            "description": "production-quality, enterprise-ready code",
            "instructions": [
                "Write robust, production-ready code with comprehensive error handling.",
                "Include detailed docstrings, type hints, and extensive comments.",
                "Consider scalability, security, and maintainability.",
                "Add logging, input validation, and proper exception handling.",
                "Follow industry best practices and design patterns."
            ]
        }
    }
    return quality_configs.get(quality, quality_configs[CodeQuality.DECENT])


def get_file_extension(language: CodeLanguage) -> str:
    """Get appropriate file extension for the language."""
    extensions = {
        CodeLanguage.PYTHON: "py",
        CodeLanguage.JAVASCRIPT: "js", 
        CodeLanguage.TYPESCRIPT: "ts",
        CodeLanguage.JAVA: "java",
        CodeLanguage.GO: "go",
        CodeLanguage.RUST: "rs"
    }
    return extensions.get(language, "txt")


def _analyze_file_structure(generated_files: Dict[str, str]) -> Dict[str, Any]:
    """Analyze the structure of generated files."""
    structure = {
        "total_files": len(generated_files),
        "file_types": {},
        "total_lines": 0,
        "directories": set()
    }
    
    for file_path, content in generated_files.items():
        # File type analysis
        extension = Path(file_path).suffix
        structure["file_types"][extension] = structure["file_types"].get(extension, 0) + 1
        
        # Line count
        lines = len(content.split('\n'))
        structure["total_lines"] += lines
        
        # Directory analysis
        directory = str(Path(file_path).parent)
        if directory != '.':
            structure["directories"].add(directory)
    
    structure["directories"] = list(structure["directories"])
    
    return structure


def _extract_dependencies(generated_files: Dict[str, str], language: CodeLanguage) -> List[str]:
    """Extract dependencies from generated code."""
    dependencies = set()
    
    for file_path, content in generated_files.items():
        if language == CodeLanguage.PYTHON:
            dependencies.update(_extract_python_dependencies(content))
        elif language == CodeLanguage.JAVASCRIPT:
            dependencies.update(_extract_javascript_dependencies(content))
        # Add more languages as needed
    
    return sorted(list(dependencies))


def _extract_python_dependencies(content: str) -> List[str]:
    """Extract Python import dependencies."""
    import re
    
    dependencies = []
    lines = content.split('\n')
    
    for line in lines:
        line = line.strip()
        
        # Standard import patterns
        if line.startswith('import '):
            match = re.match(r'import\s+([a-zA-Z_][a-zA-Z0-9_]*)', line)
            if match:
                dependencies.append(match.group(1))
        elif line.startswith('from '):
            match = re.match(r'from\s+([a-zA-Z_][a-zA-Z0-9_]*)', line)
            if match:
                dependencies.append(match.group(1))
    
    # Filter out standard library modules
    stdlib_modules = {
        'os', 'sys', 'json', 'logging', 'typing', 'pathlib', 're', 'datetime',
        'unittest', 'collections', 'itertools', 'functools', 'copy'
    }
    
    return [dep for dep in dependencies if dep not in stdlib_modules]


def _extract_javascript_dependencies(content: str) -> List[str]:
    """Extract JavaScript require/import dependencies."""
    import re
    
    dependencies = []
    lines = content.split('\n')
    
    for line in lines:
        line = line.strip()
        
        # CommonJS require patterns
        require_match = re.search(r"require\(['\"]([^'\"]+)['\"]\)", line)
        if require_match:
            dependencies.append(require_match.group(1))
            
        # ES6 import patterns
        import_match = re.search(r"from ['\"]([^'\"]+)['\"]", line)
        if import_match:
            dependencies.append(import_match.group(1))
    
    # Filter out built-in modules
    builtin_modules = {'fs', 'path', 'http', 'https', 'url', 'crypto', 'util'}
    
    return [dep for dep in dependencies if dep not in builtin_modules]


def _calculate_quality_metrics(generated_files: Dict[str, str], quality_level: CodeQuality) -> Dict[str, Any]:
    """Calculate quality metrics for generated code."""
    total_lines = 0
    total_comments = 0
    total_functions = 0
    
    for file_path, content in generated_files.items():
        lines = content.split('\n')
        total_lines += len(lines)
        
        # Count comments (simple heuristic)
        for line in lines:
            stripped = line.strip()
            if stripped.startswith('#') or stripped.startswith('//') or stripped.startswith('/*'):
                total_comments += 1
            if 'def ' in line or 'function ' in line:
                total_functions += 1
    
    comment_ratio = (total_comments / total_lines) * 100 if total_lines > 0 else 0
    avg_lines_per_file = total_lines / len(generated_files) if generated_files else 0
    
    return {
        "total_lines": total_lines,
        "total_comments": total_comments,
        "comment_ratio_percent": round(comment_ratio, 2),
        "total_functions": total_functions,
        "average_lines_per_file": round(avg_lines_per_file, 2),
        "quality_level": quality_level.value,
        "estimated_completeness": "high" if quality_level == CodeQuality.PRODUCTION else "medium"
    }