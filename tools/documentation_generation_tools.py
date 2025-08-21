# tools/creation/documentation_generation_tools.py
import logging
import re
from typing import Dict, Any, List, Optional
from enum import Enum
from dataclasses import dataclass
from pathlib import Path

# Get a logger instance for this module
logger = logging.getLogger(__name__)


class DocumentationType(Enum):
    """Types of documentation that can be generated."""
    README = "readme"
    API_DOCS = "api_docs"
    USER_GUIDE = "user_guide"
    TECHNICAL_SPEC = "technical_spec"
    CODE_COMMENTS = "code_comments"
    CHANGELOG = "changelog"
    ARCHITECTURE = "architecture"


class DocumentationFormat(Enum):
    """Output formats for documentation."""
    MARKDOWN = "markdown"
    HTML = "html"
    RST = "rst"
    PLAIN_TEXT = "plain_text"


class TemplateStyle(Enum):
    """Documentation template styles."""
    STANDARD = "standard"
    MINIMAL = "minimal"
    COMPREHENSIVE = "comprehensive"
    GITHUB = "github"
    SPHINX = "sphinx"


@dataclass
class DocumentationContext:
    """Context for documentation generation operations."""
    goal: str
    documentation_type: DocumentationType = DocumentationType.README
    output_format: DocumentationFormat = DocumentationFormat.MARKDOWN
    template_style: TemplateStyle = TemplateStyle.STANDARD
    project_info: Dict[str, Any] = None
    code_files: Dict[str, str] = None
    include_examples: bool = True
    working_directory: str = "."


@dataclass
class DocumentationResult:
    """Result of documentation generation operation."""
    success: bool
    generated_docs: Dict[str, str]
    documentation_structure: Dict[str, Any]
    quality_metrics: Dict[str, Any]
    error_details: Optional[str] = None


def generate_documentation(context: DocumentationContext, llm_client=None) -> DocumentationResult:
    """
    Generate comprehensive documentation based on context.
    
    Args:
        context: Documentation generation context
        llm_client: LLM client for intelligent documentation generation (optional)
        
    Returns:
        DocumentationResult with generated documentation and metadata
    """
    logger.info(f"Generating {context.documentation_type.value} documentation")
    
    try:
        # Generate documentation
        if llm_client:
            generated_docs = _generate_docs_with_llm(context, llm_client)
        else:
            generated_docs = _generate_docs_fallback(context)
        
        # Analyze documentation structure
        doc_structure = _analyze_documentation_structure(generated_docs)
        quality_metrics = _calculate_documentation_quality(generated_docs, context)
        
        return DocumentationResult(
            success=True,
            generated_docs=generated_docs,
            documentation_structure=doc_structure,
            quality_metrics=quality_metrics
        )
        
    except Exception as e:
        logger.error(f"Documentation generation failed: {e}")
        return DocumentationResult(
            success=False,
            generated_docs={},
            documentation_structure={},
            quality_metrics={},
            error_details=str(e)
        )


def _generate_docs_with_llm(context: DocumentationContext, llm_client) -> Dict[str, str]:
    """Generate documentation using LLM client."""
    prompt = _build_documentation_prompt(context)
    
    try:
        response = llm_client.invoke(prompt)
        return _parse_documentation_response(response, context)
        
    except Exception as e:
        logger.warning(f"LLM documentation generation failed: {e}")
        return _generate_docs_fallback(context)


def _generate_docs_fallback(context: DocumentationContext) -> Dict[str, str]:
    """Generate documentation using template-based approach."""
    
    if context.documentation_type == DocumentationType.README:
        return _generate_readme_fallback(context)
    elif context.documentation_type == DocumentationType.API_DOCS:
        return _generate_api_docs_fallback(context)
    elif context.documentation_type == DocumentationType.USER_GUIDE:
        return _generate_user_guide_fallback(context)
    elif context.documentation_type == DocumentationType.TECHNICAL_SPEC:
        return _generate_technical_spec_fallback(context)
    elif context.documentation_type == DocumentationType.CODE_COMMENTS:
        return _generate_code_comments_fallback(context)
    else:
        return _generate_generic_doc_fallback(context)


def _generate_readme_fallback(context: DocumentationContext) -> Dict[str, str]:
    """Generate README documentation."""
    project_info = context.project_info or {}
    project_name = project_info.get("name", "Project")
    description = project_info.get("description", context.goal)
    
    if context.template_style == TemplateStyle.MINIMAL:
        readme_content = f'''# {project_name}

{description}

## Usage

```python
# TODO: Add usage examples
```

## Installation

```bash
# TODO: Add installation instructions
```
'''
    
    elif context.template_style == TemplateStyle.COMPREHENSIVE:
        readme_content = f'''# {project_name}

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.7+-blue.svg)](https://python.org)

{description}

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## Features

- Feature 1: High-level feature description
- Feature 2: Another important feature
- Feature 3: Additional functionality

## Installation

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Install from PyPI

```bash
pip install {project_name.lower().replace(" ", "-")}
```

### Install from Source

```bash
git clone https://github.com/username/{project_name.lower().replace(" ", "-")}.git
cd {project_name.lower().replace(" ", "-")}
pip install -e .
```

## Quick Start

```python
import {project_name.lower().replace(" ", "_").replace("-", "_")}

# Basic usage example
result = {project_name.lower().replace(" ", "_").replace("-", "_")}.main()
print(result)
```

## Usage

### Basic Usage

```python
# TODO: Add basic usage examples
```

### Advanced Usage

```python
# TODO: Add advanced usage examples
```

## API Reference

### Main Functions

#### `main()`

Main entry point for the application.

**Returns:** Result of the main operation.

#### `helper_function(param)`

Helper function for common operations.

**Parameters:**
- `param` (str): Input parameter

**Returns:** Processed result.

## Examples

### Example 1: Basic Operation

```python
# TODO: Add example code
```

### Example 2: Advanced Operation

```python
# TODO: Add advanced example
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Thanks to all contributors
- Inspired by [relevant projects or libraries]
'''
    
    else:  # STANDARD
        readme_content = f'''# {project_name}

{description}

## Installation

```bash
pip install {project_name.lower().replace(" ", "-")}
```

## Usage

```python
import {project_name.lower().replace(" ", "_")}

# Basic usage
result = {project_name.lower().replace(" ", "_")}.main()
print(result)
```

## Features

- Core functionality implementation
- Easy-to-use API
- Comprehensive error handling

## Documentation

For detailed documentation, please refer to the [docs](docs/) directory.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.
'''
    
    return {"README.md": readme_content}


def _generate_api_docs_fallback(context: DocumentationContext) -> Dict[str, str]:
    """Generate API documentation."""
    code_files = context.code_files or {}
    
    api_docs = {}
    
    # Generate API docs for each code file
    for file_path, code_content in code_files.items():
        if file_path.endswith('.py'):
            api_doc = _generate_python_api_doc(file_path, code_content)
            doc_filename = f"api_{Path(file_path).stem}.md"
            api_docs[doc_filename] = api_doc
    
    # Generate main API index
    api_docs["api_index.md"] = _generate_api_index(api_docs.keys())
    
    return api_docs


def _generate_python_api_doc(file_path: str, code_content: str) -> str:
    """Generate API documentation for a Python file."""
    functions = _extract_function_signatures(code_content)
    classes = _extract_class_signatures(code_content)
    
    doc_content = f'''# API Reference: {Path(file_path).stem}

## Overview

This module provides functionality for {Path(file_path).stem} operations.

## Functions

'''
    
    for func_name, func_sig, func_doc in functions:
        doc_content += f'''### `{func_name}{func_sig}`

{func_doc or "Function description not available."}

'''
    
    if classes:
        doc_content += '''## Classes

'''
        for class_name, class_doc, methods in classes:
            doc_content += f'''### `{class_name}`

{class_doc or "Class description not available."}

#### Methods

'''
            for method_name, method_sig, method_doc in methods:
                doc_content += f'''##### `{method_name}{method_sig}`

{method_doc or "Method description not available."}

'''
    
    return doc_content


def _generate_user_guide_fallback(context: DocumentationContext) -> Dict[str, str]:
    """Generate user guide documentation."""
    project_info = context.project_info or {}
    project_name = project_info.get("name", "Project")
    
    user_guide = f'''# {project_name} User Guide

## Introduction

Welcome to the {project_name} user guide. This guide will help you get started and make the most of {project_name}.

## Getting Started

### System Requirements

- Python 3.7 or higher
- Operating system: Windows, macOS, or Linux

### Installation

Follow these steps to install {project_name}:

1. Install Python if you haven't already
2. Open a terminal or command prompt
3. Run the installation command:

```bash
pip install {project_name.lower().replace(" ", "-")}
```

### First Steps

After installation, you can start using {project_name}:

```python
import {project_name.lower().replace(" ", "_")}

# Your first program
{project_name.lower().replace(" ", "_")}.main()
```

## Basic Usage

### Core Concepts

- **Concept 1**: Explanation of the first important concept
- **Concept 2**: Explanation of the second important concept
- **Concept 3**: Explanation of additional concepts

### Common Tasks

#### Task 1: Basic Operation

```python
# Code example for basic operation
```

#### Task 2: Advanced Operation

```python
# Code example for advanced operation
```

## Advanced Usage

### Configuration

You can configure {project_name} using:

- Configuration files
- Environment variables
- Command-line arguments

### Best Practices

1. Always validate input data
2. Handle errors gracefully
3. Use appropriate logging
4. Follow coding standards

## Troubleshooting

### Common Issues

**Issue 1**: Problem description
- Solution: Step-by-step solution

**Issue 2**: Another problem description
- Solution: How to resolve it

### Getting Help

If you need additional help:

- Check the [FAQ](faq.md)
- Review the [API documentation](api_index.md)
- Submit an issue on GitHub

## Examples

### Example 1: Complete Workflow

```python
# Complete example showing a typical workflow
```

### Example 2: Error Handling

```python
# Example showing proper error handling
```

## Appendix

### Glossary

- **Term 1**: Definition of important term
- **Term 2**: Definition of another term

### Additional Resources

- [Official Documentation](docs/)
- [Community Forum](https://community.example.com)
- [GitHub Repository](https://github.com/username/project)
'''
    
    return {"user_guide.md": user_guide}


def _generate_technical_spec_fallback(context: DocumentationContext) -> Dict[str, str]:
    """Generate technical specification documentation."""
    project_info = context.project_info or {}
    project_name = project_info.get("name", "Project")
    
    tech_spec = f'''# Technical Specification: {project_name}

## Document Information

- **Project**: {project_name}
- **Version**: 1.0
- **Date**: {{current_date}}
- **Status**: Draft

## 1. Overview

### 1.1 Purpose

This document specifies the technical requirements and design for {project_name}.

### 1.2 Scope

{context.goal}

### 1.3 Definitions and Acronyms

- **API**: Application Programming Interface
- **CLI**: Command Line Interface
- **JSON**: JavaScript Object Notation

## 2. System Architecture

### 2.1 High-Level Architecture

```
[User Interface] -> [Application Logic] -> [Data Layer]
```

### 2.2 Component Overview

- **Component 1**: Core processing logic
- **Component 2**: Data management
- **Component 3**: User interface

### 2.3 Technology Stack

- **Language**: Python 3.7+
- **Framework**: [To be determined]
- **Database**: [If applicable]
- **Dependencies**: [List key dependencies]

## 3. Functional Requirements

### 3.1 Core Features

- **FR-001**: Primary functionality requirement
- **FR-002**: Secondary functionality requirement
- **FR-003**: Additional feature requirement

### 3.2 User Interface Requirements

- **UI-001**: Interface usability requirement
- **UI-002**: Interface accessibility requirement

### 3.3 API Requirements

- **API-001**: REST API endpoints
- **API-002**: Authentication mechanism
- **API-003**: Response format specifications

## 4. Non-Functional Requirements

### 4.1 Performance

- **Response time**: < 2 seconds for standard operations
- **Throughput**: Handle 1000 concurrent users
- **Scalability**: Horizontal scaling capability

### 4.2 Security

- **Authentication**: Secure user authentication
- **Authorization**: Role-based access control
- **Data Protection**: Encryption of sensitive data

### 4.3 Reliability

- **Availability**: 99.9% uptime
- **Error Handling**: Graceful error recovery
- **Logging**: Comprehensive audit trail

## 5. Design Constraints

### 5.1 Technical Constraints

- Must be compatible with Python 3.7+
- Should work on Windows, macOS, and Linux
- Memory usage should not exceed 512MB

### 5.2 Business Constraints

- Development timeline: [To be defined]
- Budget considerations: [To be defined]
- Compliance requirements: [If applicable]

## 6. Implementation Plan

### 6.1 Development Phases

1. **Phase 1**: Core functionality development
2. **Phase 2**: User interface implementation  
3. **Phase 3**: Testing and optimization
4. **Phase 4**: Deployment and documentation

### 6.2 Testing Strategy

- Unit testing for all components
- Integration testing for system components
- Performance testing under load
- Security testing and vulnerability assessment

## 7. Appendices

### 7.1 References

- [Reference 1]
- [Reference 2]

### 7.2 Change History

| Version | Date | Author | Description |
|---------|------|--------|-------------|
| 1.0 | TBD | Auto-generated | Initial specification |
'''
    
    return {"technical_specification.md": tech_spec}


def _generate_code_comments_fallback(context: DocumentationContext) -> Dict[str, str]:
    """Generate code comments and docstrings."""
    code_files = context.code_files or {}
    commented_files = {}
    
    for file_path, code_content in code_files.items():
        if file_path.endswith('.py'):
            commented_code = _add_python_comments(code_content)
            commented_files[f"commented_{file_path}"] = commented_code
    
    return commented_files


def _generate_generic_doc_fallback(context: DocumentationContext) -> Dict[str, str]:
    """Generate generic documentation."""
    return {
        f"{context.documentation_type.value}.md": f'''# {context.documentation_type.value.title().replace("_", " ")}

This document was automatically generated for: {context.goal}

## Content

Documentation content goes here.

## Additional Information

Please update this document with relevant information for your project.
'''
    }


def _extract_function_signatures(code_content: str) -> List[tuple]:
    """Extract function signatures and docstrings from Python code."""
    functions = []
    lines = code_content.split('\n')
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('def ') and not line.startswith('def _'):
            # Extract function signature
            func_match = re.match(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(\([^)]*\))', line)
            if func_match:
                func_name = func_match.group(1)
                func_params = func_match.group(2)
                
                # Look for docstring
                doc_string = ""
                j = i + 1
                while j < len(lines) and lines[j].strip() == "":
                    j += 1
                
                if j < len(lines) and '"""' in lines[j]:
                    # Extract docstring
                    doc_lines = []
                    if lines[j].count('"""') == 2:  # Single line docstring
                        doc_string = lines[j].strip().replace('"""', '')
                    else:  # Multi-line docstring
                        j += 1
                        while j < len(lines) and '"""' not in lines[j]:
                            doc_lines.append(lines[j].strip())
                            j += 1
                        doc_string = '\n'.join(doc_lines)
                
                functions.append((func_name, func_params, doc_string))
        i += 1
    
    return functions


def _extract_class_signatures(code_content: str) -> List[tuple]:
    """Extract class signatures and methods from Python code."""
    classes = []
    lines = code_content.split('\n')
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('class '):
            # Extract class name
            class_match = re.match(r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)', line)
            if class_match:
                class_name = class_match.group(1)
                
                # Look for class docstring
                doc_string = ""
                j = i + 1
                while j < len(lines) and lines[j].strip() == "":
                    j += 1
                
                if j < len(lines) and '"""' in lines[j]:
                    # Extract docstring (simplified)
                    doc_string = "Class docstring available"
                
                # Extract methods (simplified)
                methods = []
                while j < len(lines):
                    method_line = lines[j].strip()
                    if method_line.startswith('def '):
                        method_match = re.match(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*(\([^)]*\))', method_line)
                        if method_match:
                            method_name = method_match.group(1)
                            method_params = method_match.group(2)
                            methods.append((method_name, method_params, "Method description"))
                    elif method_line.startswith('class ') or (not method_line and j > i + 10):
                        break
                    j += 1
                
                classes.append((class_name, doc_string, methods))
        i += 1
    
    return classes


def _add_python_comments(code_content: str) -> str:
    """Add comments and docstrings to Python code."""
    lines = code_content.split('\n')
    commented_lines = []
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # Add function docstrings
        if stripped.startswith('def ') and not stripped.startswith('def _'):
            commented_lines.append(line)
            # Add docstring if not present
            if i + 1 < len(lines) and '"""' not in lines[i + 1]:
                indent = len(line) - len(line.lstrip())
                func_name = re.match(r'def\s+([a-zA-Z_][a-zA-Z0-9_]*)', stripped).group(1)
                docstring = ' ' * (indent + 4) + f'"""{func_name} function description."""'
                commented_lines.append(docstring)
        
        # Add class docstrings
        elif stripped.startswith('class '):
            commented_lines.append(line)
            if i + 1 < len(lines) and '"""' not in lines[i + 1]:
                indent = len(line) - len(line.lstrip())
                class_name = re.match(r'class\s+([a-zA-Z_][a-zA-Z0-9_]*)', stripped).group(1)
                docstring = ' ' * (indent + 4) + f'"""{class_name} class description."""'
                commented_lines.append(docstring)
        
        else:
            commented_lines.append(line)
    
    return '\n'.join(commented_lines)


def _generate_api_index(api_files: List[str]) -> str:
    """Generate API index documentation."""
    return f'''# API Index

This document provides an index of all API documentation.

## Available APIs

{chr(10).join(f"- [{Path(api_file).stem}]({api_file})" for api_file in api_files)}

## Getting Started

Choose the relevant API documentation from the list above to get started with that component.

## Common Patterns

All APIs follow consistent patterns:

- Functions return consistent data types
- Error handling follows standard conventions
- Documentation includes examples where applicable
'''


def _build_documentation_prompt(context: DocumentationContext) -> str:
    """Build prompt for LLM documentation generation."""
    project_info = context.project_info or {}
    code_info = ""
    
    if context.code_files:
        code_info = f"**Source Code Files ({len(context.code_files)}):**\n"
        for file_path, code_content in list(context.code_files.items())[:3]:  # Limit to first 3 files
            code_info += f"\n--- {file_path} ---\n```python\n{code_content[:500]}...\n```\n"
    
    return f"""
You are an expert technical writer. Generate comprehensive {context.documentation_type.value} documentation.

**Project Information:**
- Name: {project_info.get('name', 'Unknown Project')}
- Description: {project_info.get('description', context.goal)}
- Version: {project_info.get('version', '1.0.0')}

**Documentation Requirements:**
- Type: {context.documentation_type.value}
- Format: {context.output_format.value}
- Style: {context.template_style.value}
- Include Examples: {context.include_examples}

**Goal/Purpose:**
{context.goal}

{code_info}

**Output Requirements:**
1. Generate complete, professional documentation
2. Use {context.output_format.value} format
3. Follow {context.template_style.value} style conventions
4. Include appropriate sections and structure
5. Add examples if requested
6. Make it ready to use without modification

Generate the documentation content now:
"""


def _parse_documentation_response(response: str, context: DocumentationContext) -> Dict[str, str]:
    """Parse LLM response into documentation files."""
    # For now, return as single file
    file_ext = ".md" if context.output_format == DocumentationFormat.MARKDOWN else ".txt"
    filename = f"{context.documentation_type.value}{file_ext}"
    
    return {filename: response.strip()}


def _analyze_documentation_structure(generated_docs: Dict[str, str]) -> Dict[str, Any]:
    """Analyze the structure of generated documentation."""
    structure = {
        "total_files": len(generated_docs),
        "total_sections": 0,
        "total_words": 0,
        "formats": {},
        "sections_per_file": {}
    }
    
    for doc_file, doc_content in generated_docs.items():
        # Count sections (headers)
        sections = len(re.findall(r'^#+\s', doc_content, re.MULTILINE))
        structure["total_sections"] += sections
        structure["sections_per_file"][doc_file] = sections
        
        # Count words
        words = len(doc_content.split())
        structure["total_words"] += words
        
        # Identify format
        if doc_file.endswith('.md'):
            structure["formats"]["markdown"] = structure["formats"].get("markdown", 0) + 1
        elif doc_file.endswith('.rst'):
            structure["formats"]["rst"] = structure["formats"].get("rst", 0) + 1
        else:
            structure["formats"]["other"] = structure["formats"].get("other", 0) + 1
    
    return structure


def _calculate_documentation_quality(generated_docs: Dict[str, str], context: DocumentationContext) -> Dict[str, Any]:
    """Calculate quality metrics for generated documentation."""
    total_content = '\n'.join(generated_docs.values())
    total_words = len(total_content.split())
    total_lines = len(total_content.split('\n'))
    
    # Count various elements
    headers = len(re.findall(r'^#+\s', total_content, re.MULTILINE))
    code_blocks = len(re.findall(r'```', total_content))
    links = len(re.findall(r'\[([^\]]+)\]\([^)]+\)', total_content))
    
    return {
        "total_words": total_words,
        "total_lines": total_lines,
        "headers_count": headers,
        "code_blocks_count": code_blocks // 2,  # Assuming pairs of ```
        "links_count": links,
        "average_words_per_file": total_words / len(generated_docs) if generated_docs else 0,
        "documentation_completeness": _assess_completeness(generated_docs, context),
        "readability_score": _calculate_readability(total_content)
    }


def _assess_completeness(generated_docs: Dict[str, str], context: DocumentationContext) -> str:
    """Assess completeness of documentation."""
    total_content = '\n'.join(generated_docs.values()).lower()
    
    required_sections = {
        DocumentationType.README: ["installation", "usage", "example"],
        DocumentationType.API_DOCS: ["function", "parameter", "return"],
        DocumentationType.USER_GUIDE: ["getting started", "usage", "example"],
        DocumentationType.TECHNICAL_SPEC: ["requirements", "architecture", "design"]
    }
    
    required = required_sections.get(context.documentation_type, [])
    found_sections = sum(1 for section in required if section in total_content)
    
    if not required:
        return "complete"
    
    completeness_ratio = found_sections / len(required)
    
    if completeness_ratio >= 0.8:
        return "complete"
    elif completeness_ratio >= 0.6:
        return "mostly_complete"
    elif completeness_ratio >= 0.4:
        return "partial"
    else:
        return "incomplete"


def _calculate_readability(content: str) -> float:
    """Calculate basic readability score."""
    sentences = len(re.findall(r'[.!?]+', content))
    words = len(content.split())
    
    if sentences == 0:
        return 0.0
    
    avg_words_per_sentence = words / sentences
    
    # Simple readability score (lower is better)
    if avg_words_per_sentence <= 15:
        return 0.9  # Very readable
    elif avg_words_per_sentence <= 20:
        return 0.7  # Good
    elif avg_words_per_sentence <= 25:
        return 0.5  # Fair
    else:
        return 0.3  # Difficult