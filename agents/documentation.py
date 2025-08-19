# agents/documentation.py
import logging
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

# Foundational dependencies
from .base import BaseAgent
from core.context import GlobalContext
from core.models import AgentResponse, AgentResult, TaskNode

# Get a logger instance for this module
logger = logging.getLogger(__name__)


class DocumentationAgent(BaseAgent):
    """
    Specialized Tier: Documentation creation and maintenance.
    
    This agent handles all documentation-related operations.
    
    Responsibilities:
    - Generate README files, API docs, code comments
    - Create user guides and technical documentation
    - Maintain documentation consistency
    - Documentation quality analysis
    
    Does NOT: Modify application code, run tests, configure systems
    """

    def __init__(self):
        super().__init__(
            name="DocumentationAgent",
            description="Creates and maintains project documentation including READMEs, API docs, and code comments."
        )

    def required_inputs(self) -> List[str]:
        """Required inputs for DocumentationAgent execution."""
        return ["operation"]

    def optional_inputs(self) -> List[str]:
        """Optional inputs for DocumentationAgent execution."""
        return [
            "content_data",
            "target_file",
            "documentation_type",
            "code_files",
            "project_info",
            "template_style",
            "include_examples",
            "working_directory",
            "output_format"
        ]

    def execute_v2(self, goal: str, inputs: Dict[str, Any], global_context: GlobalContext) -> AgentResult:
        """
        NEW INTERFACE: Execute documentation operations.
        """
        logger.info(f"DocumentationAgent executing: '{goal}'")
        
        # Validate inputs
        try:
            self.validate_inputs(inputs)
        except Exception as validation_error:
            return self.create_result(
                success=False,
                message=str(validation_error),
                error_details={"validation_error": str(validation_error)}
            )

        # Extract inputs
        operation = inputs["operation"]
        content_data = inputs.get("content_data", {})
        target_file = inputs.get("target_file")
        documentation_type = inputs.get("documentation_type", "readme")
        code_files = inputs.get("code_files", {})
        project_info = inputs.get("project_info", {})
        template_style = inputs.get("template_style", "standard")
        include_examples = inputs.get("include_examples", True)
        working_directory = inputs.get("working_directory", str(global_context.workspace_path))
        output_format = inputs.get("output_format", "markdown")

        try:
            self.report_progress(f"Starting {operation} operation", f"Type: {documentation_type}")

            if operation == "generate":
                result = self._generate_documentation(
                    documentation_type, content_data, code_files, project_info,
                    template_style, include_examples, working_directory, output_format
                )
            elif operation == "update":
                result = self._update_documentation(
                    target_file, content_data, working_directory
                )
            elif operation == "analyze":
                result = self._analyze_documentation(
                    working_directory, code_files
                )
            elif operation == "extract_comments":
                result = self._extract_code_comments(
                    code_files, output_format
                )
            elif operation == "validate":
                result = self._validate_documentation(
                    target_file or working_directory
                )
            elif operation == "create_api_docs":
                result = self._create_api_documentation(
                    code_files, output_format, working_directory
                )
            else:
                return self.create_result(
                    success=False,
                    message=f"Unknown operation: {operation}",
                    error_details={"supported_operations": ["generate", "update", "analyze", "extract_comments", "validate", "create_api_docs"]}
                )

            self.report_progress("Documentation operation complete", result["message"])

            return self.create_result(
                success=result["success"],
                message=result["message"],
                outputs=result["outputs"]
            )

        except Exception as e:
            error_msg = f"DocumentationAgent execution failed: {e}"
            logger.error(error_msg, exc_info=True)
            return self.create_result(
                success=False,
                message=error_msg,
                error_details={"exception": str(e)}
            )

    def _generate_documentation(self, doc_type: str, content_data: Dict[str, Any],
                              code_files: Dict[str, str], project_info: Dict[str, Any],
                              template_style: str, include_examples: bool,
                              working_directory: str, output_format: str) -> Dict[str, Any]:
        """Generate documentation based on type and content."""
        
        try:
            if doc_type == "readme":
                doc_content = self._generate_readme(
                    project_info, code_files, template_style, include_examples
                )
                filename = "README.md"
            elif doc_type == "api":
                doc_content = self._generate_api_docs(
                    code_files, output_format, include_examples
                )
                filename = f"API_DOCS.{output_format.lower()}"
            elif doc_type == "user_guide":
                doc_content = self._generate_user_guide(
                    project_info, content_data, include_examples
                )
                filename = f"USER_GUIDE.{output_format.lower()}"
            elif doc_type == "contributing":
                doc_content = self._generate_contributing_guide(
                    project_info, template_style
                )
                filename = "CONTRIBUTING.md"
            elif doc_type == "changelog":
                doc_content = self._generate_changelog(
                    content_data, project_info
                )
                filename = "CHANGELOG.md"
            else:
                return {
                    "success": False,
                    "message": f"Unknown documentation type: {doc_type}",
                    "outputs": {}
                }

            # Write documentation to file
            output_path = Path(working_directory) / filename
            output_path.write_text(doc_content, encoding='utf-8')

            return {
                "success": True,
                "message": f"Successfully generated {doc_type} documentation",
                "outputs": {
                    "documentation_type": doc_type,
                    "filename": filename,
                    "file_path": str(output_path),
                    "content_length": len(doc_content),
                    "template_style": template_style
                }
            }

        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to generate {doc_type} documentation: {e}",
                "outputs": {"error": str(e)}
            }

    def _generate_readme(self, project_info: Dict[str, Any], code_files: Dict[str, str],
                        template_style: str, include_examples: bool) -> str:
        """Generate a README.md file."""
        
        project_name = project_info.get("name", "Project")
        description = project_info.get("description", "A Python project")
        version = project_info.get("version", "1.0.0")
        author = project_info.get("author", "")
        license_type = project_info.get("license", "MIT")
        
        # Analyze code structure
        main_modules = self._analyze_code_structure(code_files)
        requirements = self._extract_requirements(code_files)

        readme_content = f"""# {project_name}

{description}

## Version
{version}

## Features

"""

        # Add features based on code analysis
        if main_modules:
            readme_content += "- **Core Modules**: " + ", ".join(main_modules[:5]) + "\n"
        
        if any("test" in filename.lower() for filename in code_files.keys()):
            readme_content += "- **Testing**: Comprehensive test suite included\n"
        
        if any("api" in filename.lower() or "server" in filename.lower() for filename in code_files.keys()):
            readme_content += "- **API**: RESTful API endpoints\n"

        readme_content += f"""
## Installation

### Prerequisites
- Python 3.8 or higher
"""

        if requirements:
            readme_content += "- Required packages: " + ", ".join(requirements[:5]) + "\n"

        readme_content += """
### Setup
```bash
# Clone the repository
git clone <repository-url>
cd """ + project_name.lower().replace(" ", "-") + """

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

"""

        if include_examples:
            # Generate usage examples based on code analysis
            main_file = self._find_main_file(code_files)
            if main_file:
                readme_content += f"""### Basic Usage
```python
python {main_file}
```

"""

            # Add API examples if applicable
            if any("api" in filename.lower() for filename in code_files.keys()):
                readme_content += """### API Usage
```bash
# Start the server
python app.py

# Example API call
curl http://localhost:8000/api/endpoint
```

"""

        readme_content += f"""## Project Structure

```
{project_name.lower().replace(' ', '-')}/
├── README.md
"""

        # Add structure based on actual files
        for filename in sorted(code_files.keys())[:10]:  # Limit to 10 files
            if not filename.startswith('.'):
                readme_content += f"├── {filename}\n"

        readme_content += """└── requirements.txt
```

## Development

### Running Tests
```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=.
```

### Code Style
```bash
# Format code
black .

# Lint code
flake8 .
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

"""

        if license_type:
            readme_content += f"This project is licensed under the {license_type} License.\n"
        else:
            readme_content += "See LICENSE file for details.\n"

        if author:
            readme_content += f"""
## Author

{author}
"""

        return readme_content

    def _generate_api_docs(self, code_files: Dict[str, str], output_format: str, 
                          include_examples: bool) -> str:
        """Generate API documentation."""
        
        api_content = "# API Documentation\n\n"
        
        # Extract API endpoints from code
        endpoints = self._extract_api_endpoints(code_files)
        
        if endpoints:
            api_content += "## Endpoints\n\n"
            
            for endpoint in endpoints:
                api_content += f"### {endpoint['method']} {endpoint['path']}\n\n"
                api_content += f"{endpoint.get('description', 'API endpoint')}\n\n"
                
                if endpoint.get('parameters'):
                    api_content += "**Parameters:**\n"
                    for param in endpoint['parameters']:
                        api_content += f"- `{param['name']}` ({param['type']}): {param.get('description', '')}\n"
                    api_content += "\n"
                
                if include_examples and endpoint.get('example'):
                    api_content += "**Example:**\n"
                    api_content += f"```bash\n{endpoint['example']}\n```\n\n"
        else:
            # Generate generic API structure
            api_content += """## Base URL
```
http://localhost:8000/api/v1
```

## Authentication
Include API key in headers:
```
Authorization: Bearer <your-api-key>
```

## Response Format
All responses are returned in JSON format:
```json
{
  "status": "success|error",
  "data": {},
  "message": "Description"
}
```

## Error Codes
- `400` - Bad Request
- `401` - Unauthorized
- `404` - Not Found
- `500` - Internal Server Error
"""

        return api_content

    def _generate_user_guide(self, project_info: Dict[str, Any], content_data: Dict[str, Any], 
                           include_examples: bool) -> str:
        """Generate user guide documentation."""
        
        project_name = project_info.get("name", "Application")
        
        guide_content = f"""# {project_name} User Guide

## Getting Started

This guide will help you get started with {project_name}.

### Quick Start

1. **Installation**: Follow the installation steps in the README
2. **Configuration**: Set up your configuration files
3. **First Run**: Execute your first command

## Basic Usage

### Core Features

"""

        # Add feature descriptions from content_data
        features = content_data.get("features", [])
        for i, feature in enumerate(features[:5], 1):
            guide_content += f"#### {i}. {feature.get('name', f'Feature {i}')}\n\n"
            guide_content += f"{feature.get('description', 'Feature description')}\n\n"
            
            if include_examples and feature.get('example'):
                guide_content += f"```bash\n{feature['example']}\n```\n\n"

        guide_content += """## Advanced Usage

### Configuration Options

Customize your experience with these configuration options:

- **Option 1**: Description of option 1
- **Option 2**: Description of option 2
- **Option 3**: Description of option 3

### Troubleshooting

#### Common Issues

**Issue 1**: Problem description
- Solution: Step-by-step solution

**Issue 2**: Problem description  
- Solution: Step-by-step solution

## Best Practices

1. Always backup your data before major operations
2. Use virtual environments for Python projects
3. Keep dependencies up to date
4. Follow the documented API patterns

## Support

If you encounter issues:
1. Check the troubleshooting section
2. Search existing issues on GitHub
3. Create a new issue with detailed information
"""

        return guide_content

    def _generate_contributing_guide(self, project_info: Dict[str, Any], template_style: str) -> str:
        """Generate contributing guidelines."""
        
        project_name = project_info.get("name", "Project")
        
        return f"""# Contributing to {project_name}

Thank you for considering contributing to {project_name}! 

## How to Contribute

### Reporting Bugs

Before creating bug reports, please check existing issues. When creating a bug report, include:

- A clear and descriptive title
- Steps to reproduce the issue
- Expected vs actual behavior
- Your environment (OS, Python version, etc.)
- Any relevant logs or screenshots

### Suggesting Enhancements

Enhancement suggestions are welcome! Please provide:

- A clear and descriptive title
- A detailed description of the suggested enhancement
- Examples of how the enhancement would be used
- Any relevant mockups or diagrams

### Pull Requests

1. **Fork** the repository
2. **Create** a feature branch from `main`
3. **Make** your changes
4. **Add** or update tests as needed
5. **Run** the test suite to ensure all tests pass
6. **Update** documentation if needed
7. **Submit** a pull request

### Development Setup

```bash
# Clone your fork
git clone https://github.com/yourusername/{project_name.lower().replace(' ', '-')}.git
cd {project_name.lower().replace(' ', '-')}

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

### Code Style

- Follow PEP 8 guidelines
- Use meaningful variable and function names
- Add docstrings to functions and classes
- Keep lines under 88 characters
- Use type hints where appropriate

### Testing

- Write tests for new features
- Ensure all existing tests pass
- Aim for high test coverage
- Use descriptive test names

### Documentation

- Update relevant documentation
- Add docstrings to new functions/classes
- Update README if adding new features
- Consider adding examples

## Code of Conduct

Be respectful and inclusive. We want this to be a welcoming environment for all contributors.

## Questions?

Feel free to ask questions by opening an issue or discussion.
"""

    def _generate_changelog(self, content_data: Dict[str, Any], project_info: Dict[str, Any]) -> str:
        """Generate changelog documentation."""
        
        project_name = project_info.get("name", "Project")
        version = project_info.get("version", "1.0.0")
        
        changelog_content = f"""# Changelog

All notable changes to {project_name} will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- New features that have been added

### Changed
- Changes in existing functionality

### Deprecated
- Soon-to-be removed features

### Removed
- Features that have been removed

### Fixed
- Bug fixes

### Security
- Security improvements

## [{version}] - {self._get_current_date()}

### Added
- Initial release
- Core functionality implemented
- Basic documentation

## [0.1.0] - {self._get_current_date()}

### Added
- Project structure
- Initial setup
- Development environment

---

## Guidelines for Updating

When adding changes to this changelog:

1. **Keep entries grouped by version**
2. **Use categories**: Added, Changed, Deprecated, Removed, Fixed, Security
3. **Write for humans**: Use clear, descriptive language
4. **Link issues**: Reference GitHub issues/PRs when applicable
5. **Date releases**: Use YYYY-MM-DD format

## Version Format

We use [Semantic Versioning](https://semver.org/):
- **MAJOR**: Incompatible API changes
- **MINOR**: New functionality (backward compatible)
- **PATCH**: Bug fixes (backward compatible)
"""

        return changelog_content

    def _update_documentation(self, target_file: str, content_data: Dict[str, Any], 
                            working_directory: str) -> Dict[str, Any]:
        """Update existing documentation file."""
        
        try:
            file_path = Path(working_directory) / target_file
            
            if not file_path.exists():
                return {
                    "success": False,
                    "message": f"Documentation file not found: {target_file}",
                    "outputs": {}
                }

            current_content = file_path.read_text(encoding='utf-8')
            
            # Update content based on content_data
            updated_content = self._merge_documentation_content(current_content, content_data)
            
            file_path.write_text(updated_content, encoding='utf-8')

            return {
                "success": True,
                "message": f"Successfully updated documentation: {target_file}",
                "outputs": {
                    "file_path": str(file_path),
                    "content_length": len(updated_content),
                    "changes_applied": list(content_data.keys())
                }
            }

        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to update documentation: {e}",
                "outputs": {"error": str(e)}
            }

    def _analyze_documentation(self, working_directory: str, code_files: Dict[str, str]) -> Dict[str, Any]:
        """Analyze existing documentation quality and completeness."""
        
        try:
            workspace_path = Path(working_directory)
            
            analysis = {
                "files_found": [],
                "missing_files": [],
                "quality_issues": [],
                "coverage_analysis": {},
                "recommendations": []
            }

            # Common documentation files to check for
            expected_docs = ["README.md", "CONTRIBUTING.md", "CHANGELOG.md", "LICENSE"]
            
            for doc_file in expected_docs:
                doc_path = workspace_path / doc_file
                if doc_path.exists():
                    analysis["files_found"].append(doc_file)
                    # Analyze file quality
                    content = doc_path.read_text(encoding='utf-8')
                    quality_issues = self._analyze_doc_quality(content, doc_file)
                    if quality_issues:
                        analysis["quality_issues"].extend(quality_issues)
                else:
                    analysis["missing_files"].append(doc_file)

            # Analyze code documentation
            if code_files:
                code_doc_analysis = self._analyze_code_documentation(code_files)
                analysis["coverage_analysis"] = code_doc_analysis

            # Generate recommendations
            analysis["recommendations"] = self._generate_doc_recommendations(analysis)

            return {
                "success": True,
                "message": f"Documentation analysis complete. Found {len(analysis['files_found'])} files.",
                "outputs": analysis
            }

        except Exception as e:
            return {
                "success": False,
                "message": f"Documentation analysis failed: {e}",
                "outputs": {"error": str(e)}
            }

    def _extract_code_comments(self, code_files: Dict[str, str], output_format: str) -> Dict[str, Any]:
        """Extract comments and docstrings from code files."""
        
        try:
            extracted_docs = {}
            
            for filename, content in code_files.items():
                if filename.endswith('.py'):
                    comments = self._extract_python_comments(content)
                    if comments:
                        extracted_docs[filename] = comments

            # Format extracted documentation
            if output_format.lower() == "markdown":
                formatted_content = self._format_comments_as_markdown(extracted_docs)
            else:
                formatted_content = self._format_comments_as_text(extracted_docs)

            return {
                "success": True,
                "message": f"Extracted comments from {len(extracted_docs)} files",
                "outputs": {
                    "extracted_comments": extracted_docs,
                    "formatted_content": formatted_content,
                    "files_processed": len(code_files),
                    "files_with_comments": len(extracted_docs)
                }
            }

        except Exception as e:
            return {
                "success": False,
                "message": f"Comment extraction failed: {e}",
                "outputs": {"error": str(e)}
            }

    def _validate_documentation(self, target_path: str) -> Dict[str, Any]:
        """Validate documentation files for common issues."""
        
        try:
            path = Path(target_path)
            validation_results = {
                "files_validated": [],
                "issues_found": [],
                "suggestions": [],
                "overall_score": 0
            }

            if path.is_file():
                # Validate single file
                issues = self._validate_single_doc_file(path)
                validation_results["files_validated"] = [str(path)]
                validation_results["issues_found"] = issues
            else:
                # Validate directory
                doc_files = list(path.glob("*.md")) + list(path.glob("*.rst")) + list(path.glob("*.txt"))
                
                for doc_file in doc_files:
                    issues = self._validate_single_doc_file(doc_file)
                    validation_results["files_validated"].append(str(doc_file))
                    validation_results["issues_found"].extend(issues)

            # Calculate overall score
            total_files = len(validation_results["files_validated"])
            total_issues = len(validation_results["issues_found"])
            
            if total_files > 0:
                validation_results["overall_score"] = max(0, 100 - (total_issues * 10))
            
            # Generate suggestions
            validation_results["suggestions"] = self._generate_validation_suggestions(validation_results["issues_found"])

            return {
                "success": True,
                "message": f"Validated {total_files} files. Found {total_issues} issues.",
                "outputs": validation_results
            }

        except Exception as e:
            return {
                "success": False,
                "message": f"Documentation validation failed: {e}",
                "outputs": {"error": str(e)}
            }

    def _create_api_documentation(self, code_files: Dict[str, str], output_format: str, 
                                 working_directory: str) -> Dict[str, Any]:
        """Create comprehensive API documentation from code."""
        
        try:
            api_elements = {
                "classes": [],
                "functions": [],
                "modules": [],
                "constants": []
            }

            # Analyze code files for API elements
            for filename, content in code_files.items():
                if filename.endswith('.py'):
                    elements = self._parse_python_api_elements(content, filename)
                    for category, items in elements.items():
                        api_elements[category].extend(items)

            # Generate documentation content
            if output_format.lower() == "markdown":
                api_content = self._format_api_as_markdown(api_elements)
                filename = "API_REFERENCE.md"
            else:
                api_content = self._format_api_as_text(api_elements)
                filename = "API_REFERENCE.txt"

            # Write to file
            output_path = Path(working_directory) / filename
            output_path.write_text(api_content, encoding='utf-8')

            return {
                "success": True,
                "message": f"API documentation created: {filename}",
                "outputs": {
                    "filename": filename,
                    "file_path": str(output_path),
                    "api_elements": {k: len(v) for k, v in api_elements.items()},
                    "content_length": len(api_content)
                }
            }

        except Exception as e:
            return {
                "success": False,
                "message": f"API documentation creation failed: {e}",
                "outputs": {"error": str(e)}
            }

    # Helper methods for content analysis and generation

    def _analyze_code_structure(self, code_files: Dict[str, str]) -> List[str]:
        """Analyze code structure to identify main modules."""
        modules = []
        for filename, content in code_files.items():
            if filename.endswith('.py') and not filename.startswith('test_'):
                module_name = filename.replace('.py', '')
                if any(keyword in content for keyword in ['class ', 'def ', 'import ']):
                    modules.append(module_name)
        return modules

    def _extract_requirements(self, code_files: Dict[str, str]) -> List[str]:
        """Extract common requirements from import statements."""
        requirements = set()
        common_packages = {
            'requests', 'flask', 'django', 'fastapi', 'numpy', 'pandas', 
            'matplotlib', 'seaborn', 'pytest', 'click', 'sqlalchemy'
        }
        
        for content in code_files.values():
            for line in content.split('\n'):
                if line.strip().startswith('import ') or line.strip().startswith('from '):
                    for package in common_packages:
                        if package in line:
                            requirements.add(package)
        
        return sorted(list(requirements))

    def _find_main_file(self, code_files: Dict[str, str]) -> Optional[str]:
        """Find the main entry point file."""
        candidates = ['main.py', 'app.py', '__main__.py']
        
        for candidate in candidates:
            if candidate in code_files:
                return candidate
        
        # Look for files with if __name__ == "__main__"
        for filename, content in code_files.items():
            if 'if __name__ == "__main__"' in content:
                return filename
        
        return None

    def _extract_api_endpoints(self, code_files: Dict[str, str]) -> List[Dict[str, Any]]:
        """Extract API endpoints from code files."""
        endpoints = []
        
        for filename, content in code_files.items():
            # Look for Flask/Django/FastAPI route patterns
            route_patterns = [
                r'@app\.route\(["\']([^"\']+)["\'].*?,.*?methods=\[([^\]]+)\]',
                r'@router\.(get|post|put|delete)\(["\']([^"\']+)["\']',
                r'path\(["\']([^"\']+)["\'],.*?([A-Z]+)'
            ]
            
            for pattern in route_patterns:
                matches = re.finditer(pattern, content, re.MULTILINE)
                for match in matches:
                    endpoint = {
                        "path": match.group(1) if len(match.groups()) > 0 else "/",
                        "method": "GET",  # Default
                        "file": filename,
                        "description": "API endpoint"
                    }
                    endpoints.append(endpoint)
        
        return endpoints

    def _merge_documentation_content(self, current_content: str, content_data: Dict[str, Any]) -> str:
        """Merge new content with existing documentation."""
        # Simple merge - replace sections if they exist, append if they don't
        updated_content = current_content
        
        for section, new_content in content_data.items():
            # Look for existing section and replace it
            section_pattern = f"## {section}.*?(?=##|$)"
            if re.search(section_pattern, updated_content, re.DOTALL | re.IGNORECASE):
                updated_content = re.sub(
                    section_pattern, 
                    f"## {section}\n\n{new_content}\n", 
                    updated_content, 
                    flags=re.DOTALL | re.IGNORECASE
                )
            else:
                # Append new section
                updated_content += f"\n\n## {section}\n\n{new_content}\n"
        
        return updated_content

    def _analyze_doc_quality(self, content: str, filename: str) -> List[str]:
        """Analyze documentation quality and return issues."""
        issues = []
        
        if len(content) < 100:
            issues.append(f"{filename}: Too short (less than 100 characters)")
        
        if not re.search(r'^#', content, re.MULTILINE):
            issues.append(f"{filename}: No headings found")
        
        if filename == "README.md":
            required_sections = ["installation", "usage", "description"]
            for section in required_sections:
                if section.lower() not in content.lower():
                    issues.append(f"{filename}: Missing {section} section")
        
        return issues

    def _analyze_code_documentation(self, code_files: Dict[str, str]) -> Dict[str, Any]:
        """Analyze code documentation coverage."""
        total_functions = 0
        documented_functions = 0
        total_classes = 0
        documented_classes = 0
        
        for content in code_files.values():
            # Count functions
            functions = re.findall(r'def\s+(\w+)', content)
            total_functions += len(functions)
            
            # Count documented functions (those with docstrings)
            documented_funcs = re.findall(r'def\s+\w+.*?:\s*"""', content, re.DOTALL)
            documented_functions += len(documented_funcs)
            
            # Count classes
            classes = re.findall(r'class\s+(\w+)', content)
            total_classes += len(classes)
            
            # Count documented classes
            documented_cls = re.findall(r'class\s+\w+.*?:\s*"""', content, re.DOTALL)
            documented_classes += len(documented_cls)
        
        return {
            "function_coverage": documented_functions / max(total_functions, 1) * 100,
            "class_coverage": documented_classes / max(total_classes, 1) * 100,
            "total_functions": total_functions,
            "documented_functions": documented_functions,
            "total_classes": total_classes,
            "documented_classes": documented_classes
        }

    def _generate_doc_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate documentation improvement recommendations."""
        recommendations = []
        
        if analysis["missing_files"]:
            recommendations.append(f"Add missing documentation files: {', '.join(analysis['missing_files'])}")
        
        if analysis["quality_issues"]:
            recommendations.append("Address quality issues in existing documentation")
        
        coverage = analysis.get("coverage_analysis", {})
        if coverage.get("function_coverage", 100) < 50:
            recommendations.append("Improve function documentation coverage")
        
        if coverage.get("class_coverage", 100) < 50:
            recommendations.append("Improve class documentation coverage")
        
        return recommendations

    def _extract_python_comments(self, content: str) -> Dict[str, List[str]]:
        """Extract comments and docstrings from Python code."""
        extracted = {
            "docstrings": [],
            "comments": [],
            "module_docstring": ""
        }
        
        # Extract module docstring
        module_docstring_match = re.search(r'^"""(.*?)"""', content, re.DOTALL | re.MULTILINE)
        if module_docstring_match:
            extracted["module_docstring"] = module_docstring_match.group(1).strip()
        
        # Extract function/class docstrings
        docstring_matches = re.finditer(r'(def|class)\s+(\w+).*?:\s*"""(.*?)"""', content, re.DOTALL)
        for match in docstring_matches:
            extracted["docstrings"].append({
                "type": match.group(1),
                "name": match.group(2),
                "docstring": match.group(3).strip()
            })
        
        # Extract comments
        comment_matches = re.finditer(r'#\s*(.+)$', content, re.MULTILINE)
        for match in comment_matches:
            extracted["comments"].append(match.group(1).strip())
        
        return extracted

    def _format_comments_as_markdown(self, extracted_docs: Dict[str, Dict]) -> str:
        """Format extracted comments as Markdown."""
        markdown_content = "# Code Documentation\n\n"
        
        for filename, docs in extracted_docs.items():
            markdown_content += f"## {filename}\n\n"
            
            if docs.get("module_docstring"):
                markdown_content += f"**Module Description:**\n{docs['module_docstring']}\n\n"
            
            if docs.get("docstrings"):
                markdown_content += "### Functions and Classes\n\n"
                for item in docs["docstrings"]:
                    markdown_content += f"#### {item['type'].title()}: {item['name']}\n"
                    markdown_content += f"{item['docstring']}\n\n"
            
            if docs.get("comments"):
                markdown_content += "### Comments\n\n"
                for comment in docs["comments"][:10]:  # Limit to 10 comments
                    markdown_content += f"- {comment}\n"
                markdown_content += "\n"
        
        return markdown_content

    def _format_comments_as_text(self, extracted_docs: Dict[str, Dict]) -> str:
        """Format extracted comments as plain text."""
        text_content = "CODE DOCUMENTATION\n" + "="*50 + "\n\n"
        
        for filename, docs in extracted_docs.items():
            text_content += f"FILE: {filename}\n" + "-"*30 + "\n\n"
            
            if docs.get("module_docstring"):
                text_content += f"MODULE: {docs['module_docstring']}\n\n"
            
            for item in docs.get("docstrings", []):
                text_content += f"{item['type'].upper()}: {item['name']}\n"
                text_content += f"{item['docstring']}\n\n"
        
        return text_content

    def _validate_single_doc_file(self, file_path: Path) -> List[str]:
        """Validate a single documentation file."""
        issues = []
        
        try:
            content = file_path.read_text(encoding='utf-8')
            
            # Check file length
            if len(content) < 50:
                issues.append(f"{file_path.name}: File too short")
            
            # Check for basic structure
            if file_path.suffix.lower() == '.md':
                if not re.search(r'^#', content, re.MULTILINE):
                    issues.append(f"{file_path.name}: No markdown headings found")
            
            # Check for broken links
            broken_links = re.findall(r'\[([^\]]+)\]\(([^)]+)\)', content)
            for link_text, link_url in broken_links:
                if link_url.startswith('http') and 'localhost' in link_url:
                    issues.append(f"{file_path.name}: Localhost link found: {link_url}")
        
        except Exception as e:
            issues.append(f"{file_path.name}: Error reading file - {e}")
        
        return issues

    def _generate_validation_suggestions(self, issues: List[str]) -> List[str]:
        """Generate suggestions based on validation issues."""
        suggestions = []
        
        if any("too short" in issue.lower() for issue in issues):
            suggestions.append("Add more detailed content to short documentation files")
        
        if any("no headings" in issue.lower() for issue in issues):
            suggestions.append("Add proper headings to structure your documentation")
        
        if any("localhost" in issue.lower() for issue in issues):
            suggestions.append("Replace localhost links with production URLs")
        
        return suggestions

    def _parse_python_api_elements(self, content: str, filename: str) -> Dict[str, List[Dict]]:
        """Parse Python code for API elements."""
        elements = {
            "classes": [],
            "functions": [],
            "modules": [{"name": filename, "description": "Python module"}],
            "constants": []
        }
        
        # Extract classes
        class_matches = re.finditer(r'class\s+(\w+).*?:\s*("""(.*?)""")?', content, re.DOTALL)
        for match in class_matches:
            elements["classes"].append({
                "name": match.group(1),
                "docstring": match.group(3) if match.group(3) else "",
                "file": filename
            })
        
        # Extract functions
        func_matches = re.finditer(r'def\s+(\w+)\((.*?)\).*?:\s*("""(.*?)""")?', content, re.DOTALL)
        for match in func_matches:
            elements["functions"].append({
                "name": match.group(1),
                "parameters": match.group(2),
                "docstring": match.group(4) if match.group(4) else "",
                "file": filename
            })
        
        # Extract constants (UPPER_CASE variables)
        const_matches = re.finditer(r'^([A-Z_]+)\s*=\s*(.+)$', content, re.MULTILINE)
        for match in const_matches:
            elements["constants"].append({
                "name": match.group(1),
                "value": match.group(2),
                "file": filename
            })
        
        return elements

    def _format_api_as_markdown(self, api_elements: Dict[str, List]) -> str:
        """Format API elements as Markdown documentation."""
        markdown = "# API Reference\n\n"
        
        if api_elements["modules"]:
            markdown += "## Modules\n\n"
            for module in api_elements["modules"]:
                markdown += f"### {module['name']}\n{module.get('description', '')}\n\n"
        
        if api_elements["classes"]:
            markdown += "## Classes\n\n"
            for cls in api_elements["classes"]:
                markdown += f"### class {cls['name']}\n\n"
                if cls.get("docstring"):
                    markdown += f"{cls['docstring']}\n\n"
                markdown += f"*Defined in: {cls['file']}*\n\n"
        
        if api_elements["functions"]:
            markdown += "## Functions\n\n"
            for func in api_elements["functions"]:
                markdown += f"### {func['name']}({func['parameters']})\n\n"
                if func.get("docstring"):
                    markdown += f"{func['docstring']}\n\n"
                markdown += f"*Defined in: {func['file']}*\n\n"
        
        if api_elements["constants"]:
            markdown += "## Constants\n\n"
            for const in api_elements["constants"]:
                markdown += f"### {const['name']}\n\n"
                markdown += f"**Value:** `{const['value']}`\n\n"
                markdown += f"*Defined in: {const['file']}*\n\n"
        
        return markdown

    def _format_api_as_text(self, api_elements: Dict[str, List]) -> str:
        """Format API elements as plain text documentation."""
        text = "API REFERENCE\n" + "="*50 + "\n\n"
        
        for category, items in api_elements.items():
            if items:
                text += f"{category.upper()}\n" + "-"*20 + "\n\n"
                for item in items:
                    text += f"Name: {item.get('name', 'Unknown')}\n"
                    if 'docstring' in item and item['docstring']:
                        text += f"Description: {item['docstring']}\n"
                    if 'file' in item:
                        text += f"File: {item['file']}\n"
                    text += "\n"
        
        return text

    def _get_current_date(self) -> str:
        """Get current date in YYYY-MM-DD format."""
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d")

    # Legacy execute method for backward compatibility
    def execute(self, goal: str, context: GlobalContext, current_task: TaskNode) -> AgentResponse:
        """Legacy execute method - converts to new interface."""
        inputs = {
            'operation': 'generate',
            'documentation_type': 'readme',
            'working_directory': str(context.workspace_path),
            'include_examples': True
        }
        
        result = self.execute_v2(goal, inputs, context)
        
        return AgentResponse(
            success=result.success,
            message=result.message,
            artifacts_generated=[]
        )