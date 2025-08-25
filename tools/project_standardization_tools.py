# tools/project_standardization_tools.py
"""
Project Standardization Tools

Generates standard project configuration files for different project types
to enable proper test detection and execution.
"""

import logging
from pathlib import Path
from typing import List
from tools.file_system_tools import FilesystemResult

logger = logging.getLogger(__name__)


def detect_project_type(workspace_path: str) -> str:
    """Detect the project type based on files in the workspace."""
    workspace = Path(workspace_path)
    
    # Python project indicators
    if any(workspace.glob("*.py")) or (workspace / "requirements.txt").exists():
        return "python"
    
    # Node.js project indicators  
    if (workspace / "package.json").exists():
        return "javascript"
        
    # Go project indicators
    if (workspace / "go.mod").exists():
        return "go"
        
    # Rust project indicators
    if (workspace / "Cargo.toml").exists():
        return "rust"
        
    # Java Maven indicators
    if (workspace / "pom.xml").exists():
        return "java_maven"
        
    # Java Gradle indicators
    if any(workspace.glob("*.gradle")) or (workspace / "gradlew").exists():
        return "java_gradle"
        
    # .NET indicators
    if any(workspace.glob("*.sln")) or any(workspace.glob("*.csproj")):
        return "dotnet"
        
    return "unknown"


def has_standard_test_config(workspace_path: str, project_type: str) -> bool:
    """Check if project has standard test configuration files."""
    workspace = Path(workspace_path)
    
    if project_type == "python":
        # Check for standard Python test configs
        return any([
            (workspace / "pyproject.toml").exists(),
            (workspace / "pytest.ini").exists(),
            (workspace / "setup.cfg").exists(),
            (workspace / "tox.ini").exists()
        ])
    elif project_type == "javascript":
        package_json = workspace / "package.json"
        if package_json.exists():
            try:
                import json
                with open(package_json) as f:
                    data = json.load(f)
                    return "scripts" in data and "test" in data.get("scripts", {})
            except:
                return False
    # Add other project types as needed
    return False


def generate_python_config_files(workspace_path: str) -> FilesystemResult:
    """Generate standard Python project configuration files."""
    try:
        workspace = Path(workspace_path)
        generated_files = []
        
        # Detect project structure
        has_tests_dir = (workspace / "tests").exists()
        has_requirements = (workspace / "requirements.txt").exists()
        
        # Find main Python files
        main_files = []
        for pattern in ["main.py", "*_main.py", "app.py", "run.py"]:
            main_files.extend(workspace.glob(pattern))
        
        # Generate pyproject.toml
        pyproject_content = generate_pyproject_toml_content(
            workspace_path, has_tests_dir, has_requirements, main_files
        )
        
        pyproject_path = workspace / "pyproject.toml"
        if not pyproject_path.exists():
            with open(pyproject_path, 'w') as f:
                f.write(pyproject_content)
            generated_files.append(str(pyproject_path))
            logger.info(f"Generated pyproject.toml at {pyproject_path}")
        
        # Generate pytest.ini if no pyproject.toml test config
        pytest_ini_path = workspace / "pytest.ini"
        if not pytest_ini_path.exists() and not pyproject_path.exists():
            pytest_content = generate_pytest_ini_content(has_tests_dir)
            with open(pytest_ini_path, 'w') as f:
                f.write(pytest_content)
            generated_files.append(str(pytest_ini_path))
            logger.info(f"Generated pytest.ini at {pytest_ini_path}")
        
        from tools.file_system_tools import FileOperation
        return FilesystemResult(
            success=True,
            message=f"Generated {len(generated_files)} Python config files",
            operation=FileOperation.CREATE,
            paths_affected=generated_files,
            content={"generated_files": generated_files, "project_type": "python"}
        )
        
    except Exception as e:
        logger.error(f"Failed to generate Python config files: {e}")
        return FilesystemResult(
            success=False,
            message=f"Failed to generate Python config files: {e}",
            operation=FileOperation.CREATE,
            paths_affected=[],
            error_details={"exception": str(e)}
        )


def generate_pyproject_toml_content(workspace_path: str, has_tests_dir: bool, 
                                   has_requirements: bool, main_files: List[Path]) -> str:
    """Generate pyproject.toml content based on project structure."""
    workspace = Path(workspace_path)
    project_name = workspace.name
    
    # Read existing requirements if available
    dependencies = []
    if has_requirements:
        try:
            req_path = workspace / "requirements.txt"
            with open(req_path) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        dependencies.append(f'    "{line}",')
        except:
            pass
    
    # Generate entry points for main files
    entry_points = []
    for main_file in main_files:
        script_name = main_file.stem.replace('_', '-')
        module_name = main_file.stem
        entry_points.append(f'{script_name} = "{module_name}:main"')
    
    content = f'''[build-system]
requires = ["setuptools>=45", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "{project_name}"
version = "0.1.0"
description = "Auto-generated project configuration"
authors = [{{name = "Unknown", email = "unknown@example.com"}}]
readme = "README.md"
requires-python = ">=3.8"
dependencies = [
{chr(10).join(dependencies)}
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-asyncio>=0.21.0",
    "pytest-cov>=4.0.0",
]

'''
    
    # Add entry points if main files found
    if entry_points:
        content += f'''[project.scripts]
{chr(10).join(entry_points)}

'''
    
    # Add pytest configuration
    testpaths = '"tests"' if has_tests_dir else '"."'
    content += f'''[tool.pytest.ini_options]
testpaths = [{testpaths}]
python_files = ["test_*.py", "*_test.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
addopts = [
    "-v",
    "--tb=short",
    "--strict-markers",
    "--disable-warnings",
]
markers = [
    "slow: marks tests as slow (deselect with '-m \\"not slow\\"')",
    "integration: marks tests as integration tests",
    "unit: marks tests as unit tests",
]

[tool.coverage.run]
source = ["."]
omit = [
    "*/tests/*",
    "*/test_*.py",
    "*/*_test.py",
]
'''
    
    return content


def generate_pytest_ini_content(has_tests_dir: bool) -> str:
    """Generate pytest.ini content."""
    testpaths = "tests" if has_tests_dir else "."
    
    return f'''[tool:pytest]
testpaths = {testpaths}
python_files = test_*.py *_test.py
python_classes = Test*
python_functions = test_*
addopts = -v --tb=short --strict-markers --disable-warnings
markers =
    slow: marks tests as slow (deselect with -m "not slow")
    integration: marks tests as integration tests
    unit: marks tests as unit tests
'''


def standardize_project(workspace_path: str) -> FilesystemResult:
    """
    Main function to standardize a project by generating necessary config files.
    """
    try:
        project_type = detect_project_type(workspace_path)
        
        if project_type == "unknown":
            from tools.file_system_tools import FileOperation
            return FilesystemResult(
                success=False,
                message="Could not determine project type for standardization",
                operation=FileOperation.DISCOVER,
                paths_affected=[],
                content={"workspace_path": workspace_path}
            )
        
        # Check if already has standard config
        if has_standard_test_config(workspace_path, project_type):
            from tools.file_system_tools import FileOperation
            return FilesystemResult(
                success=True,
                message=f"Project already has standard {project_type} configuration",
                operation=FileOperation.DISCOVER,
                paths_affected=[],
                content={"project_type": project_type, "already_standardized": True}
            )
        
        # Generate config files based on project type
        if project_type == "python":
            return generate_python_config_files(workspace_path)
        else:
            from tools.file_system_tools import FileOperation
            return FilesystemResult(
                success=False,
                message=f"Standardization not yet implemented for project type: {project_type}",
                operation=FileOperation.DISCOVER,
                paths_affected=[],
                content={"project_type": project_type, "workspace_path": workspace_path}
            )
            
    except Exception as e:
        logger.error(f"Project standardization failed: {e}")
        from tools.file_system_tools import FileOperation
        return FilesystemResult(
            success=False,
            message=f"Project standardization failed: {e}",
            operation=FileOperation.DISCOVER,
            paths_affected=[],
            error_details={"exception": str(e), "workspace_path": workspace_path}
        )