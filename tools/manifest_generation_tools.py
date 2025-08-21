# tools/creation/manifest_generation_tools.py
"""
Manifest generation tools for creating project file structure from technical specifications.
Extracted from CodeManifestAgent to provide atomic, reusable manifest generation capabilities.
"""

import json
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class ProjectType(Enum):
    """Types of projects for manifest generation."""
    WEB_API = "web_api"
    CLI_TOOL = "cli_tool"
    LIBRARY = "library"
    MICROSERVICE = "microservice"
    FULL_STACK = "full_stack"
    DATA_SCIENCE = "data_science"

class ProjectStructure(Enum):
    """Different project structure styles."""
    FLAT = "flat"
    MODULAR = "modular"
    LAYERED = "layered"
    DOMAIN_DRIVEN = "domain_driven"

@dataclass
class ManifestContext:
    """Context for manifest generation."""
    technical_spec: str
    project_type: ProjectType = ProjectType.WEB_API
    structure_style: ProjectStructure = ProjectStructure.MODULAR
    target_language: str = "Python"
    existing_files: List[str] = None
    include_tests: bool = True
    include_docs: bool = True
    include_config: bool = True
    package_name: str = ""
    
    def __post_init__(self):
        if self.existing_files is None:
            self.existing_files = []

@dataclass
class FileInfo:
    """Information about a file to be created."""
    path: str
    description: str = ""
    file_type: str = ""
    dependencies: List[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []

@dataclass
class ManifestResult:
    """Result of manifest generation."""
    success: bool
    message: str
    files_to_create: List[str]
    file_details: List[FileInfo] = None
    directory_structure: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.file_details is None:
            self.file_details = []
        if self.directory_structure is None:
            self.directory_structure = {}

def build_manifest_prompt(context: ManifestContext) -> str:
    """Build a comprehensive prompt for manifest generation."""
    
    context_summary = {"existing_files": context.existing_files}
    
    if context.project_type == ProjectType.WEB_API:
        return _build_web_api_manifest_prompt(context, context_summary)
    elif context.project_type == ProjectType.CLI_TOOL:
        return _build_cli_manifest_prompt(context, context_summary)
    elif context.project_type == ProjectType.LIBRARY:
        return _build_library_manifest_prompt(context, context_summary)
    elif context.project_type == ProjectType.MICROSERVICE:
        return _build_microservice_manifest_prompt(context, context_summary)
    elif context.project_type == ProjectType.DATA_SCIENCE:
        return _build_data_science_manifest_prompt(context, context_summary)
    else:
        return _build_general_manifest_prompt(context, context_summary)

def _build_web_api_manifest_prompt(context: ManifestContext, context_summary: Dict[str, Any]) -> str:
    """Build prompt for web API project manifest generation."""
    return f"""
    **Persona**: You are an expert tech lead responsible for planning a web API project's file structure. Your task is to convert a technical specification into a JSON manifest of all the file paths that need to be created.

    **Overall Goal**: Analyze the technical specification and the existing project context to produce a logical and complete list of files for a {context.target_language} web API project.

    **Context**:
    - **Technical Specification**: 
      ---
      {context.technical_spec}
      ---
    - **Project Type**: {context.project_type.value}
    - **Structure Style**: {context.structure_style.value}
    - **Target Language**: {context.target_language}
    - **Package Name**: {context.package_name}
    - **Current Project Files**: 
      {json.dumps(context_summary, indent=2)}

    **Your Task**:
    1. **Analyze Components**: Identify all the distinct components described in the spec (e.g., data models, API routes, middleware, services).
    2. **Design a File Structure**: Create a logical file and directory structure for a web API that separates concerns.
    3. **List All Files**: Generate a complete list of all the file paths that need to be created to implement the specification.

    **Constraints**:
    - **Web API Structure**: Use standard web API project layout with routes, models, services, middleware
    - **Include Required Files**: Include all necessary files such as `__init__.py` files for packages, configuration files, middleware, and test files
    - **Follow Conventions**: Adhere to {context.target_language} web API best practices
    - **Tests**: {'Include test files' if context.include_tests else 'Exclude test files'}
    - **Documentation**: {'Include documentation files' if context.include_docs else 'Exclude documentation files'}
    - **Configuration**: {'Include configuration files' if context.include_config else 'Exclude configuration files'}

    **Output Format**: You MUST return a single, valid JSON object with one key, "files_to_create", which holds a list of strings representing the file paths.
    
    **Example Output for Python FastAPI**:
    {{
        "files_to_create": [
            "src/main.py",
            "src/__init__.py",
            "src/models/__init__.py",
            "src/models/user.py",
            "src/routes/__init__.py",
            "src/routes/auth.py",
            "src/routes/users.py",
            "src/services/__init__.py",
            "src/services/auth_service.py",
            "src/services/user_service.py",
            "src/middleware/__init__.py",
            "src/middleware/auth.py",
            "src/config.py",
            "src/database.py",
            "tests/test_auth.py",
            "tests/test_users.py",
            "requirements.txt",
            "README.md"
        ]
    }}
    """

def _build_cli_manifest_prompt(context: ManifestContext, context_summary: Dict[str, Any]) -> str:
    """Build prompt for CLI tool project manifest generation."""
    return f"""
    **Persona**: You are an expert tech lead responsible for planning a CLI tool project's file structure.

    **Context**:
    - **Technical Specification**: {context.technical_spec}
    - **Project Type**: CLI Tool
    - **Target Language**: {context.target_language}

    **Your Task**: Create a manifest for a command-line interface tool with proper command structure, argument parsing, and utilities.

    **Output Format**: Return JSON with "files_to_create" list for CLI project structure.
    """

def _build_library_manifest_prompt(context: ManifestContext, context_summary: Dict[str, Any]) -> str:
    """Build prompt for library project manifest generation."""
    return f"""
    **Persona**: You are an expert tech lead responsible for planning a library project's file structure.

    **Context**:
    - **Technical Specification**: {context.technical_spec}
    - **Project Type**: Library
    - **Target Language**: {context.target_language}

    **Your Task**: Create a manifest for a reusable library with proper module organization and public API.

    **Output Format**: Return JSON with "files_to_create" list for library project structure.
    """

def _build_microservice_manifest_prompt(context: ManifestContext, context_summary: Dict[str, Any]) -> str:
    """Build prompt for microservice project manifest generation."""
    return f"""
    **Persona**: You are an expert tech lead responsible for planning a microservice's file structure.

    **Context**:
    - **Technical Specification**: {context.technical_spec}
    - **Project Type**: Microservice
    - **Target Language**: {context.target_language}

    **Your Task**: Create a manifest for a microservice with proper separation of concerns, health checks, and containerization.

    **Output Format**: Return JSON with "files_to_create" list for microservice structure.
    """

def _build_data_science_manifest_prompt(context: ManifestContext, context_summary: Dict[str, Any]) -> str:
    """Build prompt for data science project manifest generation."""
    return f"""
    **Persona**: You are an expert data science tech lead responsible for planning a data science project's file structure.

    **Context**:
    - **Technical Specification**: {context.technical_spec}
    - **Project Type**: Data Science
    - **Target Language**: {context.target_language}

    **Your Task**: Create a manifest for a data science project with notebooks, data processing, models, and visualization.

    **Output Format**: Return JSON with "files_to_create" list for data science project structure.
    """

def _build_general_manifest_prompt(context: ManifestContext, context_summary: Dict[str, Any]) -> str:
    """Build prompt for general project manifest generation."""
    return f"""
    **Persona**: You are an expert tech lead responsible for planning a project's file structure. Your task is to convert a technical specification into a JSON manifest of all the file paths that need to be created.

    **Overall Goal**: Analyze the technical specification and the existing project context to produce a logical and complete list of files for the development team.

    **Context**:
    - **Technical Specification**: 
      ---
      {context.technical_spec}
      ---
    - **Current Project Files**: 
      {json.dumps(context_summary, indent=2)}

    **Your Task**:
    1. **Analyze Components**: Identify all the distinct components described in the spec (e.g., data models, API routes, utility functions, tests).
    2. **Design a File Structure**: Create a logical file and directory structure that separates concerns. Use standard {context.target_language} project layouts.
    3. **List All Files**: Generate a complete list of all the file paths that need to be created to implement the specification.

    **Constraints**:
    - **Be Comprehensive**: Include all necessary files, such as `__init__.py` files for packages, configuration files (`config.py`), and test files (`tests/test_...`).
    - **Follow Conventions**: Adhere to standard {context.target_language} project structure best practices.

    **Output Format**: You MUST return a single, valid JSON object with one key, "files_to_create", which holds a list of strings representing the file paths.
    
    **Example Output**:
    {{
        "files_to_create": [
            "src/main.py",
            "src/models/__init__.py",
            "src/models/user.py",
            "src/routes/__init__.py",
            "src/routes/auth.py",
            "tests/test_auth.py",
            "config.py",
            "requirements.txt"
        ]
    }}
    """

def generate_directory_structure(files: List[str]) -> Dict[str, Any]:
    """Generate a nested directory structure from a list of file paths."""
    structure = {}
    
    for file_path in files:
        parts = file_path.split('/')
        current = structure
        
        for part in parts[:-1]:  # Directory parts
            if part not in current:
                current[part] = {}
            current = current[part]
        
        # Add file
        if parts:
            filename = parts[-1]
            current[filename] = "file"
    
    return structure

def analyze_file_dependencies(files: List[str]) -> List[FileInfo]:
    """Analyze files to determine dependencies and file types."""
    file_infos = []
    
    for file_path in files:
        file_info = FileInfo(path=file_path)
        
        # Determine file type and description
        if file_path.endswith('.py'):
            file_info.file_type = "python"
            if '/models/' in file_path:
                file_info.description = "Data model definition"
            elif '/routes/' in file_path or '/api/' in file_path:
                file_info.description = "API route handler"
            elif '/services/' in file_path:
                file_info.description = "Business logic service"
            elif '/tests/' in file_path:
                file_info.description = "Test file"
            elif file_path.endswith('main.py'):
                file_info.description = "Application entry point"
            elif file_path.endswith('config.py'):
                file_info.description = "Configuration file"
            else:
                file_info.description = "Python module"
        elif file_path.endswith('.md'):
            file_info.file_type = "markdown"
            file_info.description = "Documentation file"
        elif file_path.endswith('.txt'):
            file_info.file_type = "text"
            if 'requirements' in file_path:
                file_info.description = "Python dependencies"
            else:
                file_info.description = "Text file"
        elif file_path.endswith('.json'):
            file_info.file_type = "json"
            file_info.description = "Configuration or data file"
        elif file_path.endswith('.yml') or file_path.endswith('.yaml'):
            file_info.file_type = "yaml"
            file_info.description = "Configuration file"
        else:
            file_info.file_type = "unknown"
            file_info.description = "File"
        
        # Determine basic dependencies
        if '/models/' in file_path and file_path.endswith('.py'):
            file_info.dependencies.append("database.py")
        elif '/routes/' in file_path and file_path.endswith('.py'):
            file_info.dependencies.extend(["models", "services"])
        elif '/services/' in file_path and file_path.endswith('.py'):
            file_info.dependencies.append("models")
        elif '/tests/' in file_path and file_path.endswith('.py'):
            # Test files depend on what they're testing
            test_target = file_path.replace('/tests/', '/').replace('test_', '').replace('.py', '.py')
            file_info.dependencies.append(test_target)
        
        file_infos.append(file_info)
    
    return file_infos

def validate_manifest(files: List[str], project_type: ProjectType, target_language: str) -> tuple[bool, List[str]]:
    """Validate a generated manifest for completeness."""
    errors = []
    
    if not files:
        errors.append("Manifest is empty")
        return False, errors
    
    # Check for required files based on project type and language
    if target_language.lower() == "python":
        # Python projects should have at least one .py file
        if not any(f.endswith('.py') for f in files):
            errors.append("No Python files found in manifest")
        
        # Check for __init__.py files in directories with Python modules
        python_dirs = set()
        for file_path in files:
            if file_path.endswith('.py') and '/' in file_path:
                dir_path = '/'.join(file_path.split('/')[:-1])
                python_dirs.add(dir_path)
        
        for dir_path in python_dirs:
            init_file = f"{dir_path}/__init__.py"
            if init_file not in files and dir_path != "tests":
                errors.append(f"Missing __init__.py in directory: {dir_path}")
    
    # Project type specific validations
    if project_type == ProjectType.WEB_API:
        # Web API should have routes/API files
        if not any('route' in f.lower() or 'api' in f.lower() for f in files):
            errors.append("Web API project missing route/API files")
        
        # Should have models
        if not any('model' in f.lower() for f in files):
            errors.append("Web API project missing model files")
    
    elif project_type == ProjectType.CLI_TOOL:
        # CLI should have main entry point
        if not any('main.py' in f or 'cli.py' in f or '__main__.py' in f for f in files):
            errors.append("CLI tool missing main entry point")
    
    # Check for duplicate files
    if len(files) != len(set(files)):
        errors.append("Duplicate files found in manifest")
    
    # Check for invalid file paths
    for file_path in files:
        if not file_path or file_path.startswith('/') or '..' in file_path:
            errors.append(f"Invalid file path: {file_path}")
    
    is_valid = len(errors) == 0
    return is_valid, errors

def generate_manifest(context: ManifestContext, llm_client=None) -> ManifestResult:
    """Generate a file manifest from technical specifications."""
    if llm_client is None:
        return ManifestResult(
            success=False,
            message="No LLM client provided",
            files_to_create=[]
        )
    
    try:
        # Build prompt
        prompt = build_manifest_prompt(context)
        
        # Get LLM response
        logger.info(f"Generating manifest for {context.project_type.value} project")
        response_str = llm_client.invoke(prompt)
        
        # Parse JSON response
        manifest_data = json.loads(response_str)
        
        # Validate response structure
        if "files_to_create" not in manifest_data or not isinstance(manifest_data["files_to_create"], list):
            raise ValueError("LLM response missing 'files_to_create' list")
        
        files_to_create = manifest_data["files_to_create"]
        
        # Validate manifest
        is_valid, validation_errors = validate_manifest(files_to_create, context.project_type, context.target_language)
        if not is_valid:
            logger.warning(f"Manifest validation issues: {validation_errors}")
        
        # Generate additional analysis
        directory_structure = generate_directory_structure(files_to_create)
        file_details = analyze_file_dependencies(files_to_create)
        
        logger.info(f"Successfully generated manifest with {len(files_to_create)} files")
        
        return ManifestResult(
            success=True,
            message=f"Successfully generated manifest with {len(files_to_create)} files",
            files_to_create=files_to_create,
            file_details=file_details,
            directory_structure=directory_structure
        )
        
    except json.JSONDecodeError as e:
        error_msg = f"Failed to parse LLM response as JSON: {e}"
        logger.error(error_msg)
        return ManifestResult(
            success=False,
            message=error_msg,
            files_to_create=[]
        )
        
    except ValueError as e:
        error_msg = f"Invalid manifest format: {e}"
        logger.error(error_msg)
        return ManifestResult(
            success=False,
            message=error_msg,
            files_to_create=[]
        )
        
    except Exception as e:
        error_msg = f"Unexpected error generating manifest: {e}"
        logger.error(error_msg)
        return ManifestResult(
            success=False,
            message=error_msg,
            files_to_create=[]
        )

def optimize_manifest(files: List[str], remove_duplicates: bool = True, sort_files: bool = True) -> List[str]:
    """Optimize a manifest by removing duplicates and sorting."""
    optimized_files = files.copy()
    
    if remove_duplicates:
        optimized_files = list(set(optimized_files))
    
    if sort_files:
        # Sort by directory depth first, then alphabetically
        optimized_files.sort(key=lambda x: (x.count('/'), x))
    
    return optimized_files