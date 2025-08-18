# agents/requirements_manager.py
import os
import logging
from typing import Dict, Any, List

from .base import BaseAgent
from core.context import GlobalContext
from core.models import AgentResponse, TaskNode, AgentResult

logger = logging.getLogger(__name__)

class RequirementsManagerAgent(BaseAgent):
    """
    Agent responsible for analyzing code dependencies and managing requirements.txt files.
    
    This agent:
    1. Scans Python code for import statements
    2. Maps import names to pip package names
    3. Creates or updates requirements.txt with proper dependencies
    4. Excludes standard library modules
    """
    
    def __init__(self):
        super().__init__(
            name="RequirementsManagerAgent",
            description="Analyzes code imports and manages Python dependencies in requirements.txt"
        )
    
    def required_inputs(self) -> List[str]:
        return ["code_files"]
    
    def optional_inputs(self) -> List[str]:
        return ["output_directory", "requirements_file_path"]
    
    def execute(self, goal: str, context: GlobalContext, current_task: TaskNode) -> AgentResponse:
        """Legacy execute method - tries to infer inputs from context."""
        # Try to get code files from artifacts or workspace
        code_files = {}
        
        # Check for recent artifacts that might contain code
        for artifact_key in context.list_artifacts():
            if artifact_key.endswith('.py') or 'code' in artifact_key.lower():
                content = context.get_artifact(artifact_key)
                if content and isinstance(content, str):
                    code_files[artifact_key] = content
        
        if not code_files:
            return AgentResponse(
                success=False, 
                message="No Python code files found in context to analyze for dependencies"
            )
        
        inputs = {
            'code_files': code_files,
            'output_directory': context.workspace_path
        }
        
        result = self.execute_v2(goal, inputs, context)
        
        return AgentResponse(
            success=result.success,
            message=result.message,
            artifacts_generated=[result.outputs.get('requirements_file')] if result.outputs.get('requirements_file') else []
        )
    
    def execute_v2(self, goal: str, inputs: Dict[str, Any], global_context: GlobalContext) -> AgentResult:
        """Analyze code and manage requirements.txt"""
        self.validate_inputs(inputs)
        
        code_files = inputs['code_files']
        output_dir = inputs.get('output_directory', global_context.workspace_path)
        req_file_path = inputs.get('requirements_file_path', os.path.join(output_dir, 'requirements.txt'))
        
        self.report_progress("Analyzing dependencies", f"Scanning {len(code_files)} code files")
        
        try:
            # Extract imports from all code files
            imports = self._extract_imports(code_files)
            self.report_intermediate_output("extracted_imports", imports)
            
            # Map imports to package names
            dependencies = self._map_imports_to_packages(imports)
            
            if dependencies:
                # Update requirements file
                self._update_requirements_file(req_file_path, dependencies, global_context)
                
                self.report_intermediate_output("dependencies_added", dependencies)
                
                return self.create_result(
                    True, 
                    f"Updated requirements.txt with {len(dependencies)} dependencies",
                    {
                        'requirements_file': req_file_path,
                        'dependencies': dependencies,
                        'imports_found': imports
                    }
                )
            else:
                return self.create_result(
                    True,
                    "No external dependencies found - code uses standard library only",
                    {
                        'requirements_file': None,
                        'dependencies': [],
                        'imports_found': imports
                    }
                )
                
        except Exception as e:
            error_msg = f"Requirements analysis failed: {e}"
            logger.error(error_msg, exc_info=True)
            return self.create_result(False, error_msg, error_details={'exception': str(e)})
    
    def _extract_imports(self, code_files: Dict[str, str]) -> List[str]:
        """Extract import statements from code files."""
        imports = set()
        
        for file_path, content in code_files.items():
            self.report_thinking(f"Analyzing imports in {file_path}")
            
            lines = content.split('\n')
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                
                # Handle "import module" statements
                if line.startswith('import '):
                    import_part = line[7:].strip()  # Remove "import "
                    
                    # Handle multiple imports: import a, b, c
                    if ',' in import_part:
                        modules = [m.strip().split('.')[0].split(' as ')[0] for m in import_part.split(',')]
                        imports.update(modules)
                    else:
                        # Single import: import module or import module.submodule
                        module = import_part.split('.')[0].split(' as ')[0]
                        imports.add(module)
                
                # Handle "from module import ..." statements  
                elif line.startswith('from ') and ' import ' in line:
                    from_part = line.split(' import ')[0][5:].strip()  # Remove "from "
                    module = from_part.split('.')[0]
                    imports.add(module)
        
        # Filter out empty strings and sort
        return sorted([imp for imp in imports if imp])
    
    def _map_imports_to_packages(self, imports: List[str]) -> List[str]:
        """Map import names to pip package names."""
        # Standard library modules to exclude (Python 3.7+)
        stdlib_modules = {
            # Core modules
            'os', 'sys', 'json', 'time', 'datetime', 'logging', 'pathlib',
            'subprocess', 'collections', 'itertools', 'functools', 'typing',
            'enum', 'abc', 'io', 'csv', 'sqlite3', 'uuid', 'hashlib',
            'base64', 'urllib', 'http', 'email', 'html', 'xml', 'concurrent',
            're', 'math', 'random', 'string', 'textwrap', 'tempfile',
            'shutil', 'glob', 'pickle', 'copy', 'operator', 'weakref',
            'gc', 'contextlib', 'warnings', 'inspect', 'dis', 'ast',
            'keyword', 'tokenize', 'token', 'symbol', 'parser', 'pprint',
            'reprlib', 'unittest', 'doctest', 'argparse', 'getopt',
            'configparser', 'fileinput', 'linecache', 'traceback',
            'platform', 'locale', 'gettext', 'codecs', 'unicodedata',
            'statistics', 'decimal', 'fractions', 'struct', 'array',
            'heapq', 'bisect', 'queue', 'sched', 'threading', 'multiprocessing',
            '_thread', 'socket', 'ssl', 'select', 'selectors', 'signal',
            'mmap', 'ctypes', 'getpass', 'pwd', 'grp', 'termios', 'tty',
            'pty', 'fcntl', 'pipes', 'resource', 'syslog', 'optparse'
        }
        
        # Common mappings of import names to package names
        package_mapping = {
            # Popular packages
            'requests': 'requests',
            'numpy': 'numpy', 
            'pandas': 'pandas',
            'matplotlib': 'matplotlib',
            'seaborn': 'seaborn',
            'scipy': 'scipy',
            'sklearn': 'scikit-learn',
            'cv2': 'opencv-python',
            'PIL': 'Pillow',
            'yaml': 'PyYAML',
            'toml': 'toml',
            'jsonschema': 'jsonschema',
            
            # Web frameworks
            'flask': 'Flask',
            'django': 'Django',
            'fastapi': 'fastapi',
            'tornado': 'tornado',
            'bottle': 'bottle',
            'pyramid': 'pyramid',
            
            # Database
            'sqlalchemy': 'SQLAlchemy',
            'psycopg2': 'psycopg2-binary',
            'pymongo': 'pymongo',
            'redis': 'redis',
            
            # Data & parsing
            'bs4': 'beautifulsoup4',
            'lxml': 'lxml',
            'xlrd': 'xlrd',
            'openpyxl': 'openpyxl',
            'tabulate': 'tabulate',
            
            # CLI & utilities
            'click': 'click',
            'colorama': 'colorama',
            'tqdm': 'tqdm',
            'rich': 'rich',
            
            # Testing
            'pytest': 'pytest',
            'nose': 'nose',
            'mock': 'mock',
            
            # Development
            'pydantic': 'pydantic',
            'dataclasses': 'dataclasses',  # backport for Python < 3.7
            'mypy': 'mypy',
            'black': 'black',
            'flake8': 'flake8',
            
            # Async
            'aiohttp': 'aiohttp',
            'asyncio': None,  # Built-in since Python 3.4
            
            # Other
            'jinja2': 'Jinja2',
            'werkzeug': 'Werkzeug',
            'celery': 'celery',
            'gunicorn': 'gunicorn'
        }
        
        dependencies = []
        excluded_stdlib = []
        
        for imp in imports:
            if imp in stdlib_modules:
                excluded_stdlib.append(imp)
                continue
            
            # Check if we have a specific mapping
            if imp in package_mapping:
                package = package_mapping[imp]
                if package:  # Some mappings might be None for built-ins
                    dependencies.append(package)
            else:
                # For unknown imports, assume import name = package name
                # This works for many packages like 'boto3', 'tweepy', etc.
                dependencies.append(imp)
        
        if excluded_stdlib:
            self.report_thinking(f"Excluded {len(excluded_stdlib)} standard library imports: {excluded_stdlib[:5]}")
        
        return sorted(list(set(dependencies)))  # Remove duplicates and sort
    
    def _update_requirements_file(self, req_file: str, dependencies: List[str], context: GlobalContext):
        """Create or update requirements.txt file."""
        existing_reqs = set()
        
        # Read existing requirements if file exists
        try:
            existing_content = context.workspace.get_file_content(req_file)
            if existing_content:
                for line in existing_content.split('\n'):
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # Extract package name (remove version constraints)
                        pkg_name = line.split('==')[0].split('>=')[0].split('<=')[0].split('>')[0].split('<')[0].split('~')[0].strip()
                        existing_reqs.add(pkg_name)
        except:
            # File doesn't exist or can't be read
            pass
        
        # Find new dependencies
        new_deps = [dep for dep in dependencies if dep not in existing_reqs]
        
        if new_deps or not existing_reqs:
            # Create requirements content
            content_lines = []
            
            # Add header comment for new file
            if not existing_reqs:
                content_lines.append("# Auto-generated requirements.txt")
                content_lines.append("# Dependencies extracted from Python code")
                content_lines.append("")
            
            # Add existing requirements first
            if existing_reqs:
                try:
                    existing_content = context.workspace.get_file_content(req_file)
                    for line in existing_content.split('\n'):
                        line = line.strip()
                        if line:
                            content_lines.append(line)
                except:
                    pass
            
            # Add new dependencies
            if new_deps:
                if existing_reqs:
                    content_lines.append("")
                    content_lines.append("# New dependencies")
                
                for dep in new_deps:
                    content_lines.append(dep)
            
            # Write updated file
            content = '\n'.join(content_lines)
            context.workspace.write_file_content(req_file, content, "requirements_manager")
            
            logger.info(f"Updated {req_file} with {len(new_deps)} new dependencies: {new_deps}")
        else:
            logger.info("No new dependencies to add to requirements.txt")