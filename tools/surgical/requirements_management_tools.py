# tools/surgical/requirements_management_tools.py
"""
Requirements management tools for analyzing code dependencies and managing requirements files.
Extracted from RequirementsManagerAgent to provide atomic, reusable dependency management capabilities.
"""

import os
import logging
import re
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Set, Optional, Tuple

logger = logging.getLogger(__name__)

class DependencyType(Enum):
    """Types of dependencies."""
    RUNTIME = "runtime"
    DEVELOPMENT = "development"
    TEST = "test"
    BUILD = "build"
    OPTIONAL = "optional"

class RequirementsFormat(Enum):
    """Supported requirements file formats."""
    TXT = "requirements.txt"
    PIPFILE = "Pipfile"
    POETRY = "pyproject.toml"
    SETUP_PY = "setup.py"
    SETUP_CFG = "setup.cfg"

@dataclass
class DependencyInfo:
    """Information about a dependency."""
    name: str
    package_name: str
    version: Optional[str] = None
    dependency_type: DependencyType = DependencyType.RUNTIME
    source_files: List[str] = None
    is_standard_library: bool = False
    
    def __post_init__(self):
        if self.source_files is None:
            self.source_files = []

@dataclass
class RequirementsAnalysisContext:
    """Context for requirements analysis."""
    code_files: Dict[str, str]
    project_path: str
    requirements_file_path: Optional[str] = None
    include_dev_dependencies: bool = False
    exclude_standard_library: bool = True
    version_strategy: str = "latest"  # latest, pinned, compatible
    backup_existing: bool = True

@dataclass
class RequirementsAnalysisResult:
    """Result of requirements analysis."""
    success: bool
    message: str
    dependencies: List[DependencyInfo]
    requirements_file: Optional[str] = None
    imports_found: List[str] = None
    standard_library_imports: List[str] = None
    unknown_imports: List[str] = None
    
    def __post_init__(self):
        if self.imports_found is None:
            self.imports_found = []
        if self.standard_library_imports is None:
            self.standard_library_imports = []
        if self.unknown_imports is None:
            self.unknown_imports = []

# Comprehensive standard library modules mapping (Python 3.7+)
STANDARD_LIBRARY_MODULES = {
    # Built-in modules
    '__builtin__', '__main__', '__future__', 'builtins',
    
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
    'pty', 'fcntl', 'pipes', 'resource', 'syslog', 'optparse',
    'cmd', 'shlex', 'readline', 'rlcompleter',
    
    # Data processing
    'zipfile', 'tarfile', 'gzip', 'bz2', 'lzma', 'zlib',
    
    # Network and internet
    'ftplib', 'poplib', 'imaplib', 'nntplib', 'smtplib',
    'telnetlib', 'socketserver', 'xmlrpc',
    
    # Multimedia
    'audioop', 'aifc', 'sunau', 'wave', 'chunk', 'colorsys', 'imghdr', 'sndhdr',
    
    # Development tools
    'pdb', 'profile', 'pstats', 'timeit', 'trace', 'cProfile',
    
    # Crypto
    'hmac', 'secrets',
    
    # System and OS interface
    'errno', 'stat', 'statfs', 'statvfs', 'filecmp', 'tempfile',
    
    # Python runtime services
    'atexit', 'copy', 'pickle', 'copyreg', 'shelve', 'marshal',
    'dbm', 'sqlite3', 'zoneinfo',
    
    # String services
    'string', 're', 'difflib', 'textwrap', 'unicodedata',
    'stringprep', 'readline', 'rlcompleter',
    
    # Internet data handling
    'base64', 'binascii', 'quopri', 'uu',
    
    # File processing
    'fileinput', 'stat', 'filecmp', 'tempfile', 'glob', 'fnmatch',
    'linecache', 'shutil',
    
    # Data persistence
    'pickle', 'copyreg', 'shelve', 'marshal', 'dbm', 'sqlite3',
    
    # Data compression and archiving
    'zlib', 'gzip', 'bz2', 'lzma', 'zipfile', 'tarfile',
    
    # Cryptographic services
    'hashlib', 'hmac', 'secrets',
    
    # Generic operating system services
    'os', 'io', 'time', 'argparse', 'getopt', 'logging', 'getpass',
    'curses', 'platform', 'errno', 'ctypes',
    
    # Concurrent execution
    'threading', 'multiprocessing', 'concurrent', 'subprocess',
    'sched', 'queue', '_thread',
    
    # Context variables (Python 3.7+)
    'contextvars',
    
    # Asyncio (Python 3.4+)
    'asyncio',
    
    # Type hints (Python 3.5+)
    'typing', 'typing_extensions',
    
    # Dataclasses (Python 3.7+, backport available)
    'dataclasses',
    
    # Import system
    'importlib', 'pkgutil', 'modulefinder', 'runpy',
    
    # Python language services
    'ast', 'symtable', 'symbol', 'token', 'keyword', 'tokenize',
    'tabnanny', 'pyclbr', 'py_compile', 'compileall', 'dis',
    'pickletools',
    
    # MS Windows specific
    'msilib', 'msvcrt', 'winreg', 'winsound',
    
    # Unix specific
    'posix', 'pwd', 'spwd', 'grp', 'crypt', 'termios', 'tty',
    'pty', 'fcntl', 'pipes', 'resource', 'nis', 'syslog'
}

# Common mappings of import names to package names
PACKAGE_MAPPINGS = {
    # Data science and machine learning
    'numpy': 'numpy',
    'np': 'numpy',
    'pandas': 'pandas',
    'pd': 'pandas',
    'matplotlib': 'matplotlib',
    'plt': 'matplotlib',
    'seaborn': 'seaborn',
    'sns': 'seaborn',
    'scipy': 'scipy',
    'sklearn': 'scikit-learn',
    'cv2': 'opencv-python',
    'PIL': 'Pillow',
    'torch': 'torch',
    'tensorflow': 'tensorflow',
    'tf': 'tensorflow',
    'keras': 'keras',
    
    # Web frameworks
    'flask': 'Flask',
    'django': 'Django',
    'fastapi': 'fastapi',
    'tornado': 'tornado',
    'bottle': 'bottle',
    'pyramid': 'pyramid',
    'quart': 'Quart',
    'sanic': 'sanic',
    'starlette': 'starlette',
    'uvicorn': 'uvicorn',
    'gunicorn': 'gunicorn',
    
    # Database
    'sqlalchemy': 'SQLAlchemy',
    'psycopg2': 'psycopg2-binary',
    'pymongo': 'pymongo',
    'redis': 'redis',
    'mysql': 'mysql-connector-python',
    'pymysql': 'PyMySQL',
    'sqlite3': None,  # Built-in
    'peewee': 'peewee',
    'mongoengine': 'mongoengine',
    'cassandra': 'cassandra-driver',
    'elasticsearch': 'elasticsearch',
    
    # Data parsing and formats
    'bs4': 'beautifulsoup4',
    'lxml': 'lxml',
    'xlrd': 'xlrd',
    'openpyxl': 'openpyxl',
    'tabulate': 'tabulate',
    'yaml': 'PyYAML',
    'toml': 'toml',
    'tomli': 'tomli',
    'tomllib': None,  # Built-in Python 3.11+
    'jsonschema': 'jsonschema',
    'marshmallow': 'marshmallow',
    'pydantic': 'pydantic',
    'cerberus': 'Cerberus',
    'voluptuous': 'voluptuous',
    
    # HTTP and networking
    'requests': 'requests',
    'urllib3': 'urllib3',
    'httpx': 'httpx',
    'aiohttp': 'aiohttp',
    'websockets': 'websockets',
    'paramiko': 'paramiko',
    'fabric': 'Fabric3',
    'twisted': 'Twisted',
    'gevent': 'gevent',
    'eventlet': 'eventlet',
    
    # CLI and utilities
    'click': 'click',
    'colorama': 'colorama',
    'termcolor': 'termcolor',
    'tqdm': 'tqdm',
    'rich': 'rich',
    'typer': 'typer',
    'fire': 'fire',
    'docopt': 'docopt',
    'prompt_toolkit': 'prompt-toolkit',
    
    # Testing
    'pytest': 'pytest',
    'nose': 'nose',
    'mock': 'mock',
    'unittest': None,  # Built-in
    'coverage': 'coverage',
    'hypothesis': 'hypothesis',
    'factory_boy': 'factory-boy',
    'faker': 'Faker',
    'responses': 'responses',
    'httpretty': 'httpretty',
    'vcr': 'vcrpy',
    
    # Development tools
    'mypy': 'mypy',
    'black': 'black',
    'flake8': 'flake8',
    'pylint': 'pylint',
    'autopep8': 'autopep8',
    'isort': 'isort',
    'pre_commit': 'pre-commit',
    'bandit': 'bandit',
    'safety': 'safety',
    'pyflakes': 'pyflakes',
    'mccabe': 'mccabe',
    'pycodestyle': 'pycodestyle',
    'pydocstyle': 'pydocstyle',
    
    # Async
    'asyncio': None,  # Built-in since Python 3.4
    'aiofiles': 'aiofiles',
    'aioredis': 'aioredis',
    'asyncpg': 'asyncpg',
    'aiodns': 'aiodns',
    'aiosmtplib': 'aiosmtplib',
    
    # Configuration
    'configparser': None,  # Built-in
    'python_dotenv': 'python-dotenv',
    'dotenv': 'python-dotenv',
    'dynaconf': 'dynaconf',
    'hydra': 'hydra-core',
    
    # Serialization
    'pickle': None,  # Built-in
    'dill': 'dill',
    'joblib': 'joblib',
    'cloudpickle': 'cloudpickle',
    
    # Templating
    'jinja2': 'Jinja2',
    'mako': 'Mako',
    'chameleon': 'Chameleon',
    
    # Other popular packages
    'werkzeug': 'Werkzeug',
    'celery': 'celery',
    'boto3': 'boto3',
    'botocore': 'botocore',
    's3fs': 's3fs',
    'tweepy': 'tweepy',
    'matplotlib': 'matplotlib',
    'plotly': 'plotly',
    'bokeh': 'bokeh',
    'dash': 'dash',
    'streamlit': 'streamlit',
    'jupyterlab': 'jupyterlab',
    'notebook': 'notebook',
    'ipython': 'ipython',
    'jupyter': 'jupyter',
    'nbconvert': 'nbconvert',
    'papermill': 'papermill',
    
    # Crypto and security
    'cryptography': 'cryptography',
    'pyotp': 'pyotp',
    'passlib': 'passlib',
    'bcrypt': 'bcrypt',
    'jwt': 'PyJWT',
    'pyjwt': 'PyJWT',
    'authlib': 'Authlib',
    
    # Date and time
    'dateutil': 'python-dateutil',
    'arrow': 'arrow',
    'pendulum': 'pendulum',
    'pytz': 'pytz',
    'freezegun': 'freezegun',
    
    # Logging
    'structlog': 'structlog',
    'loguru': 'loguru',
    'sentry_sdk': 'sentry-sdk',
    
    # Image processing
    'imageio': 'imageio',
    'skimage': 'scikit-image',
    'wand': 'Wand',
    
    # GUI
    'tkinter': None,  # Built-in
    'PyQt5': 'PyQt5',
    'PyQt6': 'PyQt6',
    'PySide2': 'PySide2',
    'PySide6': 'PySide6',
    'kivy': 'Kivy',
    'wxpython': 'wxPython',
    
    # Game development
    'pygame': 'pygame',
    'pyglet': 'pyglet',
    'arcade': 'arcade',
    
    # Scientific computing
    'sympy': 'sympy',
    'networkx': 'networkx',
    'igraph': 'python-igraph',
    'sage': 'sage',
    
    # Geospatial
    'geopandas': 'geopandas',
    'shapely': 'Shapely',
    'fiona': 'Fiona',
    'folium': 'folium',
    'cartopy': 'Cartopy',
}

def extract_imports_from_code(code_files: Dict[str, str]) -> List[str]:
    """Extract import statements from code files."""
    imports = set()
    
    for file_path, content in code_files.items():
        logger.debug(f"Analyzing imports in {file_path}")
        file_imports = extract_imports_from_content(content)
        imports.update(file_imports)
    
    # Filter out empty strings and sort
    return sorted([imp for imp in imports if imp])

def extract_imports_from_content(content: str) -> Set[str]:
    """Extract imports from a single file's content."""
    imports = set()
    lines = content.split('\n')
    
    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        
        # Skip comments and docstrings
        if line.startswith('#') or line.startswith('"""') or line.startswith("'''"):
            continue
        
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
            
            # Handle relative imports
            if from_part.startswith('.'):
                continue  # Skip relative imports
            
            module = from_part.split('.')[0]
            imports.add(module)
        
        # Handle try/except import blocks
        elif 'import ' in line and ('try:' in line or 'except' in line):
            # Extract imports from try/except blocks
            import_matches = re.findall(r'import\s+([a-zA-Z_][a-zA-Z0-9_]*)', line)
            imports.update(import_matches)
    
    return imports

def analyze_dependencies(context: RequirementsAnalysisContext) -> RequirementsAnalysisResult:
    """Analyze code dependencies and create requirements."""
    try:
        logger.info(f"Analyzing dependencies from {len(context.code_files)} files")
        
        # Extract imports from all code files
        imports = extract_imports_from_code(context.code_files)
        logger.info(f"Found {len(imports)} unique imports")
        
        # Categorize imports
        dependencies = []
        standard_library_imports = []
        unknown_imports = []
        
        for import_name in imports:
            dep_info = categorize_import(import_name)
            
            if dep_info.is_standard_library:
                if not context.exclude_standard_library:
                    dependencies.append(dep_info)
                standard_library_imports.append(import_name)
            elif dep_info.package_name:
                dependencies.append(dep_info)
            else:
                unknown_imports.append(import_name)
                # Still add as potential dependency
                dependencies.append(dep_info)
        
        # Update requirements file if specified
        requirements_file = None
        if context.requirements_file_path:
            requirements_file = update_requirements_file(
                context.requirements_file_path,
                dependencies,
                context.project_path,
                context.backup_existing,
                context.version_strategy
            )
        
        logger.info(f"Analysis complete: {len(dependencies)} dependencies, {len(standard_library_imports)} stdlib, {len(unknown_imports)} unknown")
        
        return RequirementsAnalysisResult(
            success=True,
            message=f"Successfully analyzed dependencies: {len(dependencies)} found",
            dependencies=dependencies,
            requirements_file=requirements_file,
            imports_found=imports,
            standard_library_imports=standard_library_imports,
            unknown_imports=unknown_imports
        )
        
    except Exception as e:
        error_msg = f"Requirements analysis failed: {e}"
        logger.error(error_msg, exc_info=True)
        return RequirementsAnalysisResult(
            success=False,
            message=error_msg,
            dependencies=[]
        )

def categorize_import(import_name: str) -> DependencyInfo:
    """Categorize an import and return dependency information."""
    # Check if it's a standard library module
    if import_name in STANDARD_LIBRARY_MODULES:
        return DependencyInfo(
            name=import_name,
            package_name="",
            is_standard_library=True
        )
    
    # Check if we have a specific mapping
    if import_name in PACKAGE_MAPPINGS:
        package_name = PACKAGE_MAPPINGS[import_name]
        if package_name is None:  # Built-in module
            return DependencyInfo(
                name=import_name,
                package_name="",
                is_standard_library=True
            )
        else:
            return DependencyInfo(
                name=import_name,
                package_name=package_name,
                is_standard_library=False
            )
    
    # For unknown imports, assume import name = package name
    return DependencyInfo(
        name=import_name,
        package_name=import_name,
        is_standard_library=False
    )

def update_requirements_file(
    requirements_file_path: str,
    dependencies: List[DependencyInfo],
    project_path: str,
    backup_existing: bool = True,
    version_strategy: str = "latest"
) -> str:
    """Update or create requirements file with dependencies."""
    req_path = Path(project_path) / requirements_file_path
    existing_reqs = set()
    
    # Read existing requirements if file exists
    if req_path.exists():
        if backup_existing:
            backup_path = req_path.with_suffix(req_path.suffix + '.backup')
            backup_path.write_text(req_path.read_text())
            logger.info(f"Backed up existing requirements to {backup_path}")
        
        try:
            existing_content = req_path.read_text()
            for line in existing_content.split('\n'):
                line = line.strip()
                if line and not line.startswith('#'):
                    # Extract package name (remove version constraints)
                    pkg_name = re.split(r'[<>=!~]', line)[0].strip()
                    existing_reqs.add(pkg_name)
        except Exception as e:
            logger.warning(f"Failed to read existing requirements: {e}")
    
    # Find new dependencies (excluding standard library)
    new_deps = []
    for dep in dependencies:
        if not dep.is_standard_library and dep.package_name and dep.package_name not in existing_reqs:
            new_deps.append(dep)
    
    if new_deps or not existing_reqs:
        # Create requirements content
        content_lines = []
        
        # Add header comment for new file
        if not existing_reqs:
            content_lines.extend([
                "# Auto-generated requirements.txt",
                "# Dependencies extracted from Python code",
                ""
            ])
        
        # Add existing requirements first
        if existing_reqs and req_path.exists():
            try:
                existing_content = req_path.read_text()
                for line in existing_content.split('\n'):
                    line = line.strip()
                    if line:
                        content_lines.append(line)
            except Exception as e:
                logger.warning(f"Failed to preserve existing requirements: {e}")
        
        # Add new dependencies
        if new_deps:
            if existing_reqs:
                content_lines.extend(["", "# New dependencies"])
            
            for dep in sorted(new_deps, key=lambda x: x.package_name):
                if version_strategy == "pinned":
                    # Would need to fetch current version - simplified here
                    content_lines.append(f"{dep.package_name}>=0.0.0")
                elif version_strategy == "compatible":
                    content_lines.append(f"{dep.package_name}~=1.0")
                else:  # latest
                    content_lines.append(dep.package_name)
        
        # Write updated file
        req_path.parent.mkdir(parents=True, exist_ok=True)
        req_path.write_text('\n'.join(content_lines))
        
        logger.info(f"Updated {req_path} with {len(new_deps)} new dependencies: {[d.package_name for d in new_deps]}")
    else:
        logger.info("No new dependencies to add to requirements file")
    
    return str(req_path)

def detect_requirements_format(project_path: str) -> Optional[RequirementsFormat]:
    """Detect the requirements format used in the project."""
    project = Path(project_path)
    
    # Check for different format files in order of preference
    format_files = [
        (RequirementsFormat.POETRY, "pyproject.toml"),
        (RequirementsFormat.PIPFILE, "Pipfile"),
        (RequirementsFormat.TXT, "requirements.txt"),
        (RequirementsFormat.SETUP_CFG, "setup.cfg"),
        (RequirementsFormat.SETUP_PY, "setup.py")
    ]
    
    for req_format, filename in format_files:
        if (project / filename).exists():
            return req_format
    
    return None

def generate_requirements_for_format(
    dependencies: List[DependencyInfo],
    req_format: RequirementsFormat,
    version_strategy: str = "latest"
) -> str:
    """Generate requirements content for a specific format."""
    if req_format == RequirementsFormat.TXT:
        return _generate_requirements_txt(dependencies, version_strategy)
    elif req_format == RequirementsFormat.PIPFILE:
        return _generate_pipfile(dependencies, version_strategy)
    elif req_format == RequirementsFormat.POETRY:
        return _generate_poetry_dependencies(dependencies, version_strategy)
    elif req_format == RequirementsFormat.SETUP_PY:
        return _generate_setup_py_dependencies(dependencies, version_strategy)
    elif req_format == RequirementsFormat.SETUP_CFG:
        return _generate_setup_cfg_dependencies(dependencies, version_strategy)
    else:
        raise ValueError(f"Unsupported requirements format: {req_format}")

def _generate_requirements_txt(dependencies: List[DependencyInfo], version_strategy: str) -> str:
    """Generate requirements.txt format."""
    lines = ["# Auto-generated requirements.txt", ""]
    
    runtime_deps = [d for d in dependencies if not d.is_standard_library and d.dependency_type == DependencyType.RUNTIME]
    dev_deps = [d for d in dependencies if not d.is_standard_library and d.dependency_type == DependencyType.DEVELOPMENT]
    
    if runtime_deps:
        lines.append("# Runtime dependencies")
        for dep in sorted(runtime_deps, key=lambda x: x.package_name):
            lines.append(_format_dependency(dep, version_strategy))
        lines.append("")
    
    if dev_deps:
        lines.append("# Development dependencies")
        for dep in sorted(dev_deps, key=lambda x: x.package_name):
            lines.append(_format_dependency(dep, version_strategy))
    
    return '\n'.join(lines)

def _generate_pipfile(dependencies: List[DependencyInfo], version_strategy: str) -> str:
    """Generate Pipfile format."""
    lines = [
        "[[source]]",
        "url = \"https://pypi.org/simple\"",
        "verify_ssl = true",
        "name = \"pypi\"",
        "",
        "[dev-packages]",
    ]
    
    dev_deps = [d for d in dependencies if not d.is_standard_library and d.dependency_type == DependencyType.DEVELOPMENT]
    for dep in sorted(dev_deps, key=lambda x: x.package_name):
        lines.append(f"{dep.package_name} = \"*\"")
    
    lines.extend(["", "[packages]"])
    
    runtime_deps = [d for d in dependencies if not d.is_standard_library and d.dependency_type == DependencyType.RUNTIME]
    for dep in sorted(runtime_deps, key=lambda x: x.package_name):
        lines.append(f"{dep.package_name} = \"*\"")
    
    lines.extend(["", "[requires]", "python_version = \"3.8\""])
    
    return '\n'.join(lines)

def _generate_poetry_dependencies(dependencies: List[DependencyInfo], version_strategy: str) -> str:
    """Generate pyproject.toml dependencies section for Poetry."""
    lines = ["[tool.poetry.dependencies]", "python = \"^3.8\""]
    
    runtime_deps = [d for d in dependencies if not d.is_standard_library and d.dependency_type == DependencyType.RUNTIME]
    for dep in sorted(runtime_deps, key=lambda x: x.package_name):
        lines.append(f"{dep.package_name} = \"*\"")
    
    lines.extend(["", "[tool.poetry.dev-dependencies]"])
    
    dev_deps = [d for d in dependencies if not d.is_standard_library and d.dependency_type == DependencyType.DEVELOPMENT]
    for dep in sorted(dev_deps, key=lambda x: x.package_name):
        lines.append(f"{dep.package_name} = \"*\"")
    
    return '\n'.join(lines)

def _generate_setup_py_dependencies(dependencies: List[DependencyInfo], version_strategy: str) -> str:
    """Generate setup.py install_requires section."""
    runtime_deps = [d for d in dependencies if not d.is_standard_library and d.dependency_type == DependencyType.RUNTIME]
    deps_list = [f'"{_format_dependency(dep, version_strategy)}"' for dep in sorted(runtime_deps, key=lambda x: x.package_name)]
    
    return f"install_requires=[\n    {',\\n    '.join(deps_list)}\n]"

def _generate_setup_cfg_dependencies(dependencies: List[DependencyInfo], version_strategy: str) -> str:
    """Generate setup.cfg install_requires section."""
    runtime_deps = [d for d in dependencies if not d.is_standard_library and d.dependency_type == DependencyType.RUNTIME]
    deps_list = [_format_dependency(dep, version_strategy) for dep in sorted(runtime_deps, key=lambda x: x.package_name)]
    
    return f"install_requires =\n    " + '\n    '.join(deps_list)

def _format_dependency(dep: DependencyInfo, version_strategy: str) -> str:
    """Format a dependency according to version strategy."""
    if version_strategy == "pinned" and dep.version:
        return f"{dep.package_name}=={dep.version}"
    elif version_strategy == "compatible" and dep.version:
        # Use compatible release operator
        return f"{dep.package_name}~={dep.version}"
    else:
        return dep.package_name

def get_package_version(package_name: str) -> Optional[str]:
    """Get the installed version of a package."""
    try:
        import pkg_resources
        return pkg_resources.get_distribution(package_name).version
    except Exception:
        return None

def validate_requirements_file(requirements_file_path: str) -> Tuple[bool, List[str]]:
    """Validate a requirements file for common issues."""
    errors = []
    req_path = Path(requirements_file_path)
    
    if not req_path.exists():
        errors.append(f"Requirements file does not exist: {requirements_file_path}")
        return False, errors
    
    try:
        content = req_path.read_text()
        lines = content.strip().split('\n')
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue
            
            # Check for common issues
            if line.startswith('-'):
                if not line.startswith('-r ') and not line.startswith('--'):
                    errors.append(f"Line {line_num}: Invalid option format: {line}")
            else:
                # Check package name format
                if not re.match(r'^[a-zA-Z0-9\-_\.]+([<>=!~].*)?$', line):
                    errors.append(f"Line {line_num}: Invalid package format: {line}")
        
        is_valid = len(errors) == 0
        return is_valid, errors
        
    except Exception as e:
        errors.append(f"Failed to read requirements file: {e}")
        return False, errors