# tools/surgical/configuration_management_tools.py
"""
Configuration management tools for precise configuration file modifications.
Extracted from ConfigurationModifierAgent to provide atomic, reusable configuration management capabilities.
"""

import json
import logging
import shutil
from configparser import ConfigParser

# Optional dependencies
try:
    import toml
    HAS_TOML = True
except ImportError:
    HAS_TOML = False

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple

logger = logging.getLogger(__name__)

class ConfigFormat(Enum):
    """Supported configuration file formats."""
    JSON = "json"
    YAML = "yaml"
    TOML = "toml"
    INI = "ini"
    CFG = "cfg"
    CONF = "conf"
    PROPERTIES = "properties"
    ENV = "env"

class ConfigOperation(Enum):
    """Types of configuration operations."""
    CREATE = "create"
    READ = "read"
    UPDATE = "update"
    DELETE = "delete"
    BACKUP = "backup"
    RESTORE = "restore"
    VALIDATE = "validate"
    MERGE = "merge"

class MergeStrategy(Enum):
    """Strategies for merging configuration data."""
    OVERWRITE = "overwrite"
    UPDATE = "update"
    MERGE_DEEP = "merge_deep"
    APPEND = "append"
    PRESERVE_EXISTING = "preserve_existing"

@dataclass
class ConfigTemplate:
    """Template for creating configuration files."""
    name: str
    format: ConfigFormat
    content: Dict[str, Any]
    description: str = ""

@dataclass
class ConfigValidationResult:
    """Result of configuration validation."""
    is_valid: bool
    file_format: Optional[ConfigFormat] = None
    error: Optional[str] = None
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []

@dataclass
class ConfigOperationContext:
    """Context for configuration operations."""
    config_file: Path
    backup_enabled: bool = True
    validate_syntax: bool = True
    merge_strategy: MergeStrategy = MergeStrategy.UPDATE
    create_directories: bool = True
    encoding: str = "utf-8"

@dataclass
class ConfigOperationResult:
    """Result of a configuration operation."""
    success: bool
    message: str
    config_file: str
    operation: ConfigOperation
    file_format: Optional[ConfigFormat] = None
    backup_created: bool = False
    validation_result: Optional[ConfigValidationResult] = None
    content: Optional[Dict[str, Any]] = None

# Configuration templates for common tools
CONFIGURATION_TEMPLATES = {
    ConfigFormat.JSON: {
        "basic": {
            "name": "basic_json",
            "format": ConfigFormat.JSON,
            "content": {
                "version": "1.0",
                "settings": {}
            },
            "description": "Basic JSON configuration"
        },
        "package_json": {
            "name": "package_json",
            "format": ConfigFormat.JSON,
            "content": {
                "name": "project",
                "version": "1.0.0",
                "description": "",
                "main": "index.js",
                "scripts": {
                    "test": "echo \"Error: no test specified\" && exit 1"
                },
                "keywords": [],
                "author": "",
                "license": "ISC"
            },
            "description": "Node.js package.json template"
        }
    },
    ConfigFormat.YAML: {
        "basic": {
            "name": "basic_yaml",
            "format": ConfigFormat.YAML,
            "content": {
                "version": "1.0",
                "settings": {}
            },
            "description": "Basic YAML configuration"
        },
        "github_workflow": {
            "name": "github_workflow",
            "format": ConfigFormat.YAML,
            "content": {
                "name": "CI",
                "on": ["push", "pull_request"],
                "jobs": {
                    "test": {
                        "runs-on": "ubuntu-latest",
                        "steps": [
                            {"uses": "actions/checkout@v2"},
                            {"name": "Run tests", "run": "echo 'Add test command here'"}
                        ]
                    }
                }
            },
            "description": "GitHub Actions workflow template"
        }
    },
    ConfigFormat.TOML: {
        "basic": {
            "name": "basic_toml",
            "format": ConfigFormat.TOML,
            "content": {
                "title": "Configuration",
                "version": "1.0"
            },
            "description": "Basic TOML configuration"
        },
        "pyproject": {
            "name": "pyproject_toml",
            "format": ConfigFormat.TOML,
            "content": {
                "build-system": {
                    "requires": ["setuptools>=45", "wheel"],
                    "build-backend": "setuptools.build_meta"
                },
                "project": {
                    "name": "project",
                    "version": "0.1.0",
                    "description": "",
                    "authors": [{"name": "Author", "email": "author@example.com"}],
                    "dependencies": []
                },
                "tool": {
                    "black": {
                        "line-length": 88,
                        "target-version": ["py38"]
                    },
                    "isort": {
                        "profile": "black"
                    }
                }
            },
            "description": "Python pyproject.toml template"
        }
    },
    ConfigFormat.INI: {
        "basic": {
            "name": "basic_ini",
            "format": ConfigFormat.INI,
            "content": {
                "DEFAULT": {
                    "version": "1.0"
                },
                "settings": {}
            },
            "description": "Basic INI configuration"
        },
        "pytest_ini": {
            "name": "pytest_ini",
            "format": ConfigFormat.INI,
            "content": {
                "tool:pytest": {
                    "testpaths": "tests",
                    "python_files": "test_*.py",
                    "python_functions": "test_*",
                    "addopts": "-v --tb=short"
                }
            },
            "description": "pytest.ini configuration template"
        }
    }
}

def detect_config_format(config_path: Path) -> ConfigFormat:
    """Detect configuration file format based on extension and content."""
    suffix = config_path.suffix.lower()
    
    # Map file extensions to formats
    extension_map = {
        '.json': ConfigFormat.JSON,
        '.yaml': ConfigFormat.YAML,
        '.yml': ConfigFormat.YAML,
        '.toml': ConfigFormat.TOML,
        '.ini': ConfigFormat.INI,
        '.cfg': ConfigFormat.CFG,
        '.conf': ConfigFormat.CONF,
        '.properties': ConfigFormat.PROPERTIES,
        '.env': ConfigFormat.ENV
    }
    
    if suffix in extension_map:
        return extension_map[suffix]
    
    # Try to detect based on filename
    filename = config_path.name.lower()
    if filename in ['pyproject.toml', 'cargo.toml', 'poetry.toml']:
        return ConfigFormat.TOML
    elif filename in ['docker-compose.yml', '.github/workflows/*.yml']:
        return ConfigFormat.YAML
    elif filename in ['package.json', 'tsconfig.json']:
        return ConfigFormat.JSON
    
    # Default to JSON
    return ConfigFormat.JSON

def read_config_content(config_path: Path, file_format: Optional[ConfigFormat] = None) -> Dict[str, Any]:
    """Read configuration file content based on format."""
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file does not exist: {config_path}")
    
    if file_format is None:
        file_format = detect_config_format(config_path)
    
    try:
        content = config_path.read_text(encoding='utf-8')
        
        if file_format == ConfigFormat.JSON:
            return json.loads(content)
        elif file_format in [ConfigFormat.YAML]:
            if not HAS_YAML:
                raise ValueError("PyYAML is required for YAML format. Install with: pip install PyYAML")
            return yaml.safe_load(content) or {}
        elif file_format == ConfigFormat.TOML:
            if not HAS_TOML:
                raise ValueError("toml is required for TOML format. Install with: pip install toml")
            return toml.loads(content)
        elif file_format in [ConfigFormat.INI, ConfigFormat.CFG, ConfigFormat.CONF]:
            parser = ConfigParser()
            parser.read_string(content)
            return {section: dict(parser.items(section)) for section in parser.sections()}
        elif file_format == ConfigFormat.PROPERTIES:
            return _parse_properties(content)
        elif file_format == ConfigFormat.ENV:
            return _parse_env_file(content)
        else:
            raise ValueError(f"Unsupported configuration format: {file_format}")
            
    except Exception as e:
        raise ValueError(f"Failed to parse {file_format.value} configuration: {e}")

def write_config_content(config_path: Path, content: Dict[str, Any], file_format: Optional[ConfigFormat] = None):
    """Write configuration content to file based on format."""
    if file_format is None:
        file_format = detect_config_format(config_path)
    
    # Create parent directories if needed
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        if file_format == ConfigFormat.JSON:
            output = json.dumps(content, indent=2, ensure_ascii=False)
        elif file_format in [ConfigFormat.YAML]:
            if not HAS_YAML:
                raise ValueError("PyYAML is required for YAML format. Install with: pip install PyYAML")
            output = yaml.dump(content, default_flow_style=False, allow_unicode=True, sort_keys=False)
        elif file_format == ConfigFormat.TOML:
            if not HAS_TOML:
                raise ValueError("toml is required for TOML format. Install with: pip install toml")
            output = toml.dumps(content)
        elif file_format in [ConfigFormat.INI, ConfigFormat.CFG, ConfigFormat.CONF]:
            output = _format_ini_content(content)
        elif file_format == ConfigFormat.PROPERTIES:
            output = _format_properties(content)
        elif file_format == ConfigFormat.ENV:
            output = _format_env_file(content)
        else:
            raise ValueError(f"Unsupported configuration format: {file_format}")
        
        config_path.write_text(output, encoding='utf-8')
        logger.info(f"Successfully wrote {file_format.value} configuration to {config_path}")
        
    except Exception as e:
        raise ValueError(f"Failed to write {file_format.value} configuration: {e}")

def create_config_template(file_format: ConfigFormat, template_name: str = "basic") -> Dict[str, Any]:
    """Create configuration content from a template."""
    if file_format not in CONFIGURATION_TEMPLATES:
        raise ValueError(f"No templates available for format: {file_format}")
    
    templates = CONFIGURATION_TEMPLATES[file_format]
    if template_name not in templates:
        available = list(templates.keys())
        raise ValueError(f"Template '{template_name}' not found. Available: {available}")
    
    return templates[template_name]["content"].copy()

def merge_config_data(existing: Dict[str, Any], updates: Dict[str, Any], strategy: MergeStrategy) -> Dict[str, Any]:
    """Merge configuration data using the specified strategy."""
    if strategy == MergeStrategy.OVERWRITE:
        return updates.copy()
    elif strategy == MergeStrategy.UPDATE:
        result = existing.copy()
        result.update(updates)
        return result
    elif strategy == MergeStrategy.MERGE_DEEP:
        return _deep_merge(existing.copy(), updates)
    elif strategy == MergeStrategy.PRESERVE_EXISTING:
        result = updates.copy()
        result.update(existing)  # Existing values take precedence
        return result
    elif strategy == MergeStrategy.APPEND:
        # Only works for list values
        result = existing.copy()
        for key, value in updates.items():
            if key in result and isinstance(result[key], list) and isinstance(value, list):
                result[key].extend(value)
            else:
                result[key] = value
        return result
    else:
        raise ValueError(f"Unknown merge strategy: {strategy}")

def get_config_value(config: Dict[str, Any], section: Optional[str] = None, key: Optional[str] = None) -> Any:
    """Get a specific value from configuration."""
    if section is None and key is None:
        return config
    
    if section is not None and section not in config:
        raise KeyError(f"Section '{section}' not found in configuration")
    
    if section is not None:
        section_data = config[section]
        if key is None:
            return section_data
        elif isinstance(section_data, dict) and key in section_data:
            return section_data[key]
        else:
            raise KeyError(f"Key '{key}' not found in section '{section}'")
    
    # section is None, look for key in root
    if key in config:
        return config[key]
    else:
        raise KeyError(f"Key '{key}' not found in configuration root")

def set_config_value(config: Dict[str, Any], value: Any, section: Optional[str] = None, key: Optional[str] = None) -> Dict[str, Any]:
    """Set a specific value in configuration."""
    result = config.copy()
    
    if section is None and key is None:
        raise ValueError("Must specify either section or key")
    
    if section is not None:
        if section not in result:
            result[section] = {}
        
        if key is None:
            # Replace entire section
            result[section] = value
        else:
            # Set specific key in section
            if not isinstance(result[section], dict):
                result[section] = {}
            result[section][key] = value
    else:
        # Set key in root
        result[key] = value
    
    return result

def delete_config_value(config: Dict[str, Any], section: Optional[str] = None, key: Optional[str] = None) -> Dict[str, Any]:
    """Delete a specific value from configuration."""
    result = config.copy()
    
    if section is None and key is None:
        raise ValueError("Must specify either section or key")
    
    if section is not None:
        if section not in result:
            raise KeyError(f"Section '{section}' not found in configuration")
        
        if key is None:
            # Delete entire section
            del result[section]
        else:
            # Delete specific key from section
            if not isinstance(result[section], dict):
                raise KeyError(f"Section '{section}' is not a dictionary")
            if key not in result[section]:
                raise KeyError(f"Key '{key}' not found in section '{section}'")
            del result[section][key]
    else:
        # Delete key from root
        if key not in result:
            raise KeyError(f"Key '{key}' not found in configuration root")
        del result[key]
    
    return result

def validate_config_syntax(config_path: Path) -> ConfigValidationResult:
    """Validate configuration file syntax."""
    if not config_path.exists():
        return ConfigValidationResult(
            is_valid=False,
            error=f"Configuration file does not exist: {config_path}"
        )
    
    file_format = detect_config_format(config_path)
    
    try:
        # Try to read and parse the file
        read_config_content(config_path, file_format)
        
        return ConfigValidationResult(
            is_valid=True,
            file_format=file_format
        )
        
    except Exception as e:
        return ConfigValidationResult(
            is_valid=False,
            file_format=file_format,
            error=str(e)
        )

def backup_config(config_path: Path) -> Path:
    """Create a backup of the configuration file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file does not exist: {config_path}")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = config_path.with_suffix(f"{config_path.suffix}.backup_{timestamp}")
    
    shutil.copy2(config_path, backup_path)
    logger.info(f"Created backup: {backup_path}")
    
    return backup_path

def restore_config(config_path: Path, backup_path: Optional[Path] = None) -> bool:
    """Restore configuration from backup."""
    if backup_path is None:
        # Find most recent backup
        backup_pattern = config_path.with_suffix(f"{config_path.suffix}.backup_*")
        backup_files = list(config_path.parent.glob(backup_pattern.name))
        
        if not backup_files:
            return False
        
        # Sort by modification time and get most recent
        backup_path = max(backup_files, key=lambda p: p.stat().st_mtime)
    
    if not backup_path.exists():
        return False
    
    shutil.copy2(backup_path, config_path)
    logger.info(f"Restored configuration from backup: {backup_path}")
    
    return True

def create_configuration(
    config_path: Path,
    content: Optional[Dict[str, Any]] = None,
    template_name: str = "basic",
    backup_existing: bool = True,
    validate_syntax: bool = True
) -> ConfigOperationResult:
    """Create a new configuration file."""
    file_format = detect_config_format(config_path)
    
    try:
        # Use template if no content provided
        if content is None:
            content = create_config_template(file_format, template_name)
        
        # Backup if file exists and backup requested
        backup_created = False
        if config_path.exists() and backup_existing:
            try:
                backup_config(config_path)
                backup_created = True
            except Exception as e:
                logger.warning(f"Failed to backup existing config: {e}")
        
        # Write configuration
        write_config_content(config_path, content, file_format)
        
        # Validate syntax if requested
        validation_result = None
        if validate_syntax:
            validation_result = validate_config_syntax(config_path)
        
        return ConfigOperationResult(
            success=True,
            message=f"Successfully created configuration file: {config_path.name}",
            config_file=str(config_path),
            operation=ConfigOperation.CREATE,
            file_format=file_format,
            backup_created=backup_created,
            validation_result=validation_result,
            content=content
        )
        
    except Exception as e:
        return ConfigOperationResult(
            success=False,
            message=f"Failed to create configuration file: {e}",
            config_file=str(config_path),
            operation=ConfigOperation.CREATE
        )

def update_configuration(
    config_path: Path,
    updates: Optional[Dict[str, Any]] = None,
    section: Optional[str] = None,
    key: Optional[str] = None,
    value: Any = None,
    merge_strategy: MergeStrategy = MergeStrategy.UPDATE,
    backup_existing: bool = True,
    validate_syntax: bool = True
) -> ConfigOperationResult:
    """Update an existing configuration file."""
    if not config_path.exists():
        return ConfigOperationResult(
            success=False,
            message=f"Configuration file does not exist: {config_path}",
            config_file=str(config_path),
            operation=ConfigOperation.UPDATE
        )
    
    try:
        file_format = detect_config_format(config_path)
        
        # Backup if requested
        backup_created = False
        if backup_existing:
            try:
                backup_config(config_path)
                backup_created = True
            except Exception as e:
                logger.warning(f"Failed to backup config: {e}")
        
        # Read existing configuration
        existing_config = read_config_content(config_path, file_format)
        
        # Apply updates
        if updates is not None:
            # Update with full config data
            updated_config = merge_config_data(existing_config, updates, merge_strategy)
        elif section is not None or key is not None:
            # Update specific key/section
            if value is None:
                return ConfigOperationResult(
                    success=False,
                    message="Value must be provided when updating specific section/key",
                    config_file=str(config_path),
                    operation=ConfigOperation.UPDATE
                )
            updated_config = set_config_value(existing_config, value, section, key)
        else:
            return ConfigOperationResult(
                success=False,
                message="No update data provided (updates or section/key/value)",
                config_file=str(config_path),
                operation=ConfigOperation.UPDATE
            )
        
        # Write updated configuration
        write_config_content(config_path, updated_config, file_format)
        
        # Validate syntax if requested
        validation_result = None
        if validate_syntax:
            validation_result = validate_config_syntax(config_path)
        
        return ConfigOperationResult(
            success=True,
            message=f"Successfully updated configuration file: {config_path.name}",
            config_file=str(config_path),
            operation=ConfigOperation.UPDATE,
            file_format=file_format,
            backup_created=backup_created,
            validation_result=validation_result,
            content=updated_config
        )
        
    except Exception as e:
        return ConfigOperationResult(
            success=False,
            message=f"Failed to update configuration file: {e}",
            config_file=str(config_path),
            operation=ConfigOperation.UPDATE
        )

def read_configuration(
    config_path: Path,
    section: Optional[str] = None,
    key: Optional[str] = None
) -> ConfigOperationResult:
    """Read configuration file content."""
    if not config_path.exists():
        return ConfigOperationResult(
            success=False,
            message=f"Configuration file does not exist: {config_path}",
            config_file=str(config_path),
            operation=ConfigOperation.READ
        )
    
    try:
        file_format = detect_config_format(config_path)
        config_content = read_config_content(config_path, file_format)
        
        # Extract specific section/key if requested
        if section is not None or key is not None:
            try:
                final_content = get_config_value(config_content, section, key)
            except KeyError as e:
                return ConfigOperationResult(
                    success=False,
                    message=str(e),
                    config_file=str(config_path),
                    operation=ConfigOperation.READ
                )
        else:
            final_content = config_content
        
        return ConfigOperationResult(
            success=True,
            message=f"Successfully read configuration from: {config_path.name}",
            config_file=str(config_path),
            operation=ConfigOperation.READ,
            file_format=file_format,
            content=final_content
        )
        
    except Exception as e:
        return ConfigOperationResult(
            success=False,
            message=f"Failed to read configuration file: {e}",
            config_file=str(config_path),
            operation=ConfigOperation.READ
        )

# Helper functions

def _deep_merge(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries."""
    result = dict1.copy()
    
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result

def _parse_properties(content: str) -> Dict[str, Any]:
    """Parse Java-style properties file."""
    result = {}
    
    for line in content.split('\n'):
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            key, value = line.split('=', 1)
            result[key.strip()] = value.strip()
    
    return result

def _parse_env_file(content: str) -> Dict[str, Any]:
    """Parse environment file (.env)."""
    result = {}
    
    for line in content.split('\n'):
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            key, value = line.split('=', 1)
            # Remove quotes if present
            value = value.strip().strip('"').strip("'")
            result[key.strip()] = value
    
    return result

def _format_ini_content(content: Dict[str, Any]) -> str:
    """Format content as INI file."""
    lines = []
    
    for section_name, section_data in content.items():
        lines.append(f"[{section_name}]")
        if isinstance(section_data, dict):
            for key, value in section_data.items():
                lines.append(f"{key} = {value}")
        lines.append("")
    
    return '\n'.join(lines)

def _format_properties(content: Dict[str, Any]) -> str:
    """Format content as properties file."""
    lines = []
    
    for key, value in content.items():
        lines.append(f"{key}={value}")
    
    return '\n'.join(lines)

def _format_env_file(content: Dict[str, Any]) -> str:
    """Format content as environment file."""
    lines = []
    
    for key, value in content.items():
        # Quote values that contain spaces
        if ' ' in str(value):
            lines.append(f'{key}="{value}"')
        else:
            lines.append(f"{key}={value}")
    
    return '\n'.join(lines)