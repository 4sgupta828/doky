# tools/configuration_tools.py
import json
import logging
import configparser
from pathlib import Path
from typing import Dict, Any

# Optional imports with fallbacks
try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

try:
    import toml
    TOML_AVAILABLE = True
except ImportError:
    TOML_AVAILABLE = False

# Get a logger instance for this module
logger = logging.getLogger(__name__)


class ConfigurationTools:
    """
    Agent-agnostic configuration file management utilities.
    
    This module provides low-level configuration file operations that can be used
    by any agent or component. It contains no agent-specific logic,
    only pure configuration file processing functions.
    """

    @staticmethod
    def detect_config_format(config_path) -> str:
        """Detect configuration file format based on extension."""
        if isinstance(config_path, str):
            config_path = Path(config_path)
        extension = config_path.suffix.lower()
        
        if extension in ['.json']:
            return "json"
        elif extension in ['.yaml', '.yml']:
            return "yaml"  
        elif extension in ['.toml']:
            return "toml"
        elif extension in ['.ini', '.cfg', '.conf']:
            return "ini"
        else:
            return "text"

    @staticmethod
    def read_config_content(config_path: Path, file_format: str = None) -> Any:
        """Read configuration content based on format."""
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file does not exist: {config_path}")

        if file_format is None:
            file_format = ConfigurationTools.detect_config_format(config_path)
        
        content = config_path.read_text()
        
        if file_format == "json":
            return json.loads(content)
        elif file_format == "yaml":
            if YAML_AVAILABLE:
                return yaml.safe_load(content)
            else:
                raise ValueError("YAML support not available. Install PyYAML: pip install PyYAML")
        elif file_format == "toml":
            if TOML_AVAILABLE:
                return toml.loads(content)
            else:
                raise ValueError("TOML support not available. Install toml: pip install toml")
        elif file_format == "ini":
            config = configparser.ConfigParser()
            config.read_string(content)
            return {section: dict(config.items(section)) for section in config.sections()}
        else:
            return content

    @staticmethod
    def write_config_content(config_path: Path, config_data: Any, file_format: str = None) -> None:
        """Write configuration content based on format."""
        if file_format is None:
            file_format = ConfigurationTools.detect_config_format(config_path)
        
        # Ensure parent directory exists
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        if file_format == "json":
            config_path.write_text(json.dumps(config_data, indent=2))
        elif file_format == "yaml":
            if YAML_AVAILABLE:
                config_path.write_text(yaml.dump(config_data, default_flow_style=False))
            else:
                raise ValueError("YAML support not available. Install PyYAML: pip install PyYAML")
        elif file_format == "toml":
            if TOML_AVAILABLE:
                config_path.write_text(toml.dumps(config_data))
            else:
                raise ValueError("TOML support not available. Install toml: pip install toml")
        elif file_format == "ini":
            config = configparser.ConfigParser()
            for section, values in config_data.items():
                config.add_section(section)
                if isinstance(values, dict):
                    for key, value in values.items():
                        config.set(section, key, str(value))
            with open(config_path, 'w') as f:
                config.write(f)
        else:
            # Plain text or unknown format
            if isinstance(config_data, str):
                config_path.write_text(config_data)
            else:
                config_path.write_text(str(config_data))

    @staticmethod
    def validate_config_syntax(config_path: Path, file_format: str = None) -> Dict[str, Any]:
        """Validate configuration file syntax."""
        if not config_path.exists():
            return {
                "is_valid": False,
                "error": f"Configuration file does not exist: {config_path}"
            }

        try:
            if file_format is None:
                file_format = ConfigurationTools.detect_config_format(config_path)
            
            # Try to parse the file
            content = config_path.read_text()
            
            if file_format == "json":
                json.loads(content)
            elif file_format == "yaml":
                if YAML_AVAILABLE:
                    yaml.safe_load(content)
                else:
                    raise ValueError("YAML support not available. Install PyYAML: pip install PyYAML")
            elif file_format == "toml":
                if TOML_AVAILABLE:
                    toml.loads(content)
                else:
                    raise ValueError("TOML support not available. Install toml: pip install toml")
            elif file_format == "ini":
                config = configparser.ConfigParser()
                config.read_string(content)

            return {
                "is_valid": True,
                "file_format": file_format,
                "error": None
            }

        except Exception as e:
            return {
                "is_valid": False,
                "file_format": file_format,
                "error": str(e)
            }

    @staticmethod
    def backup_config(config_path: Path, backup_suffix: str = ".backup") -> Path:
        """Create a backup of the configuration file."""
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file does not exist: {config_path}")

        backup_path = config_path.with_suffix(config_path.suffix + backup_suffix)
        backup_path.write_text(config_path.read_text())
        return backup_path

    @staticmethod
    def restore_config(config_path: Path, backup_suffix: str = ".backup") -> bool:
        """Restore configuration from backup."""
        backup_path = config_path.with_suffix(config_path.suffix + backup_suffix)
        
        if not backup_path.exists():
            return False

        config_path.write_text(backup_path.read_text())
        return True

    @staticmethod
    def merge_config_data(existing_config: Any, new_config: Any, merge_strategy: str = "update") -> Any:
        """Merge configuration data using specified strategy."""
        if merge_strategy == "replace":
            return new_config
        elif merge_strategy == "update":
            if isinstance(existing_config, dict) and isinstance(new_config, dict):
                merged = existing_config.copy()
                merged.update(new_config)
                return merged
            else:
                return new_config
        else:
            raise ValueError(f"Unknown merge strategy: {merge_strategy}")

    @staticmethod
    def get_config_value(config_data: Any, section: str = None, key: str = None) -> Any:
        """Extract specific value from configuration data."""
        if section and isinstance(config_data, dict):
            if section in config_data:
                section_content = config_data[section]
                if key and isinstance(section_content, dict):
                    if key in section_content:
                        return section_content[key]
                    else:
                        raise KeyError(f"Key '{key}' not found in section '{section}'")
                else:
                    return section_content
            else:
                raise KeyError(f"Section '{section}' not found")
        else:
            return config_data

    @staticmethod
    def set_config_value(config_data: Any, value: Any, section: str = None, key: str = None) -> Any:
        """Set specific value in configuration data."""
        if not isinstance(config_data, dict):
            config_data = {}

        if section and key:
            if section not in config_data:
                config_data[section] = {}
            config_data[section][key] = value
        elif section:
            config_data[section] = value
        else:
            # Replace entire config
            return value
        
        return config_data

    @staticmethod
    def delete_config_value(config_data: Any, section: str = None, key: str = None) -> Any:
        """Delete specific value from configuration data."""
        if not isinstance(config_data, dict):
            return config_data

        if section in config_data:
            if key:
                if isinstance(config_data[section], dict) and key in config_data[section]:
                    del config_data[section][key]
                else:
                    raise KeyError(f"Key '{key}' not found in section '{section}'")
            else:
                del config_data[section]
        else:
            raise KeyError(f"Section '{section}' not found")
        
        return config_data

    @staticmethod
    def create_config_template(file_format: str, template_type: str = "basic") -> Dict[str, Any]:
        """Create configuration templates for common tools."""
        templates = {
            "pytest": {
                "tool": {
                    "pytest": {
                        "ini_options": {
                            "minversion": "6.0",
                            "testpaths": ["tests"],
                            "python_files": ["test_*.py", "*_test.py"],
                            "python_classes": ["Test*"],
                            "python_functions": ["test_*"],
                            "addopts": "-v --tb=short"
                        }
                    }
                }
            },
            "black": {
                "tool": {
                    "black": {
                        "line-length": 88,
                        "target-version": ["py38"],
                        "include": "\\.pyi?$",
                        "extend-exclude": """
                        /(
                          # directories
                          \\.eggs
                          | \\.git
                          | \\.venv
                          | build
                          | dist
                        )/
                        """
                    }
                }
            },
            "mypy": {
                "tool": {
                    "mypy": {
                        "python_version": "3.8",
                        "warn_return_any": True,
                        "warn_unused_configs": True,
                        "disallow_untyped_defs": True
                    }
                }
            },
            "isort": {
                "tool": {
                    "isort": {
                        "profile": "black",
                        "multi_line_output": 3,
                        "line_length": 88
                    }
                }
            },
            "basic": {
                "project": {
                    "name": "my-project",
                    "version": "0.1.0",
                    "description": "A sample project"
                },
                "build-system": {
                    "requires": ["setuptools", "wheel"],
                    "build-backend": "setuptools.build_meta"
                }
            }
        }

        return templates.get(template_type, templates["basic"])

    @staticmethod
    def analyze_config_structure(config_data: Any) -> Dict[str, Any]:
        """Analyze the structure of configuration data."""
        analysis = {
            "data_type": type(config_data).__name__,
            "is_empty": not bool(config_data),
            "sections": [],
            "total_keys": 0,
            "depth": 0
        }

        if isinstance(config_data, dict):
            analysis["sections"] = list(config_data.keys())
            analysis["total_keys"] = len(config_data)
            
            # Calculate depth
            def calculate_depth(obj, current_depth=0):
                if isinstance(obj, dict):
                    if not obj:
                        return current_depth
                    return max(calculate_depth(v, current_depth + 1) for v in obj.values())
                elif isinstance(obj, list):
                    if not obj:
                        return current_depth
                    return max(calculate_depth(item, current_depth + 1) for item in obj)
                else:
                    return current_depth
            
            analysis["depth"] = calculate_depth(config_data)
            
            # Analyze each section
            section_analysis = {}
            for section, content in config_data.items():
                section_analysis[section] = {
                    "type": type(content).__name__,
                    "size": len(content) if hasattr(content, '__len__') else 1
                }
            analysis["section_details"] = section_analysis

        elif isinstance(config_data, list):
            analysis["total_keys"] = len(config_data)
            analysis["sections"] = [f"item_{i}" for i in range(len(config_data))]

        return analysis

    @staticmethod
    def compare_configs(config1: Any, config2: Any) -> Dict[str, Any]:
        """Compare two configuration data structures."""
        def deep_diff(obj1, obj2, path=""):
            differences = []
            
            if type(obj1) != type(obj2):
                differences.append({
                    "path": path,
                    "type": "type_mismatch",
                    "value1": type(obj1).__name__,
                    "value2": type(obj2).__name__
                })
                return differences
            
            if isinstance(obj1, dict):
                all_keys = set(obj1.keys()) | set(obj2.keys())
                for key in all_keys:
                    key_path = f"{path}.{key}" if path else key
                    if key not in obj1:
                        differences.append({
                            "path": key_path,
                            "type": "added",
                            "value": obj2[key]
                        })
                    elif key not in obj2:
                        differences.append({
                            "path": key_path,
                            "type": "removed",
                            "value": obj1[key]
                        })
                    else:
                        differences.extend(deep_diff(obj1[key], obj2[key], key_path))
            elif isinstance(obj1, list):
                if len(obj1) != len(obj2):
                    differences.append({
                        "path": path,
                        "type": "length_change",
                        "value1": len(obj1),
                        "value2": len(obj2)
                    })
                for i in range(min(len(obj1), len(obj2))):
                    differences.extend(deep_diff(obj1[i], obj2[i], f"{path}[{i}]"))
            else:
                if obj1 != obj2:
                    differences.append({
                        "path": path,
                        "type": "value_change",
                        "value1": obj1,
                        "value2": obj2
                    })
            
            return differences

        differences = deep_diff(config1, config2)
        
        return {
            "are_equal": len(differences) == 0,
            "differences_count": len(differences),
            "differences": differences
        }