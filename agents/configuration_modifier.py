# agents/configuration_modifier.py
import logging
import json
import configparser
from typing import Dict, Any, List, Optional, Union

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
from pathlib import Path

# Foundational dependencies
from .base import BaseAgent
from core.context import GlobalContext
from core.models import AgentResponse, AgentResult, TaskNode

# Get a logger instance for this module
logger = logging.getLogger(__name__)


class ConfigurationModifierAgent(BaseAgent):
    """
    Infrastructure Tier: Configuration file management ONLY.
    
    This agent handles all configuration file operations without analysis.
    
    Responsibilities:
    - Create/update pytest.ini, pyproject.toml
    - Manage linting configs, IDE settings
    - Tool configuration and settings files
    - Configuration validation and optimization
    
    Does NOT: Run tools, analyze results, install packages
    """

    def __init__(self):
        super().__init__(
            name="ConfigurationModifierAgent",
            description="Manages configuration files for development tools and project settings."
        )

    def required_inputs(self) -> List[str]:
        """Required inputs for ConfigurationModifierAgent execution."""
        return ["operation", "config_file"]

    def optional_inputs(self) -> List[str]:
        """Optional inputs for ConfigurationModifierAgent execution."""
        return [
            "config_data", 
            "config_section", 
            "config_key",
            "config_value",
            "working_directory",
            "backup_original",
            "validate_syntax",
            "merge_strategy"
        ]

    def execute_v2(self, goal: str, inputs: Dict[str, Any], global_context: GlobalContext) -> AgentResult:
        """
        NEW INTERFACE: Execute configuration file operations.
        """
        logger.info(f"ConfigurationModifierAgent executing: '{goal}'")
        
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
        config_file = inputs["config_file"]
        config_data = inputs.get("config_data", {})
        config_section = inputs.get("config_section")
        config_key = inputs.get("config_key")
        config_value = inputs.get("config_value")
        working_directory = inputs.get("working_directory", str(global_context.workspace_path))
        backup_original = inputs.get("backup_original", True)
        validate_syntax = inputs.get("validate_syntax", True)
        merge_strategy = inputs.get("merge_strategy", "update")

        try:
            self.report_progress(f"Starting {operation} on {config_file}", f"Working in {working_directory}")

            config_path = Path(working_directory) / config_file

            if operation == "create":
                result = self._create_config(
                    config_path, config_data, backup_original, validate_syntax
                )
            elif operation == "update":
                result = self._update_config(
                    config_path, config_data, config_section, config_key, 
                    config_value, backup_original, merge_strategy, validate_syntax
                )
            elif operation == "read":
                result = self._read_config(config_path, config_section, config_key)
            elif operation == "delete":
                result = self._delete_config(
                    config_path, config_section, config_key, backup_original
                )
            elif operation == "validate":
                result = self._validate_config(config_path)
            elif operation == "backup":
                result = self._backup_config(config_path)
            elif operation == "restore":
                result = self._restore_config(config_path)
            else:
                return self.create_result(
                    success=False,
                    message=f"Unknown operation: {operation}",
                    error_details={"supported_operations": ["create", "update", "read", "delete", "validate", "backup", "restore"]}
                )

            self.report_progress("Configuration operation complete", result["message"])

            return self.create_result(
                success=result["success"],
                message=result["message"],
                outputs=result["outputs"]
            )

        except Exception as e:
            error_msg = f"ConfigurationModifierAgent execution failed: {e}"
            logger.error(error_msg, exc_info=True)
            return self.create_result(
                success=False,
                message=error_msg,
                error_details={"exception": str(e)}
            )

    def _create_config(self, config_path: Path, config_data: Dict[str, Any], 
                      backup_original: bool, validate_syntax: bool) -> Dict[str, Any]:
        """Create a new configuration file."""
        
        try:
            # Backup if file exists and backup requested
            if config_path.exists() and backup_original:
                backup_result = self._backup_config(config_path)
                if not backup_result["success"]:
                    logger.warning(f"Failed to backup existing config: {backup_result['message']}")

            # Determine file format and write
            file_format = self._detect_config_format(config_path)
            
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

            # Validate syntax if requested
            validation_result = None
            if validate_syntax:
                validation_result = self._validate_config(config_path)

            return {
                "success": True,
                "message": f"Successfully created configuration file: {config_path.name}",
                "outputs": {
                    "config_file": str(config_path),
                    "file_format": file_format,
                    "validation_result": validation_result,
                    "backup_created": config_path.exists() and backup_original
                }
            }

        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to create configuration file: {e}",
                "outputs": {"error": str(e)}
            }

    def _update_config(self, config_path: Path, config_data: Dict[str, Any], 
                      config_section: Optional[str], config_key: Optional[str],
                      config_value: Any, backup_original: bool, merge_strategy: str,
                      validate_syntax: bool) -> Dict[str, Any]:
        """Update an existing configuration file."""
        
        if not config_path.exists():
            return {
                "success": False,
                "message": f"Configuration file does not exist: {config_path}",
                "outputs": {}
            }

        try:
            # Backup if requested
            if backup_original:
                backup_result = self._backup_config(config_path)
                if not backup_result["success"]:
                    logger.warning(f"Failed to backup config: {backup_result['message']}")

            # Read existing configuration
            file_format = self._detect_config_format(config_path)
            existing_config = self._read_config_content(config_path, file_format)

            # Apply updates based on what's provided
            if config_data:
                # Update with full config data
                if merge_strategy == "replace":
                    updated_config = config_data
                elif merge_strategy == "update":
                    if isinstance(existing_config, dict):
                        updated_config = existing_config.copy()
                        updated_config.update(config_data)
                    else:
                        updated_config = config_data
                else:
                    return {
                        "success": False,
                        "message": f"Unknown merge strategy: {merge_strategy}",
                        "outputs": {}
                    }
            elif config_section and config_key and config_value is not None:
                # Update specific key in section
                updated_config = existing_config.copy() if isinstance(existing_config, dict) else {}
                if config_section not in updated_config:
                    updated_config[config_section] = {}
                updated_config[config_section][config_key] = config_value
            else:
                return {
                    "success": False,
                    "message": "No update data provided (config_data or section/key/value)",
                    "outputs": {}
                }

            # Write updated configuration
            if file_format == "json":
                config_path.write_text(json.dumps(updated_config, indent=2))
            elif file_format == "yaml":
                if YAML_AVAILABLE:
                    config_path.write_text(yaml.dump(updated_config, default_flow_style=False))
                else:
                    raise ValueError("YAML support not available. Install PyYAML: pip install PyYAML")
            elif file_format == "toml":
                if TOML_AVAILABLE:
                    config_path.write_text(toml.dumps(updated_config))
                else:
                    raise ValueError("TOML support not available. Install toml: pip install toml")
            elif file_format == "ini":
                config = configparser.ConfigParser()
                for section, values in updated_config.items():
                    config.add_section(section)
                    if isinstance(values, dict):
                        for key, value in values.items():
                            config.set(section, key, str(value))
                with open(config_path, 'w') as f:
                    config.write(f)

            # Validate syntax if requested
            validation_result = None
            if validate_syntax:
                validation_result = self._validate_config(config_path)

            return {
                "success": True,
                "message": f"Successfully updated configuration file: {config_path.name}",
                "outputs": {
                    "config_file": str(config_path),
                    "file_format": file_format,
                    "merge_strategy": merge_strategy,
                    "validation_result": validation_result,
                    "backup_created": backup_original
                }
            }

        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to update configuration file: {e}",
                "outputs": {"error": str(e)}
            }

    def _read_config(self, config_path: Path, config_section: Optional[str] = None, 
                    config_key: Optional[str] = None) -> Dict[str, Any]:
        """Read configuration file content."""
        
        if not config_path.exists():
            return {
                "success": False,
                "message": f"Configuration file does not exist: {config_path}",
                "outputs": {}
            }

        try:
            file_format = self._detect_config_format(config_path)
            config_content = self._read_config_content(config_path, file_format)

            # Extract specific section/key if requested
            if config_section and isinstance(config_content, dict):
                if config_section in config_content:
                    section_content = config_content[config_section]
                    if config_key and isinstance(section_content, dict):
                        if config_key in section_content:
                            final_content = section_content[config_key]
                        else:
                            return {
                                "success": False,
                                "message": f"Key '{config_key}' not found in section '{config_section}'",
                                "outputs": {}
                            }
                    else:
                        final_content = section_content
                else:
                    return {
                        "success": False,
                        "message": f"Section '{config_section}' not found",
                        "outputs": {}
                    }
            else:
                final_content = config_content

            return {
                "success": True,
                "message": f"Successfully read configuration from: {config_path.name}",
                "outputs": {
                    "config_file": str(config_path),
                    "file_format": file_format,
                    "config_content": final_content,
                    "section": config_section,
                    "key": config_key
                }
            }

        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to read configuration file: {e}",
                "outputs": {"error": str(e)}
            }

    def _delete_config(self, config_path: Path, config_section: Optional[str] = None,
                      config_key: Optional[str] = None, backup_original: bool = True) -> Dict[str, Any]:
        """Delete configuration file or specific sections/keys."""
        
        if not config_path.exists():
            return {
                "success": False,
                "message": f"Configuration file does not exist: {config_path}",
                "outputs": {}
            }

        try:
            # If no section/key specified, delete entire file
            if not config_section and not config_key:
                if backup_original:
                    backup_result = self._backup_config(config_path)
                    if not backup_result["success"]:
                        logger.warning(f"Failed to backup before deletion: {backup_result['message']}")
                
                config_path.unlink()
                return {
                    "success": True,
                    "message": f"Successfully deleted configuration file: {config_path.name}",
                    "outputs": {
                        "config_file": str(config_path),
                        "operation": "delete_file",
                        "backup_created": backup_original
                    }
                }

            # Delete specific section or key
            if backup_original:
                backup_result = self._backup_config(config_path)
                if not backup_result["success"]:
                    logger.warning(f"Failed to backup config: {backup_result['message']}")

            file_format = self._detect_config_format(config_path)
            config_content = self._read_config_content(config_path, file_format)

            if not isinstance(config_content, dict):
                return {
                    "success": False,
                    "message": "Cannot delete section/key from non-dictionary configuration",
                    "outputs": {}
                }

            # Delete section or specific key
            if config_section in config_content:
                if config_key:
                    if isinstance(config_content[config_section], dict) and config_key in config_content[config_section]:
                        del config_content[config_section][config_key]
                        operation = f"delete_key_{config_section}.{config_key}"
                    else:
                        return {
                            "success": False,
                            "message": f"Key '{config_key}' not found in section '{config_section}'",
                            "outputs": {}
                        }
                else:
                    del config_content[config_section]
                    operation = f"delete_section_{config_section}"
            else:
                return {
                    "success": False,
                    "message": f"Section '{config_section}' not found",
                    "outputs": {}
                }

            # Write updated configuration
            if file_format == "json":
                config_path.write_text(json.dumps(config_content, indent=2))
            elif file_format == "yaml":
                config_path.write_text(yaml.dump(config_content, default_flow_style=False))
            elif file_format == "toml":
                config_path.write_text(toml.dumps(config_content))

            return {
                "success": True,
                "message": f"Successfully deleted {config_section}{'.'+config_key if config_key else ''} from {config_path.name}",
                "outputs": {
                    "config_file": str(config_path),
                    "operation": operation,
                    "backup_created": backup_original
                }
            }

        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to delete from configuration file: {e}",
                "outputs": {"error": str(e)}
            }

    def _validate_config(self, config_path: Path) -> Dict[str, Any]:
        """Validate configuration file syntax."""
        
        if not config_path.exists():
            return {
                "success": False,
                "message": f"Configuration file does not exist: {config_path}",
                "outputs": {}
            }

        try:
            file_format = self._detect_config_format(config_path)
            
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
                "success": True,
                "message": f"Configuration file syntax is valid: {config_path.name}",
                "outputs": {
                    "config_file": str(config_path),
                    "file_format": file_format,
                    "is_valid": True
                }
            }

        except Exception as e:
            return {
                "success": False,
                "message": f"Configuration file syntax error: {e}",
                "outputs": {
                    "config_file": str(config_path),
                    "is_valid": False,
                    "error": str(e)
                }
            }

    def _backup_config(self, config_path: Path) -> Dict[str, Any]:
        """Create a backup of the configuration file."""
        
        if not config_path.exists():
            return {
                "success": False,
                "message": f"Configuration file does not exist: {config_path}",
                "outputs": {}
            }

        try:
            backup_path = config_path.with_suffix(config_path.suffix + '.backup')
            backup_path.write_text(config_path.read_text())

            return {
                "success": True,
                "message": f"Successfully backed up configuration to: {backup_path.name}",
                "outputs": {
                    "original_file": str(config_path),
                    "backup_file": str(backup_path)
                }
            }

        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to backup configuration file: {e}",
                "outputs": {"error": str(e)}
            }

    def _restore_config(self, config_path: Path) -> Dict[str, Any]:
        """Restore configuration from backup."""
        
        backup_path = config_path.with_suffix(config_path.suffix + '.backup')
        
        if not backup_path.exists():
            return {
                "success": False,
                "message": f"Backup file does not exist: {backup_path}",
                "outputs": {}
            }

        try:
            config_path.write_text(backup_path.read_text())

            return {
                "success": True,
                "message": f"Successfully restored configuration from backup: {backup_path.name}",
                "outputs": {
                    "config_file": str(config_path),
                    "backup_file": str(backup_path)
                }
            }

        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to restore configuration from backup: {e}",
                "outputs": {"error": str(e)}
            }

    def _detect_config_format(self, config_path: Path) -> str:
        """Detect configuration file format based on extension."""
        
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

    def _read_config_content(self, config_path: Path, file_format: str) -> Any:
        """Read configuration content based on format."""
        
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

    # Legacy execute method for backward compatibility
    def execute(self, goal: str, context: GlobalContext, current_task: TaskNode) -> AgentResponse:
        """Legacy execute method - converts to new interface."""
        inputs = {
            'operation': 'validate',
            'config_file': 'pyproject.toml',
            'working_directory': str(context.workspace_path)
        }
        
        result = self.execute_v2(goal, inputs, context)
        
        return AgentResponse(
            success=result.success,
            message=result.message,
            artifacts_generated=[]
        )