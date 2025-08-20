# agents/configuration_modifier.py
import logging
from pathlib import Path
from typing import Dict, Any, List

# Foundational dependencies
from .base import BaseAgent
from core.context import GlobalContext
from core.models import AgentResult
from tools.configuration_tools import ConfigurationTools

# Get a logger instance for this module
logger = logging.getLogger(__name__)


class ConfigurationModifierAgent(BaseAgent):
    """
    Infrastructure Tier: Configuration file management ONLY.
    
    This agent handles all configuration file operations using
    structured inputs and leveraging configuration tools.
    
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
            description="Manages configuration files for development tools and project settings using structured inputs."
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
            "merge_strategy",
            "template_type"
        ]

    def execute_v2(self, goal: str, inputs: Dict[str, Any], global_context: GlobalContext) -> AgentResult:
        """
        Execute configuration file operations using configuration tools.
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
        template_type = inputs.get("template_type", "basic")

        try:
            self.report_progress(f"Starting {operation} on {config_file}", f"Working in {working_directory}")

            config_path = Path(working_directory) / config_file

            if operation == "create":
                result = self._create_config(
                    config_path, config_data, backup_original, validate_syntax, template_type
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
                      backup_original: bool, validate_syntax: bool, template_type: str) -> Dict[str, Any]:
        """Create a new configuration file using configuration tools."""
        try:
            # Use template if no config_data provided
            if not config_data and template_type:
                file_format = ConfigurationTools.detect_config_format(config_path)
                config_data = ConfigurationTools.create_config_template(file_format, template_type)

            # Backup if file exists and backup requested
            if config_path.exists() and backup_original:
                try:
                    ConfigurationTools.backup_config(config_path)
                except Exception as e:
                    logger.warning(f"Failed to backup existing config: {e}")

            # Write configuration using tools
            ConfigurationTools.write_config_content(config_path, config_data)

            # Validate syntax if requested
            validation_result = None
            if validate_syntax:
                validation_result = ConfigurationTools.validate_config_syntax(config_path)

            file_format = ConfigurationTools.detect_config_format(config_path)

            return {
                "success": True,
                "message": f"Successfully created configuration file: {config_path.name}",
                "outputs": {
                    "config_file": str(config_path),
                    "file_format": file_format,
                    "validation_result": validation_result,
                    "backup_created": config_path.exists() and backup_original,
                    "template_used": template_type if not config_data else None
                }
            }

        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to create configuration file: {e}",
                "outputs": {"error": str(e)}
            }

    def _update_config(self, config_path: Path, config_data: Dict[str, Any], 
                      config_section: str, config_key: str, config_value: Any, 
                      backup_original: bool, merge_strategy: str, validate_syntax: bool) -> Dict[str, Any]:
        """Update an existing configuration file using configuration tools."""
        if not config_path.exists():
            return {
                "success": False,
                "message": f"Configuration file does not exist: {config_path}",
                "outputs": {}
            }

        try:
            # Backup if requested
            if backup_original:
                try:
                    ConfigurationTools.backup_config(config_path)
                except Exception as e:
                    logger.warning(f"Failed to backup config: {e}")

            # Read existing configuration using tools
            existing_config = ConfigurationTools.read_config_content(config_path)

            # Apply updates based on what's provided
            if config_data:
                # Update with full config data
                updated_config = ConfigurationTools.merge_config_data(existing_config, config_data, merge_strategy)
            elif config_section and config_key and config_value is not None:
                # Update specific key in section
                updated_config = ConfigurationTools.set_config_value(existing_config, config_value, config_section, config_key)
            else:
                return {
                    "success": False,
                    "message": "No update data provided (config_data or section/key/value)",
                    "outputs": {}
                }

            # Write updated configuration using tools
            ConfigurationTools.write_config_content(config_path, updated_config)

            # Validate syntax if requested
            validation_result = None
            if validate_syntax:
                validation_result = ConfigurationTools.validate_config_syntax(config_path)

            file_format = ConfigurationTools.detect_config_format(config_path)

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

    def _read_config(self, config_path: Path, config_section: str = None, 
                    config_key: str = None) -> Dict[str, Any]:
        """Read configuration file content using configuration tools."""
        if not config_path.exists():
            return {
                "success": False,
                "message": f"Configuration file does not exist: {config_path}",
                "outputs": {}
            }

        try:
            file_format = ConfigurationTools.detect_config_format(config_path)
            config_content = ConfigurationTools.read_config_content(config_path, file_format)

            # Extract specific section/key if requested
            if config_section:
                try:
                    final_content = ConfigurationTools.get_config_value(config_content, config_section, config_key)
                except KeyError as e:
                    return {
                        "success": False,
                        "message": str(e),
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

    def _delete_config(self, config_path: Path, config_section: str = None,
                      config_key: str = None, backup_original: bool = True) -> Dict[str, Any]:
        """Delete configuration file or specific sections/keys using configuration tools."""
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
                    try:
                        ConfigurationTools.backup_config(config_path)
                    except Exception as e:
                        logger.warning(f"Failed to backup before deletion: {e}")
                
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
                try:
                    ConfigurationTools.backup_config(config_path)
                except Exception as e:
                    logger.warning(f"Failed to backup config: {e}")

            config_content = ConfigurationTools.read_config_content(config_path)

            try:
                updated_config = ConfigurationTools.delete_config_value(config_content, config_section, config_key)
            except KeyError as e:
                return {
                    "success": False,
                    "message": str(e),
                    "outputs": {}
                }

            # Write updated configuration
            ConfigurationTools.write_config_content(config_path, updated_config)

            operation = f"delete_key_{config_section}.{config_key}" if config_key else f"delete_section_{config_section}"

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
        """Validate configuration file syntax using configuration tools."""
        if not config_path.exists():
            return {
                "success": False,
                "message": f"Configuration file does not exist: {config_path}",
                "outputs": {}
            }

        try:
            validation_result = ConfigurationTools.validate_config_syntax(config_path)

            if validation_result["is_valid"]:
                return {
                    "success": True,
                    "message": f"Configuration file syntax is valid: {config_path.name}",
                    "outputs": {
                        "config_file": str(config_path),
                        "file_format": validation_result["file_format"],
                        "is_valid": True
                    }
                }
            else:
                return {
                    "success": False,
                    "message": f"Configuration file syntax error: {validation_result['error']}",
                    "outputs": {
                        "config_file": str(config_path),
                        "is_valid": False,
                        "error": validation_result["error"]
                    }
                }

        except Exception as e:
            return {
                "success": False,
                "message": f"Configuration validation failed: {e}",
                "outputs": {"error": str(e)}
            }

    def _backup_config(self, config_path: Path) -> Dict[str, Any]:
        """Create a backup of the configuration file using configuration tools."""
        if not config_path.exists():
            return {
                "success": False,
                "message": f"Configuration file does not exist: {config_path}",
                "outputs": {}
            }

        try:
            backup_path = ConfigurationTools.backup_config(config_path)

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
        """Restore configuration from backup using configuration tools."""
        try:
            success = ConfigurationTools.restore_config(config_path)

            if success:
                return {
                    "success": True,
                    "message": f"Successfully restored configuration from backup",
                    "outputs": {
                        "config_file": str(config_path),
                        "backup_file": str(config_path.with_suffix(config_path.suffix + '.backup'))
                    }
                }
            else:
                return {
                    "success": False,
                    "message": f"Backup file does not exist: {config_path.with_suffix(config_path.suffix + '.backup')}",
                    "outputs": {}
                }

        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to restore configuration from backup: {e}",
                "outputs": {"error": str(e)}
            }