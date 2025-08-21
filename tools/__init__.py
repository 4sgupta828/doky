# tools/surgical/__init__.py
"""
Surgical tools package for precise modifications and surgical operations on existing systems.

This package contains atomic, reusable tools for:
- Script execution and code modifications
- Requirements and dependency management  
- Configuration file management
"""

from .script_execution_tools import (
    ModificationType,
    PriorityLevel,
    CodeTarget,
    ValidationSpec,
    ScriptInstruction,
    InstructionScript,
    InstructionResult,
    ScriptExecutionResult,
    ScriptExecutionContext,
    create_fix_code_instruction,
    create_command_instruction,
    setup_backup,
    execute_instruction,
    execute_script,
    parse_script_from_json,
    parse_script_from_dict,
    validate_script
)

from .requirements_management_tools import (
    DependencyType,
    RequirementsFormat,
    DependencyInfo,
    RequirementsAnalysisContext,
    RequirementsAnalysisResult,
    extract_imports_from_code,
    extract_imports_from_content,
    analyze_dependencies,
    categorize_import,
    update_requirements_file,
    detect_requirements_format,
    generate_requirements_for_format,
    get_package_version,
    validate_requirements_file
)

from .configuration_management_tools import (
    ConfigFormat,
    ConfigOperation,
    MergeStrategy,
    ConfigTemplate,
    ConfigValidationResult,
    ConfigOperationContext,
    ConfigOperationResult,
    detect_config_format,
    read_config_content,
    write_config_content,
    create_config_template,
    merge_config_data,
    get_config_value,
    set_config_value,
    delete_config_value,
    validate_config_syntax,
    backup_config,
    restore_config,
    create_configuration,
    update_configuration,
    read_configuration
)

__all__ = [
    # Script execution tools
    "ModificationType",
    "PriorityLevel", 
    "CodeTarget",
    "ValidationSpec",
    "ScriptInstruction",
    "InstructionScript",
    "InstructionResult",
    "ScriptExecutionResult",
    "ScriptExecutionContext",
    "create_fix_code_instruction",
    "create_command_instruction",
    "setup_backup",
    "execute_instruction",
    "execute_script", 
    "parse_script_from_json",
    "parse_script_from_dict",
    "validate_script",
    
    # Requirements management tools
    "DependencyType",
    "RequirementsFormat",
    "DependencyInfo",
    "RequirementsAnalysisContext",
    "RequirementsAnalysisResult",
    "extract_imports_from_code",
    "extract_imports_from_content",
    "analyze_dependencies",
    "categorize_import",
    "update_requirements_file",
    "detect_requirements_format",
    "generate_requirements_for_format",
    "get_package_version",
    "validate_requirements_file",
    
    # Configuration management tools
    "ConfigFormat",
    "ConfigOperation",
    "MergeStrategy",
    "ConfigTemplate",
    "ConfigValidationResult",
    "ConfigOperationContext",
    "ConfigOperationResult",
    "detect_config_format",
    "read_config_content",
    "write_config_content",
    "create_config_template",
    "merge_config_data",
    "get_config_value",
    "set_config_value",
    "delete_config_value",
    "validate_config_syntax",
    "backup_config",
    "restore_config",
    "create_configuration",
    "update_configuration",
    "read_configuration"
]