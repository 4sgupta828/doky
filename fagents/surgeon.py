# fagents/surgeon.py
"""
SurgeonAgent - Foundational agent for all precise modification and surgical operations.

This agent consolidates the surgical capabilities from:
- ScriptExecutorAgent (precise code modifications)
- RequirementsManagerAgent (dependency updates)
- ConfigurationModifierAgent (config changes)

The SurgeonAgent is responsible for making precise, surgical modifications to existing code,
configurations, and project dependencies with backup and validation capabilities.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from enum import Enum

from .base import FoundationalAgent
from core.context import GlobalContext
from core.models import AgentResult

# Import all surgical tools
from tools.script_execution_tools import (
    execute_script, execute_instruction, ScriptExecutionContext, 
    InstructionScript, ScriptInstruction, parse_script_from_json,
    parse_script_from_dict, create_fix_code_instruction, create_command_instruction,
    ModificationType, PriorityLevel, CodeTarget, ValidationSpec
)
from tools.requirements_management_tools import (
    analyze_dependencies, RequirementsAnalysisContext, DependencyType,
    RequirementsFormat, update_requirements_file, detect_requirements_format,
    generate_requirements_for_format, validate_requirements_file
)
from tools.configuration_management_tools import (
    create_configuration, update_configuration, read_configuration,
    ConfigFormat, ConfigOperation, MergeStrategy, ConfigOperationContext,
    ConfigOperationResult, backup_config, restore_config, validate_config_syntax
)

logger = logging.getLogger(__name__)

class SurgicalOperation(Enum):
    """Types of surgical operations the SurgeonAgent can perform."""
    SCRIPT_EXECUTION = "script_execution"
    CODE_MODIFICATION = "code_modification"
    REQUIREMENTS_MANAGEMENT = "requirements_management"
    CONFIGURATION_MANAGEMENT = "configuration_management"
    DEPENDENCY_UPDATE = "dependency_update"
    PRECISE_REPAIR = "precise_repair"

class SurgeonAgent(FoundationalAgent):
    """
    Foundational Surgeon Agent for all precise modifications and surgical operations.
    
    This agent can perform:
    - Precise code modifications through structured scripts
    - Requirements analysis and dependency management
    - Configuration file creation, updates, and management
    - Surgical repairs with backup and rollback capabilities
    - Dependency updates and package management
    - Validation and testing of modifications
    """

    def __init__(self, llm_client: Any = None):
        super().__init__(
            name="SurgeonAgent",
            description="Foundational agent for precise modifications, surgical operations, and system maintenance"
        )
        self.llm_client = llm_client

    def get_capabilities(self) -> List[str]:
        """Return list of surgical capabilities."""
        return [
            "script_execution",           # Execute structured modification scripts
            "code_modification",          # Precise code changes and repairs
            "requirements_management",    # Analyze and manage Python dependencies
            "configuration_management",   # Create, update, and manage config files
            "dependency_updates",         # Update project dependencies
            "surgical_repairs",          # Precise fixes with backup/rollback
            "validation_and_testing"     # Validate changes and run tests
        ]

    def execute(self, goal: str, inputs: Dict[str, Any], global_context: GlobalContext) -> AgentResult:
        """
        Execute surgical operations based on the goal and inputs.
        
        Args:
            goal: Description of the surgical operation to perform
            inputs: Input data for the operation
            global_context: Global execution context
            
        Returns:
            AgentResult with operation status and details
        """
        self.report_progress("Starting surgical operation", goal)
        
        try:
            # Determine operation type from goal and inputs
            operation_type = self._determine_operation_type(goal, inputs)
            
            if operation_type == SurgicalOperation.SCRIPT_EXECUTION:
                return self._handle_script_execution(goal, inputs, global_context)
            elif operation_type == SurgicalOperation.CODE_MODIFICATION:
                return self._handle_code_modification(goal, inputs, global_context)
            elif operation_type == SurgicalOperation.REQUIREMENTS_MANAGEMENT:
                return self._handle_requirements_management(goal, inputs, global_context)
            elif operation_type == SurgicalOperation.CONFIGURATION_MANAGEMENT:
                return self._handle_configuration_management(goal, inputs, global_context)
            elif operation_type == SurgicalOperation.DEPENDENCY_UPDATE:
                return self._handle_dependency_update(goal, inputs, global_context)
            elif operation_type == SurgicalOperation.PRECISE_REPAIR:
                return self._handle_precise_repair(goal, inputs, global_context)
            else:
                return self.create_result(
                    success=False,
                    message=f"Unknown surgical operation type: {operation_type}"
                )
                
        except Exception as e:
            logger.error(f"Error in SurgeonAgent execution: {e}")
            return self.create_result(
                success=False,
                message=f"Surgical operation failed: {str(e)}"
            )

    def _determine_operation_type(self, goal: str, inputs: Dict[str, Any]) -> SurgicalOperation:
        """Determine the type of surgical operation based on goal and inputs."""
        goal_lower = goal.lower()
        
        # Check for specific keywords and input types
        if any(word in goal_lower for word in ["script", "instruction", "execute script"]):
            return SurgicalOperation.SCRIPT_EXECUTION
        elif any(word in goal_lower for word in ["requirements", "dependencies", "pip", "package"]):
            return SurgicalOperation.REQUIREMENTS_MANAGEMENT
        elif any(word in goal_lower for word in ["config", "configuration", "settings", "toml", "yaml", "json", "ini"]):
            return SurgicalOperation.CONFIGURATION_MANAGEMENT
        elif any(word in goal_lower for word in ["update dependency", "upgrade package", "version bump"]):
            return SurgicalOperation.DEPENDENCY_UPDATE
        elif any(word in goal_lower for word in ["repair", "fix", "patch", "surgical"]):
            return SurgicalOperation.PRECISE_REPAIR
        elif any(word in goal_lower for word in ["modify", "change", "edit", "update code"]):
            return SurgicalOperation.CODE_MODIFICATION
        
        # Check inputs for type hints
        if "script_artifact_key" in inputs or "instruction_script" in inputs:
            return SurgicalOperation.SCRIPT_EXECUTION
        elif "code_files" in inputs and "requirements" in goal_lower:
            return SurgicalOperation.REQUIREMENTS_MANAGEMENT
        elif "config_file" in inputs or "operation" in inputs:
            return SurgicalOperation.CONFIGURATION_MANAGEMENT
        elif "target_file" in inputs or "search_pattern" in inputs:
            return SurgicalOperation.CODE_MODIFICATION
        
        # Default to code modification for surgical tasks
        return SurgicalOperation.CODE_MODIFICATION

    def _handle_script_execution(self, goal: str, inputs: Dict[str, Any], global_context: GlobalContext) -> AgentResult:
        """Handle structured script execution tasks."""
        self.report_progress("Executing structured script", f"Goal: {goal}")
        
        try:
            # Get script from inputs or global context
            script = None
            if "instruction_script" in inputs:
                if isinstance(inputs["instruction_script"], dict):
                    script = parse_script_from_dict(inputs["instruction_script"])
                elif isinstance(inputs["instruction_script"], str):
                    script = parse_script_from_json(inputs["instruction_script"])
                else:
                    script = inputs["instruction_script"]  # Already parsed
            elif "script_artifact_key" in inputs:
                script_data = global_context.get_artifact(inputs["script_artifact_key"])
                if isinstance(script_data, str):
                    script = parse_script_from_json(script_data)
                else:
                    script = parse_script_from_dict(script_data)
            else:
                return self.create_result(
                    success=False,
                    message="No script provided (instruction_script or script_artifact_key required)"
                )
            
            # Create execution context
            context = ScriptExecutionContext(
                workspace_path=str(global_context.workspace_path) if global_context.workspace_path else ".",
                backup_enabled=inputs.get("backup_enabled", True),
                dry_run=inputs.get("dry_run", False),
                validation_mode=inputs.get("validation_mode", "standard"),
                timeout_seconds=inputs.get("timeout_seconds", 300)
            )
            
            # Execute the script
            result = execute_script(script, context)
            
            if result.success:
                self.report_intermediate_output("script_execution_result", {
                    "script_id": result.script_id,
                    "successful_instructions": result.successful_instructions,
                    "total_instructions": result.total_instructions,
                    "files_modified": result.files_modified
                })
                
                return self.create_result(
                    success=True,
                    message=result.message,
                    outputs={
                        "script_executed": True,
                        "script_id": result.script_id,
                        "instructions_completed": result.successful_instructions,
                        "total_instructions": result.total_instructions,
                        "files_modified": result.files_modified,
                        "execution_time": result.execution_time,
                        "rollback_available": result.rollback_available,
                        "dry_run": context.dry_run
                    }
                )
            else:
                return self.create_result(
                    success=False,
                    message=result.message,
                    outputs={
                        "script_executed": False,
                        "instructions_completed": result.successful_instructions,
                        "total_instructions": result.total_instructions,
                        "rollback_available": result.rollback_available
                    }
                )
                
        except Exception as e:
            return self.create_result(
                success=False,
                message=f"Script execution failed: {str(e)}"
            )

    def _handle_code_modification(self, goal: str, inputs: Dict[str, Any], global_context: GlobalContext) -> AgentResult:
        """Handle direct code modification tasks."""
        self.report_progress("Performing code modification", f"Goal: {goal}")
        
        try:
            # Create a simple script instruction for the modification
            target_file = inputs.get("target_file", inputs.get("file_path", ""))
            if not target_file:
                return self.create_result(
                    success=False,
                    message="No target file specified (target_file or file_path required)"
                )
            
            modification_type = inputs.get("modification_type", ModificationType.FIX_CODE_ISSUE.value)
            try:
                mod_type_enum = ModificationType(modification_type)
            except ValueError:
                mod_type_enum = ModificationType.FIX_CODE_ISSUE
            
            # Create code target
            target = CodeTarget(
                file_path=target_file,
                line_start=inputs.get("line_start"),
                line_end=inputs.get("line_end"),
                function_name=inputs.get("function_name"),
                search_pattern=inputs.get("search_pattern")
            )
            
            # Create validation if test command provided
            validation = None
            if inputs.get("test_command"):
                validation = ValidationSpec(test_command=inputs["test_command"])
            
            # Create instruction
            instruction = ScriptInstruction(
                instruction_id="code_mod_001",
                instruction_type=mod_type_enum,
                description=goal,
                target=target,
                content=inputs.get("content", inputs.get("new_content", "")),
                command=inputs.get("command"),
                validation=validation,
                priority=PriorityLevel(inputs.get("priority", PriorityLevel.MEDIUM.value))
            )
            
            # Create execution context
            context = ScriptExecutionContext(
                workspace_path=str(global_context.workspace_path) if global_context.workspace_path else ".",
                backup_enabled=inputs.get("backup_enabled", True),
                dry_run=inputs.get("dry_run", False)
            )
            
            # Execute the instruction
            result = execute_instruction(instruction, context)
            
            if result.success:
                self.report_intermediate_output("code_modification_result", {
                    "files_modified": result.files_modified,
                    "validation_passed": result.validation_passed
                })
                
                return self.create_result(
                    success=True,
                    message=result.message,
                    outputs={
                        "modification_applied": True,
                        "files_modified": result.files_modified,
                        "validation_passed": result.validation_passed,
                        "execution_time": result.duration_seconds
                    }
                )
            else:
                return self.create_result(
                    success=False,
                    message=result.message
                )
                
        except Exception as e:
            return self.create_result(
                success=False,
                message=f"Code modification failed: {str(e)}"
            )

    def _handle_requirements_management(self, goal: str, inputs: Dict[str, Any], global_context: GlobalContext) -> AgentResult:
        """Handle requirements analysis and management tasks."""
        self.report_progress("Managing requirements", f"Goal: {goal}")
        
        try:
            # Get code files to analyze
            code_files = inputs.get("code_files", {})
            if not code_files and global_context.workspace:
                # Try to get code files from workspace
                workspace_files = global_context.workspace.list_files()
                code_files = {}
                for file_path in workspace_files:
                    if file_path.endswith('.py'):
                        try:
                            content = global_context.workspace.get_file_content(file_path)
                            if content:
                                code_files[file_path] = content
                        except Exception as e:
                            logger.warning(f"Failed to read {file_path}: {e}")
            
            if not code_files:
                return self.create_result(
                    success=False,
                    message="No code files provided for analysis"
                )
            
            # Create analysis context
            context = RequirementsAnalysisContext(
                code_files=code_files,
                project_path=str(global_context.workspace_path) if global_context.workspace_path else ".",
                requirements_file_path=inputs.get("requirements_file_path", "requirements.txt"),
                include_dev_dependencies=inputs.get("include_dev_dependencies", False),
                exclude_standard_library=inputs.get("exclude_standard_library", True),
                version_strategy=inputs.get("version_strategy", "latest"),
                backup_existing=inputs.get("backup_existing", True)
            )
            
            # Analyze dependencies
            result = analyze_dependencies(context)
            
            if result.success:
                self.report_intermediate_output("requirements_analysis", {
                    "dependencies_found": len(result.dependencies),
                    "imports_found": len(result.imports_found),
                    "standard_library_imports": len(result.standard_library_imports),
                    "unknown_imports": len(result.unknown_imports)
                })
                
                dependency_names = [dep.package_name for dep in result.dependencies if not dep.is_standard_library and dep.package_name]
                
                return self.create_result(
                    success=True,
                    message=result.message,
                    outputs={
                        "requirements_updated": result.requirements_file is not None,
                        "requirements_file": result.requirements_file,
                        "dependencies": dependency_names,
                        "imports_found": result.imports_found,
                        "standard_library_imports": result.standard_library_imports,
                        "unknown_imports": result.unknown_imports,
                        "dependency_details": [
                            {"name": dep.name, "package": dep.package_name, "type": dep.dependency_type.value}
                            for dep in result.dependencies
                        ]
                    }
                )
            else:
                return self.create_result(
                    success=False,
                    message=result.message
                )
                
        except Exception as e:
            return self.create_result(
                success=False,
                message=f"Requirements management failed: {str(e)}"
            )

    def _handle_configuration_management(self, goal: str, inputs: Dict[str, Any], global_context: GlobalContext) -> AgentResult:
        """Handle configuration file management tasks."""
        self.report_progress("Managing configuration", f"Goal: {goal}")
        
        try:
            operation = inputs.get("operation", "update")
            config_file = inputs.get("config_file", "")
            if not config_file:
                return self.create_result(
                    success=False,
                    message="No config_file specified"
                )
            
            # Determine workspace path
            workspace_path = str(global_context.workspace_path) if global_context.workspace_path else "."
            config_path = Path(workspace_path) / config_file
            
            # Execute based on operation type
            if operation == "create":
                result = create_configuration(
                    config_path=config_path,
                    content=inputs.get("config_data"),
                    template_name=inputs.get("template_type", "basic"),
                    backup_existing=inputs.get("backup_original", True),
                    validate_syntax=inputs.get("validate_syntax", True)
                )
            elif operation == "update":
                result = update_configuration(
                    config_path=config_path,
                    updates=inputs.get("config_data"),
                    section=inputs.get("config_section"),
                    key=inputs.get("config_key"),
                    value=inputs.get("config_value"),
                    merge_strategy=MergeStrategy(inputs.get("merge_strategy", "update")),
                    backup_existing=inputs.get("backup_original", True),
                    validate_syntax=inputs.get("validate_syntax", True)
                )
            elif operation == "read":
                result = read_configuration(
                    config_path=config_path,
                    section=inputs.get("config_section"),
                    key=inputs.get("config_key")
                )
            elif operation == "backup":
                backup_path = backup_config(config_path)
                result = ConfigOperationResult(
                    success=True,
                    message=f"Configuration backed up to: {backup_path.name}",
                    config_file=str(config_path),
                    operation=ConfigOperation.BACKUP,
                    backup_created=True
                )
            elif operation == "validate":
                validation_result = validate_config_syntax(config_path)
                result = ConfigOperationResult(
                    success=validation_result.is_valid,
                    message=f"Configuration syntax {'valid' if validation_result.is_valid else 'invalid'}: {validation_result.error or ''}",
                    config_file=str(config_path),
                    operation=ConfigOperation.VALIDATE,
                    validation_result=validation_result
                )
            else:
                return self.create_result(
                    success=False,
                    message=f"Unknown configuration operation: {operation}"
                )
            
            if result.success:
                self.report_intermediate_output("config_operation_result", {
                    "operation": result.operation.value,
                    "config_file": result.config_file,
                    "file_format": result.file_format.value if result.file_format else None,
                    "backup_created": result.backup_created
                })
                
                return self.create_result(
                    success=True,
                    message=result.message,
                    outputs={
                        "operation": result.operation.value,
                        "config_file": result.config_file,
                        "file_format": result.file_format.value if result.file_format else None,
                        "backup_created": result.backup_created,
                        "content": result.content,
                        "validation_result": {
                            "is_valid": result.validation_result.is_valid,
                            "error": result.validation_result.error
                        } if result.validation_result else None
                    }
                )
            else:
                return self.create_result(
                    success=False,
                    message=result.message
                )
                
        except Exception as e:
            return self.create_result(
                success=False,
                message=f"Configuration management failed: {str(e)}"
            )

    def _handle_dependency_update(self, goal: str, inputs: Dict[str, Any], global_context: GlobalContext) -> AgentResult:
        """Handle dependency update tasks."""
        self.report_progress("Updating dependencies", f"Goal: {goal}")
        
        # This combines requirements management with script execution for updates
        try:
            # First, analyze current dependencies
            requirements_inputs = {
                "code_files": inputs.get("code_files", {}),
                "requirements_file_path": inputs.get("requirements_file", "requirements.txt"),
                "version_strategy": inputs.get("version_strategy", "latest"),
                "backup_existing": True
            }
            
            req_result = self._handle_requirements_management(
                "Analyze current dependencies", 
                requirements_inputs, 
                global_context
            )
            
            if not req_result.success:
                return req_result
            
            # Then execute update commands if specified
            if inputs.get("update_commands"):
                update_commands = inputs["update_commands"]
                if not isinstance(update_commands, list):
                    update_commands = [update_commands]
                
                # Create script for dependency updates
                instructions = []
                for i, command in enumerate(update_commands):
                    instruction = create_command_instruction(
                        instruction_id=f"dep_update_{i+1}",
                        command=command,
                        description=f"Update dependencies: {command}",
                        priority=PriorityLevel.HIGH
                    )
                    instructions.append(instruction)
                
                script = InstructionScript(
                    script_id="dependency_update_script",
                    title="Dependency Update Script",
                    description="Update project dependencies",
                    created_by="SurgeonAgent",
                    target_issue="Dependency updates",
                    instructions=instructions,
                    backup_required=True
                )
                
                # Execute the update script
                script_inputs = {
                    "instruction_script": script,
                    "backup_enabled": True,
                    "dry_run": inputs.get("dry_run", False)
                }
                
                script_result = self._handle_script_execution(
                    "Execute dependency updates",
                    script_inputs,
                    global_context
                )
                
                # Combine results
                return self.create_result(
                    success=script_result.success,
                    message=f"Dependency update completed: {script_result.message}",
                    outputs={
                        "requirements_analysis": req_result.outputs,
                        "update_execution": script_result.outputs if script_result.success else None
                    }
                )
            
            return req_result
            
        except Exception as e:
            return self.create_result(
                success=False,
                message=f"Dependency update failed: {str(e)}"
            )

    def _handle_precise_repair(self, goal: str, inputs: Dict[str, Any], global_context: GlobalContext) -> AgentResult:
        """Handle precise repair tasks that may combine multiple surgical operations."""
        self.report_progress("Performing precise repair", f"Goal: {goal}")
        
        try:
            repair_plan = inputs.get("repair_plan", [])
            if not repair_plan:
                # Create a simple repair based on inputs
                repair_plan = [{
                    "operation": "code_modification",
                    "inputs": inputs
                }]
            
            results = []
            all_successful = True
            
            # Execute each repair step
            for i, step in enumerate(repair_plan):
                operation = step.get("operation", "code_modification")
                step_inputs = step.get("inputs", {})
                
                self.report_progress(f"Repair step {i+1}/{len(repair_plan)}", f"Operation: {operation}")
                
                if operation == "code_modification":
                    result = self._handle_code_modification(f"Repair step {i+1}", step_inputs, global_context)
                elif operation == "configuration_management":
                    result = self._handle_configuration_management(f"Repair step {i+1}", step_inputs, global_context)
                elif operation == "requirements_management":
                    result = self._handle_requirements_management(f"Repair step {i+1}", step_inputs, global_context)
                elif operation == "script_execution":
                    result = self._handle_script_execution(f"Repair step {i+1}", step_inputs, global_context)
                else:
                    result = self.create_result(
                        success=False,
                        message=f"Unknown repair operation: {operation}"
                    )
                
                results.append({
                    "step": i+1,
                    "operation": operation,
                    "success": result.success,
                    "message": result.message,
                    "outputs": result.outputs if result.success else None
                })
                
                if not result.success:
                    all_successful = False
                    break
            
            if all_successful:
                self.report_intermediate_output("precise_repair_results", results)
                return self.create_result(
                    success=True,
                    message=f"Precise repair completed successfully: {len(results)} steps executed",
                    outputs={
                        "repair_completed": True,
                        "steps_executed": len(results),
                        "results": results
                    }
                )
            else:
                return self.create_result(
                    success=False,
                    message=f"Precise repair failed at step {len(results)}",
                    outputs={
                        "repair_completed": False,
                        "steps_executed": len(results),
                        "results": results
                    }
                )
                
        except Exception as e:
            return self.create_result(
                success=False,
                message=f"Precise repair failed: {str(e)}"
            )

    def required_inputs(self) -> List[str]:
        """Return required inputs based on operation type."""
        # SurgeonAgent is flexible and can work with various input combinations
        return []

    def supports_goal(self, goal: str) -> bool:
        """Check if this agent supports the given goal."""
        surgical_keywords = [
            "modify", "change", "update", "fix", "repair", "patch", "execute", "script",
            "requirements", "dependencies", "config", "configuration", "precise", "surgical"
        ]
        goal_lower = goal.lower()
        return any(keyword in goal_lower for keyword in surgical_keywords)