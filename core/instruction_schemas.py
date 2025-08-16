# core/instruction_schemas.py
from typing import Dict, Any, List, Optional, Union, Literal
from pydantic import BaseModel, Field
from enum import Enum


class InstructionType(Enum):
    """Types of structured instructions agents can execute."""
    FIX_CODE_ISSUE = "fix_code_issue"
    REPLACE_FUNCTION = "replace_function"
    INSERT_CODE = "insert_code"
    DELETE_CODE = "delete_code"
    RUN_COMMAND = "run_command"
    VALIDATE_FIX = "validate_fix"
    ROLLBACK_CHANGES = "rollback_changes"


class CodeTarget(BaseModel):
    """Specifies a precise code location for modification."""
    file_path: str = Field(description="Absolute path to the target file")
    function_name: Optional[str] = Field(default=None, description="Function name to target")
    class_name: Optional[str] = Field(default=None, description="Class name to target")
    line_start: Optional[int] = Field(default=None, description="Starting line number (1-indexed)")
    line_end: Optional[int] = Field(default=None, description="Ending line number (1-indexed)")
    search_pattern: Optional[str] = Field(default=None, description="Pattern to locate code")


class ValidationSpec(BaseModel):
    """Specifies how to validate an instruction's execution."""
    test_command: Optional[str] = Field(default=None, description="Command to run for validation")
    expected_files: Optional[List[str]] = Field(default=None, description="Files that should exist after execution")
    success_criteria: Optional[str] = Field(default=None, description="Success criteria description")


class StructuredInstruction(BaseModel):
    """Base structured instruction that agents can parse and execute."""
    instruction_id: str = Field(description="Unique identifier for this instruction")
    instruction_type: InstructionType = Field(description="Type of instruction to execute")
    description: str = Field(description="Human-readable description of the instruction")
    priority: Literal["low", "medium", "high", "critical"] = Field(default="medium", description="Instruction priority")
    
    # Execution details
    target: Optional[CodeTarget] = Field(default=None, description="Code target for modification")
    content: Optional[str] = Field(default=None, description="Content to insert/replace")
    command: Optional[str] = Field(default=None, description="Command to execute")
    
    # Validation and rollback
    validation: Optional[ValidationSpec] = Field(default=None, description="How to validate execution")
    rollback_info: Optional[Dict[str, Any]] = Field(default=None, description="Information needed for rollback")
    
    # Dependencies and context
    depends_on: Optional[List[str]] = Field(default=None, description="Instruction IDs this depends on")
    context_data: Optional[Dict[str, Any]] = Field(default=None, description="Additional context data")


class InstructionScript(BaseModel):
    """A collection of structured instructions forming a repair script."""
    script_id: str = Field(description="Unique identifier for this script")
    title: str = Field(description="Human-readable title of the repair script")
    description: str = Field(description="Detailed description of what this script accomplishes")
    
    # Execution metadata
    created_by: str = Field(description="Agent that created this script")
    target_issue: str = Field(description="Description of the issue being fixed")
    estimated_duration: Optional[str] = Field(default=None, description="Estimated execution time")
    
    # Instructions and execution order
    instructions: List[StructuredInstruction] = Field(description="Ordered list of instructions to execute")
    execution_order: Optional[List[str]] = Field(default=None, description="Custom execution order if different from list order")
    
    # Safety and rollback
    backup_required: bool = Field(default=True, description="Whether to backup files before changes")
    rollback_strategy: Optional[str] = Field(default=None, description="Strategy for rolling back if script fails")
    
    # Validation
    final_validation: Optional[ValidationSpec] = Field(default=None, description="Final validation after all instructions")


class InstructionResult(BaseModel):
    """Result of executing a structured instruction."""
    instruction_id: str = Field(description="ID of the executed instruction")
    success: bool = Field(description="Whether execution succeeded")
    message: str = Field(description="Result message or error description")
    
    # Execution details
    executed_at: Optional[str] = Field(default=None, description="ISO timestamp of execution")
    duration_seconds: Optional[float] = Field(default=None, description="Execution duration")
    
    # Changes made
    files_modified: Optional[List[str]] = Field(default=None, description="Files that were modified")
    commands_executed: Optional[List[str]] = Field(default=None, description="Commands that were run")
    
    # Validation results
    validation_passed: Optional[bool] = Field(default=None, description="Whether validation passed")
    validation_output: Optional[str] = Field(default=None, description="Output from validation")
    
    # Rollback information
    rollback_data: Optional[Dict[str, Any]] = Field(default=None, description="Data needed to rollback this instruction")


class ScriptExecutionResult(BaseModel):
    """Result of executing an entire instruction script."""
    script_id: str = Field(description="ID of the executed script")
    success: bool = Field(description="Whether the entire script succeeded")
    message: str = Field(description="Overall result message")
    
    # Execution summary
    total_instructions: int = Field(description="Total number of instructions in script")
    successful_instructions: int = Field(description="Number of successfully executed instructions")
    failed_instructions: int = Field(description="Number of failed instructions")
    
    # Detailed results
    instruction_results: List[InstructionResult] = Field(description="Results for each instruction")
    
    # Files and validation
    files_modified: List[str] = Field(default_factory=list, description="All files modified by the script")
    final_validation_passed: Optional[bool] = Field(default=None, description="Whether final validation passed")
    
    # Rollback capability
    rollback_available: bool = Field(default=False, description="Whether rollback is available")
    rollback_script_id: Optional[str] = Field(default=None, description="ID of rollback script if created")


class ToolingInstruction(BaseModel):
    """Structured instruction for ToolingAgent operations - replaces free-form text commands."""
    instruction_id: str = Field(description="Unique identifier for this tooling operation")
    command_type: Literal["diagnostic", "build", "test", "install", "deploy", "custom"] = Field(description="Type of tooling operation")
    commands: List[str] = Field(description="Ordered list of shell commands to execute")
    purpose: str = Field(description="Clear description of what this operation accomplishes")
    
    # Execution parameters
    timeout: Optional[int] = Field(default=120, description="Timeout in seconds per command")
    working_directory: Optional[str] = Field(default=None, description="Directory to execute commands in")
    environment_vars: Optional[Dict[str, str]] = Field(default=None, description="Environment variables to set")
    
    # Expected outcomes
    expected_artifacts: Optional[List[str]] = Field(default=None, description="Expected output files/artifacts")
    success_criteria: Optional[str] = Field(default=None, description="How to determine if operation succeeded")
    
    # Safety and overrides
    safety_overrides: Optional[List[str]] = Field(default=None, description="Commands to allow despite safety checks")
    ignore_errors: bool = Field(default=False, description="Continue execution even if some commands fail")


class ToolingExecutionResult(BaseModel):
    """Result of executing a ToolingInstruction."""
    instruction_id: str = Field(description="ID of the executed tooling instruction")
    success: bool = Field(description="Whether the tooling operation succeeded")
    message: str = Field(description="Overall result message")
    
    # Command execution details
    commands_executed: List[str] = Field(description="Commands that were executed")
    command_results: List[Dict[str, Any]] = Field(description="Detailed results for each command")
    total_duration: float = Field(description="Total execution time in seconds")
    
    # Output and artifacts
    combined_output: str = Field(description="Combined stdout/stderr from all commands")
    artifacts_generated: List[str] = Field(default_factory=list, description="Artifacts created by the operation")
    
    # Error handling
    failed_commands: List[str] = Field(default_factory=list, description="Commands that failed")
    error_details: Optional[str] = Field(default=None, description="Detailed error information")


# Helper functions for creating common instruction types

def create_fix_code_instruction(
    instruction_id: str,
    file_path: str,
    issue_description: str,
    fix_content: str,
    target_function: str = None,
    line_start: int = None,
    line_end: int = None,
    test_command: str = None
) -> StructuredInstruction:
    """Helper to create a code fix instruction."""
    return StructuredInstruction(
        instruction_id=instruction_id,
        instruction_type=InstructionType.FIX_CODE_ISSUE,
        description=f"Fix code issue: {issue_description}",
        target=CodeTarget(
            file_path=file_path,
            function_name=target_function,
            line_start=line_start,
            line_end=line_end
        ),
        content=fix_content,
        validation=ValidationSpec(test_command=test_command) if test_command else None
    )


def create_command_instruction(
    instruction_id: str,
    command: str,
    description: str,
    expected_success_criteria: str = None
) -> StructuredInstruction:
    """Helper to create a command execution instruction."""
    return StructuredInstruction(
        instruction_id=instruction_id,
        instruction_type=InstructionType.RUN_COMMAND,
        description=description,
        command=command,
        validation=ValidationSpec(success_criteria=expected_success_criteria) if expected_success_criteria else None
    )


def create_validation_instruction(
    instruction_id: str,
    test_command: str,
    description: str = None
) -> StructuredInstruction:
    """Helper to create a validation instruction."""
    return StructuredInstruction(
        instruction_id=instruction_id,
        instruction_type=InstructionType.VALIDATE_FIX,
        description=description or f"Validate fix by running: {test_command}",
        validation=ValidationSpec(test_command=test_command)
    )


# Helper functions for creating ToolingInstructions

def create_diagnostic_instruction(
    instruction_id: str,
    commands: List[str],
    purpose: str,
    timeout: int = 120,
    expected_artifacts: List[str] = None
) -> ToolingInstruction:
    """Helper to create diagnostic tooling instructions."""
    return ToolingInstruction(
        instruction_id=instruction_id,
        command_type="diagnostic",
        commands=commands,
        purpose=purpose,
        timeout=timeout,
        expected_artifacts=expected_artifacts
    )


def create_test_instruction(
    instruction_id: str,
    test_commands: List[str],
    purpose: str = "Execute test suite",
    timeout: int = 300,
    expected_artifacts: List[str] = None
) -> ToolingInstruction:
    """Helper to create test execution instructions."""
    return ToolingInstruction(
        instruction_id=instruction_id,
        command_type="test",
        commands=test_commands,
        purpose=purpose,
        timeout=timeout,
        expected_artifacts=expected_artifacts,
        success_criteria="All tests pass with exit code 0"
    )


def create_build_instruction(
    instruction_id: str,
    build_commands: List[str],
    purpose: str = "Build project",
    timeout: int = 600,
    expected_artifacts: List[str] = None
) -> ToolingInstruction:
    """Helper to create build instructions."""
    return ToolingInstruction(
        instruction_id=instruction_id,
        command_type="build",
        commands=build_commands,
        purpose=purpose,
        timeout=timeout,
        expected_artifacts=expected_artifacts,
        success_criteria="Build completes successfully with exit code 0"
    )


def create_install_instruction(
    instruction_id: str,
    install_commands: List[str],
    purpose: str = "Install dependencies",
    timeout: int = 300
) -> ToolingInstruction:
    """Helper to create dependency installation instructions."""
    return ToolingInstruction(
        instruction_id=instruction_id,
        command_type="install",
        commands=install_commands,
        purpose=purpose,
        timeout=timeout,
        success_criteria="All dependencies installed successfully"
    )