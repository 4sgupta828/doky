# tools/surgical/script_execution_tools.py
"""
Script execution tools for precise code modifications and surgical operations.
Extracted from ScriptExecutorAgent to provide atomic, reusable script execution capabilities.
"""

import json
import logging
import os
import subprocess
import tempfile
import time
import shutil
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

class ModificationType(Enum):
    """Types of code modifications."""
    FIX_CODE_ISSUE = "fix_code_issue"
    REPLACE_FUNCTION = "replace_function"
    INSERT_CODE = "insert_code"
    DELETE_CODE = "delete_code"
    RUN_COMMAND = "run_command"
    VALIDATE_FIX = "validate_fix"

class PriorityLevel(Enum):
    """Priority levels for script instructions."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class CodeTarget:
    """Represents a target location in code for modifications."""
    file_path: str
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    function_name: Optional[str] = None
    class_name: Optional[str] = None
    search_pattern: Optional[str] = None

@dataclass
class ValidationSpec:
    """Specification for validating script execution."""
    test_command: Optional[str] = None
    expected_files: List[str] = None
    expected_output: Optional[str] = None
    timeout_seconds: int = 120
    
    def __post_init__(self):
        if self.expected_files is None:
            self.expected_files = []

@dataclass
class ScriptInstruction:
    """Represents a single script instruction."""
    instruction_id: str
    instruction_type: ModificationType
    description: str
    target: Optional[CodeTarget] = None
    content: Optional[str] = None
    command: Optional[str] = None
    validation: Optional[ValidationSpec] = None
    priority: PriorityLevel = PriorityLevel.MEDIUM

@dataclass
class InstructionScript:
    """Represents a complete script with multiple instructions."""
    script_id: str
    title: str
    description: str
    created_by: str
    target_issue: str
    instructions: List[ScriptInstruction]
    execution_order: Optional[List[str]] = None
    backup_required: bool = True
    estimated_duration: Optional[str] = None
    final_validation: Optional[ValidationSpec] = None
    
    def __post_init__(self):
        if self.execution_order is None:
            self.execution_order = [inst.instruction_id for inst in self.instructions]

@dataclass
class InstructionResult:
    """Result of executing a single instruction."""
    instruction_id: str
    success: bool
    message: str
    executed_at: str
    duration_seconds: float
    files_modified: List[str] = None
    validation_passed: Optional[bool] = None
    validation_output: Optional[str] = None
    
    def __post_init__(self):
        if self.files_modified is None:
            self.files_modified = []

@dataclass
class ScriptExecutionResult:
    """Result of executing an entire script."""
    script_id: str
    success: bool
    message: str
    total_instructions: int
    successful_instructions: int
    failed_instructions: int
    instruction_results: List[InstructionResult]
    files_modified: List[str]
    execution_time: float = 0.0
    final_validation_passed: Optional[bool] = None
    rollback_available: bool = False

@dataclass
class ScriptExecutionContext:
    """Context for script execution."""
    workspace_path: str
    backup_enabled: bool = True
    dry_run: bool = False
    validation_mode: str = "standard"
    timeout_seconds: int = 300

def create_fix_code_instruction(
    instruction_id: str,
    file_path: str,
    issue_description: str,
    fix_content: str,
    line_start: Optional[int] = None,
    line_end: Optional[int] = None,
    function_name: Optional[str] = None,
    search_pattern: Optional[str] = None,
    priority: PriorityLevel = PriorityLevel.MEDIUM
) -> ScriptInstruction:
    """Create a code fix instruction."""
    target = CodeTarget(
        file_path=file_path,
        line_start=line_start,
        line_end=line_end,
        function_name=function_name,
        search_pattern=search_pattern
    )
    
    return ScriptInstruction(
        instruction_id=instruction_id,
        instruction_type=ModificationType.FIX_CODE_ISSUE,
        description=issue_description,
        target=target,
        content=fix_content,
        priority=priority
    )

def create_command_instruction(
    instruction_id: str,
    command: str,
    description: str,
    validation_command: Optional[str] = None,
    priority: PriorityLevel = PriorityLevel.MEDIUM
) -> ScriptInstruction:
    """Create a command execution instruction."""
    validation = None
    if validation_command:
        validation = ValidationSpec(test_command=validation_command)
    
    return ScriptInstruction(
        instruction_id=instruction_id,
        instruction_type=ModificationType.RUN_COMMAND,
        description=description,
        command=command,
        validation=validation,
        priority=priority
    )

def setup_backup(context: ScriptExecutionContext, script: InstructionScript) -> Optional[Path]:
    """Create backup of files that will be modified."""
    if not script.backup_required or not context.backup_enabled or context.dry_run:
        return None
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    workspace = Path(context.workspace_path)
    backup_dir = workspace / f"backups/script_{script.script_id}_{timestamp}"
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect all files that will be modified
    files_to_backup = set()
    for instruction in script.instructions:
        if instruction.target and instruction.target.file_path:
            files_to_backup.add(instruction.target.file_path)
    
    # Backup each file
    for file_path in files_to_backup:
        try:
            source_path = workspace / file_path
            if source_path.exists():
                backup_path = backup_dir / file_path
                backup_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Copy original file to backup
                shutil.copy2(source_path, backup_path)
                logger.info(f"Backed up {file_path} to {backup_path}")
                
        except Exception as e:
            logger.warning(f"Failed to backup {file_path}: {e}")
    
    return backup_dir

def execute_instruction(
    instruction: ScriptInstruction, 
    context: ScriptExecutionContext
) -> InstructionResult:
    """Execute a single script instruction."""
    start_time = time.time()
    logger.info(f"Executing instruction: {instruction.instruction_id} ({instruction.instruction_type.value})")
    
    try:
        # Execute based on instruction type
        if instruction.instruction_type == ModificationType.FIX_CODE_ISSUE:
            success, message, files_modified = fix_code_issue(instruction, context)
        elif instruction.instruction_type == ModificationType.REPLACE_FUNCTION:
            success, message, files_modified = replace_function(instruction, context)
        elif instruction.instruction_type == ModificationType.INSERT_CODE:
            success, message, files_modified = insert_code(instruction, context)
        elif instruction.instruction_type == ModificationType.DELETE_CODE:
            success, message, files_modified = delete_code(instruction, context)
        elif instruction.instruction_type == ModificationType.RUN_COMMAND:
            success, message, files_modified = run_command(instruction, context)
        elif instruction.instruction_type == ModificationType.VALIDATE_FIX:
            success, message, files_modified = validate_instruction(instruction, context)
        else:
            success, message, files_modified = False, f"Unknown instruction type: {instruction.instruction_type}", []
        
        # Run validation if specified and execution was successful
        validation_passed = None
        validation_output = None
        if success and instruction.validation:
            validation_passed = run_validation(instruction.validation, context)
            validation_output = f"Validation {'passed' if validation_passed else 'failed'}"
            if not validation_passed:
                success = False
                message += f" (Validation failed: {validation_output})"
        
        duration = time.time() - start_time
        
        return InstructionResult(
            instruction_id=instruction.instruction_id,
            success=success,
            message=message,
            executed_at=datetime.now().isoformat(),
            duration_seconds=duration,
            files_modified=files_modified,
            validation_passed=validation_passed,
            validation_output=validation_output
        )
        
    except Exception as e:
        logger.error(f"Instruction {instruction.instruction_id} failed with exception: {e}")
        return InstructionResult(
            instruction_id=instruction.instruction_id,
            success=False,
            message=f"Execution failed: {e}",
            executed_at=datetime.now().isoformat(),
            duration_seconds=time.time() - start_time
        )

def fix_code_issue(instruction: ScriptInstruction, context: ScriptExecutionContext) -> Tuple[bool, str, List[str]]:
    """Execute a code fix instruction."""
    if not instruction.target or not instruction.content:
        return False, "Missing target or content for code fix", []
    
    workspace = Path(context.workspace_path)
    file_path = workspace / instruction.target.file_path
    
    if context.dry_run:
        return True, f"[DRY RUN] Would apply code fix to {instruction.target.file_path}", [instruction.target.file_path]
    
    try:
        # Read current file content
        if not file_path.exists():
            return False, f"Target file does not exist: {file_path}", []
        
        original_content = file_path.read_text()
        
        # Apply the fix based on target specification
        if instruction.target.line_start and instruction.target.line_end:
            # Line-based replacement
            lines = original_content.splitlines()
            new_lines = lines[:instruction.target.line_start-1] + \
                       instruction.content.splitlines() + \
                       lines[instruction.target.line_end:]
            new_content = '\n'.join(new_lines)
        elif instruction.target.search_pattern:
            # Pattern-based replacement
            new_content = original_content.replace(instruction.target.search_pattern, instruction.content)
            if new_content == original_content:
                return False, f"Search pattern not found: {instruction.target.search_pattern}", []
        elif instruction.target.function_name:
            # Simple function replacement - replace entire function
            import re
            func_pattern = rf"def {re.escape(instruction.target.function_name)}\([^)]*\):.*?(?=\ndef|\nclass|\Z)"
            new_content = re.sub(func_pattern, instruction.content, original_content, flags=re.DOTALL)
            if new_content == original_content:
                return False, f"Function {instruction.target.function_name} not found for replacement", []
        else:
            # Default: append content to end of file
            new_content = original_content + "\n\n" + instruction.content
        
        # Write the fixed content
        file_path.write_text(new_content)
        logger.info(f"Applied code fix to {file_path}")
        
        return True, f"Successfully applied code fix to {instruction.target.file_path}", [instruction.target.file_path]
        
    except Exception as e:
        return False, f"Failed to apply code fix: {e}", []

def replace_function(instruction: ScriptInstruction, context: ScriptExecutionContext) -> Tuple[bool, str, List[str]]:
    """Replace a specific function in a file."""
    return fix_code_issue(instruction, context)

def insert_code(instruction: ScriptInstruction, context: ScriptExecutionContext) -> Tuple[bool, str, List[str]]:
    """Insert code at a specific location."""
    if not instruction.target or not instruction.content:
        return False, "Missing target or content for code insertion", []
    
    workspace = Path(context.workspace_path)
    file_path = workspace / instruction.target.file_path
    
    if context.dry_run:
        return True, f"[DRY RUN] Would insert code into {instruction.target.file_path}", [instruction.target.file_path]
    
    try:
        if not file_path.exists():
            # Create new file
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(instruction.content)
        else:
            # Insert into existing file
            original_content = file_path.read_text()
            lines = original_content.splitlines()
            
            insert_line = instruction.target.line_start or len(lines)
            new_lines = lines[:insert_line] + instruction.content.splitlines() + lines[insert_line:]
            file_path.write_text('\n'.join(new_lines))
        
        return True, f"Successfully inserted code into {instruction.target.file_path}", [instruction.target.file_path]
        
    except Exception as e:
        return False, f"Failed to insert code: {e}", []

def delete_code(instruction: ScriptInstruction, context: ScriptExecutionContext) -> Tuple[bool, str, List[str]]:
    """Delete code from a specific location."""
    if not instruction.target:
        return False, "Missing target for code deletion", []
    
    workspace = Path(context.workspace_path)
    file_path = workspace / instruction.target.file_path
    
    if context.dry_run:
        return True, f"[DRY RUN] Would delete code from {instruction.target.file_path}", [instruction.target.file_path]
    
    try:
        if not file_path.exists():
            return False, f"Target file does not exist: {file_path}", []
        
        original_content = file_path.read_text()
        
        if instruction.target.line_start and instruction.target.line_end:
            # Delete specific lines
            lines = original_content.splitlines()
            new_lines = lines[:instruction.target.line_start-1] + lines[instruction.target.line_end:]
            file_path.write_text('\n'.join(new_lines))
        elif instruction.target.search_pattern:
            # Delete pattern
            new_content = original_content.replace(instruction.target.search_pattern, '')
            if new_content == original_content:
                return False, f"Search pattern not found: {instruction.target.search_pattern}", []
            file_path.write_text(new_content)
        else:
            return False, "No deletion target specified", []
        
        return True, f"Successfully deleted code from {instruction.target.file_path}", [instruction.target.file_path]
        
    except Exception as e:
        return False, f"Failed to delete code: {e}", []

def run_command(instruction: ScriptInstruction, context: ScriptExecutionContext) -> Tuple[bool, str, List[str]]:
    """Execute a shell command."""
    if not instruction.command:
        return False, "No command specified", []
    
    if context.dry_run:
        return True, f"[DRY RUN] Would execute command: {instruction.command}", []
    
    try:
        # Run command in workspace directory
        result = subprocess.run(
            instruction.command,
            shell=True,
            cwd=context.workspace_path,
            capture_output=True,
            text=True,
            timeout=context.timeout_seconds
        )
        
        if result.returncode == 0:
            return True, f"Command executed successfully: {instruction.command}", []
        else:
            return False, f"Command failed (exit {result.returncode}): {result.stderr}", []
            
    except subprocess.TimeoutExpired:
        return False, f"Command timed out: {instruction.command}", []
    except Exception as e:
        return False, f"Failed to execute command: {e}", []

def validate_instruction(instruction: ScriptInstruction, context: ScriptExecutionContext) -> Tuple[bool, str, List[str]]:
    """Run validation for an instruction."""
    if not instruction.validation:
        return False, "No validation specified", []
    
    success = run_validation(instruction.validation, context)
    message = f"Validation {'passed' if success else 'failed'}"
    
    return success, message, []

def run_validation(validation_spec: ValidationSpec, context: ScriptExecutionContext) -> bool:
    """Run validation according to the spec."""
    if context.dry_run:
        return True  # Assume validation would pass in dry run
    
    if validation_spec.test_command:
        try:
            result = subprocess.run(
                validation_spec.test_command,
                shell=True,
                cwd=context.workspace_path,
                capture_output=True,
                text=True,
                timeout=validation_spec.timeout_seconds
            )
            return result.returncode == 0
        except Exception as e:
            logger.error(f"Validation command failed: {e}")
            return False
    
    if validation_spec.expected_files:
        workspace = Path(context.workspace_path)
        for file_path in validation_spec.expected_files:
            full_path = workspace / file_path
            if not full_path.exists():
                return False
    
    # If no specific validation, assume success
    return True

def execute_script(script: InstructionScript, context: ScriptExecutionContext) -> ScriptExecutionResult:
    """Execute all instructions in a script."""
    start_time = time.time()
    
    # Setup backup if required
    backup_dir = None
    if script.backup_required and context.backup_enabled and not context.dry_run:
        backup_dir = setup_backup(context, script)
    
    # Determine execution order
    execution_order = script.execution_order or [inst.instruction_id for inst in script.instructions]
    instruction_map = {inst.instruction_id: inst for inst in script.instructions}
    
    instruction_results = []
    files_modified = []
    successful_count = 0
    failed_count = 0
    
    # Execute each instruction in order
    for i, instruction_id in enumerate(execution_order):
        instruction = instruction_map.get(instruction_id)
        if not instruction:
            logger.warning(f"Instruction '{instruction_id}' not found in script")
            continue
        
        logger.info(f"Executing instruction {i+1}/{len(execution_order)}: {instruction.description}")
        
        result = execute_instruction(instruction, context)
        instruction_results.append(result)
        
        if result.success:
            successful_count += 1
            if result.files_modified:
                files_modified.extend(result.files_modified)
        else:
            failed_count += 1
            logger.error(f"Instruction {instruction_id} failed: {result.message}")
            
            # Check if we should stop on failure
            if instruction.priority == PriorityLevel.CRITICAL:
                logger.error(f"Critical instruction failed, stopping execution: {result.message}")
                break
    
    # Run final validation if specified
    final_validation_passed = None
    if script.final_validation:
        logger.info(f"Running final validation: {script.final_validation.test_command or 'Custom validation'}")
        final_validation_passed = run_validation(script.final_validation, context)
    
    # Create execution result
    execution_time = time.time() - start_time
    success = failed_count == 0 and (final_validation_passed is not False)
    
    result = ScriptExecutionResult(
        script_id=script.script_id,
        success=success,
        message=f"Executed {successful_count}/{len(execution_order)} instructions successfully" + 
               (f", final validation {'passed' if final_validation_passed else 'failed'}" if final_validation_passed is not None else ""),
        total_instructions=len(execution_order),
        successful_instructions=successful_count,
        failed_instructions=failed_count,
        instruction_results=instruction_results,
        files_modified=list(set(files_modified)),
        execution_time=execution_time,
        final_validation_passed=final_validation_passed,
        rollback_available=backup_dir is not None
    )
    
    return result

def parse_script_from_json(script_json: str) -> InstructionScript:
    """Parse a script from JSON string."""
    try:
        script_dict = json.loads(script_json)
        return parse_script_from_dict(script_dict)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")

def parse_script_from_dict(script_dict: Dict[str, Any]) -> InstructionScript:
    """Parse a script from dictionary."""
    instructions = []
    for inst_dict in script_dict.get("instructions", []):
        # Parse target if present
        target = None
        if "target" in inst_dict:
            target_dict = inst_dict["target"]
            target = CodeTarget(**target_dict)
        
        # Parse validation if present
        validation = None
        if "validation" in inst_dict:
            validation_dict = inst_dict["validation"]
            validation = ValidationSpec(**validation_dict)
        
        # Create instruction
        instruction = ScriptInstruction(
            instruction_id=inst_dict["instruction_id"],
            instruction_type=ModificationType(inst_dict["instruction_type"]),
            description=inst_dict["description"],
            target=target,
            content=inst_dict.get("content"),
            command=inst_dict.get("command"),
            validation=validation,
            priority=PriorityLevel(inst_dict.get("priority", "medium"))
        )
        instructions.append(instruction)
    
    # Parse final validation if present
    final_validation = None
    if "final_validation" in script_dict:
        final_validation = ValidationSpec(**script_dict["final_validation"])
    
    return InstructionScript(
        script_id=script_dict["script_id"],
        title=script_dict["title"],
        description=script_dict["description"],
        created_by=script_dict["created_by"],
        target_issue=script_dict["target_issue"],
        instructions=instructions,
        execution_order=script_dict.get("execution_order"),
        backup_required=script_dict.get("backup_required", True),
        estimated_duration=script_dict.get("estimated_duration"),
        final_validation=final_validation
    )

def validate_script(script: InstructionScript) -> Tuple[bool, List[str]]:
    """Validate a script for common issues."""
    errors = []
    
    if not script.instructions:
        errors.append("Script has no instructions")
    
    # Check for duplicate instruction IDs
    instruction_ids = [inst.instruction_id for inst in script.instructions]
    if len(instruction_ids) != len(set(instruction_ids)):
        errors.append("Duplicate instruction IDs found")
    
    # Validate execution order
    if script.execution_order:
        for inst_id in script.execution_order:
            if inst_id not in instruction_ids:
                errors.append(f"Execution order references unknown instruction: {inst_id}")
    
    # Validate individual instructions
    for instruction in script.instructions:
        if instruction.instruction_type in [ModificationType.FIX_CODE_ISSUE, ModificationType.INSERT_CODE, ModificationType.DELETE_CODE]:
            if not instruction.target or not instruction.target.file_path:
                errors.append(f"Instruction {instruction.instruction_id} missing target file_path")
        
        if instruction.instruction_type == ModificationType.RUN_COMMAND:
            if not instruction.command:
                errors.append(f"Instruction {instruction.instruction_id} missing command")
    
    is_valid = len(errors) == 0
    return is_valid, errors