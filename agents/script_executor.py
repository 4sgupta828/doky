# agents/script_executor.py
import json
import logging
import os
import subprocess
import tempfile
import time
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

# Foundational dependencies
from .base import BaseAgent
from core.context import GlobalContext
from core.models import AgentResponse, TaskNode
from core.instruction_schemas import (
    InstructionScript, StructuredInstruction, InstructionResult, ScriptExecutionResult,
    InstructionType, CodeTarget
)

# Get a logger instance for this module
logger = logging.getLogger(__name__)


class ScriptExecutorAgent(BaseAgent):
    """
    A specialized agent designed to execute structured repair scripts with precision.
    
    This agent is the tactical counterpart to the strategic debugging agent. It:
    1. Parses structured instruction scripts
    2. Executes code modifications with surgical precision
    3. Validates each step and provides rollback capabilities
    4. Reports back with detailed execution status
    
    Unlike CodeGenerationAgent (which creates new code from specs), this agent
    executes specific, tactical changes to existing code.
    """

    def __init__(self, llm_client: Any = None, backup_enabled: bool = True):
        super().__init__(
            name="ScriptExecutorAgent",
            description="Executes structured repair scripts with precision, validation, and rollback capabilities."
        )
        self.llm_client = llm_client
        self.backup_enabled = backup_enabled
        self.backup_dir = None
        self.execution_history: List[ScriptExecutionResult] = []

    def execute(self, goal: str, context: GlobalContext, current_task: TaskNode) -> AgentResponse:
        """Execute a structured instruction script."""
        logger.info(f"ScriptExecutorAgent executing with goal: '{goal}'")
        self.report_progress("Initializing script execution", f"Processing: '{goal[:80]}...'")
        
        try:
            # 1. Look for instruction script in context
            script = self._load_instruction_script(context, current_task)
            if not script:
                return AgentResponse(
                    success=False,
                    message="No instruction script found in context. Expected 'instruction_script.json' artifact."
                )
            
            self.report_thinking(f"Loaded instruction script '{script.title}' with {len(script.instructions)} instructions")
            self.report_intermediate_output("instruction_script", {
                "title": script.title,
                "description": script.description,
                "instructions_count": len(script.instructions),
                "estimated_duration": script.estimated_duration
            })
            
            # 2. Setup backup if required
            if script.backup_required and self.backup_enabled:
                self._setup_backup(context, script)
                self.report_progress("Backup created", f"Files backed up to: {self.backup_dir}")
            
            # 3. Execute the script
            self.report_progress("Executing script", f"Running {len(script.instructions)} instructions")
            execution_result = self._execute_script(script, context, current_task)
            
            # 4. Store results and history
            self.execution_history.append(execution_result)
            context.add_artifact(
                f"script_execution_result_{script.script_id}.json", 
                execution_result.model_dump_json(indent=2), 
                current_task.task_id
            )
            
            # 5. Report final results
            if execution_result.success:
                self.report_progress("Script execution successful", 
                    f"Completed {execution_result.successful_instructions}/{execution_result.total_instructions} instructions")
                self.report_thinking(f"Successfully executed script '{script.title}'. All validations passed.")
                
                return AgentResponse(
                    success=True,
                    message=f"Successfully executed script '{script.title}': {execution_result.message}",
                    artifacts_generated=[f"script_execution_result_{script.script_id}.json"] + execution_result.files_modified
                )
            else:
                self.report_progress("Script execution failed", 
                    f"Failed at instruction {execution_result.failed_instructions}/{execution_result.total_instructions}")
                
                # Offer rollback if available
                rollback_msg = ""
                if execution_result.rollback_available:
                    rollback_msg = " Rollback is available if needed."
                    
                return AgentResponse(
                    success=False,
                    message=f"Script execution failed: {execution_result.message}{rollback_msg}",
                    artifacts_generated=[f"script_execution_result_{script.script_id}.json"]
                )
                
        except Exception as e:
            error_msg = f"ScriptExecutorAgent encountered unexpected error: {e}"
            logger.error(error_msg, exc_info=True)
            return AgentResponse(success=False, message=error_msg)

    def _load_instruction_script(self, context: GlobalContext, current_task: TaskNode) -> Optional[InstructionScript]:
        """Load instruction script from context artifacts."""
        # Look for script in different possible artifact keys
        script_keys = [
            "instruction_script.json",
            "repair_script.json",
            f"instruction_script_{current_task.task_id}.json"
        ]
        
        for key in script_keys:
            script_data = context.get_artifact(key)
            if script_data:
                try:
                    if isinstance(script_data, str):
                        script_dict = json.loads(script_data)
                    else:
                        script_dict = script_data
                    
                    return InstructionScript(**script_dict)
                except Exception as e:
                    logger.warning(f"Failed to parse script from artifact '{key}': {e}")
                    continue
        
        return None

    def _setup_backup(self, context: GlobalContext, script: InstructionScript):
        """Create backup of files that will be modified."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        workspace = Path(context.workspace_path) if isinstance(context.workspace_path, str) else context.workspace_path
        self.backup_dir = workspace / f"backups/script_{script.script_id}_{timestamp}"
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Collect all files that will be modified
        files_to_backup = set()
        for instruction in script.instructions:
            if instruction.target and instruction.target.file_path:
                files_to_backup.add(instruction.target.file_path)
        
        # Backup each file
        for file_path in files_to_backup:
            try:
                workspace = Path(context.workspace_path) if isinstance(context.workspace_path, str) else context.workspace_path
                source_path = workspace / file_path
                if source_path.exists():
                    backup_path = self.backup_dir / file_path
                    backup_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    # Copy original file to backup
                    import shutil
                    shutil.copy2(source_path, backup_path)
                    logger.info(f"Backed up {file_path} to {backup_path}")
                    
            except Exception as e:
                logger.warning(f"Failed to backup {file_path}: {e}")

    def _execute_script(self, script: InstructionScript, context: GlobalContext, current_task: TaskNode) -> ScriptExecutionResult:
        """Execute all instructions in the script."""
        start_time = time.time()
        
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
            
            self.report_progress(f"Executing instruction {i+1}/{len(execution_order)}", 
                               f"{instruction.instruction_type.value}: {instruction.description}")
            
            result = self._execute_instruction(instruction, context, current_task)
            instruction_results.append(result)
            
            if result.success:
                successful_count += 1
                if result.files_modified:
                    files_modified.extend(result.files_modified)
            else:
                failed_count += 1
                logger.error(f"Instruction {instruction_id} failed: {result.message}")
                
                # Check if we should stop on failure
                if instruction.priority == "critical":
                    self.report_thinking(f"Critical instruction failed, stopping execution: {result.message}")
                    break
        
        # Run final validation if specified
        final_validation_passed = None
        if script.final_validation:
            self.report_progress("Running final validation", script.final_validation.test_command or "Custom validation")
            final_validation_passed = self._run_validation(script.final_validation, context)
            
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
            final_validation_passed=final_validation_passed,
            rollback_available=self.backup_dir is not None
        )
        
        return result

    def _execute_instruction(self, instruction: StructuredInstruction, context: GlobalContext, current_task: TaskNode) -> InstructionResult:
        """Execute a single structured instruction."""
        start_time = time.time()
        logger.info(f"Executing instruction: {instruction.instruction_id} ({instruction.instruction_type.value})")
        
        try:
            # Execute based on instruction type
            if instruction.instruction_type == InstructionType.FIX_CODE_ISSUE:
                success, message, files_modified = self._fix_code_issue(instruction, context)
            elif instruction.instruction_type == InstructionType.REPLACE_FUNCTION:
                success, message, files_modified = self._replace_function(instruction, context)
            elif instruction.instruction_type == InstructionType.INSERT_CODE:
                success, message, files_modified = self._insert_code(instruction, context)
            elif instruction.instruction_type == InstructionType.DELETE_CODE:
                success, message, files_modified = self._delete_code(instruction, context)
            elif instruction.instruction_type == InstructionType.RUN_COMMAND:
                success, message, files_modified = self._run_command(instruction, context)
            elif instruction.instruction_type == InstructionType.VALIDATE_FIX:
                success, message, files_modified = self._validate_instruction(instruction, context)
            else:
                success, message, files_modified = False, f"Unknown instruction type: {instruction.instruction_type}", []
            
            # Run validation if specified
            validation_passed = None
            validation_output = None
            if success and instruction.validation:
                validation_passed = self._run_validation(instruction.validation, context)
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

    def _fix_code_issue(self, instruction: StructuredInstruction, context: GlobalContext) -> tuple[bool, str, List[str]]:
        """Execute a code fix instruction."""
        if not instruction.target or not instruction.content:
            return False, "Missing target or content for code fix", []
        
        workspace = Path(context.workspace_path) if isinstance(context.workspace_path, str) else context.workspace_path
        file_path = workspace / instruction.target.file_path
        
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
            elif instruction.target.function_name:
                # Simple function replacement - replace entire function
                # This is a basic implementation - in practice would use AST parsing
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

    def _replace_function(self, instruction: StructuredInstruction, context: GlobalContext) -> tuple[bool, str, List[str]]:
        """Replace a specific function in a file."""
        # This would be more sophisticated in practice, using AST parsing
        return self._fix_code_issue(instruction, context)

    def _insert_code(self, instruction: StructuredInstruction, context: GlobalContext) -> tuple[bool, str, List[str]]:
        """Insert code at a specific location."""
        if not instruction.target or not instruction.content:
            return False, "Missing target or content for code insertion", []
        
        workspace = Path(context.workspace_path) if isinstance(context.workspace_path, str) else context.workspace_path
        file_path = workspace / instruction.target.file_path
        
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

    def _delete_code(self, instruction: StructuredInstruction, context: GlobalContext) -> tuple[bool, str, List[str]]:
        """Delete code from a specific location."""
        if not instruction.target:
            return False, "Missing target for code deletion", []
        
        workspace = Path(context.workspace_path) if isinstance(context.workspace_path, str) else context.workspace_path
        file_path = workspace / instruction.target.file_path
        
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
                file_path.write_text(new_content)
            else:
                return False, "No deletion target specified", []
            
            return True, f"Successfully deleted code from {instruction.target.file_path}", [instruction.target.file_path]
            
        except Exception as e:
            return False, f"Failed to delete code: {e}", []

    def _run_command(self, instruction: StructuredInstruction, context: GlobalContext) -> tuple[bool, str, List[str]]:
        """Execute a shell command."""
        if not instruction.command:
            return False, "No command specified", []
        
        try:
            # Run command in workspace directory
            result = subprocess.run(
                instruction.command,
                shell=True,
                cwd=context.workspace_path,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                return True, f"Command executed successfully: {instruction.command}", []
            else:
                return False, f"Command failed (exit {result.returncode}): {result.stderr}", []
                
        except subprocess.TimeoutExpired:
            return False, f"Command timed out: {instruction.command}", []
        except Exception as e:
            return False, f"Failed to execute command: {e}", []

    def _validate_instruction(self, instruction: StructuredInstruction, context: GlobalContext) -> tuple[bool, str, List[str]]:
        """Run validation for an instruction."""
        if not instruction.validation:
            return False, "No validation specified", []
        
        success = self._run_validation(instruction.validation, context)
        message = f"Validation {'passed' if success else 'failed'}"
        
        return success, message, []

    def _run_validation(self, validation_spec, context: GlobalContext) -> bool:
        """Run validation according to the spec."""
        if validation_spec.test_command:
            try:
                result = subprocess.run(
                    validation_spec.test_command,
                    shell=True,
                    cwd=context.workspace_path,
                    capture_output=True,
                    text=True,
                    timeout=120  # 2 minute timeout for tests
                )
                return result.returncode == 0
            except Exception as e:
                logger.error(f"Validation command failed: {e}")
                return False
        
        if validation_spec.expected_files:
            for file_path in validation_spec.expected_files:
                workspace = Path(context.workspace_path) if isinstance(context.workspace_path, str) else context.workspace_path
                full_path = workspace / file_path
                if not full_path.exists():
                    return False
        
        # If no specific validation, assume success
        return True


# --- Self-Testing Block ---
if __name__ == "__main__":
    import unittest
    from unittest.mock import MagicMock
    import shutil
    from pathlib import Path
    from utils.logger import setup_logger
    from core.instruction_schemas import create_fix_code_instruction

    setup_logger(default_level=logging.INFO)

    class TestScriptExecutorAgent(unittest.TestCase):

        def setUp(self):
            self.test_workspace_path = "./temp_script_executor_test_ws"
            if Path(self.test_workspace_path).exists():
                shutil.rmtree(self.test_workspace_path)
            
            self.context = GlobalContext(workspace_path=self.test_workspace_path)
            self.agent = ScriptExecutorAgent()
            self.task = TaskNode(
                goal="Execute repair script",
                assigned_agent="ScriptExecutorAgent"
            )

        def tearDown(self):
            if Path(self.test_workspace_path).exists():
                shutil.rmtree(self.test_workspace_path)

        def test_successful_script_execution(self):
            """Tests successful execution of a simple repair script."""
            print("\n--- [Test Case 1: Successful Script Execution] ---")
            
            # Create test file
            test_file = Path(self.test_workspace_path) / "test.py"
            test_file.parent.mkdir(parents=True, exist_ok=True)
            test_file.write_text("def broken_function():\n    return 'broken'")
            
            # Create instruction script
            instruction = create_fix_code_instruction(
                instruction_id="fix_001",
                file_path="test.py",
                issue_description="Fix broken function",
                fix_content="def fixed_function():\n    return 'fixed'",
                line_start=1,
                line_end=2
            )
            
            script = InstructionScript(
                script_id="test_script_001",
                title="Test Repair Script",
                description="Simple test script",
                created_by="TestCase",
                target_issue="Test issue",
                instructions=[instruction]
            )
            
            # Add script to context
            self.context.add_artifact("instruction_script.json", script.model_dump_json(), "test")
            
            # Execute
            response = self.agent.execute(self.task.goal, self.context, self.task)
            
            self.assertTrue(response.success)
            self.assertIn("Successfully executed script", response.message)
            
            # Verify file was modified
            modified_content = test_file.read_text()
            self.assertIn("fixed_function", modified_content)
            
            logger.info("✅ test_successful_script_execution: PASSED")

    # Run the tests
    unittest.main(argv=['first-arg-is-ignored'], exit=False)