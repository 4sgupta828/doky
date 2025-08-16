# agents/tooling.py
import logging
import subprocess
from typing import Optional, List

# Foundational dependencies
from .base import BaseAgent
from core.context import GlobalContext
from core.models import AgentResponse, TaskNode
from core.instruction_schemas import ToolingInstruction, ToolingExecutionResult

# Get a logger instance for this module
logger = logging.getLogger(__name__)

class ToolingAgent(BaseAgent):
    """
    The team's DevOps specialist. It is the only agent authorized to run
    shell commands. This centralization ensures that all environment
    interactions are safe, logged, and controlled.
    """

    def __init__(self):
        super().__init__(
            name="ToolingAgent",
            description="Executes shell commands for build, dependency, and environment tasks."
        )
        # A set of commands that are not allowed for safety reasons.
        self.disallowed_commands = {"rm", "mv", "dd", "mkfs", "shutdown", "reboot"}

    def _is_command_safe(self, command: str) -> bool:
        """A simple safety check to prevent destructive commands."""
        if not command.strip():
            logger.error("Empty command blocked for safety.")
            return False
        first_word = command.strip().split()[0]
        if first_word in self.disallowed_commands:
            logger.error(f"Execution of disallowed command '{first_word}' was blocked.")
            return False
        # Log successful safety validation for transparency
        logger.debug(f"Command '{first_word}' passed safety validation.")
        return True

    def execute(self, goal: str, context: GlobalContext, current_task: TaskNode) -> AgentResponse:
        """Execute structured tooling instructions instead of free-form commands."""
        logger.info(f"ToolingAgent executing structured operation: '{goal}'")
        
        # Report meaningful progress to user
        self.report_progress("Loading tooling instruction", f"Processing: '{goal[:60]}...'")
        
        try:
            # 1. Look for structured tooling instruction in context
            tooling_instruction = self._load_tooling_instruction(context, current_task)
            if not tooling_instruction:
                # Fallback: treat goal as single command for backward compatibility
                logger.warning("No structured tooling instruction found, falling back to legacy mode")
                return self._execute_legacy_command(goal, context, current_task)
            
            self.report_thinking(f"Loaded structured tooling instruction '{tooling_instruction.instruction_id}' with {len(tooling_instruction.commands)} commands of type '{tooling_instruction.command_type}'")
            self.report_intermediate_output("tooling_instruction", {
                "instruction_id": tooling_instruction.instruction_id,
                "command_type": tooling_instruction.command_type,
                "purpose": tooling_instruction.purpose,
                "commands_count": len(tooling_instruction.commands),
                "timeout": tooling_instruction.timeout
            })
            
            # 2. Execute the structured instruction
            self.report_progress("Executing structured commands", f"Running {len(tooling_instruction.commands)} commands for: {tooling_instruction.purpose}")
            execution_result = self._execute_tooling_instruction(tooling_instruction, context, current_task)
            
            # 3. Store results and create artifacts
            context.add_artifact(
                f"tooling_execution_result_{tooling_instruction.instruction_id}.json", 
                execution_result.model_dump_json(indent=2), 
                current_task.task_id
            )
            
            # 4. Report final results
            if execution_result.success:
                self.report_progress("Tooling operation successful", 
                    f"Completed {len(execution_result.commands_executed)}/{len(tooling_instruction.commands)} commands successfully")
                self.report_thinking(f"Successfully executed tooling operation '{tooling_instruction.purpose}'. All commands completed without errors.")
                
                return AgentResponse(
                    success=True,
                    message=f"Successfully executed tooling operation: {tooling_instruction.purpose}",
                    artifacts_generated=[f"tooling_execution_result_{tooling_instruction.instruction_id}.json"] + execution_result.artifacts_generated
                )
            else:
                self.report_progress("Tooling operation failed", 
                    f"Failed commands: {len(execution_result.failed_commands)}/{len(tooling_instruction.commands)}")
                
                return AgentResponse(
                    success=False,
                    message=f"Tooling operation failed: {execution_result.message}",
                    artifacts_generated=[f"tooling_execution_result_{tooling_instruction.instruction_id}.json"]
                )
                
        except Exception as e:
            error_msg = f"ToolingAgent encountered unexpected error: {e}"
            logger.error(error_msg, exc_info=True)
            return AgentResponse(success=False, message=error_msg)

    def _load_tooling_instruction(self, context: GlobalContext, current_task: TaskNode) -> Optional[ToolingInstruction]:
        """Load structured tooling instruction from context artifacts."""
        # Look for instruction in different possible artifact keys
        instruction_keys = [
            "tooling_instruction.json",
            f"tooling_instruction_{current_task.task_id}.json",
            "diagnostic_instruction.json"
        ]
        
        for key in instruction_keys:
            instruction_data = context.get_artifact(key)
            if instruction_data:
                try:
                    if isinstance(instruction_data, str):
                        import json
                        instruction_dict = json.loads(instruction_data)
                    else:
                        instruction_dict = instruction_data
                    
                    return ToolingInstruction(**instruction_dict)
                except Exception as e:
                    logger.warning(f"Failed to parse tooling instruction from artifact '{key}': {e}")
                    continue
        
        return None

    def _execute_tooling_instruction(self, instruction: ToolingInstruction, context: GlobalContext, current_task: TaskNode) -> ToolingExecutionResult:
        """Execute a structured tooling instruction."""
        import time
        from datetime import datetime
        
        start_time = time.time()
        logger.info(f"Executing tooling instruction: {instruction.instruction_id} ({instruction.command_type})")
        
        commands_executed = []
        command_results = []
        failed_commands = []
        artifacts_generated = []
        combined_output = ""
        
        # Determine working directory
        work_dir = instruction.working_directory or str(context.workspace.repo_path)
        
        # Execute each command in sequence
        for i, command in enumerate(instruction.commands):
            self.report_progress(f"Executing command {i+1}/{len(instruction.commands)}", 
                               f"{instruction.command_type}: {command[:50]}...")
            
            # Check if command is allowed (with safety overrides)
            if not self._is_command_safe_with_overrides(command, instruction.safety_overrides):
                failed_commands.append(command)
                command_result = {
                    "command": command,
                    "success": False,
                    "exit_code": -1,
                    "stdout": "",
                    "stderr": "Command blocked by safety check",
                    "duration": 0
                }
                command_results.append(command_result)
                
                if not instruction.ignore_errors:
                    break
                continue
            
            # Execute the command
            command_start = time.time()
            try:
                # Set up environment if specified
                env = None
                if instruction.environment_vars:
                    import os
                    env = os.environ.copy()
                    env.update(instruction.environment_vars)
                
                result = subprocess.run(
                    command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    cwd=work_dir,
                    timeout=instruction.timeout,
                    env=env
                )
                
                command_duration = time.time() - command_start
                
                command_result = {
                    "command": command,
                    "success": result.returncode == 0,
                    "exit_code": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                    "duration": command_duration
                }
                
                commands_executed.append(command)
                command_results.append(command_result)
                combined_output += f"=== Command: {command} ===\n"
                combined_output += f"Exit code: {result.returncode}\n"
                combined_output += f"STDOUT:\n{result.stdout}\n"
                combined_output += f"STDERR:\n{result.stderr}\n\n"
                
                # Report command completion
                if result.returncode == 0:
                    self.report_intermediate_output(f"command_{i+1}_stdout", result.stdout[:500] if result.stdout.strip() else "No output")
                else:
                    failed_commands.append(command)
                    self.report_intermediate_output(f"command_{i+1}_error", result.stderr[:500] if result.stderr.strip() else f"Exit code: {result.returncode}")
                    
                    if not instruction.ignore_errors:
                        logger.error(f"Command failed: {command} (exit code: {result.returncode})")
                        break
                
            except subprocess.TimeoutExpired:
                command_duration = time.time() - command_start
                failed_commands.append(command)
                command_result = {
                    "command": command,
                    "success": False,
                    "exit_code": -2,
                    "stdout": "",
                    "stderr": f"Command timed out after {instruction.timeout} seconds",
                    "duration": command_duration
                }
                command_results.append(command_result)
                combined_output += f"=== Command: {command} ===\n"
                combined_output += f"TIMEOUT after {instruction.timeout} seconds\n\n"
                
                if not instruction.ignore_errors:
                    break
                    
            except Exception as e:
                command_duration = time.time() - command_start
                failed_commands.append(command)
                command_result = {
                    "command": command,
                    "success": False,
                    "exit_code": -3,
                    "stdout": "",
                    "stderr": f"Execution error: {e}",
                    "duration": command_duration
                }
                command_results.append(command_result)
                combined_output += f"=== Command: {command} ===\n"
                combined_output += f"ERROR: {e}\n\n"
                
                if not instruction.ignore_errors:
                    break
        
        # Check for expected artifacts
        if instruction.expected_artifacts:
            for artifact_path in instruction.expected_artifacts:
                full_path = context.workspace.repo_path / artifact_path
                if full_path.exists():
                    artifacts_generated.append(artifact_path)
        
        # Store combined output as artifact
        output_artifact_key = f"{instruction.instruction_id}_combined_output.txt"
        context.add_artifact(output_artifact_key, combined_output, current_task.task_id)
        artifacts_generated.append(output_artifact_key)
        
        # Determine overall success
        total_duration = time.time() - start_time
        success = len(failed_commands) == 0
        
        # Create detailed result message
        if success:
            message = f"Successfully executed {len(commands_executed)} commands for {instruction.purpose}"
        else:
            message = f"Failed to execute {len(failed_commands)} out of {len(instruction.commands)} commands: {', '.join(failed_commands[:3])}{'...' if len(failed_commands) > 3 else ''}"
        
        return ToolingExecutionResult(
            instruction_id=instruction.instruction_id,
            success=success,
            message=message,
            commands_executed=commands_executed,
            command_results=command_results,
            total_duration=total_duration,
            combined_output=combined_output,
            artifacts_generated=artifacts_generated,
            failed_commands=failed_commands,
            error_details=f"Failed commands: {failed_commands}" if failed_commands else None
        )

    def _is_command_safe_with_overrides(self, command: str, safety_overrides: Optional[List[str]] = None) -> bool:
        """Check if command is safe, considering safety overrides."""
        if not command.strip():
            logger.error("Empty command blocked for safety.")
            return False
            
        first_word = command.strip().split()[0]
        
        # Check if this command is specifically allowed via overrides
        if safety_overrides and first_word in safety_overrides:
            logger.info(f"Command '{first_word}' allowed via safety override")
            return True
            
        # Use standard safety check
        return self._is_command_safe(command)

    def _execute_legacy_command(self, command_to_run: str, context: GlobalContext, current_task: TaskNode) -> AgentResponse:
        """Legacy method for backward compatibility - executes single command."""
        logger.warning("Using legacy command execution mode - consider migrating to structured tooling instructions")
        
        # Report thinking about safety validation
        self.report_thinking(f"Legacy mode: Validating command safety for '{command_to_run.split()[0] if command_to_run.strip() else 'empty'}'")

        if not self._is_command_safe(command_to_run):
            self.report_progress("Command blocked", f"Safety check failed for command: {command_to_run.split()[0] if command_to_run.strip() else 'empty'}")
            return AgentResponse(success=False, message=f"Command '{command_to_run}' is disallowed for safety reasons.")

        try:
            # Report execution start with transparency about environment
            self.report_progress("Executing legacy command", f"Running in workspace: {str(context.workspace.repo_path)}")
            self.report_thinking(f"Legacy command will execute with 120-second timeout in directory: {str(context.workspace.repo_path)}")
            
            # The command is executed within the context of the agent's workspace.
            result = subprocess.run(
                command_to_run,
                shell=True,
                capture_output=True,
                text=True,
                cwd=str(context.workspace.repo_path),
                timeout=120  # A timeout is a crucial safety feature.
            )

            # Report command completion status
            self.report_progress("Legacy command completed", f"Exit code: {result.returncode}, stdout: {len(result.stdout)} chars, stderr: {len(result.stderr)} chars")
            
            # Create an artifact with the combined output for easier debugging.
            output_content = f"--- STDOUT ---\n{result.stdout}\n\n--- STDERR ---\n{result.stderr}"
            output_artifact_key = f"{current_task.task_id}_legacy_output.txt"
            context.add_artifact(output_artifact_key, output_content, current_task.task_id)
            
            # Report intermediate output for transparency
            if result.stdout.strip():
                self.report_intermediate_output("command_stdout", result.stdout.strip()[:1000])  # Truncate for display
            if result.stderr.strip():
                self.report_intermediate_output("command_stderr", result.stderr.strip()[:1000])  # Truncate for display

            if result.returncode != 0:
                msg = f"Legacy command failed with exit code {result.returncode}."
                logger.error(f"{msg}\n{output_content}")
                self.report_thinking(f"Legacy command failed with exit code {result.returncode}. Consider using structured tooling instructions for better error handling.")
                return AgentResponse(success=False, message=msg, artifacts_generated=[output_artifact_key])

            self.report_progress("Legacy command successful", f"Execution completed successfully with exit code 0")
            return AgentResponse(success=True, message="Legacy command executed successfully.", artifacts_generated=[output_artifact_key])

        except subprocess.TimeoutExpired:
            msg = f"Legacy command '{command_to_run}' timed out after 120 seconds."
            logger.error(msg)
            self.report_thinking(f"Legacy command exceeded timeout. Consider using structured tooling instructions with custom timeouts.")
            return AgentResponse(success=False, message=msg)
        except Exception as e:
            error_msg = f"An unexpected error occurred while running legacy command: {e}"
            logger.critical(error_msg, exc_info=True)
            self.report_thinking(f"Unexpected error during legacy command execution: {e}. Consider migrating to structured tooling instructions.")
            return AgentResponse(success=False, message=error_msg)


# --- Self-Testing Block ---
if __name__ == "__main__":
    import unittest
    from unittest.mock import patch
    import shutil
    from pathlib import Path
    from utils.logger import setup_logger

    setup_logger(default_level=logging.INFO)

    class TestToolingAgent(unittest.TestCase):

        def setUp(self):
            self.test_workspace_path = "./temp_tooling_test_ws"
            if Path(self.test_workspace_path).exists():
                shutil.rmtree(self.test_workspace_path)
            self.context = GlobalContext(workspace_path=self.test_workspace_path)
            self.agent = ToolingAgent()

        def tearDown(self):
            shutil.rmtree(self.test_workspace_path)

        @patch('subprocess.run')
        def test_tooling_agent_success(self, mock_subprocess_run):
            """Tests the ToolingAgent's successful command execution."""
            print("\n--- [Test Case 1: ToolingAgent Success] ---")
            mock_subprocess_run.return_value = subprocess.CompletedProcess(
                args=['ls', '-l'], returncode=0, stdout='total 0\n-rw-r--r-- 1 user group 0 Aug 13 18:20 file.txt', stderr=''
            )
            task = TaskNode(goal="ls -l", assigned_agent="ToolingAgent")

            response = self.agent.execute(task.goal, self.context, task)

            mock_subprocess_run.assert_called_once_with(
                "ls -l", shell=True, capture_output=True, text=True, cwd=str(self.context.workspace.repo_path), timeout=120
            )
            self.assertTrue(response.success)
            output_artifact = self.context.get_artifact(f"{task.task_id}_output.txt")
            self.assertIn("file.txt", output_artifact)
            logger.info("✅ test_tooling_agent_success: PASSED")

        @patch('subprocess.run')
        def test_tooling_agent_failure(self, mock_subprocess_run):
            """Tests the ToolingAgent when a command fails (non-zero exit code)."""
            print("\n--- [Test Case 2: ToolingAgent Command Failure] ---")
            mock_subprocess_run.return_value = subprocess.CompletedProcess(
                args=['cat', 'nonexistent.txt'], returncode=1, stdout='', stderr='cat: nonexistent.txt: No such file or directory'
            )
            task = TaskNode(goal="cat nonexistent.txt", assigned_agent="ToolingAgent")

            response = self.agent.execute(task.goal, self.context, task)

            self.assertFalse(response.success)
            self.assertIn("Command failed with exit code 1", response.message)
            output_artifact = self.context.get_artifact(f"{task.task_id}_output.txt")
            self.assertIn("No such file or directory", output_artifact)
            logger.info("✅ test_tooling_agent_failure: PASSED")

        def test_tooling_agent_disallowed_command(self):
            """Tests that the ToolingAgent blocks unsafe commands."""
            print("\n--- [Test Case 3: ToolingAgent Disallowed Command] ---")
            task = TaskNode(goal="rm -rf /", assigned_agent="ToolingAgent")
            response = self.agent.execute(task.goal, self.context, task)
            self.assertFalse(response.success)
            self.assertIn("disallowed for safety reasons", response.message)
            logger.info("✅ test_tooling_agent_disallowed_command: PASSED")

    unittest.main(argv=['first-arg-is-ignored'], exit=False)