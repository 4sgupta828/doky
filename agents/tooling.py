# agents/tooling.py
import logging
import json
import uuid
from typing import Optional, List, Dict, Any

# Foundational dependencies
from .base import BaseAgent
from core.context import GlobalContext
from core.models import AgentResponse, AgentResult, TaskNode
from core.instruction_schemas import ToolingInstruction, ToolingExecutionResult
# --- NEW: Import the atomic tool for all shell operations ---
from tools.shell import execute_shell_command

# Get a logger instance for this module
logger = logging.getLogger(__name__)

class ToolingAgent(BaseAgent):
    """
    The team's DevOps specialist. It executes shell commands for build,
    dependency, and environment tasks using a safe, structured, and atomic tool.
    This agent now operates exclusively with a function-call (v2) interface.
    """

    def __init__(self):
        super().__init__(
            name="ToolingAgent",
            description="Executes shell commands for build, dependency, and environment tasks."
        )
    
    # === FUNCTION-CALL INTERFACE (V2) ===
    
    def required_inputs(self) -> List[str]:
        """Required inputs for ToolingAgent execution."""
        return ["commands", "purpose"]
    
    def optional_inputs(self) -> List[str]:
        """Optional inputs for ToolingAgent execution."""
        return ["working_directory", "timeout", "environment_vars", "safety_overrides", "ignore_errors"]
    
    def execute_v2(self, goal: str, inputs: Dict[str, Any], global_context: GlobalContext) -> AgentResult:
        """
        Executes tooling commands with explicit, validated inputs.
        """
        logger.info(f"ToolingAgent executing with function-call interface: '{goal}'")
        
        # Fail-fast validation
        self.validate_inputs(inputs)
        
        # Convert the function-call inputs into a structured ToolingInstruction.
        # This ensures a single, consistent execution path for all operations.
        instruction = ToolingInstruction(
            instruction_id=f"v2_exec_{uuid.uuid4().hex[:8]}",
            command_type="custom",
            commands=inputs["commands"],
            purpose=inputs["purpose"],
            working_directory=inputs.get("working_directory"),
            timeout=inputs.get("timeout", 120),
            environment_vars=inputs.get("environment_vars"),
            safety_overrides=inputs.get("safety_overrides", []),
            ignore_errors=inputs.get("ignore_errors", False)
        )
        
        try:
            # Execute the instruction
            execution_result = self._execute_tooling_instruction(instruction, global_context)
            
            # The agent's primary job is to return a structured result.
            # The orchestrator will be responsible for saving artifacts.
            return self.create_result(
                success=execution_result.success,
                message=execution_result.message,
                outputs={
                    "commands_executed": execution_result.commands_executed,
                    "command_results": execution_result.command_results,
                    "combined_output": execution_result.combined_output,
                    "duration_seconds": execution_result.total_duration
                }
            )
                
        except Exception as e:
            error_msg = f"ToolingAgent execution error: {e}"
            logger.error(error_msg, exc_info=True)
            return self.create_result(
                success=False,
                message=error_msg,
                error_details={"exception": str(e), "exception_type": type(e).__name__}
            )

    # === CORE LOGIC ===

    def _execute_tooling_instruction(self, instruction: ToolingInstruction, context: GlobalContext) -> ToolingExecutionResult:
        """Executes a structured tooling instruction by calling the atomic tool."""
        import time
        
        start_time = time.time()
        logger.info(f"Executing tooling instruction: {instruction.instruction_id}")
        
        command_results = []
        failed_commands = []
        combined_output = ""
        
        work_dir = instruction.working_directory or str(context.workspace.repo_path)
        
        for i, command in enumerate(instruction.commands):
            self.report_progress(f"Executing command {i+1}/{len(instruction.commands)}", f"{command[:50]}...")
            
            # The tool handles safety and environment setup internally.
            tool_output = execute_shell_command(
                command=command, 
                working_dir=work_dir,
                env=instruction.environment_vars
            )
            
            # Store the original command in the result for clarity
            tool_output['command'] = command
            command_results.append(tool_output)
            
            combined_output += f"=== Command: {command} ===\n"
            combined_output += f"Exit code: {tool_output['exit_code']}\n"
            combined_output += f"STDOUT:\n{tool_output['stdout']}\n"
            combined_output += f"STDERR:\n{tool_output['stderr']}\n\n"
            
            if tool_output['exit_code'] != 0:
                failed_commands.append(command)
                if not instruction.ignore_errors:
                    logger.error(f"Command failed, stopping execution: {command}")
                    break
        
        total_duration = time.time() - start_time
        success = len(failed_commands) == 0
        
        message = f"Executed {len(instruction.commands)} commands. Success: {success}"
        
        # Store combined output as an artifact for debugging purposes
        output_artifact_key = f"{instruction.instruction_id}_output.txt"
        context.add_artifact(output_artifact_key, combined_output, "ToolingAgent")
        
        return ToolingExecutionResult(
            instruction_id=instruction.instruction_id,
            success=success,
            message=message,
            commands_executed=[res['command'] for res in command_results],
            command_results=command_results,
            total_duration=total_duration,
            combined_output=combined_output,
            artifacts_generated=[output_artifact_key],
            failed_commands=failed_commands
        )
