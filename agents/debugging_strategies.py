# agents/debugging_strategies.py
import logging
import json
import re
import uuid
from typing import Dict, Any, List, Optional

# Foundational dependencies
from .base import BaseAgent
from core.context import GlobalContext
from core.models import AgentResult
from core.instruction_schemas import InstructionScript, InstructionType

logger = logging.getLogger(__name__)

def _discover_log_configurations(agent_registry: Dict[str, BaseAgent], context: GlobalContext) -> List[str]:
    """
    A helper strategy to discover logging configurations within the codebase.
    Returns a list of potential log file paths.
    """
    discovered_paths = []
    tooling_agent = agent_registry.get("ToolingAgent")

    if not tooling_agent:
        logger.warning("ToolingAgent not available for log discovery.")
        return []

    # Grep for logging setup in Python files to find explicit filenames
    grep_command = "grep -r 'logging.basicConfig' . --include '*.py'"
    grep_result = tooling_agent.execute_v2(
        goal="Find logging setup in Python files",
        inputs={"commands": [grep_command], "purpose": "Log Config Discovery"},
        global_context=context
    )

    if grep_result.success:
        for line in grep_result.outputs.get("stdout", "").splitlines():
            match = re.search(r"filename=['\"]([^'\"]+)['\"]", line)
            if match:
                discovered_paths.append(match.group(1))

    # Clean up and return unique paths
    return sorted(list(set(p for p in discovered_paths if p)))


def scan_for_logs(problem_description: str, failure_timestamp: str, agent_registry: Dict[str, BaseAgent], context: GlobalContext) -> AgentResult:
    """
    An investigation strategy to actively scan for relevant error logs. It first
    tries to discover logging configurations and then performs a targeted, time-based search.
    """
    tooling_agent = agent_registry.get("ToolingAgent")
    if not tooling_agent:
        return AgentResult(success=False, message="ToolingAgent not available for log scanning.")

    logger.info("Discovering logging configurations in the codebase...")
    configured_log_files = _discover_log_configurations(agent_registry, context)
    
    # Define search patterns beyond simple keywords
    patterns = [
        "Traceback (most recent call last):",
        "CRITICAL",
        "FATAL",
        "ERROR",
        "Exception",
        "ValueError",
        "TypeError",
        "KeyError",
        "IndexError"
    ]
    
    # Add keywords from the problem description
    patterns.extend(problem_description.split()[:5])
    search_pattern = '|'.join(set(patterns)) # Use a unique set of patterns
    
    log_scan_commands = []

    # Prioritize searching in configured log files
    if configured_log_files:
        logger.info(f"Found potential log files: {configured_log_files}. Performing targeted, time-based scan.")
        files_to_scan = " ".join([f'"{path}"' for path in configured_log_files])
        # Use awk to filter logs since the failure timestamp
        log_scan_commands.append(f"awk '$0 >= \"{failure_timestamp}\"' {files_to_scan} | grep -iE '{search_pattern}' | tail -n 200")
    
    # Add fallback search for common locations
    logger.info("Adding fallback search for common project and system log locations.")
    log_scan_commands.extend([
        f"find . -name '*.log' -newermt '{failure_timestamp}' -print0 | xargs -0 grep -iE '{search_pattern}' | tail -n 100",
        f"journalctl --since '{failure_timestamp}' | grep -iE '{search_pattern}' | tail -n 100" # For systemd logs
    ])

    return tooling_agent.execute_v2(
        goal="Scan for relevant error logs since the last failure.",
        inputs={
            "commands": log_scan_commands,
            "purpose": "Time-based Log Scanning",
            "ignore_errors": True # Don't stop if one command fails
        },
        global_context=context
    )

def instrument_code_for_debugging(hypothesis: Dict, code_context: Dict, agent_registry: Dict[str, BaseAgent], context: GlobalContext) -> AgentResult:
    """
    An investigation strategy to dynamically add debug statements to the code.
    """
    script_executor = agent_registry.get("ScriptExecutorAgent")
    llm_client = agent_registry.get("DebuggingAgent").llm_client

    if not script_executor or not llm_client:
        return AgentResult(success=False, message="Required agents not available for instrumentation.")

    prompt = f"""
    Based on this hypothesis and code, where should I add debug print statements to get more information?
    Hypothesis: {json.dumps(hypothesis)}
    Code: {json.dumps(code_context)}

    Return a JSON object where keys are file paths and values are dictionaries mapping line numbers to the Python print statements to insert.
    Example:
    {{
      "src/main.py": {{
        "42": "print(f'DEBUG: user_id={{user.id}}, status={{user.status}}')"
      }}
    }}
    """
    try:
        response_str = llm_client.invoke(prompt)
        instrumentation_plan = json.loads(response_str)

        if not isinstance(instrumentation_plan, dict):
            return AgentResult(success=False, message="LLM failed to generate a valid instrumentation plan.")
        
        instructions = []
        for file_path, line_map in instrumentation_plan.items():
            for line_num, content in line_map.items():
                instructions.append({
                    "instruction_id": f"instrument_{file_path}_{line_num}",
                    "instruction_type": InstructionType.INSERT_CODE.value,
                    "target": {"file_path": file_path, "line_start": int(line_num)},
                    "content": content
                })
        
        script = InstructionScript(
            script_id=f"instrument_{uuid.uuid4().hex[:8]}",
            title="Dynamic Code Instrumentation",
            description="Insert debug statements to gather runtime information.",
            instructions=instructions
        )
        
        context.add_artifact("instrumentation_script.json", script.model_dump_json(indent=2), "DebuggingAgent")

        return script_executor.execute_v2(
            goal="Apply code instrumentation.",
            inputs={"instruction_script": script.model_dump()},
            global_context=context
        )

    except Exception as e:
        return AgentResult(success=False, message=f"Failed to generate or apply instrumentation: {e}")
