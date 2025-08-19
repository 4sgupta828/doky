# agents/debugging_strategies.py
import logging
import json
import re
from typing import Dict, Any, List

# Foundational dependencies
from .base import BaseAgent
from core.context import GlobalContext
from core.models import AgentResult

logger = logging.getLogger(__name__)

def _discover_log_configurations(agent_registry: Dict[str, BaseAgent], context: GlobalContext) -> List[str]:
    """
    A helper strategy to discover logging configurations within the codebase.
    Returns a list of potential log file paths.
    """
    discovered_paths = []
    file_system_agent = agent_registry.get("FileSystemAgent")
    tooling_agent = agent_registry.get("ToolingAgent")

    if not file_system_agent or not tooling_agent:
        logger.warning("Required agents for log discovery are not available.")
        return []

    # 1. Look for common logging configuration files
    discovery_result = file_system_agent.execute_v2(
        goal="Discover logging configuration files",
        inputs={
            "operation": "discover",
            "patterns": ["logging.conf", "logging.ini", "logging.yaml", "*.log"],
            "recursive": True
        },
        global_context=context
    )
    if discovery_result.success and discovery_result.outputs.get("discovered_files"):
        discovered_paths.extend(discovery_result.outputs["discovered_files"])

    # 2. Grep for logging setup in Python files
    grep_command = "grep -r 'logging.basicConfig' . --include '*.py'"
    grep_result = tooling_agent.execute_v2(
        goal="Find logging setup in Python files",
        inputs={"commands": [grep_command], "purpose": "Log Config Discovery"},
        global_context=context
    )

    if grep_result.success:
        # Parse the output to find filename arguments
        for line in grep_result.outputs.get("stdout", "").splitlines():
            match = re.search(r"filename=['\"]([^'\"]+)['\"]", line)
            if match:
                discovered_paths.append(match.group(1))

    # Clean up and return unique paths
    return sorted(list(set(p for p in discovered_paths if p)))


def scan_for_logs(problem_description: str, agent_registry: Dict[str, BaseAgent], context: GlobalContext) -> AgentResult:
    """
    An investigation strategy to actively scan for relevant error logs. It first
    tries to discover logging configurations and falls back to common locations.
    """
    tooling_agent = agent_registry.get("ToolingAgent")
    if not tooling_agent:
        return AgentResult(success=False, message="ToolingAgent not available for log scanning.")

    # --- NEW: Two-Step Investigation ---
    # 1. Discover configured log files
    logger.info("Discovering logging configurations in the codebase...")
    configured_log_files = _discover_log_configurations(agent_registry, context)
    
    keywords = problem_description.split()[:5]
    keyword_pattern = '|'.join(keywords)
    log_scan_commands = []

    # 2. Create targeted search commands if configs are found
    if configured_log_files:
        logger.info(f"Found potential log files: {configured_log_files}. Performing targeted scan.")
        # Create a single, powerful grep command for all discovered files
        files_to_scan = " ".join([f'"{path}"' for path in configured_log_files])
        log_scan_commands.append(f"grep -iE '{keyword_pattern}' {files_to_scan} | tail -n 100")
    
    # 3. Always include fallback search commands
    logger.info("Adding fallback search for common log locations.")
    log_scan_commands.extend([
        f"find . -name '*.log' -print0 | xargs -0 grep -iE '{keyword_pattern}' | tail -n 50",
        f"grep -iE '{keyword_pattern}' /var/log/*.log /var/log/syslog | tail -n 50"
    ])

    logger.info(f"Scanning for logs with {len(log_scan_commands)} command groups.")

    return tooling_agent.execute_v2(
        goal="Scan for relevant error logs based on the problem description.",
        inputs={
            "commands": log_scan_commands,
            "purpose": "Log Scanning",
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

        from core.instruction_schemas import InstructionScript, InstructionType
        
        instructions = []
        for file_path, line_map in instrumentation_plan.items():
            for line_num, content in line_map.items():
                instructions.append({
                    "instruction_id": f"instrument_{file_path}_{line_num}",
                    "instruction_type": InstructionType.INSERT_CODE,
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
