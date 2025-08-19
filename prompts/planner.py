# prompts/planner.py
import json
from typing import Dict, Any, List
from agents.planner import PlanningQuality # Assuming planner.py is in agents directory

def build_intent_analysis_prompt(goal: str, context_summary: Dict) -> str:
    """Builds a prompt to analyze intent and determine planning quality."""
    return f"""
    You are an expert software development analyst. Analyze the user's request to understand their true intent and determine the appropriate planning quality.

    USER GOAL: "{goal}"
    
    PROJECT CONTEXT:
    {json.dumps(context_summary, indent=2)}
    
    Analyze the goal and context to determine:
    1.  **intent**: A concise, one-sentence summary of the user's core objective.
    2.  **planning_quality**: The appropriate quality level for the plan. Choose one of: "FAST", "DECENT", "PRODUCTION".
        - Use "FAST" for simple requests, prototypes, or quick fixes.
        - Use "DECENT" for standard feature development.
        - Use "PRODUCTION" for complex, critical, or enterprise-grade features.

    Your response must be a single JSON object with two keys: "intent" and "planning_quality".
    
    Example:
    {{
        "intent": "Refactor the existing user authentication module to use a more secure password hashing algorithm and add integration tests.",
        "planning_quality": "PRODUCTION"
    }}
    """

def build_planning_prompt(intent: str, context_summary: Dict[str, Any], quality: PlanningQuality, quality_instructions: Dict, agent_capabilities: List[Dict]) -> str:
    """Constructs a detailed prompt to guide the LLM in generating the TaskGraph."""
    
    quality_instructions_str = "\n        ".join([f"- {inst}" for inst in quality_instructions["instructions"]])

    return f"""
    You are the PlannerAgent, a master strategist for an AI agent collective.
    Your mission is to decompose a user's intent into a detailed TaskGraph in JSON format.
    
    **User Intent:**
    {intent}

    **Planning Quality Level: {quality.value.upper()}**
    **Instructions for this quality level:**
    {quality_instructions_str}

    **Available Agents (Your Tools):**
    {json.dumps(agent_capabilities, indent=2)}

    **Current Workspace Context:**
    {json.dumps(context_summary, indent=2)}

    **General Instructions:**
    1.  Create a 'TaskNode' for each step with a unique `task_id`.
    2.  Assign the most appropriate agent from the list of available agents.
    3.  Define `dependencies` for each task using the `task_id` of prerequisite tasks.
    4.  Define `input_artifact_keys` and `output_artifact_keys` for data flow.
    5.  CRITICAL RULE: After any task that modifies code (e.g., CoderAgent), you MUST add a subsequent task to verify the change (e.g., TestRunnerAgent or CodeAnalysisAgent).
    6.  Your output MUST be a valid JSON object representing the TaskGraph.

    **JSON Output Format:**
    {{
        "nodes": {{
            "task_id_1": {{ "task_id": "...", "goal": "...", ... }},
            "task_id_2": {{ "task_id": "...", "goal": "...", ... }}
        }}
    }}

    Now, generate the TaskGraph JSON for the provided user intent.
    """
