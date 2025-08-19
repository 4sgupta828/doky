# agents/planner.py
import json
import logging
from typing import Dict, Any, List
from enum import Enum

# Foundational dependencies
from .base import BaseAgent
from core.context import GlobalContext
from core.models import AgentResponse, AgentResult, TaskGraph, TaskNode, ValidationError

# --- NEW: Import the externalized prompt builders ---
from prompts.planner import build_intent_analysis_prompt, build_planning_prompt

# Get a logger instance for this module.
logger = logging.getLogger(__name__)


# --- Planning Quality Levels ---
class PlanningQuality(Enum):
    """Defines different planning quality levels for speed vs detail trade-offs."""
    FAST = "fast"
    DECENT = "decent"
    PRODUCTION = "production"


class PlannerAgent(BaseAgent):
    """
    The master strategist of the collective. It performs a two-step process:
    1.  Analyzes the user's goal to understand the underlying intent and required quality.
    2.  Decomposes the intent into a structured, logical TaskGraph that can be
        executed by the other specialized agents.
    """

    def __init__(self, llm_client: Any = None, agent_capabilities: List[Dict] = None):
        """
        Initializes the PlannerAgent.
        """
        super().__init__(
            name="PlannerAgent",
            description="Decomposes high-level goals into a detailed, executable TaskGraph."
        )
        self.llm_client = llm_client
        self.agent_capabilities = agent_capabilities or []

    # --- V2 INTERFACE IMPLEMENTATION ---

    def required_inputs(self) -> List[str]:
        """The PlannerAgent requires a 'goal' to start planning."""
        return ["goal"]

    def execute_v2(self, goal: str, inputs: Dict[str, Any], global_context: GlobalContext) -> AgentResult:
        """
        Performs the two-step process of intent analysis followed by plan generation.
        This is the primary, standardized entry point for the agent.
        """
        logger.info(f"PlannerAgent executing with goal: '{goal}'")
        
        # The 'goal' parameter in v2 is the high-level objective, but the specific
        # user request is passed in the 'inputs' dictionary.
        user_goal = inputs["goal"]
        context_summary = {"files_in_workspace": global_context.workspace.list_files()}
        
        try:
            # --- Step 1: Intent and Quality Analysis ---
            self.report_progress("Analyzing user intent and determining plan quality...")
            intent_prompt = build_intent_analysis_prompt(user_goal, context_summary)
            intent_response_str = self.llm_client.invoke(intent_prompt)
            intent_data = json.loads(intent_response_str)

            user_intent = intent_data.get("intent")
            quality_str = intent_data.get("planning_quality", "DECENT")
            quality_level = PlanningQuality(quality_str.lower())

            if not user_intent:
                return self.create_result(success=False, message="Failed to analyze user intent.")
            
            self.report_thinking(f"Analyzed Intent: {user_intent}")
            self.report_thinking(f"Determined Plan Quality: {quality_level.value.upper()}")

            # --- Step 2: Plan Generation ---
            self.report_progress("Generating detailed plan...")
            logger.info(f"Using planning quality level: {quality_level.value.upper()}")
            
            quality_instructions = self._get_quality_instructions(quality_level)
            planning_prompt = build_planning_prompt(
                intent=user_intent,
                context_summary=context_summary,
                quality=quality_level,
                quality_instructions=quality_instructions,
                agent_capabilities=self.agent_capabilities
            )
            
            plan_response_str = self.llm_client.invoke(planning_prompt)
            plan_data = json.loads(plan_response_str)

            # Validate and store the plan
            task_graph = TaskGraph(**plan_data)
            if not task_graph.nodes:
                return self.create_result(success=False, message="LLM failed to generate any tasks for the plan.")

            global_context.task_graph.nodes.update(task_graph.nodes)
            global_context.log_event("plan_generated", {"task_count": len(task_graph.nodes), "quality": quality_level.value})

            return self.create_result(
                success=True,
                message=f"Successfully generated a {quality_level.value} quality plan with {len(task_graph.nodes)} tasks.",
                outputs={"task_graph": task_graph.model_dump()}
            )
        
        except (json.JSONDecodeError, ValidationError, ValueError, KeyError) as e:
            error_msg = f"PlannerAgent failed to parse or validate LLM response. Error: {e}"
            logger.error(error_msg, exc_info=True)
            return self.create_result(success=False, message=error_msg)
        except Exception as e:
            error_msg = f"An unexpected error occurred in PlannerAgent: {e}"
            logger.critical(error_msg, exc_info=True)
            return self.create_result(success=False, message=error_msg)

    # --- Helper Methods ---

    def _get_quality_instructions(self, quality: PlanningQuality) -> Dict[str, str]:
        """Returns quality-specific instructions and descriptions."""
        quality_configs = {
            PlanningQuality.FAST: {
                "description": "a minimal, efficient task plan",
                "instructions": [
                    "Focus on essential steps only; avoid over-engineering.",
                    "Skip optional tasks like documentation unless requested.",
                    "Prioritize speed and a direct path to the goal."
                ]
            },
            PlanningQuality.DECENT: {
                "description": "a well-structured, balanced task plan",
                "instructions": [
                    "Create logical, well-organized task sequences.",
                    "Include essential validation and testing steps.",
                    "Balance thoroughness with practicality."
                ]
            },
            PlanningQuality.PRODUCTION: {
                "description": "a comprehensive, enterprise-ready task plan",
                "instructions": [
                    "Create detailed, thorough task breakdowns.",
                    "Include comprehensive error handling, testing, and quality assurance tasks.",
                    "Consider security, scalability, and maintainability."
                ]
            }
        }
        return quality_configs.get(quality, quality_configs[PlanningQuality.DECENT])
