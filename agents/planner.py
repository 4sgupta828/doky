# agents/planner.py
import json
import logging
from typing import Dict, Any, List
from enum import Enum

# Foundational dependencies from Tier 1
from .base import BaseAgent, ContextTooLargeError
from core.context import GlobalContext
from core.models import AgentResponse, TaskGraph, TaskNode, ValidationError

# Get a logger instance for this module.
logger = logging.getLogger(__name__)


# --- Planning Quality Levels ---
class PlanningQuality(Enum):
    """Defines different planning quality levels for speed vs detail trade-offs."""
    FAST = "fast"          # Quick, minimal plans - prioritizes speed
    DECENT = "decent"      # Balanced approach - good structure, reasonable detail (default)
    PRODUCTION = "production"  # Comprehensive, detailed plans with full considerations


# --- Real LLM Integration Placeholder ---
# In a real system, this client would live in a dedicated `integrations` directory.
# It would handle API keys, request/response logic, and error handling for a specific LLM service.
class LLMClient:
    """
    A placeholder for a real LLM client (e.g., OpenAI, Anthropic, Gemini).
    This class defines the interface that our agents will use to interact with an LLM.
    """
    def invoke(self, prompt: str) -> str:
        """
        Invokes the LLM with a given prompt and returns the text response.

        NOTE: This is where you would add the actual API call logic.
              For example, using the `openai` library:

              from openai import OpenAI
              client = OpenAI(api_key="YOUR_API_KEY")
              response = client.chat.completions.create(
                  model="gpt-4-turbo",
                  messages=[{"role": "user", "content": prompt}]
              )
              return response.choices[0].message.content
        """
        # This placeholder returns an error message. In a real scenario,
        # it would make a network request to an LLM API.
        raise NotImplementedError(
            "LLMClient.invoke is a placeholder and has not been implemented. "
            "You must configure a real LLM client (e.g., OpenAI)."
        )

# --- Agent Implementation ---

class PlannerAgent(BaseAgent):
    """
    The master strategist of the collective. Its sole responsibility is to take a
    high-level goal and break it down into a structured, logical TaskGraph that
    can be executed by the other specialized agents.
    """

    def __init__(self, llm_client: Any = None, agent_capabilities: List[Dict] = None, default_quality: PlanningQuality = PlanningQuality.FAST):
        """
        Initializes the PlannerAgent.

        Args:
            llm_client: An instance of a real LLM client.
            agent_capabilities: A list of capabilities of all available agents,
                                used to inform the LLM about its available tools.
        """
        super().__init__(
            name="PlannerAgent",
            description="Decomposes high-level goals into a detailed, executable TaskGraph."
        )
        self.llm_client = llm_client or LLMClient()
        self.agent_capabilities = agent_capabilities or []
        self.default_quality = default_quality

    def _get_quality_instructions(self, quality: PlanningQuality) -> Dict[str, str]:
        """Returns quality-specific instructions and descriptions."""
        quality_configs = {
            PlanningQuality.FAST: {
                "description": "a minimal, efficient task plan",
                "instructions": [
                    "Keep plans as SIMPLE as possible - create 1-2 tasks maximum for straightforward requests",
                    "Focus on essential steps only, avoid over-engineering",
                    "Skip unnecessary documentation or testing tasks unless explicitly requested",
                    "Use basic task descriptions without extensive detail",
                    "Prioritize speed and getting started quickly"
                ]
            },
            PlanningQuality.DECENT: {
                "description": "a well-structured, balanced task plan",
                "instructions": [
                    "Create logical, well-organized task sequences",
                    "Include reasonable detail in task descriptions",
                    "Consider basic dependencies and data flow",
                    "Balance thoroughness with practicality",
                    "Include essential validation and testing steps when appropriate"
                ]
            },
            PlanningQuality.PRODUCTION: {
                "description": "a comprehensive, enterprise-ready task plan",
                "instructions": [
                    "Create detailed, thorough task breakdowns with full considerations",
                    "Include comprehensive error handling and validation tasks",
                    "Plan for documentation, testing, and quality assurance",
                    "Consider scalability, security, and maintainability aspects",
                    "Include detailed dependency analysis and risk mitigation",
                    "Plan for deployment, monitoring, and operational considerations"
                ]
            }
        }
        return quality_configs[quality]
    
    def _detect_quality_level(self, goal: str, context: GlobalContext) -> PlanningQuality:
        """Detects the desired planning quality level from the goal and context."""
        goal_lower = goal.lower()
        
        # Check for explicit quality keywords in the goal
        if any(keyword in goal_lower for keyword in ['fast', 'quick', 'rapid', 'simple', 'minimal']):
            logger.info("Detected FAST planning quality level from goal keywords")
            return PlanningQuality.FAST
        elif any(keyword in goal_lower for keyword in ['decent', 'structured', 'balanced', 'organized', 'thorough']):
            logger.info("Detected DECENT planning quality level from goal keywords")
            return PlanningQuality.DECENT
        elif any(keyword in goal_lower for keyword in ['production', 'enterprise', 'comprehensive', 'detailed', 'robust']):
            logger.info("Detected PRODUCTION planning quality level from goal keywords")
            return PlanningQuality.PRODUCTION
        
        # Check context for quality preferences
        if hasattr(context, 'planning_quality_preference'):
            return context.planning_quality_preference
            
        # Default to FAST for speed optimization
        logger.info("Using default FAST planning quality level")
        return self.default_quality

    def _build_prompt(self, goal: str, context_summary: Dict[str, Any], quality: PlanningQuality = None) -> str:
        """Constructs a detailed, high-quality prompt to guide the LLM."""
        if quality is None:
            quality = self.default_quality
            
        quality_config = self._get_quality_instructions(quality)
        
        # Build quality-specific instructions
        quality_instructions = "\n        ".join([f"{i+1}. {instruction}" for i, instruction in enumerate(quality_config["instructions"])])
        
        return f"""
        You are the PlannerAgent, a master strategist for an autonomous AI agent collective.
        Your mission is to decompose a high-level user goal into {quality_config["description"]} in JSON format.
        
        **Planning Quality Level: {quality.value.upper()}**

        **User Goal:**
        {goal}

        **Available Agents (Your Tools):**
        {json.dumps(self.agent_capabilities, indent=2)}

        **Current Workspace Context:**
        {json.dumps(context_summary, indent=2)}

        **Quality-Specific Planning Instructions:**
        {quality_instructions}
        
        **General Instructions:**
        1. Analyze the user goal and the available agents
        2. For each step, create a 'TaskNode' with a unique `task_id`
        3. Assign the most appropriate agent from the list of available agents to each task
        4. Define the `dependencies` for each task using the `task_id` of prerequisite tasks
        5. Define the `input_artifact_keys` and `output_artifact_keys` for data flow between tasks
        6. You MUST return your TaskGraph as structured JSON data
        7. Ensure the keys in the 'nodes' dictionary are the same as the 'task_id' within each node object

        **JSON Output Format Example:**
        {{
            "nodes": {{
                "task_clarify_intent": {{
                    "task_id": "task_clarify_intent",
                    "goal": "Clarify API requirements with the user.",
                    "assigned_agent": "IntentClarificationAgent",
                    "dependencies": [],
                    "output_artifact_keys": ["clarified_requirements.md"]
                }},
                "task_generate_spec": {{
                    "task_id": "task_generate_spec",
                    "goal": "Create a detailed technical specification.",
                    "assigned_agent": "SpecGenerationAgent",
                    "dependencies": ["task_clarify_intent"],
                    "input_artifact_keys": ["clarified_requirements.md"],
                    "output_artifact_keys": ["technical_spec.md"]
                }}
            }}
        }}

        Now, generate the TaskGraph JSON for the provided user goal.
        """

    def execute(self, goal: str, context: GlobalContext, current_task: TaskNode) -> AgentResponse:
        """
        Generates a TaskGraph by calling an LLM and validating its response.
        """
        logger.info(f"PlannerAgent executing with goal: '{goal}'")
        
        # Report meaningful progress
        self.report_progress("Creating plan", f"Analyzing goal: '{goal[:80]}...'")

        context_summary = {"files_in_workspace": context.workspace.list_files()}
        
        # Detect quality level from goal and context
        quality_level = self._detect_quality_level(goal, context)
        logger.info(f"Using planning quality level: {quality_level.value.upper()}")
        
        # Show thinking process for important decisions
        self.report_thinking(f"I'll create a {quality_level.value} quality plan. Let me analyze what agents are available and how to break down this goal effectively.")
        
        prompt = self._build_prompt(goal, context_summary, quality_level)

        # Define JSON schema for guaranteed structured response
        task_graph_schema = {
            "type": "object",
            "properties": {
                "nodes": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "object",
                        "properties": {
                            "task_id": {"type": "string"},
                            "goal": {"type": "string"},
                            "assigned_agent": {"type": "string"},
                            "dependencies": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "input_artifact_keys": {
                                "type": "array",
                                "items": {"type": "string"}
                            },
                            "output_artifact_keys": {
                                "type": "array",
                                "items": {"type": "string"}
                            }
                        },
                        "required": ["task_id", "goal", "assigned_agent"]
                    }
                }
            },
            "required": ["nodes"]
        }

        try:
            # Generate the plan using LLM
            logger.debug("Using regular invoke method (primary approach)")
            llm_response_str = self.llm_client.invoke(prompt)
            
            # Validate and handle LLM response
            try:
                test_parse = json.loads(llm_response_str)
                if not isinstance(test_parse, dict) or not test_parse.get('nodes'):
                    raise ValueError("Invalid or empty response")
                logger.debug("Regular invoke succeeded with valid JSON response")
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Regular invoke failed to produce valid JSON ({e}), trying function calling")
                self.report_thinking("Initial response wasn't valid JSON. Retrying with structured output.")
                
                if hasattr(self.llm_client, 'invoke_with_schema'):
                    try:
                        logger.debug(f"Planner prompt length: {len(prompt)} characters")
                        logger.debug(f"Planner schema: {task_graph_schema}")
                        llm_response_str = self.llm_client.invoke_with_schema(prompt, task_graph_schema)
                        logger.debug("Function calling fallback succeeded")
                    except Exception as fallback_error:
                        logger.error(f"Function calling fallback also failed: {fallback_error}")
                        self.fail_step("Failed to generate valid plan", 
                                     ["Check LLM client configuration", "Verify API keys", "Try simplifying the goal"])
                        return AgentResponse(success=False, message=f"LLM failed to generate a valid plan: {fallback_error}")
                else:
                    logger.warning("Function calling not available, keeping original response")
            
            # Parse and validate the plan
            plan_data = json.loads(llm_response_str)
            logger.debug(f"Parsed plan data: {plan_data}")

            # Use Pydantic to validate the structure
            task_graph = TaskGraph(**plan_data)
            logger.debug(f"TaskGraph nodes: {task_graph.nodes}")
            logger.debug(f"TaskGraph nodes length: {len(task_graph.nodes) if task_graph.nodes else 0}")

            if not task_graph.nodes:
                self.fail_step("Empty plan generated", ["Make the goal more specific", "Provide more context", "Try a different quality level"])
                return AgentResponse(success=False, message="LLM failed to generate a plan with any tasks.")

            # Show the generated plan to the user
            plan_summary = []
            for task_id, task in task_graph.nodes.items():
                deps = f" (depends on: {', '.join(task.dependencies)})" if task.dependencies else ""
                plan_summary.append(f"• {task_id}: {task.goal} → {task.assigned_agent}{deps}")
            
            self.report_intermediate_output("generated_plan", "\n".join(plan_summary))

            # Store the plan in context
            context.task_graph.nodes.update(task_graph.nodes)
            context.log_event("plan_generated", {"task_count": len(task_graph.nodes), "source_task": current_task.task_id})

            return AgentResponse(
                success=True,
                message=f"Successfully generated a plan with {len(task_graph.nodes)} tasks.",
                artifacts_generated=["task_graph"]
            )
        
        except ContextTooLargeError as e:
            return self.handle_context_error(e, goal)
        except NotImplementedError:
            msg = "LLMClient is not implemented. Cannot generate a plan."
            logger.critical(msg)
            return AgentResponse(success=False, message=msg)
        except json.JSONDecodeError as e:
            error_msg = f"PlannerAgent failed to parse LLM response as JSON. Error: {e}"
            logger.error(error_msg, exc_info=True)
            return AgentResponse(success=False, message=error_msg)
        except ValidationError as e:
            error_msg = f"PlannerAgent received a structurally invalid plan from the LLM. Details:\n{e}"
            logger.error(error_msg, exc_info=True)
            return AgentResponse(success=False, message=error_msg)
        except Exception as e:
            error_msg = f"An unexpected error occurred in PlannerAgent: {e}"
            logger.critical(error_msg, exc_info=True)
            return AgentResponse(success=False, message=error_msg)


# --- Self-Testing Block ---
# This block uses unittest.mock to simulate the LLMClient, allowing for robust testing
# without making actual API calls.
if __name__ == "__main__":
    import unittest
    from unittest.mock import MagicMock
    import shutil
    from pathlib import Path
    from utils.logger import setup_logger

    setup_logger(default_level=logging.INFO)

    class TestPlannerAgent(unittest.TestCase):

        def setUp(self):
            """Set up a clean environment for each test."""
            self.test_workspace_path = "./temp_planner_test_ws"
            if Path(self.test_workspace_path).exists():
                shutil.rmtree(self.test_workspace_path)
            
            self.mock_llm_client = MagicMock(spec=LLMClient)
            self.context = GlobalContext(workspace_path=self.test_workspace_path)
            self.planner_task = TaskNode(goal="Test Goal", assigned_agent="PlannerAgent")
            self.planner = PlannerAgent(llm_client=self.mock_llm_client, agent_capabilities=[{"name": "TestAgent", "description": "A test agent"}])

        def tearDown(self):
            """Clean up the environment after each test."""
            shutil.rmtree(self.test_workspace_path)

        def test_successful_plan_generation(self):
            """Tests the ideal case where the LLM returns a valid plan."""
            print("\n--- [Test Case 1: Successful Plan Generation] ---")
            # Configure the mock to return a valid JSON string.
            valid_plan_json = json.dumps({
                "nodes": {
                    "task_1": {"task_id": "task_1", "goal": "First step", "assigned_agent": "TestAgent"},
                    "task_2": {"task_id": "task_2", "goal": "Second step", "assigned_agent": "TestAgent", "dependencies": ["task_1"]}
                }
            })
            self.mock_llm_client.invoke.return_value = valid_plan_json

            response = self.planner.execute("A valid goal", self.context, self.planner_task)

            self.mock_llm_client.invoke.assert_called_once()
            self.assertTrue(response.success)
            self.assertIn("Successfully generated a plan", response.message)
            self.assertEqual(len(self.context.task_graph.nodes), 2)
            self.assertIn("task_1", self.context.task_graph.nodes)
            self.assertEqual(self.context.task_graph.get_task("task_2").dependencies, ["task_1"])
            logger.info("✅ test_successful_plan_generation: PASSED")

        def test_llm_returns_invalid_json(self):
            """Tests how the agent handles a malformed JSON string from the LLM."""
            print("\n--- [Test Case 2: Invalid JSON Response] ---")
            self.mock_llm_client.invoke.return_value = '{"nodes": {"task_1": "incomplete"'

            response = self.planner.execute("A goal", self.context, self.planner_task)

            self.assertTrue(not response.success)
            self.assertIn("Failed to parse LLM response as JSON", response.message)
            self.assertEqual(len(self.context.task_graph.nodes), 0) # No tasks should be added.
            logger.info("✅ test_llm_returns_invalid_json: PASSED")

        def test_llm_returns_incomplete_plan(self):
            """Tests how the agent handles a structurally invalid plan (missing required fields)."""
            print("\n--- [Test Case 3: Incomplete Plan (Validation Error)] ---")
            # JSON is valid, but the data doesn't match the Pydantic model.
            incomplete_plan_json = json.dumps({
                "nodes": {
                    "task_1": {"task_id": "task_1"} # Missing 'goal' and 'assigned_agent'
                }
            })
            self.mock_llm_client.invoke.return_value = incomplete_plan_json

            response = self.planner.execute("A goal", self.context, self.planner_task)

            self.assertTrue(not response.success)
            self.assertIn("structurally invalid plan", response.message)
            logger.info("✅ test_llm_returns_incomplete_plan: PASSED")

        def test_llm_not_implemented(self):
            """Tests the placeholder case where the LLM client hasn't been configured."""
            print("\n--- [Test Case 4: LLM Not Implemented] ---")
            # Use a real (unimplemented) client instead of a mock.
            unimplemented_planner = PlannerAgent(llm_client=LLMClient())
            
            response = unimplemented_planner.execute("A goal", self.context, self.planner_task)
            
            self.assertTrue(not response.success)
            self.assertIn("LLMClient is not implemented", response.message)
            logger.info("✅ test_llm_not_implemented: PASSED")

    # Run the tests
    unittest.main()