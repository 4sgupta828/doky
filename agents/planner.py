# agents/planner.py
import json
import logging
from typing import Dict, Any, List

# Foundational dependencies from Tier 1
from .base import BaseAgent
from core.context import GlobalContext
from core.models import AgentResponse, TaskGraph, TaskNode, ValidationError

# Get a logger instance for this module.
logger = logging.getLogger(__name__)


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

    def __init__(self, llm_client: Any = None, agent_capabilities: List[Dict] = None):
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

    def _build_prompt(self, goal: str, context_summary: Dict[str, Any]) -> str:
        """Constructs a detailed, high-quality prompt to guide the LLM."""
        return f"""
        You are the PlannerAgent, a master strategist for an autonomous AI agent collective.
        Your mission is to decompose a high-level user goal into a structured, dependency-aware
        TaskGraph in JSON format.

        **User Goal:**
        {goal}

        **Available Agents (Your Tools):**
        {json.dumps(self.agent_capabilities, indent=2)}

        **Current Workspace Context:**
        {json.dumps(context_summary, indent=2)}

        **Instructions:**
        1.  Analyze the user goal and the available agents.
        2.  Break the goal down into a series of logical, sequential steps.
        3.  For each step, create a 'TaskNode'. Assign a unique `task_id`.
        4.  Assign the most appropriate agent from the list of available agents to each task.
        5.  Define the `dependencies` for each task using the `task_id` of prerequisite tasks.
        6.  Define the `input_artifact_keys` and `output_artifact_keys` for data flow between tasks.
        7.  The final output MUST be a single JSON object representing the TaskGraph, containing a 'nodes' dictionary.
        8.  Ensure the keys in the 'nodes' dictionary are the same as the 'task_id' within each node object.

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

        context_summary = {"files_in_workspace": context.workspace.list_files()}
        prompt = self._build_prompt(goal, context_summary)

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
            # Use function calling for guaranteed JSON response
            if hasattr(self.llm_client, 'invoke_with_schema'):
                llm_response_str = self.llm_client.invoke_with_schema(prompt, task_graph_schema)
            else:
                # Fallback to regular invoke for backward compatibility
                llm_response_str = self.llm_client.invoke(prompt)
            
            plan_data = json.loads(llm_response_str)

            # Use Pydantic to parse and validate the entire structure.
            # This is a critical step for ensuring data integrity from the LLM.
            task_graph = TaskGraph(**plan_data)

            if not task_graph.nodes:
                return AgentResponse(success=False, message="LLM failed to generate a plan with any tasks.")

            # The plan is valid, so we add it to the global context.
            # We don't overwrite the whole graph, but update it with the new nodes.
            context.task_graph.nodes.update(task_graph.nodes)
            context.log_event("plan_generated", {"task_count": len(task_graph.nodes), "source_task": current_task.task_id})

            return AgentResponse(
                success=True,
                message=f"Successfully generated a plan with {len(task_graph.nodes)} tasks.",
                artifacts_generated=["task_graph"]
            )
        
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