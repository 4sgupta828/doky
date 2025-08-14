# agents/plan_refiner.py
import json
import logging
from typing import Dict, Any

# Foundational dependencies
from .base import BaseAgent
from core.context import GlobalContext
from core.models import AgentResponse, TaskNode, TaskGraph, ValidationError

# Get a logger instance for this module
logger = logging.getLogger(__name__)


# --- Real LLM Integration Placeholder ---
class LLMClient:
    """A placeholder for a real LLM client."""
    def invoke(self, prompt: str) -> str:
        raise NotImplementedError("LLMClient.invoke must be implemented.")

# --- Agent Implementation ---

class PlanRefinementAgent(BaseAgent):
    """
    An expert agent that modifies an existing TaskGraph based on user feedback.
    It does not plan from scratch; it performs surgical edits, additions, or
    removals to an existing plan.
    """

    def __init__(self, llm_client: Any = None):
        super().__init__(
            name="PlanRefinementAgent",
            description="Refines an existing task plan based on user feedback, questions, or comments."
        )
        self.llm_client = llm_client or LLMClient()

    def _build_prompt(self, current_plan: Dict, user_feedback: str) -> str:
        """Constructs a prompt to guide the LLM in refining a plan."""
        return f"""
        You are an expert project manager. Your task is to intelligently modify an
        existing project plan (a TaskGraph) based on a user's feedback, question, or comment.

        **Current Project Plan (TaskGraph):**
        ---
        {json.dumps(current_plan, indent=2)}
        ---

        **User's Feedback / Question / Refinement:**
        ---
        "{user_feedback}"
        ---

        **Instructions:**
        1.  Analyze the user's feedback in the context of the current plan.
        2.  If the user is asking a question, provide a concise answer.
        3.  If the user is providing a comment or refinement, modify the TaskGraph accordingly. This may involve adding new tasks, removing tasks, or changing dependencies.
        4.  Your output MUST be a single, valid JSON object with two keys:
            - `response_to_user`: A string containing a natural language response to the user (either an answer to their question or a confirmation of the change).
            - `updated_task_graph`: The complete, modified TaskGraph in the same JSON format as the input. If no changes were needed (e.g., just answering a question), return the original TaskGraph.

        **JSON Output Format Example:**
        {{
            "response_to_user": "Good idea. I've added a linting step before the tests as you suggested.",
            "updated_task_graph": {{ ... complete new TaskGraph JSON ... }}
        }}

        Now, process the user's feedback.
        """

    def execute(self, goal: str, context: GlobalContext, current_task: TaskNode) -> AgentResponse:
        # The 'goal' for this agent is the user's refinement feedback.
        user_feedback = goal
        logger.info(f"PlanRefinementAgent executing with feedback: '{user_feedback}'")

        # Get the current plan from the context.
        current_plan_json = context.task_graph.model_dump(mode='json')
        if not current_plan_json.get("nodes"):
            return AgentResponse(success=False, message="No active plan to refine.")

        # Define JSON schema for guaranteed structured response
        refinement_schema = {
            "type": "object",
            "properties": {
                "response_to_user": {
                    "type": "string",
                    "description": "Natural language response to the user's feedback"
                },
                "updated_task_graph": {
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
            },
            "required": ["response_to_user", "updated_task_graph"]
        }

        try:
            prompt = self._build_prompt(current_plan_json, user_feedback)
            
            # Use function calling for guaranteed JSON response
            if hasattr(self.llm_client, 'invoke_with_schema'):
                llm_response_str = self.llm_client.invoke_with_schema(prompt, refinement_schema)
            else:
                # Fallback to regular invoke for backward compatibility
                llm_response_str = self.llm_client.invoke(prompt)
            
            refinement_data = json.loads(llm_response_str)

            response_to_user = refinement_data.get("response_to_user")
            updated_graph_data = refinement_data.get("updated_task_graph")

            if not response_to_user or not updated_graph_data:
                raise ValueError("LLM response is missing required keys.")

            # Validate and update the task graph in the context.
            new_task_graph = TaskGraph(**updated_graph_data)
            context.task_graph = new_task_graph

            return AgentResponse(success=True, message=response_to_user)

        except (json.JSONDecodeError, ValidationError, ValueError) as e:
            msg = f"Failed to process refinement. The LLM may have returned an invalid structure. Error: {e}"
            logger.error(msg)
            return AgentResponse(success=False, message=msg)
        except Exception as e:
            msg = f"An unexpected error occurred during plan refinement: {e}"
            logger.critical(msg, exc_info=True)
            return AgentResponse(success=False, message=msg)


# --- Self-Testing Block ---
if __name__ == "__main__":
    import unittest
    from unittest.mock import MagicMock
    from utils.logger import setup_logger

    setup_logger(default_level=logging.INFO)

    class TestPlanRefinementAgent(unittest.TestCase):
        def setUp(self):
            self.mock_llm_client = MagicMock(spec=LLMClient)
            self.agent = PlanRefinementAgent(llm_client=self.mock_llm_client)
            self.context = GlobalContext()
            # Create a simple initial plan
            self.initial_graph = TaskGraph()
            self.initial_graph.add_task(TaskNode(task_id="task_code", goal="Write code", assigned_agent="Coder"))
            self.initial_graph.add_task(TaskNode(task_id="task_test", goal="Run tests", assigned_agent="Tester", dependencies=["task_code"]))
            self.context.task_graph = self.initial_graph
            self.task = TaskNode(goal="Refine the plan", assigned_agent="PlanRefinementAgent")

        def test_successful_refinement(self):
            print("\n--- [Test Case 1: Successful Refinement] ---")
            user_feedback = "Add a linting step before the tests."
            
            # The new graph the LLM is expected to return
            refined_graph_json = {
                "nodes": {
                    "task_code": {"task_id": "task_code", "goal": "Write code", "assigned_agent": "Coder"},
                    "task_lint": {"task_id": "task_lint", "goal": "Run linter", "assigned_agent": "ToolingAgent", "dependencies": ["task_code"]},
                    "task_test": {"task_id": "task_test", "goal": "Run tests", "assigned_agent": "Tester", "dependencies": ["task_lint"]}
                }
            }
            mock_llm_response = json.dumps({
                "response_to_user": "Great suggestion! I've added a linting step.",
                "updated_task_graph": refined_graph_json
            })
            self.mock_llm_client.invoke.return_value = mock_llm_response

            response = self.agent.execute(user_feedback, self.context, self.task)

            self.assertTrue(response.success)
            self.assertIn("added a linting step", response.message)
            # Check that the graph in the context was updated
            self.assertEqual(len(self.context.task_graph.nodes), 3)
            self.assertIn("task_lint", self.context.task_graph.nodes)
            self.assertEqual(self.context.task_graph.get_task("task_test").dependencies, ["task_lint"])
            logger.info("✅ test_successful_refinement: PASSED")

    unittest.main(argv=['first-arg-is-ignored'], exit=False)