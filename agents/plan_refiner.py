# agents/plan_refiner.py
import json
import logging
from typing import Dict, Any, List

# Foundational dependencies
from .base import BaseAgent
from core.context import GlobalContext
from core.models import AgentResult, TaskNode, TaskGraph, ValidationError

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

    def required_inputs(self) -> List[str]:
        """Required inputs for plan refinement."""
        return ["user_feedback"]

    def optional_inputs(self) -> List[str]:
        """Optional inputs for plan refinement."""
        return []

    def execute_v2(self, goal: str, inputs: Dict[str, Any], global_context: GlobalContext) -> AgentResult:
        """
        V2 interface for plan refinement.
        
        Args:
            goal: Description of what refinement to perform
            inputs: Must contain 'user_feedback' with the refinement request
            global_context: The shared context containing the task graph to refine
            
        Returns:
            AgentResult with success status and refined task graph details
        """
        try:
            self.validate_inputs(inputs)
            user_feedback = inputs["user_feedback"]
            
            logger.info(f"PlanRefinementAgent executing with feedback: '{user_feedback}'")
            self.report_progress("Processing refinement request", f"Feedback: {user_feedback[:60]}...")
            
            # Get the current plan from the context.
            current_plan_json = global_context.task_graph.model_dump(mode='json')
            if not current_plan_json.get("nodes"):
                return self.create_result(
                    success=False,
                    message="No active plan to refine.",
                    error_details={"issue": "empty_task_graph"}
                )

            self.report_thinking("Analyzing current plan structure and user feedback")
            
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

            self.report_progress("Generating refinement", "Invoking LLM for plan modification")
            
            try:
                prompt = self._build_prompt(current_plan_json, user_feedback)
                
                # Use regular invoke by default, fall back to function calling if needed
                logger.debug("Using regular invoke method (primary approach)")
                llm_response_str = self.llm_client.invoke(prompt)
                
                # Check if the response is valid JSON with actual content
                try:
                    test_parse = json.loads(llm_response_str)
                    if not isinstance(test_parse, dict) or not test_parse.get('updated_task_graph'):
                        raise ValueError("Invalid or empty response")
                    logger.debug("Regular invoke succeeded with valid JSON response")
                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning(f"Regular invoke failed to produce valid JSON ({e}), trying function calling")
                    if hasattr(self.llm_client, 'invoke_with_schema'):
                        try:
                            llm_response_str = self.llm_client.invoke_with_schema(prompt, refinement_schema)
                            logger.debug("Function calling fallback succeeded")
                        except Exception as fallback_error:
                            logger.error(f"Function calling fallback also failed: {fallback_error}")
                            # Keep the original response from regular invoke for error reporting
                            pass
                    else:
                        logger.warning("Function calling not available, keeping original response")
                
                refinement_data = json.loads(llm_response_str)

                response_to_user = refinement_data.get("response_to_user")
                updated_graph_data = refinement_data.get("updated_task_graph")

                if not response_to_user or not updated_graph_data:
                    raise ValueError("LLM response is missing required keys.")

                self.report_progress("Updating task graph", "Validating and applying changes")
                
                # Validate and update the task graph in the context.
                new_task_graph = TaskGraph(**updated_graph_data)
                global_context.task_graph = new_task_graph

                self.report_progress("Refinement completed", f"Successfully updated plan based on feedback")
                
                return self.create_result(
                    success=True,
                    message=response_to_user,
                    outputs={
                        "refinement_applied": True,
                        "updated_nodes_count": len(new_task_graph.nodes),
                        "response_to_user": response_to_user
                    }
                )

            except (json.JSONDecodeError, ValidationError, ValueError) as e:
                error_msg = f"Failed to process refinement. The LLM may have returned an invalid structure. Error: {e}"
                logger.error(error_msg)
                self.fail_step(error_msg, ["Check LLM response format", "Verify task graph structure"])
                return self.create_result(
                    success=False,
                    message=error_msg,
                    error_details={"exception": str(e), "type": "parsing_error"}
                )
                
        except Exception as e:
            error_msg = f"Unexpected error during plan refinement: {e}"
            logger.error(error_msg, exc_info=True)
            self.fail_step(error_msg)
            return self.create_result(
                success=False,
                message=error_msg,
                error_details={"exception": str(e), "type": "unexpected_error"}
            )

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