# core/learning_core.py
import json
import logging
from typing import Dict, Any, List

# Foundational dependencies
from core.context import GlobalContext
from core.models import TaskGraph

# Get a logger instance for this module
logger = logging.getLogger(__name__)


# --- Real LLM Integration Placeholder ---
class LLMClient:
    """A placeholder for a real LLM client (e.g., OpenAI, Gemini)."""
    def invoke(self, prompt: str) -> str:
        raise NotImplementedError("LLMClient.invoke must be implemented by a concrete class.")

# --- Agent Implementation ---

class LearningCore:
    """
    The post-mission analysis and self-improvement engine.
    This component analyzes the final state of a completed mission to extract valuable
    insights and generate heuristics that can improve future planning and execution.
    """

    def __init__(self, llm_client: Any = None):
        """
        Initializes the LearningCore.

        Args:
            llm_client: An instance of a real LLM client for analysis.
        """
        self.llm_client = llm_client or LLMClient()

    def _build_analysis_prompt(self, mission_log: List[Dict], task_graph: TaskGraph) -> str:
        """Constructs a detailed prompt for the LLM to analyze a completed mission."""
        # Serialize the plan and log for the LLM.
        plan_summary = {
            task_id: {"goal": task.goal, "status": task.status}
            for task_id, task in task_graph.nodes.items()
        }

        return f"""
        You are a senior AI engineering manager responsible for process optimization.
        Your task is to analyze the following completed mission log and task plan to
        extract valuable, reusable heuristics.

        **Final Task Plan Summary:**
        ---
        {json.dumps(plan_summary, indent=2)}
        ---

        **Detailed Mission Event Log:**
        ---
        {json.dumps(mission_log, indent=2)}
        ---

        **Instructions:**
        Analyze the data above to identify two types of insights:
        1.  **Successful Workflow Patterns**: Look for sequences of successful tasks that represent a common, effective workflow. For example, a sequence of "Clarify -> Spec -> Manifest -> Code -> Test" is a strong pattern for new feature development.
        2.  **Failure Recovery Heuristics**: Identify instances where the AdaptiveEngine was triggered. Analyze the failed task and the subsequent recovery plan. Formulate a heuristic in the format: "If a task by [Agent] fails with an error like '[Error Type]', then a good recovery plan often starts with a task for [Recovery Agent]."

        **Your output MUST be a single, valid JSON object with two keys:**
        1.  `successful_patterns`: A list of strings, where each string describes a common successful workflow.
        2.  `recovery_heuristics`: A list of strings, where each string describes a learned failure recovery rule.

        **JSON Output Format Example:**
        {{
            "successful_patterns": [
                "For new feature creation, the following agent sequence is effective: IntentValidationAgent -> SpecValidationAgent -> CodeManifestAgent -> CodeGenerationAgent."
            ],
            "recovery_heuristics": [
                "If a `TestRunnerAgent` task fails, the recovery plan should immediately start with a `DebuggingAgent` task to analyze the test report."
            ]
        }}

        If no meaningful patterns or heuristics can be found, return empty lists. Now, perform the analysis.
        """

    def analyze_completed_mission(self, final_context: GlobalContext):
        """
        Analyzes the final state of a completed mission to extract learnings.
        In a production system, these learnings would be saved to a persistent
        knowledge base (e.g., a vector database or a simple JSON file) that the
        PlannerAgent could query in the future.

        Args:
            final_context: The GlobalContext object at the end of the mission.
        """
        logger.info("--- LEARNING CORE: ANALYZING COMPLETED MISSION ---")

        if not final_context.mission_log or not final_context.task_graph.nodes:
            logger.warning("Mission log or task graph is empty. Skipping analysis.")
            return

        try:
            prompt = self._build_analysis_prompt(final_context.mission_log, final_context.task_graph)
            llm_response_str = self.llm_client.invoke(prompt)
            learnings = json.loads(llm_response_str)

            patterns = learnings.get("successful_patterns", [])
            heuristics = learnings.get("recovery_heuristics", [])

            if patterns:
                logger.info(f"Learned {len(patterns)} successful workflow patterns:")
                for p in patterns:
                    logger.info(f"  - PATTERN: {p}")
                # In a real system: self.knowledge_base.save_patterns(patterns)

            if heuristics:
                logger.info(f"Learned {len(heuristics)} failure recovery heuristics:")
                for h in heuristics:
                    logger.info(f"  - HEURISTIC: {h}")
                # In a real system: self.knowledge_base.save_heuristics(heuristics)

            if not patterns and not heuristics:
                logger.info("No new significant patterns or heuristics were identified in this mission.")

        except NotImplementedError as e:
            logger.error(f"Cannot perform mission analysis: {e}")
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse LLM response for mission analysis. Error: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred during mission analysis: {e}", exc_info=True)
        
        logger.info("--- MISSION ANALYSIS COMPLETE ---")


# --- Self-Testing Block ---
if __name__ == "__main__":
    import unittest
    from unittest.mock import MagicMock
    from utils.logger import setup_logger

    setup_logger(default_level=logging.INFO)

    class TestLearningCore(unittest.TestCase):

        def setUp(self):
            self.mock_llm_client = MagicMock(spec=LLMClient)
            self.learning_core = LearningCore(llm_client=self.mock_llm_client)
            self.context = GlobalContext()

        def test_analysis_of_successful_mission(self):
            """Tests that the core can identify successful patterns from a clean run."""
            print("\n--- [Test Case 1: Analysis of Successful Mission] ---")
            # Setup a mock context representing a successful mission
            self.context.task_graph.add_task(TaskNode(task_id="t1", goal="Clarify", assigned_agent="Clarifier", status="success"))
            self.context.task_graph.add_task(TaskNode(task_id="t2", goal="Spec", assigned_agent="SpecGen", status="success"))
            self.context.mission_log.append({"event": "task_succeeded", "details": {"task_id": "t1"}})
            self.context.mission_log.append({"event": "task_succeeded", "details": {"task_id": "t2"}})

            # Configure the mock LLM to return a pattern
            mock_learnings = json.dumps({
                "successful_patterns": ["Clarifier -> SpecGen is a common successful pattern."],
                "recovery_heuristics": []
            })
            self.mock_llm_client.invoke.return_value = mock_learnings

            # We capture logs to verify the output
            with self.assertLogs(logger, level='INFO') as cm:
                self.learning_core.analyze_completed_mission(self.context)
                # Verify that the learned pattern was logged
                self.assertTrue(any("PATTERN: Clarifier -> SpecGen" in msg for msg in cm.output))

            self.mock_llm_client.invoke.assert_called_once()
            logger.info("✅ test_analysis_of_successful_mission: PASSED")

        def test_analysis_of_mission_with_failure(self):
            """Tests that the core can identify recovery heuristics from a failed run."""
            print("\n--- [Test Case 2: Analysis of Mission with Failure] ---")
            # Setup a mock context representing a mission with a failure and recovery
            self.context.task_graph.add_task(TaskNode(task_id="t1_fail", goal="Test code", assigned_agent="TestRunner", status="failed"))
            self.context.task_graph.add_task(TaskNode(task_id="t2_recover", goal="Debug", assigned_agent="Debugger", status="success"))
            self.context.mission_log.append({"event": "task_failed", "details": {"task_id": "t1_fail"}})
            self.context.mission_log.append({"event": "adaptive_engine_triggered", "details": {"failed_task_id": "t1_fail"}})
            self.context.mission_log.append({"event": "task_succeeded", "details": {"task_id": "t2_recover"}})

            mock_learnings = json.dumps({
                "successful_patterns": [],
                "recovery_heuristics": ["If TestRunner fails, a good recovery is to use the Debugger."]
            })
            self.mock_llm_client.invoke.return_value = mock_learnings

            with self.assertLogs(logger, level='INFO') as cm:
                self.learning_core.analyze_completed_mission(self.context)
                self.assertTrue(any("HEURISTIC: If TestRunner fails" in msg for msg in cm.output))

            self.mock_llm_client.invoke.assert_called_once()
            logger.info("✅ test_analysis_of_mission_with_failure: PASSED")

        def test_llm_returns_invalid_json(self):
            """Tests graceful handling of a malformed LLM response."""
            print("\n--- [Test Case 3: Analysis with Invalid JSON] ---")
            self.context.task_graph.add_task(TaskNode(task_id="t1", goal="Do something", status="success"))
            self.mock_llm_client.invoke.return_value = "this is not json"

            with self.assertLogs(logger, level='ERROR') as cm:
                self.learning_core.analyze_completed_mission(self.context)
                self.assertTrue(any("Failed to parse LLM response" in msg for msg in cm.output))
            
            logger.info("✅ test_llm_returns_invalid_json: PASSED")

    unittest.main(argv=['first-arg-is-ignored'], exit=False)