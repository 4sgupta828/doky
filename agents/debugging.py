# agents/debugging.py
import json
import logging
from typing import Dict, Any

# Foundational dependencies
from .base import BaseAgent
from core.context import GlobalContext
from core.models import AgentResponse, TaskNode

# Get a logger instance for this module
logger = logging.getLogger(__name__)


# --- Real LLM Integration Placeholder ---
class LLMClient:
    """A placeholder for a real LLM client (e.g., OpenAI, Gemini)."""
    def invoke(self, prompt: str) -> str:
        raise NotImplementedError("LLMClient.invoke must be implemented by a concrete class.")

# --- Agent Implementation ---

class DebuggingAgent(BaseAgent):
    """
    The team's expert troubleshooter. When a test fails, this agent is
    activated to perform a root-cause analysis. It examines the code, the
    error message, and the stack trace to pinpoint the source of the problem.
    """

    def __init__(self, llm_client: Any = None):
        super().__init__(
            name="DebuggingAgent",
            description="Analyzes failed tests and stack traces to find root causes and suggest fixes."
        )
        self.llm_client = llm_client or LLMClient()

    def _build_prompt(self, failed_test_report: Dict, code_context: str) -> str:
        """Constructs a detailed prompt to guide the LLM in debugging the code."""
        return f"""
        You are an expert Python debugger. Your task is to analyze a failed test report
        and the relevant source code to identify the root cause of the failure and
        propose a fix in the form of a code diff.

        **Failed Test Report:**
        ---
        {json.dumps(failed_test_report, indent=2)}
        ---

        **Relevant Source Code:**
        ---
        {code_context}
        ---

        **Instructions:**
        1.  **Analyze the Root Cause**: Carefully examine the error message, stack trace, and source code to pinpoint the exact line and reason for the failure.
        2.  **Formulate an Explanation**: Write a clear, concise explanation of the bug in Markdown format.
        3.  **Propose a Fix**: Generate a code patch in the standard 'diff' format to correct the bug. The diff should only contain the necessary changes to fix the issue.

        **Your output MUST be a single, valid JSON object with two keys:**
        1.  `root_cause_analysis`: A string containing the Markdown explanation of the bug.
        2.  `suggested_fix_diff`: A string containing the code patch in standard diff format.

        **JSON Output Format Example:**
        {{
            "root_cause_analysis": "### Root Cause Analysis\\n\\nThe `add` function fails on non-integer inputs because it does not perform type checking. The test failed when passing a string, which caused a `TypeError`.",
            "suggested_fix_diff": "--- a/src/calculator.py\\n+++ b/src/calculator.py\\n@@ -1,2 +1,4 @@\\n def add(a, b):\\n+    if not isinstance(a, int) or not isinstance(b, int):\\n+        raise TypeError(\\"Both inputs must be integers.\\")\\n     return a + b"
        }}

        Now, perform the debugging analysis.
        """

    def execute(self, goal: str, context: GlobalContext, current_task: TaskNode) -> AgentResponse:
        logger.info(f"DebuggingAgent executing with goal: '{goal}'")

        # 1. Retrieve necessary artifacts from the context.
        report_key = "failed_test_report.json"
        context_key = "targeted_code_context.txt"
        
        failed_report = context.get_artifact(report_key)
        code_context = context.get_artifact(context_key)

        if not failed_report or not code_context:
            msg = f"Missing required artifacts: needs '{report_key}' and '{context_key}'."
            logger.error(msg)
            return AgentResponse(success=False, message=msg)

        # 2. Invoke the LLM to perform the analysis.
        # Define JSON schema for guaranteed structured response
        debug_schema = {
            "type": "object",
            "properties": {
                "root_cause_analysis": {
                    "type": "string",
                    "description": "Detailed analysis of the root cause"
                },
                "suggested_fix_diff": {
                    "type": "string", 
                    "description": "Suggested fix in diff format"
                }
            },
            "required": ["root_cause_analysis", "suggested_fix_diff"]
        }

        try:
            prompt = self._build_prompt(failed_report, code_context)
            
            # Use function calling for guaranteed JSON response
            if hasattr(self.llm_client, 'invoke_with_schema'):
                llm_response_str = self.llm_client.invoke_with_schema(prompt, debug_schema)
            else:
                # Fallback to regular invoke for backward compatibility
                llm_response_str = self.llm_client.invoke(prompt)
            
            debug_results = json.loads(llm_response_str)

            analysis = debug_results.get("root_cause_analysis")
            suggested_fix = debug_results.get("suggested_fix_diff")

            if not analysis or not suggested_fix:
                raise ValueError("LLM response is missing 'root_cause_analysis' or 'suggested_fix_diff'.")

        except NotImplementedError as e:
            return AgentResponse(success=False, message=f"Cannot execute debugging: {e}")
        except (json.JSONDecodeError, ValueError) as e:
            return AgentResponse(success=False, message=f"Failed to parse LLM response for debugging. Error: {e}")
        except Exception as e:
            return AgentResponse(success=False, message=f"An unexpected error occurred during debugging: {e}")

        # 3. Add the new analysis and diff artifacts to the context.
        analysis_key = "root_cause_analysis.md"
        fix_key = "suggested_fix.diff"
        
        context.add_artifact(analysis_key, analysis, current_task.task_id)
        context.add_artifact(fix_key, suggested_fix, current_task.task_id)

        return AgentResponse(
            success=True,
            message="Successfully analyzed failure and generated a root cause analysis and a suggested fix.",
            artifacts_generated=[analysis_key, fix_key]
        )


# --- Self-Testing Block ---
if __name__ == "__main__":
    import unittest
    from unittest.mock import MagicMock
    import shutil
    from pathlib import Path
    from utils.logger import setup_logger

    setup_logger(default_level=logging.INFO)

    class TestDebuggingAgent(unittest.TestCase):

        def setUp(self):
            self.test_workspace_path = "./temp_debugging_test_ws"
            if Path(self.test_workspace_path).exists():
                shutil.rmtree(self.test_workspace_path)
            
            self.mock_llm_client = MagicMock(spec=LLMClient)
            self.context = GlobalContext(workspace_path=self.test_workspace_path)
            self.agent = DebuggingAgent(llm_client=self.mock_llm_client)
            self.task = TaskNode(
                goal="Debug a failed test",
                assigned_agent="DebuggingAgent",
                input_artifact_keys=["failed_test_report.json", "targeted_code_context.txt"]
            )
            # Pre-populate context with necessary artifacts
            self.context.add_artifact("failed_test_report.json", {"summary": {"failed": 1}, "error": "TypeError"}, "task_test")
            self.context.add_artifact("targeted_code_context.txt", "def add(a, b): return a + b", "task_context")

        def tearDown(self):
            shutil.rmtree(self.test_workspace_path)

        def test_successful_debugging(self):
            """Tests the ideal case where the LLM returns a valid analysis and diff."""
            print("\n--- [Test Case 1: DebuggingAgent Success] ---")
            # Configure the mock LLM to return a valid debug analysis.
            mock_debug_output = json.dumps({
                "root_cause_analysis": "The function fails due to a TypeError.",
                "suggested_fix_diff": "--- a/file.py\n+++ b/file.py\n@@ -1 +1,2 @@\n+    if not isinstance(a, int): raise TypeError\n     return a + b"
            })
            self.mock_llm_client.invoke.return_value = mock_debug_output

            response = self.agent.execute(self.task.goal, self.context, self.task)

            self.mock_llm_client.invoke.assert_called_once()
            self.assertTrue(response.success)
            self.assertIn("root_cause_analysis.md", response.artifacts_generated)
            self.assertIn("suggested_fix.diff", response.artifacts_generated)
            
            analysis = self.context.get_artifact("root_cause_analysis.md")
            self.assertIn("TypeError", analysis)
            logger.info("✅ test_successful_debugging: PASSED")

        def test_failure_on_missing_artifacts(self):
            """Tests that the agent fails gracefully if prerequisites are not in the context."""
            print("\n--- [Test Case 2: DebuggingAgent Missing Artifacts] ---")
            empty_context = GlobalContext(workspace_path=self.test_workspace_path) # Use a fresh context
            
            response = self.agent.execute(self.task.goal, empty_context, self.task)

            self.assertFalse(response.success)
            self.assertIn("Missing required artifacts", response.message)
            self.mock_llm_client.invoke.assert_not_called()
            logger.info("✅ test_failure_on_missing_artifacts: PASSED")

        def test_llm_returns_invalid_json(self):
            """Tests how the agent handles a malformed JSON string from the LLM."""
            print("\n--- [Test Case 3: DebuggingAgent Invalid JSON] ---")
            self.mock_llm_client