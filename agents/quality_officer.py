# agents/quality_officer.py
import json
import logging
from typing import Dict, Any, List

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

class QualityOfficerAgent(BaseAgent):
    """
    The team's quality and security supervisor. This agent runs a suite of
    advanced checks to ensure the codebase is not just functional, but also
    secure, maintainable, and high-quality.
    """

    def __init__(self, llm_client: Any = None):
        super().__init__(
            name="QualityOfficerAgent",
            description="Performs static analysis, dependency audits, and security scans."
        )
        self.llm_client = llm_client or LLMClient()

    def _build_prompt(self, all_code: Dict[str, str]) -> str:
        """Constructs a detailed prompt to guide the LLM in auditing the code."""
        code_blocks = "\n\n".join(
            f"--- File: {path} ---\n```python\n{content}\n```"
            for path, content in all_code.items()
        )

        return f"""
        You are an expert security researcher and principal software engineer. Your task is to
        conduct a thorough security and quality audit of the following Python codebase.

        **Source Code:**
        ---
        {code_blocks}
        ---

        **Instructions:**
        Analyze the code for the following categories of issues:
        1.  **Security Vulnerabilities**: Look for common vulnerabilities like SQL injection, Cross-Site Scripting (XSS), insecure deserialization, hardcoded secrets, and improper error handling.
        2.  **Code Smells & Maintainability**: Identify anti-patterns, overly complex functions (high cyclomatic complexity), "magic numbers", code duplication, and poor naming conventions.
        3.  **Best Practice Deviations**: Check for violations of PEP 8, missing docstrings, and non-idiomatic Python usage.

        **Your output MUST be a single, valid JSON object with one key: `issues`**.
        The `issues` key must hold a list of objects, where each object represents a single identified issue and has the following structure:
        - `file_path`: The path to the file containing the issue.
        - `line_number`: The line number where the issue occurs.
        - `severity`: The severity of the issue ('Critical', 'High', 'Medium', 'Low').
        - `category`: The type of issue ('Security', 'Maintainability', 'Best Practice').
        - `description`: A clear, concise explanation of the issue and a suggestion for how to fix it.

        **JSON Output Format Example:**
        {{
            "issues": [
                {{
                    "file_path": "src/database.py",
                    "line_number": 42,
                    "severity": "High",
                    "category": "Security",
                    "description": "A hardcoded password was found. Use environment variables or a secret manager instead."
                }},
                {{
                    "file_path": "src/utils.py",
                    "line_number": 15,
                    "severity": "Low",
                    "category": "Best Practice",
                    "description": "Function `calculate` is missing a docstring explaining its purpose, arguments, and return value."
                }}
            ]
        }}

        If no issues are found, return an object with an empty `issues` list. Now, conduct the audit.
        """

    def execute(self, goal: str, context: GlobalContext, current_task: TaskNode) -> AgentResponse:
        logger.info(f"ChiefQualityOfficerAgent executing with goal: '{goal}'")
        
        # Report meaningful progress
        self.report_progress("Quality audit", "Analyzing code quality and security")

        # 1. Gather all application code from the workspace.
        all_files = context.workspace.list_files()
        app_code_files = [
            f for f in all_files 
            if f.endswith('.py') and not f.startswith('tests/')
        ]
        
        if not app_code_files:
            return AgentResponse(success=True, message="No application code (.py files) found to audit.")

        all_code = {}
        for file_path in app_code_files:
            content = context.workspace.get_file_content(file_path)
            if content:
                all_code[file_path] = content

        if not all_code:
            return AgentResponse(success=True, message="Application code files were found, but they are all empty.")

        # 2. Invoke the LLM to perform the audit.
        # Define JSON schema for guaranteed structured response
        quality_schema = {
            "type": "object",
            "properties": {
                "issues": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "severity": {"type": "string"},
                            "category": {"type": "string"},
                            "file": {"type": "string"},
                            "line": {"type": "number"},
                            "message": {"type": "string"},
                            "fix": {"type": "string"}
                        },
                        "required": ["severity", "category", "file", "message"]
                    },
                    "description": "Array of quality issues found in the code"
                }
            },
            "required": ["issues"]
        }

        try:
            prompt = self._build_prompt(all_code)
            
            # Use regular invoke by default, fall back to function calling if needed
            logger.debug("Using regular invoke method (primary approach)")
            llm_response_str = self.llm_client.invoke(prompt)
            
            # Check if the response is valid JSON with actual content
            try:
                test_parse = json.loads(llm_response_str)
                if not isinstance(test_parse, dict) or "issues" not in test_parse:
                    raise ValueError("Invalid or empty response")
                logger.debug("Regular invoke succeeded with valid JSON response")
            except (json.JSONDecodeError, ValueError) as e:
                logger.warning(f"Regular invoke failed to produce valid JSON ({e}), trying function calling")
                if hasattr(self.llm_client, 'invoke_with_schema'):
                    try:
                        llm_response_str = self.llm_client.invoke_with_schema(prompt, quality_schema)
                        logger.debug("Function calling fallback succeeded")
                    except Exception as fallback_error:
                        logger.error(f"Function calling fallback also failed: {fallback_error}")
                        # Keep the original response from regular invoke for error reporting
                        pass
                else:
                    logger.warning("Function calling not available, keeping original response")
            
            quality_report = json.loads(llm_response_str)

            if "issues" not in quality_report or not isinstance(quality_report["issues"], list):
                raise ValueError("LLM response is missing the 'issues' list.")

        except NotImplementedError as e:
            return AgentResponse(success=False, message=f"Cannot execute audit: {e}")
        except (json.JSONDecodeError, ValueError) as e:
            return AgentResponse(success=False, message=f"Failed to parse LLM response for audit report. Error: {e}")
        except Exception as e:
            return AgentResponse(success=False, message=f"An unexpected error occurred during code audit: {e}")

        # 3. Add the report artifact to the context.
        output_artifact_key = "quality_and_security_report.json"
        context.add_artifact(output_artifact_key, quality_report, current_task.task_id)

        issue_count = len(quality_report["issues"])
        msg = f"Audit complete. Found {issue_count} potential issues." if issue_count > 0 else "Audit complete. No issues found."
        
        return AgentResponse(
            success=True,
            message=msg,
            artifacts_generated=[output_artifact_key]
        )


# --- Self-Testing Block ---
if __name__ == "__main__":
    import unittest
    from unittest.mock import MagicMock
    import shutil
    from pathlib import Path
    from utils.logger import setup_logger

    setup_logger(default_level=logging.INFO)

    class TestChiefQualityOfficerAgent(unittest.TestCase):

        def setUp(self):
            self.test_workspace_path = "./temp_quality_officer_test_ws"
            if Path(self.test_workspace_path).exists():
                shutil.rmtree(self.test_workspace_path)
            
            self.mock_llm_client = MagicMock(spec=LLMClient)
            self.context = GlobalContext(workspace_path=self.test_workspace_path)
            self.agent = ChiefQualityOfficerAgent(llm_client=self.mock_llm_client)
            self.task = TaskNode(goal="Audit the codebase", assigned_agent="ChiefQualityOfficerAgent")
            
            # Setup a file in the workspace for testing
            self.context.workspace.write_file_content("src/main.py", "password = '12345'", "task_setup")

        def tearDown(self):
            shutil.rmtree(self.test_workspace_path)

        def test_successful_audit_with_issues(self):
            """Tests the ideal case where the LLM finds issues and returns a valid report."""
            print("\n--- [Test Case 1: Successful Audit with Issues] ---")
            # Configure the mock LLM to return a valid report.
            mock_report = json.dumps({
                "issues": [{
                    "file_path": "src/main.py",
                    "line_number": 1,
                    "severity": "Critical",
                    "category": "Security",
                    "description": "Hardcoded secret 'password' found."
                }]
            })
            self.mock_llm_client.invoke.return_value = mock_report

            response = self.agent.execute(self.task.goal, self.context, self.task)

            self.mock_llm_client.invoke.assert_called_once()
            self.assertTrue(response.success)
            self.assertIn("Found 1 potential issues", response.message)
            
            report_artifact = self.context.get_artifact("quality_and_security_report.json")
            self.assertEqual(len(report_artifact["issues"]), 1)
            self.assertEqual(report_artifact["issues"][0]["severity"], "Critical")
            logger.info("✅ test_successful_audit_with_issues: PASSED")

        def test_successful_audit_no_issues(self):
            """Tests the case where the code is clean and the LLM finds no issues."""
            print("\n--- [Test Case 2: Successful Audit with No Issues] ---")
            self.mock_llm_client.invoke.return_value = json.dumps({"issues": []})
            
            # Overwrite file with clean code
            self.context.workspace.write_file_content("src/main.py", "print('hello')", "task_setup_clean")
            
            response = self.agent.execute(self.task.goal, self.context, self.task)
            
            self.assertTrue(response.success)
            self.assertIn("No issues found", response.message)
            report_artifact = self.context.get_artifact("quality_and_security_report.json")
            self.assertEqual(len(report_artifact["issues"]), 0)
            logger.info("✅ test_successful_audit_no_issues: PASSED")

        def test_llm_returns_invalid_json(self):
            """Tests how the agent handles a malformed JSON string from the LLM."""
            print("\n--- [Test Case 3: Audit with Invalid JSON Response] ---")
            self.mock_llm_client.invoke.return_value = "This is not json"

            response = self.agent.execute(self.task.goal, self.context, self.task)

            self.assertFalse(response.success)
            self.assertIn("Failed to parse LLM response", response.message)
            logger.info("✅ test_llm_returns_invalid_json: PASSED")
            
    unittest.main(argv=['first-arg-is-ignored'], exit=False)