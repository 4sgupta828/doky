# agents/context_builder.py
import logging
from typing import List

# Foundational dependencies
from .base import BaseAgent
from core.context import GlobalContext
from core.models import AgentResponse, TaskNode

# Get a logger instance for this module
logger = logging.getLogger(__name__)

class ContextBuilderAgent(BaseAgent):
    """
    The team's research specialist. It reads specified files from the workspace
    to build a focused context for other agents, especially for code modification
    or debugging tasks.
    """

    def __init__(self):
        super().__init__(
            name="ContextBuilderAgent",
            description="Gathers relevant code snippets from the workspace to provide context."
        )

    def execute(self, goal: str, context: GlobalContext, current_task: TaskNode) -> AgentResponse:
        logger.info(f"ContextBuilderAgent executing with goal: '{goal}'")

        # The goal for this agent should specify which files to read.
        # This information typically comes from a 'file_manifest.json' artifact
        # or from the output of a file-finding command.
        files_to_read_key = current_task.input_artifact_keys[0] if current_task.input_artifact_keys else "files_to_read.json"
        files_to_read_manifest = context.get_artifact(files_to_read_key)

        if not files_to_read_manifest or "files" not in files_to_read_manifest:
            msg = f"Missing or invalid artifact '{files_to_read_key}' with a 'files' list. Cannot build context."
            return AgentResponse(success=False, message=msg)

        files_list = files_to_read_manifest["files"]
        if not isinstance(files_list, list):
            return AgentResponse(success=False, message=f"Artifact '{files_to_read_key}' does not contain a valid list of files.")

        # Read the content of each requested file.
        context_content = ""
        read_files = []
        for file_path in files_list:
            content = context.workspace.get_file_content(file_path)
            if content is not None:
                context_content += f"--- File: {file_path} ---\n```\n{content}\n```\n\n"
                read_files.append(file_path)
            else:
                logger.warning(f"Could not read file '{file_path}' while building context.")

        if not read_files:
            return AgentResponse(success=False, message="Could not read any of the specified files to build context.")

        output_artifact_key = "targeted_code_context.txt"
        context.add_artifact(
            key=output_artifact_key,
            value=context_content,
            source_task_id=current_task.task_id
        )

        return AgentResponse(
            success=True,
            message=f"Successfully built context from {len(read_files)} files.",
            artifacts_generated=[output_artifact_key]
        )

# --- Self-Testing Block ---
if __name__ == "__main__":
    import unittest
    import shutil
    from pathlib import Path
    from utils.logger import setup_logger

    setup_logger(default_level=logging.INFO)

    class TestContextBuilderAgent(unittest.TestCase):

        def setUp(self):
            self.test_workspace_path = "./temp_context_test_ws"
            if Path(self.test_workspace_path).exists():
                shutil.rmtree(self.test_workspace_path)
            self.context = GlobalContext(workspace_path=self.test_workspace_path)
            self.agent = ContextBuilderAgent()

        def tearDown(self):
            shutil.rmtree(self.test_workspace_path)

        def test_context_builder_success(self):
            """Tests the ContextBuilderAgent's successful flow."""
            print("\n--- [Test Case 1: ContextBuilderAgent Success] ---")
            task = TaskNode(
                goal="Build context for app.py",
                assigned_agent="ContextBuilderAgent",
                input_artifact_keys=["files_to_read.json"]
            )
            # Setup context
            self.context.workspace.write_file_content("app.py", "print('hello')", "task_setup")
            self.context.add_artifact("files_to_read.json", {"files": ["app.py", "nonexistent.py"]}, "task_setup")

            response = self.agent.execute(task.goal, self.context, task)

            self.assertTrue(response.success)
            self.assertIn("built context from 1 files", response.message)
            context_artifact = self.context.get_artifact("targeted_code_context.txt")
            self.assertIn("--- File: app.py ---", context_artifact)
            self.assertIn("print('hello')", context_artifact)
            self.assertNotIn("nonexistent.py", context_artifact)
            logger.info("✅ test_context_builder_success: PASSED")

        def test_context_builder_missing_artifact(self):
            """Tests ContextBuilderAgent's failure when the input manifest is missing."""
            print("\n--- [Test Case 2: ContextBuilderAgent Missing Artifact] ---")
            task = TaskNode(goal="Build context", assigned_agent="ContextBuilderAgent")
            
            response = self.agent.execute(task.goal, self.context, task)
            
            self.assertFalse(response.success)
            self.assertIn("Missing or invalid artifact", response.message)
            logger.info("✅ test_context_builder_missing_artifact: PASSED")

    unittest.main(argv=['first-arg-is-ignored'], exit=False)