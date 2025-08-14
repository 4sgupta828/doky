# interfaces/collaboration_ui.py
import logging
import tempfile
import subprocess
import os
from typing import Dict, Any

# Foundational dependencies
from core.context import GlobalContext
from core.models import TaskNode

# Get a logger instance for this module
logger = logging.getLogger(__name__)

class CollaborationUI:
    """
    Provides a terminal-based interface for human-in-the-loop collaboration.
    This class handles all direct interaction with the user, such as displaying
    the system's status and prompting for input or confirmation.
    """

    def display_status(self, context: GlobalContext):
        """
        Displays a summary of the current mission status to the user.
        This provides transparency into the agent's plan and progress.
        """
        print("\n" + "="*50)
        print(" MISSION STATUS UPDATE")
        print("="*50)
        
        if not context.task_graph.nodes:
            print("No plan has been generated yet.")
            return

        print("\n--- Task Plan ---")
        # Sort tasks for a consistent display order
        sorted_tasks = sorted(context.task_graph.nodes.values(), key=lambda t: t.task_id)
        for task in sorted_tasks:
            status_icon = {
                "pending": "⚪️",
                "running": "🔵",
                "success": "✅",
                "failed": "❌",
                "obsolete": "⚫️"
            }.get(task.status, "❓")
            
            print(f"{status_icon} [{task.status.upper()}] {task.task_id}: {task.goal} (Agent: {task.assigned_agent})")
            if task.dependencies:
                print(f"    - Depends on: {', '.join(task.dependencies)}")
        
        print("\n--- Recent Events ---")
        for log in context.mission_log[-5:]: # Display the last 5 events
            print(f"- {log['event']}: {log['details']}")
        
        print("="*50)

    def prompt_for_input(self, question: str) -> str:
        """
        Pauses execution and asks the user a question, waiting for their input.

        Args:
            question: The question to display to the user.

        Returns:
            The user's response as a string.
        """
        print("\n" + "-"*50)
        print("
<img src="https://placehold.co/15x15/3b82f6/ffffff?text=Q" alt="Question Icon">
 **ACTION REQUIRED**")
        print(f"The agent needs your input to proceed:")
        print(f"\n  > {question}")
        print("-"*50)
        
        # In a real application, this directly captures user input from the console.
        user_response = input("Your response: ")
        
        logger.info(f"User was asked '{question}' and responded: '{user_response}'")
        return user_response.strip()

    def allow_artifact_edit(self, artifact_key: str, context: GlobalContext) -> bool:
        """
        Allows the user to manually inspect and edit a text-based artifact.
        It opens the artifact in the system's default text editor (e.g., nano, vim, notepad).

        Args:
            artifact_key: The key of the artifact to be edited.
            context: The shared GlobalContext.

        Returns:
            True if the artifact was successfully edited and saved, False otherwise.
        """
        artifact_content = context.get_artifact(artifact_key)
        if artifact_content is None or not isinstance(artifact_content, str):
            logger.error(f"Artifact '{artifact_key}' cannot be edited. It is either missing or not a string.")
            print(f"Error: Artifact '{artifact_key}' is not available for editing.")
            return False

        # Use the system's default editor (e.g., from the $EDITOR environment variable)
        editor = os.environ.get('EDITOR', 'nano') 

        try:
            # Create a temporary file to hold the artifact content for editing.
            with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix=".md") as tmpfile:
                tmpfile.write(artifact_content)
                tmpfile_path = tmpfile.name
            
            print("\n" + "-"*50)
            print("
<img src="https://placehold.co/15x15/f59e0b/ffffff?text=E" alt="Edit Icon">
 **EDIT ARTIFACT**")
            print(f"Opening artifact '{artifact_key}' in your default editor ({editor})...")
            print("Please save and close the editor when you are finished.")
            print("-"*50)

            # Open the editor as a subprocess, which pauses our script until the editor is closed.
            subprocess.run([editor, tmpfile_path], check=True)

            # Read the potentially modified content back from the temporary file.
            with open(tmpfile_path, 'r') as tmpfile:
                new_content = tmpfile.read()
            
            if new_content != artifact_content:
                # If the content has changed, update the artifact in the context.
                context.add_artifact(artifact_key, new_content, "user_edit")
                logger.info(f"User successfully edited artifact '{artifact_key}'.")
                print(f"Artifact '{artifact_key}' has been updated.")
            else:
                logger.info(f"User closed the editor without changing artifact '{artifact_key}'.")
                print("No changes were made to the artifact.")

            return True
        except FileNotFoundError:
            msg = f"Editor '{editor}' not found. Please set your $EDITOR environment variable."
            logger.error(msg)
            print(f"Error: {msg}")
            return False
        except Exception as e:
            msg = f"An unexpected error occurred while editing artifact: {e}"
            logger.error(msg, exc_info=True)
            print(f"Error: {msg}")
            return False
        finally:
            # Clean up the temporary file.
            if 'tmpfile_path' in locals() and os.path.exists(tmpfile_path):
                os.remove(tmpfile_path)

# --- Self-Testing Block ---
if __name__ == "__main__":
    import unittest
    from unittest.mock import patch, MagicMock
    from utils.logger import setup_logger

    setup_logger(default_level=logging.INFO)

    class TestCollaborationUI(unittest.TestCase):

        def setUp(self):
            self.ui = CollaborationUI()
            self.context = GlobalContext()
            self.context.task_graph.add_task(TaskNode(task_id="task_1", goal="First task", assigned_agent="TestAgent", status="success"))
            self.context.task_graph.add_task(TaskNode(task_id="task_2", goal="Second task", assigned_agent="TestAgent", status="pending", dependencies=["task_1"]))

        @patch('builtins.print')
        def test_display_status(self, mock_print):
            """Tests that the status display prints the correct information."""
            print("\n--- [Test Case 1: Display Status] ---")
            self.ui.display_status(self.context)
            
            # Check if key elements of the status were printed
            mock_print.assert_any_call("✅ [SUCCESS] task_1: First task (Agent: TestAgent)")
            mock_print.assert_any_call("⚪️ [PENDING] task_2: Second task (Agent: TestAgent)")
            mock_print.assert_any_call("    - Depends on: task_1")
            logger.info("✅ test_display_status: PASSED")

        @patch('builtins.input', return_value="JWT Authentication")
        def test_prompt_for_input(self, mock_input):
            """Tests that the prompt function returns the simulated user input."""
            print("\n--- [Test Case 2: Prompt for Input] ---")
            question = "What auth method should be used?"
            response = self.ui.prompt_for_input(question)
            
            mock_input.assert_called_once()
            self.assertEqual(response, "JWT Authentication")
            logger.info("✅ test_prompt_for_input: PASSED")

        @patch('subprocess.run')
        @patch('os.environ.get', return_value='mock_editor')
        def test_allow_artifact_edit(self, mock_os_environ, mock_subprocess_run):
            """Tests the artifact editing flow, simulating a user changing a file."""
            print("\n--- [Test Case 3: Allow Artifact Edit] ---")
            artifact_key = "tech_spec.md"
            original_content = "# Original Spec"
            edited_content = "# Edited Spec by User"
            self.context.add_artifact(artifact_key, original_content, "task_spec")

            # This is a clever way to mock the user editing a file.
            # We patch `open` to simulate reading the edited content back.
            with patch('builtins.open', MagicMock()) as mock_open:
                # When our code tries to read the temp file, this mock will return the edited content.
                mock_open.return_value.__enter__.return_value.read.return_value = edited_content
                
                result = self.ui.allow_artifact_edit(artifact_key, self.context)

            # Verify that the editor was called and the artifact was updated.
            mock_subprocess_run.assert_called_once()
            self.assertTrue(result)
            self.assertEqual(self.context.get_artifact(artifact_key), edited_content)
            logger.info("✅ test_allow_artifact_edit: PASSED")

    unittest.main(argv=['first-arg-is-ignored'], exit=False)