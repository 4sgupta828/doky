# interfaces/collaboration_ui.py
import logging
import os
import subprocess
import tempfile
import sys
from typing import Dict, Any

# Foundational dependencies
from core.context import GlobalContext
from core.models import TaskGraph, TaskNode

# Get a logger instance for this module
logger = logging.getLogger(__name__)

# --- ANSI Escape Codes for Rich Terminal Formatting ---
class Style:
    """A helper class for ANSI terminal styling."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    STRIKETHROUGH = "\033[9m"
    
    class Fg:
        GREEN = "\033[32m"
        BLUE = "\033[34m"
        YELLOW = "\033[33m"
        RED = "\033[31m"
        GRAY = "\033[90m"

class CollaborationUI:
    """
    Provides a rich, interactive, terminal-based interface for human-in-the-loop collaboration.
    This class manages all direct user interaction, providing a fluid and intuitive experience.
    """

    def display_status(self, context: GlobalContext, execution_summary: str = None):
        """
        Displays a beautifully formatted summary of the mission status, clearly indicating
        completed, running, and pending tasks.
        """
        print("\n" + "="*80)
        print(f"{Style.BOLD}
<img src="https://placehold.co/15x15/7c3aed/ffffff?text=S" alt="Status Icon">
 MISSION STATUS{Style.RESET}")
        print("="*80)

        if not context.task_graph.nodes:
            print("No plan is currently active. Waiting for your next goal.")
            return

        # Separate tasks by status for clear grouping
        tasks_by_status = {
            "completed": [], "running": [], "pending": [], "failed": [], "obsolete": []
        }
        for task in context.task_graph.nodes.values():
            if task.status in ["success", "obsolete"]:
                tasks_by_status["completed"].append(task)
            elif task.status in tasks_by_status:
                tasks_by_status[task.status].append(task)

        if tasks_by_status["completed"]:
            print(f"\n✅ {Style.Fg.GREEN}{Style.BOLD}Completed Tasks{Style.RESET}")
            for task in sorted(tasks_by_status["completed"], key=lambda t: t.task_id):
                status_text = "OBSOLETE" if task.status == "obsolete" else "SUCCESS"
                print(f"   - {Style.Fg.GRAY}{Style.STRIKETHROUGH}[{status_text}] {task.task_id}: {task.goal}{Style.RESET}")

        if tasks_by_status["running"]:
            print(f"\n🔵 {Style.Fg.BLUE}{Style.BOLD}In Progress{Style.RESET}")
            for task in sorted(tasks_by_status["running"], key=lambda t: t.task_id):
                print(f"   - [RUNNING] {task.task_id}: {task.goal}")
        
        if tasks_by_status["failed"]:
            print(f"\n❌ {Style.Fg.RED}{Style.BOLD}Failed Tasks{Style.RESET}")
            for task in sorted(tasks_by_status["failed"], key=lambda t: t.task_id):
                print(f"   - [FAILED] {task.task_id}: {task.goal}")
                if task.result:
                    print(f"     {Style.Fg.RED}Reason: {task.result.message}{Style.RESET}")

        if tasks_by_status["pending"]:
            print(f"\n⚪️ {Style.BOLD}Remaining Tasks{Style.RESET}")
            for task in sorted(tasks_by_status["pending"], key=lambda t: t.task_id):
                print(f"   - [PENDING] {task.task_id}: {task.goal}")
        
        if execution_summary:
            print("-" * 80)
            print(f"🤖 {Style.Fg.YELLOW}[System] {execution_summary}{Style.RESET}")

        print("="*80)

    def prompt_for_input(self, prompt: str) -> str:
        """Asks the user an open-ended question."""
        print("\n" + "-"*80)
        print(f"
<img src="https://placehold.co/15x15/3b82f6/ffffff?text=Q" alt="Question Icon">
 {Style.Fg.BLUE}{Style.BOLD}ACTION REQUIRED{Style.RESET}")
        print(f"  > {prompt}")
        print("-"*80)
        user_response = input("Your response: ")
        logger.info(f"User was prompted with '{prompt}' and responded: '{user_response}'")
        return user_response.strip()

    def prompt_for_confirmation(self, question: str) -> bool:
        """
        Asks the user a yes/no question for approval of a critical action.
        This is the generic approval mechanism for plans, commands, etc.
        """
        print("\n" + "-"*80)
        print(f"
<img src="https://placehold.co/15x15/f59e0b/ffffff?text=A" alt="Approval Icon">
 {Style.Fg.YELLOW}{Style.BOLD}APPROVAL REQUIRED{Style.RESET}")
        print(f"  > {question}")
        print("-"*80)
        response = input("Do you want to proceed? (yes/no): ").lower().strip()
        return response in ["y", "yes"]

    def display_system_message(self, message: str, is_error: bool = False):
        """Prints a formatted system message for updates or errors."""
        if is_error:
            print(f"\n❌ {Style.Fg.RED}[Error]{Style.RESET} {message}")
        else:
            print(f"\n🤖 {Style.Fg.BLUE}[System]{Style.RESET} {message}")


# --- Self-Testing Block ---
if __name__ == "__main__":
    import unittest
    from unittest.mock import patch
    from core.models import TaskNode
    from utils.logger import setup_logger

    setup_logger(default_level=logging.INFO)

    class TestCollaborationUI(unittest.TestCase):

        def setUp(self):
            self.ui = CollaborationUI()
            self.context = GlobalContext()
            self.context.task_graph.add_task(TaskNode(task_id="task_1", goal="Completed task", status="success"))
            self.context.task_graph.add_task(TaskNode(task_id="task_2", goal="Running task", status="running"))
            self.context.task_graph.add_task(TaskNode(task_id="task_3", goal="Pending task", status="pending"))

        @patch('builtins.print')
        def test_display_status_formatting(self, mock_print):
            print("\n--- [Test Case 1: Rich Status Display] ---")
            self.ui.display_status(self.context)
            
            # Convert mock calls to a single string for easy searching
            output = "\n".join([call.args[0] for call in mock_print.call_args_list])

            self.assertIn("Completed Tasks", output)
            self.assertIn(Style.STRIKETHROUGH, output) # Check for strikethrough
            self.assertIn("[SUCCESS]", output)
            
            self.assertIn("In Progress", output)
            self.assertIn("[RUNNING]", output)
            
            self.assertIn("Remaining Tasks", output)
            self.assertIn("[PENDING]", output)
            logger.info("✅ test_display_status_formatting: PASSED")

        @patch('builtins.input', return_value="yes")
        def test_prompt_for_confirmation_yes(self, mock_input):
            print("\n--- [Test Case 2: Confirmation Prompt (Yes)] ---")
            result = self.ui.prompt_for_confirmation("Run `pip install`?")
            self.assertTrue(result)
            logger.info("✅ test_prompt_for_confirmation_yes: PASSED")

        @patch('builtins.input', return_value="no")
        def test_prompt_for_confirmation_no(self, mock_input):
            print("\n--- [Test Case 3: Confirmation Prompt (No)] ---")
            result = self.ui.prompt_for_confirmation("Apply code changes?")
            self.assertFalse(result)
            logger.info("✅ test_prompt_for_confirmation_no: PASSED")

    unittest.main(argv=['first-arg-is-ignored'], exit=False)