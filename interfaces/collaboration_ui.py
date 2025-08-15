# interfaces/collaboration_ui.py
import logging
import os
import sys
import time
from typing import Any, List

# Foundational dependencies
from core.context import GlobalContext
from core.models import TaskNode, AgentResponse
from utils.input_handler import get_input_handler
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.context import GlobalContext

# Get a logger instance for this module
logger = logging.getLogger(__name__)

# --- GitHub-inspired Color Theme ---
class Style:
    """Clean, professional color theme inspired by GitHub."""
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    STRIKETHROUGH = "\033[9m"
    
    class Fg:
        # Success states - soft green
        SUCCESS = "\033[38;2;40;167;69m"    # GitHub green
        # Info/progress - calm blue  
        INFO = "\033[38;2;88;166;255m"      # GitHub blue
        # Warnings - amber (less jarring than yellow)
        WARNING = "\033[38;2;251;188;5m"   # GitHub amber
        # Errors - muted red
        ERROR = "\033[38;2;248;81;73m"     # GitHub red
        # Secondary text - soft gray
        MUTED = "\033[38;2;106;115;125m"   # GitHub gray
        # Code/output - purple accent
        CODE = "\033[38;2;171;142;255m"    # GitHub purple

class CollaborationUI:
    """
    Provides a rich, interactive, terminal-based interface for human-in-the-loop collaboration.
    This class manages all direct user interaction, providing a fluid and intuitive experience.
    """
    
    def __init__(self):
        """Initialize the CollaborationUI with enhanced input capabilities."""
        # Store command history in system-appropriate location, separate from project data
        if sys.platform == "darwin":  # macOS
            history_dir = os.path.expanduser("~/Library/Application Support/doky")
        elif sys.platform == "win32":  # Windows
            history_dir = os.path.expanduser("~/AppData/Local/doky")
        else:  # Linux/Unix
            history_dir = os.path.expanduser("~/.local/share/doky")
        
        os.makedirs(history_dir, exist_ok=True)
        history_file = os.path.join(history_dir, "command_history.txt")
        self.input_handler = get_input_handler(history_file)

    def display_status(self, context: GlobalContext, execution_summary: str = None):
        """
        Displays a beautifully formatted summary of the mission status, clearly indicating
        completed, running, and pending tasks.
        """
        print("\n" + "="*80)
        print(f"{Style.BOLD}🎯 MISSION STATUS{Style.RESET}")
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
            print(f"\n✅ {Style.Fg.SUCCESS}{Style.BOLD}Completed Tasks{Style.RESET}")
            for task in sorted(tasks_by_status["completed"], key=lambda t: t.task_id):
                status_text = "OBSOLETE" if task.status == "obsolete" else "SUCCESS"
                print(f"   - {Style.Fg.MUTED}{Style.STRIKETHROUGH}[{status_text}] {task.task_id}: {task.goal}{Style.RESET}")

        if tasks_by_status["running"]:
            print(f"\n🔄 {Style.Fg.INFO}{Style.BOLD}In Progress{Style.RESET}")
            for task in sorted(tasks_by_status["running"], key=lambda t: t.task_id):
                print(f"   - [RUNNING] {task.task_id}: {task.goal}")
        
        if tasks_by_status["failed"]:
            print(f"\n❌ {Style.Fg.ERROR}{Style.BOLD}Failed Tasks{Style.RESET}")
            for task in sorted(tasks_by_status["failed"], key=lambda t: t.task_id):
                print(f"   - [FAILED] {task.task_id}: {task.goal}")
                if task.result:
                    print(f"     {Style.Fg.ERROR}Reason: {task.result.message}{Style.RESET}")

        if tasks_by_status["pending"]:
            print(f"\n⏳ {Style.BOLD}Remaining Tasks{Style.RESET}")
            for task in sorted(tasks_by_status["pending"], key=lambda t: t.task_id):
                print(f"   - [PENDING] {task.task_id}: {task.goal}")
        
        if execution_summary:
            print("-" * 80)
            print(f"ℹ️  {Style.Fg.INFO}[System] {execution_summary}{Style.RESET}")

        print("="*80)

    def prompt_for_input(self, prompt: str) -> str:
        """Asks the user an open-ended question."""
        print("\n" + "-"*80)
        print(f"💬 {Style.Fg.INFO}{Style.BOLD}INPUT REQUIRED{Style.RESET}")
        print(f"  > {prompt}")
        print("-"*80)
        user_response = self.input_handler.prompt("Your response: ")
        logger.info(f"User was prompted with '{prompt}' and responded: '{user_response}'")
        return user_response

    def prompt_for_confirmation(self, question: str) -> bool:
        """
        Asks the user a yes/no question for approval of a critical action.
        This is the generic approval mechanism for plans, commands, etc.
        """
        print("\n" + "-"*80)
        print(f"⚡ {Style.Fg.WARNING}{Style.BOLD}APPROVAL REQUIRED{Style.RESET}")
        print(f"  > {question}")
        print("-"*80)
        response = self.input_handler.prompt("Do you want to proceed? (yes/no): ").lower()
        return response in ["y", "yes"]

    def display_system_message(self, message: str, is_error: bool = False):
        """Prints a formatted system message for updates or errors."""
        if is_error:
            print(f"\n❌ {Style.Fg.ERROR}[Error]{Style.RESET} {message}")
        else:
            print(f"\nℹ️  {Style.Fg.INFO}[System]{Style.RESET} {message}")

    def present_plan_for_approval(self, task_graph) -> bool:
        """
        Presents the generated plan to the user and asks for approval.
        Returns True if approved, False if rejected.
        """
        print("\n" + "="*80)
        print(f"📋 {Style.BOLD}{Style.Fg.INFO}GENERATED PLAN{Style.RESET}")
        print("="*80)
        
        if not task_graph.nodes:
            print("No tasks in the plan.")
            return False
        
        # Display tasks in dependency order
        for i, (task_id, task) in enumerate(task_graph.nodes.items(), 1):
            status_icon = "✅" if task.status == "success" else "🔄" if task.status == "running" else "⏸️"
            print(f"{i:2}. {status_icon} {task.goal}")
            print(f"    Agent: {task.assigned_agent}")
            if task.dependencies:
                print(f"    Dependencies: {', '.join(task.dependencies)}")
            print()
        
        print("="*80)
        print("Options:")
        print("  approve - Execute this plan")
        print("  refine  - Modify or improve the plan") 
        print("  cancel  - Cancel and start over")
        print("-" * 80)
        
        while True:
            choice = self.input_handler.prompt("Your choice (approve/refine/cancel): ").lower()
            if choice in ["approve", "refine", "cancel"]:
                logger.info(f"User chose: {choice}")
                return choice
            print("Please enter 'approve', 'refine', or 'cancel'")

    def display_help(self, help_text: str):
        """Displays a formatted help menu."""
        print("\n" + "="*80)
        print(f"📚 {Style.BOLD}COMMAND HELP{Style.RESET}")
        print("="*80)
        print("To run a high-level goal, just type your request.")
        print("To invoke a specific agent, use one of the following commands:")
        print(help_text)
        print("\nSession commands:")
        print("  /clear      - Reset conversation context (saves snapshot first)")
        print("  /reset      - Same as /clear")
        print("  exit        - Exit the session")
        print("  quit        - Same as exit")
        print("="*80)

    def _display_artifact_content(self, artifact_key: str, content: Any):
        """Helper method to format and display artifact content."""
        print(f"\n   📄 {Style.BOLD}{Style.Fg.CODE}{artifact_key}:{Style.RESET}")
        print(f"   {Style.Fg.MUTED}" + "─" * 76 + f"{Style.RESET}")
        
        if isinstance(content, str):
            # For text content, display with proper indentation
            lines = content.split('\n')
            for line in lines[:50]:  # Limit to first 50 lines to avoid overwhelming output
                print(f"   {line}")
            if len(lines) > 50:
                print(f"   {Style.Fg.MUTED}... ({len(lines) - 50} more lines){Style.RESET}")
        elif isinstance(content, (dict, list)):
            # For structured data, display as formatted JSON
            import json
            json_str = json.dumps(content, indent=2, ensure_ascii=False)
            lines = json_str.split('\n')
            for line in lines[:30]:  # Limit to first 30 lines for JSON
                print(f"   {line}")
            if len(lines) > 30:
                print(f"   {Style.Fg.MUTED}... ({len(lines) - 30} more lines){Style.RESET}")
        else:
            # For other types, show string representation
            print(f"   {str(content)[:500]}{'...' if len(str(content)) > 500 else ''}")
        
        print(f"   {Style.Fg.MUTED}" + "─" * 76 + f"{Style.RESET}")

    def display_agent_progress(self, agent_name: str, step: str, details: str = None):
        """Shows real-time progress updates during agent execution."""
        timestamp = time.strftime("%H:%M:%S")
        print(f"\n🔄 [{Style.Fg.MUTED}{timestamp}{Style.RESET}] {Style.Fg.INFO}{agent_name}{Style.RESET}: {step}")
        if details:
            print(f"   {Style.Fg.MUTED}└─ {details}{Style.RESET}")

    def display_agent_thinking(self, agent_name: str, thought: str):
        """Shows the agent's reasoning or thought process."""
        print(f"\n💭 {Style.Fg.CODE}{agent_name} thinking:{Style.RESET}")
        print(f"   {Style.Fg.MUTED}\"{thought}\"{Style.RESET}")

    def display_intermediate_output(self, agent_name: str, output_type: str, content: Any, preview_lines: int = 10):
        """Shows intermediate outputs like generated code, specs, etc. with preview."""
        print(f"\n📄 {Style.Fg.INFO}{agent_name}{Style.RESET} generated {Style.Fg.CODE}{output_type}{Style.RESET}:")
        print(f"   {Style.Fg.MUTED}" + "─" * 76 + f"{Style.RESET}")
        
        if isinstance(content, str):
            lines = content.split('\n')
            for i, line in enumerate(lines[:preview_lines]):
                print(f"   {Style.Fg.MUTED}{i+1:3}{Style.RESET} | {line}")
            if len(lines) > preview_lines:
                print(f"   {Style.Fg.MUTED}... ({len(lines) - preview_lines} more lines){Style.RESET}")
        else:
            print(f"   {str(content)[:300]}{'...' if len(str(content)) > 300 else ''}")
        print(f"   {Style.Fg.MUTED}" + "─" * 76 + f"{Style.RESET}")

    def display_failure_analysis(self, agent_name: str, error: str, troubleshooting_steps: List[str] = None):
        """Shows detailed failure analysis with troubleshooting suggestions."""
        print(f"\n❌ {Style.Fg.ERROR}{Style.BOLD}{agent_name} FAILURE ANALYSIS{Style.RESET}")
        print(f"   {Style.Fg.MUTED}" + "─" * 76 + f"{Style.RESET}")
        print(f"   {Style.BOLD}Error:{Style.RESET} {error}")
        
        if troubleshooting_steps:
            print(f"\n   {Style.BOLD}Suggested Troubleshooting Steps:{Style.RESET}")
            for i, step in enumerate(troubleshooting_steps, 1):
                print(f"   {Style.Fg.MUTED}{i}.{Style.RESET} {step}")
        print(f"   {Style.Fg.MUTED}" + "─" * 76 + f"{Style.RESET}")

    def display_direct_command_result(self, agent_name: str, response: AgentResponse, context: 'GlobalContext' = None):
        """Displays the formatted result of a single agent's execution."""
        print(f"\n{Style.Fg.MUTED}" + "-"*80 + f"{Style.RESET}")
        if response.success:
            print(f"✅ {Style.Fg.SUCCESS}{Style.BOLD}{agent_name} finished successfully.{Style.RESET}")
        else:
            print(f"❌ {Style.Fg.ERROR}{Style.BOLD}{agent_name} failed.{Style.RESET}")
        
        print(f"   - {Style.BOLD}Message:{Style.RESET} {response.message}")
        
        if response.artifacts_generated:
            print(f"   - {Style.BOLD}Artifacts Created/Updated:{Style.RESET} {Style.Fg.CODE}{', '.join(response.artifacts_generated)}{Style.RESET}")
            
            # Display artifact content if context is provided
            if context:
                for artifact_key in response.artifacts_generated:
                    artifact_content = context.get_artifact(artifact_key)
                    if artifact_content is not None:
                        self._display_artifact_content(artifact_key, artifact_content)
                    else:
                        print(f"   {Style.Fg.WARNING}⚠️ Artifact '{artifact_key}' not found in context{Style.RESET}")
        print(f"{Style.Fg.MUTED}" + "-"*80 + f"{Style.RESET}")
        
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
            self.context.task_graph.add_task(TaskNode(task_id="task_1", goal="Completed task", assigned_agent="TestAgent", status="success"))
            self.context.task_graph.add_task(TaskNode(task_id="task_2", goal="Running task", assigned_agent="TestAgent", status="running"))
            self.context.task_graph.add_task(TaskNode(task_id="task_3", goal="Pending task", assigned_agent="TestAgent", status="pending"))

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

        @patch('builtins.print')
        def test_artifact_display(self, mock_print):
            print("\n--- [Test Case 4: Artifact Display] ---")
            # Create a test response with artifacts
            response = AgentResponse(success=True, message="Generated spec", artifacts_generated=["test_spec.md"])
            
            # Add test artifact to context
            self.context.add_artifact("test_spec.md", "# Test Specification\nThis is a test spec.", "test_task")
            
            # Display the result
            self.ui.display_direct_command_result("TestAgent", response, self.context)
            
            # Verify artifact content is displayed
            output = "\n".join([str(call.args[0]) for call in mock_print.call_args_list if call.args])
            self.assertIn("test_spec.md", output)
            self.assertIn("Test Specification", output)
            logger.info("✅ test_artifact_display: PASSED")

    unittest.main(argv=['first-arg-is-ignored'], exit=False)