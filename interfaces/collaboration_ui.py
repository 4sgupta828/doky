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
        
        # Store the last displayed content for Ctrl+R functionality
        self.last_full_content = {}

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
            return "cancel"
        
        # Display tasks in dependency order
        for i, (task_id, task) in enumerate(task_graph.nodes.items(), 1):
            status_icon = "✅" if task.status == "success" else "🔄" if task.status == "running" else "⏸️"
            print(f"{i:2}. {status_icon} {task.goal}")
            print(f"    Agent: {task.assigned_agent}")
            if task.dependencies:
                print(f"    Dependencies: {', '.join(task.dependencies)}")
            print()
        
        print("="*80)
        print("Review the plan above:")
        print("  • Press ENTER to approve and execute this plan")
        print("  • Type any feedback to refine the plan")
        print("  • Type 'cancel' to cancel and start over")
        print("-" * 80)
        
        user_input = self.input_handler.prompt("Your response: ").strip()
        
        # Empty input (just Enter) means approval
        if not user_input:
            logger.info("User approved the plan (empty input)")
            return "approve"
        elif user_input.lower() == "cancel":
            logger.info("User cancelled the plan")
            return "cancel"
        else:
            logger.info(f"User provided feedback for plan refinement: {user_input}")
            return "refine"

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
        print("  Ctrl+R      - Show full output history (complete code, diffs, results)")
        print("  exit        - Exit the session")
        print("  quit        - Same as exit")
        print("="*80)

    def _display_artifact_content(self, artifact_key: str, content: Any, show_full: bool = False):
        """Helper method to format and display artifact content."""
        print(f"\n   📄 {Style.BOLD}{Style.Fg.CODE}{artifact_key}:{Style.RESET}")
        print(f"   " + "─" * 76)
        
        # Store full content for later retrieval
        content_key = f"artifact_{artifact_key}"
        self.last_full_content[content_key] = content
        
        if isinstance(content, str):
            # For text content, display with proper indentation
            lines = content.split('\n')
            display_lines = lines if show_full else lines[:50]
            for line in display_lines:
                print(f"   {line}")
            if not show_full and len(lines) > 50:
                print(f"   {Style.Fg.MUTED}... ({len(lines) - 50} more lines) - Press Ctrl+R to see full output{Style.RESET}")
        elif isinstance(content, (dict, list)):
            # For structured data, display as formatted JSON
            import json
            json_str = json.dumps(content, indent=2, ensure_ascii=False)
            lines = json_str.split('\n')
            display_lines = lines if show_full else lines[:30]
            for line in display_lines:
                print(f"   {line}")
            if not show_full and len(lines) > 30:
                print(f"   {Style.Fg.MUTED}... ({len(lines) - 30} more lines) - Press Ctrl+R to see full output{Style.RESET}")
        else:
            # For other types, show string representation
            str_content = str(content)
            if show_full or len(str_content) <= 500:
                print(f"   {str_content}")
            else:
                print(f"   {str_content[:500]}...")
                print(f"   {Style.Fg.MUTED}... (truncated) - Press Ctrl+R to see full output{Style.RESET}")
        
        print(f"   " + "─" * 76)

    def display_agent_progress(self, agent_name: str, step: str, details: str = None):
        """Shows real-time progress updates during agent execution."""
        timestamp = time.strftime("%H:%M:%S")
        print(f"\n🔄 [{timestamp}] {agent_name}: {step}")
        if details:
            print(f"   └─ {details}")

    def display_agent_thinking(self, agent_name: str, thought: str):
        """Shows the agent's reasoning or thought process."""
        print(f"\n💭 {agent_name} thinking:")
        print(f"   \"{thought}\"")

    def display_agent_input(self, agent_name: str, goal: str, inputs: dict, reasoning: str = None):
        """Shows clean, formatted agent input with smart trimming and boundaries."""
        from utils.content_trimmer import trim_content
        
        print(f"\n{'═' * 80}")
        print(f"🤖 {Style.Fg.INFO}{Style.BOLD}[AGENT INPUT] {agent_name}{Style.RESET}")
        if reasoning:
            print(f"🧠 {Style.Fg.MUTED}Routing reason: {reasoning}{Style.RESET}")
        print(f"{'─' * 80}")
        
        # Display goal
        print(f"🎯 {Style.BOLD}Goal:{Style.RESET} {goal}")
        
        # Display inputs with smart trimming
        if inputs:
            print(f"📥 {Style.BOLD}Inputs:{Style.RESET}")
            for key, value in inputs.items():
                # Trim the value smartly
                trim_result = trim_content(value, "auto")
                
                print(f"   {Style.Fg.CODE}{key}:{Style.RESET}")
                if trim_result.was_truncated:
                    print(f"      {trim_result.content}")
                    print(f"      {Style.Fg.MUTED}... {trim_result.truncation_info}{Style.RESET}")
                else:
                    # For short content, format nicely
                    content_lines = trim_result.content.split('\n')
                    for line in content_lines:
                        print(f"      {line}")
        else:
            print(f"📥 {Style.Fg.MUTED}No inputs provided{Style.RESET}")
        
        print(f"{'─' * 80}")

    def display_agent_output(self, agent_name: str, success: bool, message: str, outputs: dict = None):
        """Shows clean, formatted agent output with smart trimming and boundaries."""
        from utils.content_trimmer import trim_content
        
        # Status indicator
        status_icon = "✅" if success else "❌"
        status_color = Style.Fg.SUCCESS if success else Style.Fg.ERROR
        
        print(f"📤 {Style.BOLD}Output:{Style.RESET}")
        print(f"   {status_icon} {status_color}Status:{Style.RESET} {'SUCCESS' if success else 'FAILED'}")
        print(f"   💬 {Style.BOLD}Message:{Style.RESET} {message}")
        
        # Display outputs with smart trimming  
        if outputs:
            print(f"   📋 {Style.BOLD}Data:{Style.RESET}")
            for key, value in outputs.items():
                # Trim the value smartly
                trim_result = trim_content(value, "auto")
                
                print(f"      {Style.Fg.CODE}{key}:{Style.RESET}")
                if trim_result.was_truncated:
                    # Show trimmed content with indentation
                    content_lines = trim_result.content.split('\n')
                    for line in content_lines:
                        print(f"         {line}")
                    print(f"         {Style.Fg.MUTED}... {trim_result.truncation_info}{Style.RESET}")
                else:
                    # For short content, format nicely  
                    content_lines = trim_result.content.split('\n')
                    for line in content_lines:
                        print(f"         {line}")
        
        print(f"{'═' * 80}")

    def display_routing_decision(self, from_agent: str, to_agent: str, confidence: float, reasoning: str):
        """Shows intelligent routing decisions between agents."""
        print(f"\n🔀 {Style.Fg.INFO}{Style.BOLD}[ROUTING DECISION]{Style.RESET}")
        print(f"   {from_agent} → {Style.Fg.SUCCESS}{to_agent}{Style.RESET} (confidence: {confidence:.2f})")
        print(f"   💡 {Style.Fg.MUTED}{reasoning}{Style.RESET}")

    def display_llm_communication(self, agent_name: str, prompt_preview: str, response_preview: str):
        """Shows LLM communication for transparency."""
        from utils.content_trimmer import trim_content
        
        print(f"\n🧠 {Style.Fg.CODE}{Style.BOLD}[LLM COMMUNICATION] {agent_name}{Style.RESET}")
        print(f"{'─' * 80}")
        
        # Show prompt preview - increased limit for better visibility
        prompt_trim = trim_content(prompt_preview, "text", {"text": 800})
        print(f"📤 {Style.BOLD}Prompt:{Style.RESET}")
        prompt_lines = prompt_trim.content.split('\n')
        for line in prompt_lines:
            print(f"   {line}")
        if prompt_trim.was_truncated:
            print(f"   {Style.Fg.MUTED}... {prompt_trim.truncation_info}{Style.RESET}")
        
        # Show response preview - increased limit for better visibility
        response_trim = trim_content(response_preview, "text", {"text": 1000})
        print(f"📥 {Style.BOLD}Response:{Style.RESET}")
        response_lines = response_trim.content.split('\n')
        for line in response_lines:
            print(f"   {line}")
        if response_trim.was_truncated:
            print(f"   {Style.Fg.MUTED}... {response_trim.truncation_info}{Style.RESET}")
        
        print(f"{'─' * 80}")

    def display_intermediate_output(self, agent_name: str, output_type: str, content: Any, preview_lines: int = 10):
        """Shows intermediate outputs like generated code, specs, etc. with preview."""
        if output_type == "code_diff" and isinstance(content, dict):
            self.display_code_diff(agent_name, content)
        elif output_type == "code_snippet":
            if isinstance(content, dict) and "content" in content and "filename" in content:
                self.display_code_snippet(agent_name, content["content"], content["filename"])
            elif isinstance(content, str):
                self.display_code_snippet(agent_name, content, "code")
            else:
                self._display_original_intermediate_output(agent_name, output_type, content, preview_lines)
        elif output_type == "code" and isinstance(content, str):
            self.display_code_snippet(agent_name, content, "code")
        elif output_type == "code_files" and isinstance(content, dict):
            self.display_code_files(agent_name, content)
        else:
            # Fallback to original display method
            self._display_original_intermediate_output(agent_name, output_type, content, preview_lines)
    
    def _display_original_intermediate_output(self, agent_name: str, output_type: str, content: Any, preview_lines: int = 10, show_full: bool = False):
        """Original intermediate output display method."""
        print(f"\n📄 {agent_name} generated {output_type}:")
        print(f"   " + "─" * 76)
        
        # Store full content for later retrieval
        content_key = f"output_{agent_name}_{output_type}"
        self.last_full_content[content_key] = content
        
        if isinstance(content, str):
            lines = content.split('\n')
            display_lines = lines if show_full else lines[:preview_lines]
            for i, line in enumerate(display_lines, 1):
                print(f"   {i:3} | {line}")
            if not show_full and len(lines) > preview_lines:
                print(f"   ... ({len(lines) - preview_lines} more lines) - Press Ctrl+R to see full output")
        else:
            str_content = str(content)
            if show_full or len(str_content) <= 1000:  # Increased from 300
                print(f"   {str_content}")
            else:
                print(f"   {str_content[:1000]}...")  # Increased from 300
                print(f"   {Style.Fg.MUTED}... (truncated) - Press Ctrl+R to see full output{Style.RESET}")
        print(f"   " + "─" * 76)
    
    def display_code_snippet(self, agent_name: str, content: str, filename: str = "code", show_full: bool = False):
        """Displays a code snippet with syntax highlighting and line numbers."""
        print(f"\n📝 {agent_name} generated {filename}:")
        print(f"   ┌" + "─" * 76 + "┐")
        
        # Store full content for later retrieval
        content_key = f"code_{agent_name}_{filename}"
        self.last_full_content[content_key] = content
        
        lines = content.split('\n')
        max_line_num_width = len(str(len(lines)))
        
        # Show full content or truncated based on show_full parameter
        display_lines = lines if show_full else lines[:50]  # Limit to 50 lines for code
        
        for i, line in enumerate(display_lines, 1):
            line_num = f"{i:>{max_line_num_width}}"
            print(f"   │ {line_num} │ {line}")
        
        if not show_full and len(lines) > 50:
            print(f"   │ ... ({len(lines) - 50} more lines) - Press Ctrl+R to see full output │")
        
        print(f"   └" + "─" * 76 + "┘")
        print(f"   Lines: {len(lines)}")
    
    def display_code_diff(self, agent_name: str, diff_data: dict):
        """Displays GitHub-style diff for code changes."""
        print(f"\n🔄 {agent_name} generated code diff:")
        
        for file_path, changes in diff_data.items():
            print(f"\n   📁 {file_path}")
            print(f"   ──────────────────────────────────────────")
            
            if isinstance(changes, dict) and 'old' in changes and 'new' in changes:
                old_lines = changes['old'].split('\n') if changes['old'] else []
                new_lines = changes['new'].split('\n') if changes['new'] else []
                
                # Simple diff implementation
                self._display_simple_diff(old_lines, new_lines)
            elif isinstance(changes, str):
                # New file
                print(f"   +++ New file")
                lines = changes.split('\n')
                for i, line in enumerate(lines, 1):
                    print(f"   +{i:3} │ {line}")
    
    def display_code_files(self, agent_name: str, files_data: dict, show_full: bool = False):
        """Displays multiple code files with syntax highlighting."""
        print(f"\n📁 {agent_name} generated {len(files_data)} files:")
        
        # Store full content for later retrieval
        content_key = f"files_{agent_name}"
        self.last_full_content[content_key] = files_data
        
        for file_path, content in files_data.items():
            print(f"\n   📄 {file_path}")
            print(f"   ┌" + "─" * 76 + "┐")
            
            if isinstance(content, str):
                lines = content.split('\n')
                max_line_num_width = len(str(len(lines))) if lines else 1
                
                # Show full content or first 20 lines based on show_full parameter
                display_lines = lines if show_full else lines[:20]
                for i, line in enumerate(display_lines, 1):
                    line_num = f"{i:>{max_line_num_width}}"
                    print(f"   │ {line_num} │ {line}")
                
                if not show_full and len(lines) > 20:
                    print(f"   │ ... ({len(lines) - 20} more lines) - Press Ctrl+R to see full output │")
                    
            print(f"   └" + "─" * 76 + "┘")
            print(f"   Lines: {len(content.split('\n')) if isinstance(content, str) else 0}")
    
    def _display_simple_diff(self, old_lines: list, new_lines: list):
        """Displays a simple unified diff between old and new lines."""
        import difflib
        
        diff = list(difflib.unified_diff(
            old_lines, new_lines, 
            fromfile='old', tofile='new', 
            lineterm='', n=3
        ))
        
        for line in diff[3:]:  # Skip the diff header lines
            print(f"   {line}")

    def display_failure_analysis(self, agent_name: str, error: str, troubleshooting_steps: List[str] = None):
        """Shows detailed failure analysis with troubleshooting suggestions."""
        print(f"\n❌ {Style.Fg.ERROR}{Style.BOLD}{agent_name} FAILURE ANALYSIS{Style.RESET}")
        print(f"   " + "─" * 76)
        print(f"   {Style.BOLD}Error:{Style.RESET} {error}")
        
        if troubleshooting_steps:
            print(f"\n   {Style.BOLD}Suggested Troubleshooting Steps:{Style.RESET}")
            for i, step in enumerate(troubleshooting_steps, 1):
                print(f"   {Style.Fg.MUTED}{i}.{Style.RESET} {step}")
        print(f"   " + "─" * 76)

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
    
    def display_full_output(self):
        """Display all stored full content when Ctrl+R is pressed."""
        if not self.last_full_content:
            print(f"\n{Style.Fg.MUTED}No recent output to display in full.{Style.RESET}")
            return
        
        print(f"\n{Style.BOLD}📜 FULL OUTPUT HISTORY (Ctrl+R){Style.RESET}")
        print("=" * 80)
        
        for content_key, content in self.last_full_content.items():
            print(f"\n{Style.BOLD}{Style.Fg.INFO}🔍 {content_key}:{Style.RESET}")
            print("─" * 80)
            
            if content_key.startswith("artifact_"):
                # Display artifact content in full
                artifact_name = content_key.replace("artifact_", "")
                self._display_artifact_content(artifact_name, content, show_full=True)
            elif content_key.startswith("output_"):
                # Display intermediate output in full
                parts = content_key.replace("output_", "").split("_", 1)
                agent_name = parts[0] if parts else "Unknown"
                output_type = parts[1] if len(parts) > 1 else "output"
                self._display_original_intermediate_output(agent_name, output_type, content, show_full=True)
            elif content_key.startswith("code_"):
                # Display code snippet in full
                parts = content_key.replace("code_", "").split("_", 1)
                agent_name = parts[0] if parts else "Unknown"
                filename = parts[1] if len(parts) > 1 else "code"
                self.display_code_snippet(agent_name, content, filename, show_full=True)
            elif content_key.startswith("files_"):
                # Display code files in full
                agent_name = content_key.replace("files_", "")
                self.display_code_files(agent_name, content, show_full=True)
            else:
                # Fallback display
                if isinstance(content, str):
                    lines = content.split('\n')
                    for i, line in enumerate(lines, 1):
                        print(f"{i:4} | {line}")
                else:
                    print(str(content))
            
            print("─" * 80)
        
        print(f"\n{Style.Fg.MUTED}End of full output history{Style.RESET}")
        print("=" * 80)
    
    def clear_output_history(self):
        """Clear the stored output history."""
        self.last_full_content.clear()
        print(f"\n{Style.Fg.INFO}Output history cleared.{Style.RESET}")
        
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