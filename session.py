# session.py
import logging
import sys
from typing import Optional

# Foundational dependencies
from orchestrator import Orchestrator
from interfaces.collaboration_ui import CollaborationUI, Style
from core.models import AgentResponse
from agents import AGENT_ALIASES, get_agent_help

# Get a logger instance for this module
logger = logging.getLogger(__name__)

class InteractiveSession:
    """
    Manages the continuous, interactive session with the user, orchestrating the
    entire Read-Plan-Execute-Loop (RPEL) with a refinement stage.
    """

    def __init__(self, workspace_path: Optional[str] = None, resume_snapshot: Optional[str] = None):
        """
        Initializes the interactive session.

        Args:
            workspace_path: The file path for the mission's workspace. If None,
                          auto-generates a timestamped directory in /Users/sgupta/
            resume_snapshot: Path to snapshot file to resume from (for crash recovery)
        """
        try:
            self.ui = CollaborationUI()
            
            # Handle session resume from snapshot
            if resume_snapshot:
                from core.context import GlobalContext
                self.global_context = GlobalContext.load_from_snapshot(resume_snapshot)
                self.orchestrator = Orchestrator(global_context=self.global_context, ui_interface=self.ui)
                self.ui.display_system_message(f"Session resumed from snapshot: {resume_snapshot}")
            else:
                self.orchestrator = Orchestrator(workspace_path=workspace_path, ui_interface=self.ui)
                self.global_context = self.orchestrator.global_context
            logging.info("InteractiveSession initialized successfully.")
        except Exception as e:
            logging.critical("Failed to initialize core components for InteractiveSession.", exc_info=True)
            raise RuntimeError(f"Failed to initialize session: {e}")

    def start(self):
        """
        The main Read-Plan-Execute-Loop (RPEL). This is the primary entry point
        that runs the interactive session.
        """
        self.ui.display_system_message(f"{Style.BOLD}Sovereign Agent Collective: Interactive Session Activated{Style.RESET}")
        
        # Communicate workspace information to user
        workspace_path = self.global_context.workspace_path
        if self.global_context.workspace_was_auto_generated:
            self.ui.display_system_message(f"📁 Created new workspace: {Style.BOLD}{workspace_path}{Style.RESET}")
        else:
            self.ui.display_system_message(f"📁 Using workspace: {Style.BOLD}{workspace_path}{Style.RESET}")
        
        # Show session data directory
        session_dir = self.global_context.session_dir
        self.ui.display_system_message(f"💾 Session data: {session_dir}")
        
        print("Type your goal and press Enter. Type 'exit' or 'quit' to end.")

        while True:
            try:
                user_input = self.ui.prompt_for_input("Your command")
                if user_input.lower() in ["exit", "quit"]:
                    # Save session data before exiting
                    self._save_session_data()
                    self.ui.display_system_message("Session terminated. Goodbye!")
                    break

                if not user_input:
                    continue

                # --- COMMAND DISPATCHER ---
                if user_input.lower() in ["/clear", "/reset"]:
                    self._handle_clear_command()
                elif user_input.startswith('@'):
                    self._handle_direct_command(user_input)
                else:
                    self._handle_goal_command(user_input)

                self.ui.display_status(self.global_context, "Ready for your next command.")

            except KeyboardInterrupt:
                # Save session data before exiting
                self._save_session_data()
                self.ui.display_system_message("\nSession interrupted by user. Goodbye!")
                break
            except Exception as e:
                logger.critical("A critical error occurred in the session loop.", exc_info=True)
                self.ui.display_system_message(f"A fatal error occurred: {e}", is_error=True)

    def _handle_goal_command(self, goal: str):
        """
        Orchestrates the plan, confirm, and execute stages, including a
        refinement loop for the user to perfect the plan.
        """
        current_goal = goal
        plan_generated = False
        
        while True: # This is the Plan-Refine-Confirm loop
            # --- PLAN STAGE (only on first iteration or after plan failure) ---
            if not plan_generated:
                self.ui.display_system_message(f"Analyzing goal and generating a plan for: '{current_goal}'")
                plan_response: AgentResponse = self.orchestrator.plan_mission(current_goal)

                if not plan_response.success:
                    self.ui.display_system_message(f"Failed to create a plan: {plan_response.message}", is_error=True)
                    return # Exit the handler and wait for a new goal
                    
                plan_generated = True

            # --- CONFIRM STAGE ---
            user_choice = self.ui.present_plan_for_approval(self.global_context.task_graph)

            if user_choice == "approve":
                # The user is happy with the plan, so we break the loop and proceed to execution.
                break 
            
            elif user_choice == "refine":
                # The user wants to change the goal. We prompt for feedback and loop again.
                refinement_feedback = self.ui.prompt_for_input("Please provide your feedback, question, or refinement")
                if not refinement_feedback:
                    self.ui.display_system_message("No refinement provided. Presenting previous plan again.", is_error=True)
                    continue
                
                # Invoke the PlanRefinementAgent via the orchestrator
                self.ui.display_system_message("Incorporating your feedback and refining the plan...")
                refinement_response = self.orchestrator.refine_mission_plan(refinement_feedback)
                
                # Display the agent's natural language response
                self.ui.display_system_message(refinement_response.message)

                if not refinement_response.success:
                    self.ui.display_system_message("Failed to refine the plan. Presenting previous plan for another action.", is_error=True)
                
                # The loop will now continue, presenting the newly refined plan.
                continue

            elif user_choice == "cancel":
                # The user wants to scrap this plan entirely.
                self.ui.display_system_message("Plan cancelled. Awaiting next goal.")
                self.global_context.task_graph.nodes.clear()
                return # Exit the handler and wait for a completely new goal

        # --- EXECUTE STAGE ---
        self.ui.display_system_message("Plan approved. Executing plan...")
        final_status = self.orchestrator.execute_plan()
        self.ui.display_system_message(f"Plan execution finished. {final_status}")

    def _handle_direct_command(self, user_input: str):
        """Parses and executes a direct agent command (e.g., '@coder write a function')."""
        parts = user_input.strip().split(maxsplit=1)
        alias = parts[0]
        goal = parts[1] if len(parts) > 1 else ""

        if alias == "@help":
            help_text = get_agent_help()
            self.ui.display_help(help_text)
            return

        agent_name = AGENT_ALIASES.get(alias)
        if not agent_name:
            self.ui.display_system_message(f"Unknown command '{alias}'. Type '@help' to see available commands.", is_error=True)
            return
            
        if not goal:
            self.ui.display_system_message(f"Please provide a goal for the {agent_name}.", is_error=True)
            return

        self.ui.display_system_message(f"Directly invoking {agent_name}...")
        response = self.orchestrator.execute_single_task(goal, agent_name)
        self.ui.display_direct_command_result(agent_name, response, self.global_context)

    def _handle_clear_command(self):
        """Handle the /clear or /reset command to reset conversation context."""
        try:
            self.ui.display_system_message("Saving snapshot before clearing context...")
            
            # Save a final snapshot before clearing
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            snapshot_name = f"pre_clear_snapshot_{timestamp}.json"
            snapshot_path = str(self.global_context.session_dir / snapshot_name)
            self.global_context.save_snapshot(snapshot_path)
            
            # Save session summary with clear event
            task_count = len(self.global_context.task_graph.nodes)
            artifact_count = len(self.global_context.artifacts)
            
            clear_summary = {
                "session_cleared": True,
                "clear_timestamp": datetime.now().isoformat(),
                "tasks_cleared": task_count,
                "artifacts_cleared": artifact_count,
                "events_cleared": len(self.global_context.mission_log),
                "snapshot_saved": snapshot_path
            }
            
            self.global_context.save_session_memory(clear_summary)
            
            # Clear in-memory context
            self.global_context.task_graph.nodes.clear()
            self.global_context.artifacts.clear()
            self.global_context.mission_log.clear()
            
            # Log the reset event
            self.global_context.log_event(
                "session_reset",
                {"cleared_at": datetime.now().isoformat(), "snapshot_path": snapshot_path}
            )
            
            self.ui.display_system_message(f"✅ Context cleared successfully. Snapshot saved to: {snapshot_name}")
            self.ui.display_system_message("Memory reset complete. Ready for new tasks.")
            
        except Exception as e:
            logger.error(f"Failed to clear context: {e}", exc_info=True)
            self.ui.display_system_message(f"❌ Failed to clear context: {e}", is_error=True)
    
    def _save_session_data(self):
        """Save session data and create a final snapshot before exit."""
        try:
            # Save final snapshot
            self.global_context.save_snapshot()
            
            # Save command history
            if hasattr(self.ui, 'input_handler'):
                self.ui.input_handler.save_history()
                logger.info("Command history saved.")
            
            # Save session summary
            task_count = len(self.global_context.task_graph.nodes)
            artifact_count = len(self.global_context.artifacts)
            
            session_summary = {
                "session_ended": True,
                "task_count": task_count,
                "artifact_count": artifact_count,
                "total_events": len(self.global_context.mission_log),
                "workspace_path": self.global_context.workspace_path,
                "workspace_was_auto_generated": self.global_context.workspace_was_auto_generated
            }
            
            self.global_context.save_session_memory(session_summary)
            logger.info("Session data saved successfully.")
            
        except Exception as e:
            logger.error(f"Failed to save session data: {e}", exc_info=True)


# --- Self-Testing Block ---
if __name__ == "__main__":
    import unittest
    from unittest.mock import patch, MagicMock

    # We need to import the real classes to mock them
    from orchestrator import Orchestrator
    from interfaces.collaboration_ui import CollaborationUI
    from core.models import AgentResponse, TaskGraph
    from utils.logger import setup_logger

    setup_logger(default_level=logging.INFO)

    class TestInteractiveSession(unittest.TestCase):

        @patch('session.Orchestrator')
        @patch('session.CollaborationUI')
        def setUp(self, MockCollaborationUI, MockOrchestrator):
            """Set up a new session with mocked dependencies for each test."""
            self.mock_orchestrator = MockOrchestrator.return_value
            self.mock_ui = MockCollaborationUI.return_value
            
            from core.context import GlobalContext
            self.mock_orchestrator.global_context = MagicMock(spec=GlobalContext)
            self.mock_orchestrator.global_context.task_graph = TaskGraph()

            self.session = InteractiveSession(workspace_path="./mock_ws")

        def test_successful_plan_and_execute(self):
            print("\n--- [Test Case 1: Successful Plan and Execute] ---")
            self.mock_orchestrator.plan_mission.return_value = AgentResponse(success=True, message="Plan created")
            self.mock_ui.present_plan_for_approval.return_value = "approve"
            self.mock_orchestrator.execute_plan.return_value = "Plan executed successfully."
            
            self.session._handle_goal_command("build an api")
            
            self.mock_orchestrator.plan_mission.assert_called_once_with("build an api")
            self.mock_ui.present_plan_for_approval.assert_called_once()
            self.mock_orchestrator.execute_plan.assert_called_once()
            logger.info("✅ test_successful_plan_and_execute: PASSED")

        def test_user_cancels_plan(self):
            print("\n--- [Test Case 2: User Cancels Plan] ---")
            self.mock_orchestrator.plan_mission.return_value = AgentResponse(success=True, message="Plan created")
            self.mock_ui.present_plan_for_approval.return_value = "cancel"
            
            self.session._handle_goal_command("build an api")

            self.mock_orchestrator.plan_mission.assert_called_once_with("build an api")
            self.mock_ui.present_plan_for_approval.assert_called_once()
            self.mock_orchestrator.execute_plan.assert_not_called()
            self.assertEqual(len(self.mock_orchestrator.global_context.task_graph.nodes), 0)
            logger.info("✅ test_user_cancels_plan: PASSED")

        def test_plan_refinement_loop(self):
            """Tests the full Plan-Refine-Confirm loop."""
            print("\n--- [Test Case 3: Plan Refinement Loop] ---")
            # The user will choose 'refine' once, then 'approve'.
            self.mock_ui.present_plan_for_approval.side_effect = ["refine", "approve"]
            # The user provides refinement text when asked.
            self.mock_ui.prompt_for_input.return_value = "add a linting step"
            
            # Mock the orchestrator's planning and refining methods.
            self.mock_orchestrator.plan_mission.return_value = AgentResponse(success=True, message="Initial plan created")
            self.mock_orchestrator.refine_mission_plan.return_value = AgentResponse(success=True, message="Plan refined with linting step.")
            self.mock_orchestrator.execute_plan.return_value = "Plan executed successfully."

            self.session._handle_goal_command("build an api")
            
            # Verify the flow
            self.assertEqual(self.mock_orchestrator.plan_mission.call_count, 1) # Initial plan
            self.assertEqual(self.mock_ui.present_plan_for_approval.call_count, 2) # First look, then after refinement
            self.mock_ui.prompt_for_input.assert_called_once_with("Please provide your feedback, question, or refinement")
            self.mock_orchestrator.refine_mission_plan.assert_called_once_with("add a linting step")
            self.mock_orchestrator.execute_plan.assert_called_once() # Finally executed
            logger.info("✅ test_plan_refinement_loop: PASSED")
            
        @patch('builtins.input', side_effect=['exit'])
        def test_exit_command(self, mock_input):
            """Tests that the main start() loop terminates when the user types 'exit'."""
            print("\n--- [Test Case 4: Exit Command] ---")
            self.mock_ui.prompt_for_input.return_value = "exit"
            
            self.session.start()
            
            self.mock_ui.prompt_for_input.assert_called_once_with("Your goal")
            self.mock_orchestrator.plan_mission.assert_not_called()
            self.mock_ui.display_system_message.assert_any_call("Session terminated. Goodbye!")
            logger.info("✅ test_exit_command: PASSED")

    unittest.main(argv=['first-arg-is-ignored'], exit=False)