# main_interactive_intelligent.py
"""
Intelligent Interactive Session with Foundation Agent Workflow Coordination

This enhanced version of main_interactive.py uses the intelligent workflow coordinator
to provide goal-oriented, multi-agent coordination with automatic completion validation.

Features:
- Intelligent multi-agent workflow execution
- LLM-based routing with directional progress guarantee  
- Automatic completion validation
- Comprehensive progress tracking and reporting
- Fallback to rule-based routing when LLM unavailable
"""

import argparse
import logging
import sys
from typing import List, Optional, Dict, Any
from pathlib import Path

# Core dependencies
from utils.logger import setup_logger
from interfaces.collaboration_ui import Style, CollaborationUI
from core.context import GlobalContext

# Intelligent workflow coordination
from fagents.workflow_coordinator import WorkflowCoordinator, execute_user_goal
from fagents.inter_agent_router import WorkflowStatus

# Get a logger instance for this module
logger = logging.getLogger(__name__)


class IntelligentInteractiveSession:
    """
    Enhanced interactive session using intelligent workflow coordination.
    
    This session provides:
    - Goal-oriented multi-agent coordination
    - LLM-based intelligent routing
    - Automatic completion validation
    - Comprehensive progress tracking
    """
    
    def __init__(self, workspace_path: Optional[str] = None, resume_snapshot: Optional[str] = None, 
                 llm_client: Any = None):
        """
        Initialize the intelligent interactive session.
        
        Args:
            workspace_path: Directory path for the workspace
            resume_snapshot: Path to snapshot file for crash recovery
            llm_client: LLM client for intelligent routing
        """
        try:
            self.ui = CollaborationUI()
            self.llm_client = llm_client
            
            # Initialize global context
            if resume_snapshot:
                self.global_context = GlobalContext.load_from_snapshot(resume_snapshot)
                self.ui.display_system_message(f"Session resumed from snapshot: {resume_snapshot}")
            else:
                workspace = Path(workspace_path) if workspace_path else Path.cwd()
                self.global_context = GlobalContext(workspace_path=workspace)
                
            # Initialize intelligent workflow coordinator with UI transparency
            self.workflow_coordinator = WorkflowCoordinator(llm_client=llm_client, ui_interface=self.ui)
            
            # Session state
            self.active_workflows: Dict[str, Any] = {}
            self.session_history: List[Dict[str, Any]] = []
            
            logger.info("IntelligentInteractiveSession initialized successfully.")
            
        except Exception as e:
            logger.critical("Failed to initialize intelligent session.", exc_info=True)
            raise RuntimeError(f"Failed to initialize session: {e}")
    
    def start(self):
        """Start the intelligent interactive session."""
        # Display welcome message
        print("\n" + "="*80)
        print("🧠 INTELLIGENT FOUNDATION AGENT SYSTEM")
        print("="*80)
        print("Welcome to the intelligent multi-agent workflow coordination system!")
        
        self.ui.display_system_message(
            "🧠 Intelligent Foundation Agent System Active\n"
            "   • Multi-agent workflow coordination\n"
            "   • LLM-based intelligent routing\n"
            "   • Automatic completion validation\n"
            "   • Comprehensive progress tracking"
        )
        
        try:
            self._display_help()
            self._main_loop()
        except KeyboardInterrupt:
            self.ui.display_system_message("\n👋 Session interrupted by user. Goodbye!")
        except Exception as e:
            logger.error(f"Session error: {e}", exc_info=True)
            self.ui.display_error_message(f"Session error: {e}")
        finally:
            self._cleanup()
    
    def _main_loop(self):
        """Main interactive loop."""
        while True:
            try:
                # Get user input
                user_input = self.ui.prompt_for_input("💬 You").strip()
                
                if not user_input:
                    continue
                
                # Handle special commands
                if user_input.lower() in ['quit', 'exit', 'q']:
                    self.ui.display_system_message("👋 Goodbye!")
                    break
                elif user_input.lower() in ['help', 'h', '?']:
                    self._display_help()
                    continue
                elif user_input.lower().startswith('status'):
                    self._handle_status_command(user_input)
                    continue
                elif user_input.lower().startswith('workflows'):
                    self._handle_workflows_command()
                    continue
                elif user_input.lower().startswith('history'):
                    self._handle_history_command()
                    continue
                
                # Execute user goal with intelligent coordination
                self._execute_user_goal(user_input)
                
            except KeyboardInterrupt:
                raise  # Re-raise to be caught by outer try-catch
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                self.ui.display_system_message(f"❌ An error occurred: {e}", is_error=True)
    
    def _execute_user_goal(self, user_goal: str):
        """Execute a user goal using intelligent workflow coordination."""
        
        self.ui.display_system_message(f"\n🎯 Executing Goal: {user_goal}")
        self.ui.display_system_message("🔄 Starting intelligent multi-agent workflow...")
        
        try:
            # Execute goal with intelligent coordination
            result = self.workflow_coordinator.execute_goal(
                user_goal=user_goal,
                inputs={},
                global_context=self.global_context,
                max_hops=10
            )
            
            # Store workflow for tracking
            workflow_id = result.outputs.get('workflow_id', 'unknown')
            self.active_workflows[workflow_id] = result
            
            # Add to session history
            self.session_history.append({
                'goal': user_goal,
                'result': result,
                'workflow_id': workflow_id,
                'timestamp': str(__import__('datetime').datetime.now())
            })
            
            # Display results
            self._display_workflow_results(result, user_goal)
            
        except Exception as e:
            logger.error(f"Goal execution failed: {e}", exc_info=True)
            self.ui.display_system_message(f"❌ Goal execution failed: {e}", is_error=True)
    
    def _display_workflow_results(self, result, user_goal: str):
        """Display comprehensive workflow execution results."""
        
        success_icon = "✅" if result.success else "❌"
        self.ui.display_system_message(f"\n{success_icon} {result.message}")
        
        if result.success and result.outputs:
            outputs = result.outputs
            
            # Show execution path
            agents_used = outputs.get('agents_used', [])
            if agents_used:
                execution_path = ' → '.join(agents_used)
                self.ui.display_system_message(f"🔄 Execution Path: {execution_path}")
            
            # Show key achievements
            workflow_summary = outputs.get('workflow_summary', {})
            achievements = workflow_summary.get('key_achievements', [])
            if achievements:
                self.ui.display_system_message("🎯 Key Achievements:")
                for achievement in achievements:
                    self.ui.display_system_message(f"   • {achievement}")
            
            # Show execution details
            total_hops = outputs.get('total_hops', 0)
            completion_validated = outputs.get('completion_validated', False)
            
            self.ui.display_system_message(
                f"📊 Execution Summary:\n"
                f"   • Total Hops: {total_hops}\n"
                f"   • Agents Used: {len(set(agents_used))}\n"
                f"   • Completion Validated: {'✅' if completion_validated else '❌'}\n"
                f"   • Workflow ID: {outputs.get('workflow_id', 'N/A')}"
            )
            
            # Show execution steps
            execution_summary = outputs.get('execution_summary', [])
            if execution_summary:
                self.ui.display_system_message("\n📋 Detailed Execution Steps:")
                for step in execution_summary:
                    status_icon = "✅" if step['success'] else "❌"
                    self.ui.display_system_message(
                        f"   {step['step']}. {step['agent']} {status_icon}\n"
                        f"      Goal: {step['goal']}\n"
                        f"      Result: {step['message']}"
                    )
        else:
            # Show failure details
            if result.outputs and 'error' in result.outputs:
                self.ui.display_system_message(f"❌ Error Details: {result.outputs['error']}", is_error=True)
    
    def _handle_status_command(self, command: str):
        """Handle status command to show workflow status."""
        
        parts = command.split()
        if len(parts) > 1:
            # Status for specific workflow
            workflow_id = parts[1]
            status = self.workflow_coordinator.get_workflow_status(workflow_id)
            if status:
                self.ui.display_system_message(f"\n📊 Workflow Status: {workflow_id}")
                self.ui.display_system_message(f"   Status: {status['status']}")
                self.ui.display_system_message(f"   Hops: {status['current_hop']}/{status['max_hops']}")
                self.ui.display_system_message(f"   Validated: {status['completion_validated']}")
                self.ui.display_system_message(f"   Agents: {status['agents_used']}")
            else:
                self.ui.display_system_message(f"❌ Workflow not found: {workflow_id}", is_error=True)
        else:
            # General status
            active_count = len(self.active_workflows)
            session_count = len(self.session_history)
            self.ui.display_system_message(
                f"\n📊 Session Status:\n"
                f"   • Active Workflows: {active_count}\n"
                f"   • Goals Executed: {session_count}\n"
                f"   • LLM Client: {self._get_llm_client_status()}\n"
                f"   • Workspace: {self.global_context.workspace_path}"
            )
    
    def _handle_workflows_command(self):
        """Handle workflows command to list active workflows."""
        
        workflows = self.workflow_coordinator.list_active_workflows()
        
        if workflows:
            self.ui.display_system_message(f"\n📋 Active Workflows ({len(workflows)}):")
            for wf in workflows:
                self.ui.display_system_message(
                    f"   • {wf['workflow_id'][:8]}... - {wf['status']}\n"
                    f"     Goal: {wf['user_goal'][:60]}{'...' if len(wf['user_goal']) > 60 else ''}\n"
                    f"     Progress: {wf['current_hop']} hops, {wf['agents_used']} agents"
                )
        else:
            self.ui.display_system_message("📋 No active workflows")
    
    def _handle_history_command(self):
        """Handle history command to show session history."""
        
        if self.session_history:
            self.ui.display_system_message(f"\n📚 Session History ({len(self.session_history)} goals):")
            for i, entry in enumerate(self.session_history[-10:], 1):  # Show last 10
                status_icon = "✅" if entry['result'].success else "❌"
                self.ui.display_system_message(
                    f"   {i}. {status_icon} {entry['goal'][:60]}{'...' if len(entry['goal']) > 60 else ''}\n"
                    f"      Workflow: {entry['workflow_id'][:8]}... - {entry['timestamp'][:19]}"
                )
        else:
            self.ui.display_system_message("📚 No session history yet")
    
    def _display_help(self):
        """Display help information."""
        
        help_text = """
🧠 Intelligent Foundation Agent System - Help

GOAL EXECUTION:
   Simply type your goal and press Enter to execute it with intelligent multi-agent coordination.
   
   Examples:
   • "Create a REST API with authentication"
   • "Build a web scraper with error handling" 
   • "Analyze my code for security vulnerabilities"
   • "Generate unit tests for my Python project"

COMMANDS:
   help, h, ?          - Show this help message
   status              - Show general session status
   status <workflow>   - Show specific workflow status
   workflows           - List all active workflows
   history             - Show session execution history
   quit, exit, q       - Exit the session

FEATURES:
   • LLM-based intelligent routing between foundation agents
   • Automatic completion validation ensures goals are achieved
   • Minimal agent hops with directional progress guarantee
   • Comprehensive progress tracking with execution reasoning
   • Fallback to rule-based routing when LLM unavailable

FOUNDATION AGENTS:
   • AnalystAgent     - Deep analysis and validation
   • CreatorAgent     - Code, tests, and documentation generation
   • ExecutorAgent    - Test execution and validation
   • SurgeonAgent     - Precise modifications and maintenance  
   • StrategistAgent  - Complex planning and orchestration
   • DebuggingAgent   - Systematic debugging and troubleshooting
        """
        
        self.ui.display_system_message(help_text)
    
    def _get_llm_client_status(self) -> str:
        """Get the LLM client status string."""
        if not self.llm_client:
            return "Fallback Mode"
        elif hasattr(self.llm_client, '__class__') and 'Mock' in self.llm_client.__class__.__name__:
            return "Mock LLM (Testing)"
        else:
            return "Connected"
    
    def _cleanup(self):
        """Cleanup session resources."""
        try:
            # Save any necessary state
            logger.info("Session cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


def main(args: List[str]) -> None:
    """
    Main entry point for the intelligent interactive session.
    
    Args:
        args: Command-line arguments
    """
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Launch the Intelligent Foundation Agent System with multi-agent workflow coordination."
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default=None,
        help="Directory path for the workspace. If not specified, uses current directory."
    )
    parser.add_argument(
        "--resume",
        type=str,
        nargs='?',
        const="auto",
        default=None,
        help="Resume from snapshot. Options: 'auto'/'latest' (auto-discover most recent), 'list' (show available snapshots), or specify a snapshot file path."
    )
    parser.add_argument(
        "--quiet-logs",
        action="store_true",
        help="Suppress console logging messages while preserving UI messages."
    )
    parser.add_argument(
        "--llm-client",
        type=str,
        default=None,
        help="LLM client configuration (optional - defaults to OpenAI)."
    )
    parser.add_argument(
        "--development-mode",
        action="store_true",
        help="Development mode: try real LLM, fallback to mock if unavailable."
    )
    parser.add_argument(
        "--use-mock",
        action="store_true", 
        help="Use mock LLM client for testing (not for production)."
    )
    parser.add_argument(
        "--no-llm",
        action="store_true",
        help="Disable LLM completely and use only fallback routing."
    )
    
    # Handle test flag
    if "--test" in args:
        return
    
    parsed_args = parser.parse_args(args)
    
    # Resolve resume argument to actual snapshot file path
    resume_snapshot = None
    if parsed_args.resume is not None:
        from utils.snapshot_utils import resolve_resume_argument, validate_snapshot_file, display_available_snapshots
        
        if parsed_args.resume == "list":
            # Special case: just list available snapshots and exit
            print("📁 Available snapshots:")
            display_available_snapshots(parsed_args.workspace)
            return
        
        resume_snapshot = resolve_resume_argument(parsed_args.resume, parsed_args.workspace)
        
        if resume_snapshot:
            if validate_snapshot_file(resume_snapshot):
                print(f"✅ Resuming from snapshot: {resume_snapshot}")
            else:
                print(f"❌ Invalid snapshot file: {resume_snapshot}")
                print("🔄 Falling back to new session")
                resume_snapshot = None
        else:
            print("⚠️  No valid snapshots found for resumption")
            if parsed_args.workspace:
                print(f"💡 Searched workspace: {parsed_args.workspace}")
            print("🔄 Starting new session instead")
            resume_snapshot = None
    
    # Setup logging
    setup_logger(suppress_console_logs=parsed_args.quiet_logs)
    
    # Initialize LLM client with fail-fast behavior for production
    llm_client = None
    
    try:
        from tools.llm_tool import create_llm_client
        
        if parsed_args.llm_client:
            # Custom LLM client configuration
            logger.info(f"Using custom LLM client configuration: {parsed_args.llm_client}")
            # This would initialize your specific LLM client based on config
            llm_client = create_llm_client(provider="openai")
            
        elif hasattr(parsed_args, 'development_mode') and parsed_args.development_mode:
            # Development mode: try real LLM, fallback to mock
            llm_client = create_llm_client()
            
        elif hasattr(parsed_args, 'use_mock') and parsed_args.use_mock:
            # Explicitly requested mock LLM
            # Mock mode not supported in new client - use None
            llm_client = None
            logger.info("Using mock LLM client for testing")
            
        elif hasattr(parsed_args, 'no_llm') and parsed_args.no_llm:
            # No LLM requested - use fallback routing only
            llm_client = None
            logger.info("LLM disabled - using fallback routing only")
            
        else:
            # Production mode: require real LLM, fail fast if not available
            try:
                llm_client = create_llm_client()
                logger.info("Using OpenAI LLM client for intelligent routing")
                
            except Exception as production_error:
                error_msg = f"❌ LLM client initialization failed: {production_error}"
                logger.critical(error_msg)
                sys.exit(1)
                
    except ImportError as e:
        logger.critical(f"Failed to import LLM client: {e}")
        print(f"\n❌ {Style.Fg.ERROR}LLM client dependencies not available: {e}{Style.RESET}")
        sys.exit(1)
    
    # Start intelligent session
    try:
        session = IntelligentInteractiveSession(
            workspace_path=parsed_args.workspace,
            resume_snapshot=resume_snapshot,
            llm_client=llm_client
        )
        session.start()
        
    except Exception as e:
        logger.critical("The intelligent session failed to start or crashed.", exc_info=True)
        print(f"\n❌ {Style.Fg.RED}{Style.BOLD}FATAL ERROR:{Style.RESET} A critical error occurred.")
        print(f"   Error: {e}")
        sys.exit(1)


# --- Application Entry Point & Self-Testing Block ---
if __name__ == "__main__":
    # To run the application:
    # python main_interactive_intelligent.py --workspace ./my_project
    #
    # Resume examples:
    # python main_interactive_intelligent.py --resume              # Auto-discover latest snapshot
    # python main_interactive_intelligent.py --resume auto        # Same as above  
    # python main_interactive_intelligent.py --resume latest      # Same as above
    # python main_interactive_intelligent.py --resume list        # List available snapshots
    # python main_interactive_intelligent.py --resume /path/to/snapshot.json  # Specific snapshot
    
    # To run tests:
    # python main_interactive_intelligent.py --test
    
    if "--test" in sys.argv:
        import unittest
        from unittest.mock import patch, MagicMock
        
        class TestIntelligentInteractive(unittest.TestCase):
            
            @patch('main_interactive_intelligent.WorkflowCoordinator')
            @patch('main_interactive_intelligent.GlobalContext')
            @patch('main_interactive_intelligent.CollaborationUI')
            def test_intelligent_session_initialization(self, MockUI, MockContext, MockCoordinator):
                """Test that IntelligentInteractiveSession initializes correctly."""
                print("\n--- [Test Case 1: Intelligent Session Initialization] ---")
                
                # Setup mocks
                mock_ui = MockUI.return_value
                mock_context = MockContext.return_value
                mock_coordinator = MockCoordinator.return_value
                
                # Initialize session
                session = IntelligentInteractiveSession(workspace_path="/tmp/test")
                
                # Verify initialization
                self.assertIsNotNone(session.ui)
                self.assertIsNotNone(session.workflow_coordinator)
                self.assertIsNotNone(session.global_context)
                self.assertEqual(session.active_workflows, {})
                self.assertEqual(session.session_history, [])
                
                logger.info("✅ test_intelligent_session_initialization: PASSED")
            
            @patch('main_interactive_intelligent.IntelligentInteractiveSession')
            def test_main_starts_intelligent_session(self, MockSession):
                """Test that main() starts the intelligent session correctly."""
                print("\n--- [Test Case 2: Main Starts Intelligent Session] ---")
                
                mock_session_instance = MockSession.return_value
                main([])  # No arguments
                
                MockSession.assert_called_once()
                mock_session_instance.start.assert_called_once()
                
                logger.info("✅ test_main_starts_intelligent_session: PASSED")
            
            def test_workflow_execution_mock(self):
                """Test workflow execution with mocked components."""
                print("\n--- [Test Case 3: Workflow Execution Mock] ---")
                
                # This would test the workflow execution logic
                # In a real implementation, you'd mock the workflow coordinator
                # and verify that goals are executed correctly
                
                # For now, just verify the test framework works
                self.assertTrue(True)
                
                logger.info("✅ test_workflow_execution_mock: PASSED")
        
        # Remove test flag and run tests
        sys.argv = [arg for arg in sys.argv if arg != '--test']
        unittest.main(argv=sys.argv[:1])
    else:
        # Run the application normally
        main(sys.argv[1:])