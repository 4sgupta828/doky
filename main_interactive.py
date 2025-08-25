# main_interactive.py
import argparse
import logging
import sys
from typing import List

# Foundational dependencies from the project structure
from session import InteractiveSession
from utils.logger import setup_logger

# We need this for the testing block
from interfaces.collaboration_ui import Style

# Get a logger instance for this module
logger = logging.getLogger(__name__)

def main(args: List[str]) -> None:
    """
    The main entry point for initiating a mission with the Sovereign Agent Collective.

    This function is responsible for:
    1.  Setting up the application-wide logger.
    2.  Parsing command-line arguments to capture any session configurations.
    3.  Instantiating and starting the main InteractiveSession.
    4.  Handling any critical startup errors gracefully.

    Args:
        args: A list of command-line arguments, typically from `sys.argv[1:]`.
    """
    # 1. Parse command-line arguments first to get the quiet-logs flag.
    parser = argparse.ArgumentParser(
        description="Launch the Sovereign Agent Collective in an interactive CLI session."
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default=None,
        help="The directory path for the mission's workspace. If not specified, auto-generates a timestamped directory."
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
        help="Suppress console logging messages while preserving UI transparency/collaboration messages."
    )

    # We check for a special '--test' flag to run the built-in test suite.
    if "--test" in args:
        # This will be handled in the __main__ block.
        return

    parsed_args = parser.parse_args(args)
    workspace_path = parsed_args.workspace
    resume_arg = parsed_args.resume
    quiet_logs = parsed_args.quiet_logs
    
    # Resolve resume argument to actual snapshot file path
    resume_snapshot = None
    if resume_arg is not None:
        from utils.snapshot_utils import resolve_resume_argument, validate_snapshot_file, display_available_snapshots
        
        if resume_arg == "list":
            # Special case: just list available snapshots and exit
            print("📁 Available snapshots:")
            display_available_snapshots(workspace_path)
            return
        
        resume_snapshot = resolve_resume_argument(resume_arg, workspace_path)
        
        if resume_snapshot:
            if validate_snapshot_file(resume_snapshot):
                print(f"✅ Resuming from snapshot: {resume_snapshot}")
            else:
                print(f"❌ Invalid snapshot file: {resume_snapshot}")
                print("🔄 Falling back to new session")
                resume_snapshot = None
        else:
            print("⚠️  No valid snapshots found for resumption")
            if workspace_path:
                print(f"💡 Searched workspace: {workspace_path}")
            print("🔄 Starting new session instead")
            resume_snapshot = None

    # 2. Setup application-wide logging with optional console suppression.
    setup_logger(suppress_console_logs=quiet_logs)

    # 3. Instantiate and start the interactive session.
    try:
        session = InteractiveSession(workspace_path=workspace_path, resume_snapshot=resume_snapshot)
        session.start()

    except Exception as e:
        logger.critical("The interactive session failed to start or crashed.", exc_info=True)
        print(f"\n❌ {Style.Fg.RED}{Style.BOLD}FATAL ERROR:{Style.RESET} A critical error occurred. Please check 'logs/mission.log' for details.")
        print(f"   Error: {e}")
        sys.exit(1)

# --- Application Entry Point & Self-Testing Block ---
if __name__ == "__main__":
    # To run the real application from your terminal:
    # python main_interactive.py --workspace ./my_project
    #
    # Resume examples:
    # python main_interactive.py --resume                 # Auto-discover latest snapshot
    # python main_interactive.py --resume auto           # Same as above  
    # python main_interactive.py --resume latest         # Same as above
    # python main_interactive.py --resume list           # List available snapshots
    # python main_interactive.py --resume /path/to/snapshot.json  # Specific snapshot
    
    # To run the tests for this file:
    # python main_interactive.py --test
    
    if "--test" in sys.argv:
        import unittest
        from unittest.mock import patch

        class TestMainInteractive(unittest.TestCase):

            @patch('main_interactive.InteractiveSession')
            def test_main_starts_session_with_default_workspace(self, MockInteractiveSession):
                """
                Tests that main() correctly instantiates and starts the session
                with the default workspace path when no args are provided.
                """
                print("\n--- [Test Case 1: Start Session with Default Workspace] ---")
                
                mock_session_instance = MockInteractiveSession.return_value
                main([]) # Simulate running with no command-line arguments

                MockInteractiveSession.assert_called_once_with(workspace_path=None)
                mock_session_instance.start.assert_called_once()
                logger.info("✅ test_main_starts_session_with_default_workspace: PASSED")

            @patch('main_interactive.InteractiveSession')
            def test_main_starts_session_with_custom_workspace(self, MockInteractiveSession):
                """
                Tests that main() correctly parses the --workspace argument and
                passes it to the InteractiveSession.
                """
                print("\n--- [Test Case 2: Start Session with Custom Workspace] ---")
                mock_session_instance = MockInteractiveSession.return_value
                
                test_args = ["--workspace", "/tmp/custom_agent_ws"]
                main(test_args)

                MockInteractiveSession.assert_called_once_with(workspace_path="/tmp/custom_agent_ws")
                mock_session_instance.start.assert_called_once()
                logger.info("✅ test_main_starts_session_with_custom_workspace: PASSED")

            @patch('main_interactive.InteractiveSession')
            def test_main_handles_startup_exception(self, MockInteractiveSession):
                """
                Tests that main() catches exceptions during session startup and exits gracefully.
                """
                print("\n--- [Test Case 3: Handle Startup Exception] ---")
                
                mock_session_instance = MockInteractiveSession.return_value
                mock_session_instance.start.side_effect = RuntimeError("Failed to initialize core component")

                # We patch sys.exit to prevent the test runner from exiting the process.
                with patch('sys.exit') as mock_exit:
                    main([])
                    # Verify that the program tried to exit with a non-zero status code.
                    mock_exit.assert_called_once_with(1)

                logger.info("✅ test_main_handles_startup_exception: PASSED")
        
        # We must remove our custom '--test' flag before handing control over to unittest.
        sys.argv = [arg for arg in sys.argv if arg != '--test']
        unittest.main(argv=sys.argv[:1]) # Pass only the script name to unittest
    else:
        # If not testing, run the application normally with the provided command-line arguments.
        main(sys.argv[1:])