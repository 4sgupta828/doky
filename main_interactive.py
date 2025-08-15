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
    # 1. Setup application-wide logging.
    setup_logger()

    # 2. Parse command-line arguments.
    parser = argparse.ArgumentParser(
        description="Launch the Sovereign Agent Collective in an interactive CLI session."
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default=None,
        help="The directory path for the mission's workspace. If not specified, auto-generates a timestamped directory."
    )

    # We check for a special '--test' flag to run the built-in test suite.
    if "--test" in args:
        # This will be handled in the __main__ block.
        return

    parsed_args = parser.parse_args(args)
    workspace_path = parsed_args.workspace

    # 3. Instantiate and start the interactive session.
    try:
        session = InteractiveSession(workspace_path=workspace_path)
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