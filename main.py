# main.py
import argparse
import logging
import sys
from typing import List

# Foundational dependencies from the project structure
from orchestrator import Orchestrator
from utils.logger import setup_logger

# Get a logger instance for this module
logger = logging.getLogger(__name__)

def main(args: List[str]) -> None:
    """
    The main entry point for initiating a mission with the Sovereign Agent Collective.

    This function is responsible for:
    1.  Setting up the application-wide logger.
    2.  Parsing command-line arguments to capture the user's high-level goal.
    3.  Instantiating and running the main Orchestrator.
    4.  Printing the final outcome of the mission.

    Args:
        args: A list of command-line arguments, typically from `sys.argv[1:]`.
    """
    # 1. Setup application-wide logging.
    setup_logger()

    # 2. Parse command-line arguments.
    parser = argparse.ArgumentParser(
        description="Sovereign Agent Collective: An autonomous software development agency."
    )
    parser.add_argument(
        "mission_goal",
        type=str,
        help="The high-level goal for the agent collective to accomplish (e.g., 'Build a simple Flask API for user auth')."
    )
    parser.add_argument(
        "--workspace",
        type=str,
        default="./mission_workspace",
        help="The directory path for the mission's workspace."
    )

    # In a real application, you might add a verbosity flag
    # parser.add_argument('-v', '--verbose', action='store_true', help='Enable debug logging')

    parsed_args = parser.parse_args(args)
    mission_goal = parsed_args.mission_goal
    workspace_path = parsed_args.workspace

    logger.info(f"Starting Sovereign Agent Collective with mission: '{mission_goal}'")

    try:
        # 3. Instantiate the main orchestrator, passing the workspace path.
        # The orchestrator's __init__ handles the setup of all core components.
        orchestrator = Orchestrator(workspace_path=workspace_path)

        # 4. Start the mission. This is the primary blocking call.
        final_outcome = orchestrator.execute_mission(mission_goal)

        # 5. Print the final result of the mission.
        print("\n" + "="*60)
        print(" MISSION CONCLUDED")
        print("="*60)
        print(f"Final Outcome: {final_outcome}")
        print(f"Review the complete log in 'logs/mission.log'")
        print(f"Final state snapshot saved to '{workspace_path}/final_mission_state.json'")
        print("="*60)

    except Exception as e:
        logger.critical("A critical unhandled exception occurred in the main execution flow.", exc_info=True)
        print(f"\nFATAL ERROR: A critical error occurred. Please check the logs for details. Error: {e}")
        sys.exit(1)


# --- Self-Testing Block ---
# This block allows for testing the main entry point's argument parsing and
# its interaction with the Orchestrator without running a full mission.
if __name__ == "__main__":
    # To run the real application from your terminal:
    # python main.py "Your mission goal here" --workspace ./my_project
    
    # To run the tests for this file:
    # python main.py --test
    
    if "--test" in sys.argv:
        import unittest
        from unittest.mock import patch

        class TestMain(unittest.TestCase):

            @patch('main.Orchestrator')
            def test_main_flow_successful_mission(self, MockOrchestrator):
                """
                Tests the main function's happy path, ensuring it correctly
                instantiates and calls the orchestrator with default arguments.
                """
                print("\n--- [Test Case 1: Successful Mission Flow] ---")
                
                # Configure the mock orchestrator instance
                mock_instance = MockOrchestrator.return_value
                mock_instance.execute_mission.return_value = "Mission completed successfully by mock."

                # Define the command-line arguments to simulate
                test_args = ["Build a simple Flask API"]
                
                main(test_args)

                # Assert that the Orchestrator was initialized with the default workspace
                MockOrchestrator.assert_called_once_with(workspace_path="./mission_workspace")
                
                # Assert that execute_mission was called with the correct goal
                mock_instance.execute_mission.assert_called_once_with("Build a simple Flask API")
                logger.info("✅ test_main_flow_successful_mission: PASSED")

            @patch('main.Orchestrator')
            def test_main_flow_with_custom_workspace(self, MockOrchestrator):
                """
                Tests that custom command-line arguments like `--workspace` are parsed correctly.
                """
                print("\n--- [Test Case 2: Custom Workspace Argument] ---")
                mock_instance = MockOrchestrator.return_value
                mock_instance.execute_mission.return_value = "Mission completed."

                test_args = ["A different goal", "--workspace", "/tmp/custom_ws"]
                main(test_args)

                # Check initialization with the custom path
                MockOrchestrator.assert_called_once_with(workspace_path="/tmp/custom_ws")
                mock_instance.execute_mission.assert_called_once_with("A different goal")
                logger.info("✅ test_main_flow_with_custom_workspace: PASSED")
        
        # Remove the '--test' argument to prevent argparse from getting confused
        # and to allow unittest to run properly.
        sys.argv = [arg for arg in sys.argv if arg != '--test']
        unittest.main(argv=sys.argv[:1]) # Pass only the script name to unittest
    else:
        # If not testing, run the application normally with the provided command-line arguments.
        main(sys.argv[1:])