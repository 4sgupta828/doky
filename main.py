# main.py
from orchestrator import Orchestrator

def main():
    """
    The main entry point for initiating a mission with the Sovereign Agent Collective.
    This function sets the high-level goal and kicks off the orchestration process.
    """
    # The high-level goal provided by the user.
    mission_goal = "Build a simple REST API for user registration and login."

    # Instantiate the main orchestrator.
    orchestrator = Orchestrator()

    # Start the mission. The orchestrator will handle the entire lifecycle.
    final_outcome = orchestrator.execute_mission(mission_goal)

    # Print the final result of the mission.
    print(f"\n--- MISSION COMPLETE ---")
    print(f"Final Outcome: {final_outcome}")


if __name__ == "__main__":
    main()