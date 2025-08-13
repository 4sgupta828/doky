# orchestrator.py
from core.context import GlobalContext
from core.adaptive_engine import AdaptiveEngine
from core.models import TaskNode, AgentResponse
from agents.base import BaseAgent  # This would be part of a registry system
import logging

# Configure logging for debuggability
# from utils.logger import setup_logger
# setup_logger()


class Orchestrator:
    """
    The heart of the system, responsible for driving the entire mission from start to finish.
    It manages the main execution loop, invokes agents to perform tasks, and coordinates
    with the AdaptiveEngine when failures occur. It does not contain any agent-specific
    logic itself, acting purely as a conductor.
    """

    def __init__(self):
        """
        Initializes the Orchestrator and all core components of the agent collective.
        This setup phase is critical for establishing the mission's environment.
        """
        self.global_context = GlobalContext()
        self.adaptive_engine = AdaptiveEngine()
        # In a real implementation, agents would be loaded dynamically into a registry.
        # self.agent_registry = self._load_agents()
        logging.info("Orchestrator initialized with all core components.")

    def execute_mission(self, mission_goal: str) -> str:
        """
        Starts and manages the entire lifecycle of a mission.

        Args:
            mission_goal: The high-level objective from the user.

        Returns:
            A string summarizing the final outcome of the mission.
        """
        logging.info(f"--- NEW MISSION INITIATED --- Goal: {mission_goal}")
        
        # The first step is always to invoke the PlannerAgent to create the initial TaskGraph.
        self._invoke_agent(agent_name="PlannerAgent", goal=mission_goal)
        
        # Run the main execution loop until all tasks are completed or a critical failure occurs.
        self._run_main_loop()

        # Once the loop is finished, determine the final mission status.
        return "Mission finished successfully." # Or provide a failure summary.

    def _run_main_loop(self):
        """
        The core execution loop that processes the TaskGraph.
        It continuously finds and executes the next available task until none are left.
        This loop is designed for maximum debuggability, with clear states and logs.
        """
        while True:
            next_task = self._get_next_executable_task()
            
            if not next_task:
                logging.info("No more executable tasks. Main loop concluding.")
                break

            # Mark the task as running to prevent it from being picked up again.
            next_task.status = "running"
            logging.info(f"Executing Task '{next_task.task_id}': {next_task.goal}")
            
            # Invoke the responsible agent to execute the task.
            response = self._invoke_agent(
                agent_name=next_task.assigned_agent, 
                goal=next_task.goal
            )
            
            if response.success:
                next_task.status = "success"
                logging.info(f"Task '{next_task.task_id}' completed successfully.")
            else:
                next_task.status = "failed"
                logging.error(f"Task '{next_task.task_id}' failed. Reason: {response.message}")
                
                # A failure occurred, trigger the AdaptiveEngine to handle it.
                is_recovery_possible = self.adaptive_engine.handle_failure(
                    failed_task=next_task, 
                    context=self.global_context
                )
                
                if not is_recovery_possible:
                    logging.critical("Adaptive recovery failed. Aborting mission.")
                    break

    def _get_next_executable_task(self) -> TaskNode | None:
        """
        Scans the TaskGraph in the GlobalContext to find a task that is ready to be executed.
        A task is ready if its status is 'pending' and all of its dependencies have a
        status of 'success'.

        Returns:
            A TaskNode object that is ready for execution, or None if no tasks are ready.
        """
        pass

    def _invoke_agent(self, agent_name: str, goal: str) -> AgentResponse:
        """
        Finds the specified agent in the registry and calls its execute method.
        This acts as a controlled gateway for all agent actions.

        Args:
            agent_name: The name of the specialized agent to invoke.
            goal: The specific sub-problem or goal for the agent to solve.

        Returns:
            An AgentResponse object indicating the outcome of the agent's execution.
        """
        # agent_to_run = self.agent_registry.get(agent_name)
        # return agent_to_run.execute(goal, self.global_context)
        pass