# orchestrator.py
import logging
import time
from typing import Dict, Optional

# Foundational dependencies from Tier 1 & 2
from core.context import GlobalContext
from core.adaptive_engine import AdaptiveEngine
from core.models import TaskNode, AgentResponse, TaskGraph
from agents import get_agent, AGENT_REGISTRY, BaseAgent
from utils.logger import setup_logger

# Configure the logger for this module
setup_logger()

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
        self.agent_registry: Dict[str, BaseAgent] = self._load_agents()
        logging.info(f"Orchestrator initialized with agents: {list(self.agent_registry.keys())}")

    def _load_agents(self) -> Dict[str, BaseAgent]:
        """Loads agent instances from the central agent registry."""
        registry = {}
        for agent_name in AGENT_REGISTRY:
            try:
                # We need to pass the registry to the planner so it knows what tools are available.
                if agent_name == "PlannerAgent":
                    # Pass a dictionary of agent capabilities to the planner.
                    all_capabilities = {name: cls().get_capabilities() for name, cls in AGENT_REGISTRY.items()}
                    registry[agent_name] = get_agent(agent_name, agent_capabilities=all_capabilities)
                else:
                    registry[agent_name] = get_agent(agent_name)
            except Exception as e:
                logging.error(f"Failed to load agent '{agent_name}': {e}", exc_info=True)
        return registry

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
        initial_plan_task = TaskNode(
            task_id="task_0_plan",
            goal=mission_goal,
            assigned_agent="PlannerAgent",
        )
        self.global_context.task_graph.add_task(initial_plan_task)

        # Run the main execution loop.
        self._run_main_loop()

        # Determine the final mission status based on the state of the graph.
        final_status = "Mission finished successfully."
        if any(task.status == 'failed' for task in self.global_context.task_graph.nodes.values()):
            final_status = "Mission concluded with unrecoverable failures."
        elif any(task.status == 'pending' for task in self.global_context.task_graph.nodes.values()):
             final_status = "Mission concluded with pending tasks due to a deadlock or unmet dependencies."


        logging.info(f"--- MISSION CONCLUDED --- Outcome: {final_status}")
        self.global_context.save_snapshot("final_mission_state.json")
        return final_status

    def _run_main_loop(self):
        """
        The core execution loop that processes the TaskGraph. It continuously finds
        and executes the next available task until none are left or a deadlock occurs.
        """
        max_loops_without_progress = 10  # Prevents infinite loops
        loops_without_progress = 0

        while True:
            next_task = self._get_next_executable_task()

            if not next_task:
                # Check for mission completion
                if not any(task.status in ["pending", "running"] for task in self.global_context.task_graph.nodes.values()):
                    logging.info("No more executable or pending tasks. Main loop concluding.")
                    break
                
                # Check for deadlock
                loops_without_progress += 1
                if loops_without_progress >= max_loops_without_progress:
                    logging.critical("No executable tasks found for multiple cycles. Deadlock suspected. Aborting mission.")
                    break
                
                logging.warning("No executable tasks found, but pending tasks remain. Waiting...")
                time.sleep(1)
                continue
            
            loops_without_progress = 0 # Reset counter since we made progress
            
            # Execute the task
            next_task.status = "running"
            self.global_context.log_event("task_started", {"task_id": next_task.task_id, "goal": next_task.goal})

            response = self._invoke_agent(next_task)
            next_task.result = response

            if response.success:
                next_task.status = "success"
                self.global_context.log_event("task_succeeded", {"task_id": next_task.task_id})
            else:
                next_task.status = "failed"
                self.global_context.log_event("task_failed", {"task_id": next_task.task_id, "reason": response.message})
                
                # A failure occurred, trigger the AdaptiveEngine.
                recovery_possible = self.adaptive_engine.handle_failure(
                    failed_task=next_task,
                    context=self.global_context,
                    planner=self.agent_registry["PlannerAgent"]
                )
                
                if not recovery_possible:
                    logging.critical(f"Adaptive recovery failed for task {next_task.task_id}. Aborting mission.")
                    break

    def _get_next_executable_task(self) -> Optional[TaskNode]:
        """
        Scans the TaskGraph to find a task that is ready to be executed. A task
        is ready if its status is 'pending' and all its dependencies are 'success'.
        """
        for task in sorted(self.global_context.task_graph.nodes.values(), key=lambda t: t.task_id):
            if task.status == "pending":
                if all(
                    self.global_context.task_graph.get_task(dep_id)
                    and self.global_context.task_graph.get_task(dep_id).status == "success"
                    for dep_id in task.dependencies
                ):
                    return task
        return None

    def _invoke_agent(self, current_task: TaskNode) -> AgentResponse:
        """
        Finds the specified agent in the registry and calls its execute method.
        This provides a controlled, single point of execution for all agent actions.
        """
        agent_name = current_task.assigned_agent
        agent_to_run = self.agent_registry.get(agent_name)

        if not agent_to_run:
            error_msg = f"Agent '{agent_name}' not found in registry."
            logging.error(error_msg)
            return AgentResponse(success=False, message=error_msg)

        logging.info(f"Invoking agent '{agent_name}' for task '{current_task.task_id}'...")
        try:
            return agent_to_run.execute(current_task.goal, self.global_context, current_task)
        except Exception as e:
            error_msg = f"Uncaught exception in agent '{agent_name}': {e}"
            logging.critical(error_msg, exc_info=True)
            return AgentResponse(success=False, message=error_msg)

# --- Self-Testing Block ---
if __name__ == "__main__":
    from utils.logger import setup_logger
    import shutil
    from pathlib import Path

    setup_logger(default_level=logging.INFO)

    # --- Mock Agents for Isolated Testing ---
    class MockSuccessAgent(BaseAgent):
        def execute(self, goal: str, context: GlobalContext, current_task: TaskNode) -> AgentResponse:
            context.add_artifact(f"{current_task.task_id}_output", "success_data", current_task.task_id)
            return AgentResponse(success=True, message=f"{self.name} completed successfully.")

    class MockFailureAgent(BaseAgent):
        def execute(self, goal: str, context: GlobalContext, current_task: TaskNode) -> AgentResponse:
            return AgentResponse(success=False, message="I am designed to fail.")

    class MockPlanner(BaseAgent):
        def __init__(self, should_fail=False):
            super().__init__("PlannerAgent", "Mock Planner")
            self.should_fail = should_fail
        
        def execute(self, goal: str, context: GlobalContext, current_task: TaskNode) -> AgentResponse:
            if "fail" in goal.lower() or self.should_fail:
                return AgentResponse(success=False, message="Planner failed to create a plan.")
            
            # On first run, create a standard plan
            if "build a simple api" in goal.lower():
                graph = TaskGraph(nodes={
                    "task_code": TaskNode(task_id="task_code", goal="Write code", assigned_agent="CodeGenerationAgent", dependencies=[]),
                    "task_test": TaskNode(task_id="task_test", goal="Run tests", assigned_agent="TestRunnerAgent", dependencies=["task_code"])
                })
                context.task_graph.nodes.update(graph.nodes)
                return AgentResponse(success=True, message="Initial plan created.")

            # On recovery run, create a recovery plan
            if "failed" in goal.lower():
                graph = TaskGraph(nodes={
                    "task_debug": TaskNode(task_id="task_debug", goal="Debug the failure", assigned_agent="DebuggingAgent", dependencies=[]),
                    "task_fix": TaskNode(task_id="task_fix", goal="Fix the code", assigned_agent="CodeGenerationAgent", dependencies=["task_debug"])
                })
                # Splice the new plan in, replacing the old one
                context.task_graph.nodes = graph.nodes
                return AgentResponse(success=True, message="Recovery plan created.")
            
            return AgentResponse(success=False, message="Mock planner did not understand goal.")

    print("\n--- Testing Orchestrator ---")
    TEST_WORKSPACE = "./temp_orchestrator_test_ws"

    def setup_test_env():
        if Path(TEST_WORKSPACE).exists():
            shutil.rmtree(TEST_WORKSPACE)
        
        # Override the real registry with mock agents for this test
        global AGENT_REGISTRY
        AGENT_REGISTRY = {
            "PlannerAgent": MockPlanner,
            "CodeGenerationAgent": MockSuccessAgent,
            "TestRunnerAgent": MockSuccessAgent,
            "DebuggingAgent": MockSuccessAgent,
        }

    # 1. Test Golden Path (Successful Mission)
    print("\n[1] Testing successful mission execution (golden path)...")
    setup_test_env()
    orchestrator = Orchestrator()
    orchestrator.global_context.workspace.repo_path = Path(TEST_WORKSPACE) # Point to test dir
    result = orchestrator.execute_mission("Build a simple API")
    assert "successfully" in result
    assert orchestrator.global_context.task_graph.get_task("task_code").status == "success"
    assert orchestrator.global_context.task_graph.get_task("task_test").status == "success"
    logger.info("Golden path test passed.")

    # 2. Test Failure and Recovery Path
    print("\n[2] Testing mission with failure and successful recovery...")
    setup_test_env()
    # Make the coder fail this time
    AGENT_REGISTRY["CodeGenerationAgent"] = MockFailureAgent
    orchestrator = Orchestrator()
    orchestrator.global_context.workspace.repo_path = Path(TEST_WORKSPACE)
    result = orchestrator.execute_mission("Build a simple API")
    assert "successfully" in result # Mission still succeeds because it recovered
    
    # Check the state of the graph
    original_failed_task = orchestrator.global_context.task_graph.get_task("task_code")
    original_dependent_task = orchestrator.global_context.task_graph.get_task("task_test")
    
    # These tasks don't exist in the final graph because the planner replaced them
    assert original_failed_task is None
    assert original_dependent_task is None

    # The new recovery tasks should exist and be successful
    assert orchestrator.global_context.task_graph.get_task("task_debug").status == "success"
    assert orchestrator.global_context.task_graph.get_task("task_fix").status == "success"
    logger.info("Failure and recovery path test passed.")

    # 3. Test Unrecoverable Failure (Planner fails to recover)
    print("\n[3] Testing mission with unrecoverable failure...")
    setup_test_env()
    AGENT_REGISTRY["CodeGenerationAgent"] = MockFailureAgent
    # This special mock planner will also fail on the recovery attempt
    AGENT_REGISTRY["PlannerAgent"] = lambda: MockPlanner(should_fail=True)
    orchestrator = Orchestrator()
    orchestrator.global_context.workspace.repo_path = Path(TEST_WORKSPACE)
    result = orchestrator.execute_mission("Build a simple API")
    assert "unrecoverable" in result
    logger.info("Unrecoverable failure test passed.")

    # Cleanup
    shutil.rmtree(TEST_WORKSPACE)
    print("\n--- All Orchestrator Tests Passed Successfully ---")