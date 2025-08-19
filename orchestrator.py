# orchestrator.py
import logging
import time
from typing import Dict, Optional, Any

# Foundational dependencies
from core.context import GlobalContext
from core.adaptive_engine import AdaptiveEngine
from core.models import TaskNode, AgentResponse, AgentResult
from agents import AGENT_REGISTRY, BaseAgent
from utils.logger import setup_logger
from interfaces.progress_tracker import ProgressTracker

# Configure the logger for this module
setup_logger()

class Orchestrator:
    """
    The heart of the system, responsible for driving the entire mission.
    It manages the main execution loop, invokes agents to perform tasks, and coordinates
    with the AdaptiveEngine when failures occur. It acts as the central Executor.
    """

    def __init__(self, workspace_path: Optional[str] = None, ui_interface: Any = None, global_context: Optional[GlobalContext] = None):
        """
        Initializes the Orchestrator and all core components of the agent collective.
        """
        self.ui_interface = ui_interface
        self.progress_tracker = ProgressTracker(ui_interface=ui_interface)
        
        # Use provided context or create new one
        if global_context:
            self.global_central_context = global_context
        else:
            self.global_context = GlobalContext(workspace_path=workspace_path)

        # Initialize real LLM client
        from real_llm_client import create_llm_client
        self.llm_client = create_llm_client()
        logging.info(f"Initialized LLM client: {self.llm_client.provider} with model {self.llm_client.model}")
        
        self.adaptive_engine = AdaptiveEngine()
        self.agent_registry: Dict[str, BaseAgent] = self._load_agents()
        logging.info(f"Orchestrator initialized with agents: {list(self.agent_registry.keys())}")

    def _load_agents(self) -> Dict[str, BaseAgent]:
        """
        Loads agent instances from the central registry, providing them with any
        dependencies they are designed to accept (e.g., llm_client, ui_interface).
        This is a flexible dependency injection system.
        """
        registry = {}
        # Prepare all available capabilities that can be injected.
        all_capabilities = [cls().get_capabilities() for name, cls in AGENT_REGISTRY.items() if name != "PlannerAgent"]
        
        available_dependencies = {
            "llm_client": self.llm_client,
            "ui_interface": self.ui_interface,
            "agent_registry": registry, # This will be passed to agents that need to call others
            "agent_capabilities": all_capabilities # Specifically for the Planner
        }

        for agent_name, agent_class in AGENT_REGISTRY.items():
            try:
                # Inspect the agent's __init__ method to see which dependencies it accepts.
                import inspect
                init_params = inspect.signature(agent_class.__init__).parameters
                
                # Build the kwargs for this specific agent.
                kwargs_for_agent = {}
                for param_name, dependency in available_dependencies.items():
                    if param_name in init_params:
                        kwargs_for_agent[param_name] = dependency
                
                # Instantiate the agent with the dependencies it can accept.
                agent_instance = agent_class(**kwargs_for_agent)
                
                # --- NEW: Ensure every agent has a progress tracker ---
                if hasattr(agent_instance, 'set_progress_tracker'):
                    agent_instance.set_progress_tracker(self.progress_tracker, None) # Set tracker, task_id will be updated on invocation
                
                registry[agent_name] = agent_instance
                        
            except Exception as e:
                logging.error(f"Failed to load agent '{agent_name}': {e}", exc_info=True)
        
        return registry

    def intelligent_mission_planning(self, mission_goal: str) -> AgentResponse:
        """
        The primary entry point for planning. It uses the unified PlannerAgent.
        """
        logging.info(f"Orchestrator received planning request for: '{mission_goal}'")
        self.global_context.task_graph.nodes.clear() # Clear any previous plan
        
        planning_task = TaskNode(goal=mission_goal, assigned_agent="PlannerAgent")
        return self._invoke_agent(planning_task)

    def refine_mission_plan(self, user_feedback: str) -> AgentResponse:
        """Invokes the PlanRefinementAgent to modify the current TaskGraph."""
        logging.info(f"Orchestrator received plan refinement request: '{user_feedback}'")
        
        if not self.global_context.task_graph.nodes:
            return AgentResponse(success=False, message="There is no active plan to refine.")

        refinement_task = TaskNode(
            goal=user_feedback,
            assigned_agent="PlanRefinementAgent"
        )
        return self._invoke_agent(refinement_task)

    def execute_plan(self) -> str:
        """Executes the TaskGraph currently loaded in the GlobalContext."""
        if not self.global_context.task_graph.nodes:
            return "Execution failed: No plan is loaded in the context."

        self._run_main_loop()

        if any(task.status == 'failed' for task in self.global_context.task_graph.nodes.values()):
            return "Plan concluded with unrecoverable failures."
        elif any(task.status == 'pending' for task in self.global_context.task_graph.nodes.values()):
             return "Plan concluded with pending tasks due to a deadlock."
        return "Plan executed successfully."

    def execute_single_task(self, goal: str, agent_name: str) -> AgentResponse:
        """
        Creates and executes a single, standalone task immediately.
        """
        if agent_name not in self.agent_registry:
            return AgentResponse(success=False, message=f"Agent '{agent_name}' not found.")

        single_task = TaskNode(goal=goal, assigned_agent=agent_name)
        response = self._invoke_agent(single_task)
        
        status = "succeeded" if response.success else "failed"
        self.global_context.log_event(
            f"direct_command_{status}",
            {"agent": agent_name, "goal": goal, "message": response.message}
        )
        return response

    def _run_main_loop(self):
        """The core execution loop that processes the TaskGraph."""
        max_loops_without_progress = 10
        loops_without_progress = 0

        while True:
            next_task = self._get_next_executable_task()

            if not next_task:
                if not any(task.status in ["pending", "running"] for task in self.global_context.task_graph.nodes.values()):
                    logging.info("No more executable or pending tasks. Main loop concluding.")
                    break
                
                loops_without_progress += 1
                if loops_without_progress >= max_loops_without_progress:
                    logging.critical("No executable tasks found for multiple cycles. Deadlock suspected. Aborting.")
                    break
                
                time.sleep(1)
                continue
            
            loops_without_progress = 0
            
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
                
                recovery_possible = self.adaptive_engine.handle_failure(
                    failed_task=next_task,
                    context=self.global_context,
                    planner=self.agent_registry["PlannerAgent"]
                )
                
                if not recovery_possible:
                    logging.critical(f"Adaptive recovery failed for task {next_task.task_id}. Aborting mission.")
                    break

    def _get_next_executable_task(self) -> Optional[TaskNode]:
        """Scans the TaskGraph to find a task that is ready to be executed."""
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
        Finds the specified agent and calls its modern execute_v2 method.
        """
        agent_name = current_task.assigned_agent
        agent_to_run = self.agent_registry.get(agent_name)

        if not agent_to_run:
            error_msg = f"Agent '{agent_name}' not found in registry."
            logging.error(error_msg)
            return AgentResponse(success=False, message=error_msg)

        logging.info(f"Invoking agent '{agent_name}' for task '{current_task.task_id}'...")
        
        print(f"\n🎯 Starting: {agent_name}")
        print(f"   Task: {current_task.goal[:80]}{'...' if len(current_task.goal) > 80 else ''}")
        
        # The progress tracker is now set during agent loading, but we update the task_id here.
        if hasattr(agent_to_run, 'set_progress_tracker'):
            agent_to_run.set_progress_tracker(self.progress_tracker, current_task.task_id)
        
        self.progress_tracker.start_agent_progress(agent_name, current_task.task_id, current_task.goal)
        
        try:
            # --- UPDATED: Exclusively use the v2 interface ---
            logger.debug(f"Using modern v2 interface for {agent_name}")
            
            # The 'goal' is the high-level objective. Specific inputs for the agent
            # are typically derived from artifacts in the context. For direct commands
            # and planning, the goal itself is the primary input.
            inputs = {"goal": current_task.goal} 
            
            result: AgentResult = agent_to_run.execute_v2(current_task.goal, inputs, self.global_context)
            
            # Convert v2 AgentResult back to v1 AgentResponse for the main loop and task result storage
            response = AgentResponse(
                success=result.success,
                message=result.message,
                artifacts_generated=result.outputs.get("artifacts_generated", [])
            )
            
            if response.success:
                print(f"✅ Completed: {agent_name}")
            else:
                print(f"❌ Failed: {agent_name}")
            
            self.progress_tracker.finish_agent_progress(current_task.task_id, success=response.success)
            
            return response
            
        except Exception as e:
            error_msg = f"Uncaught exception in agent '{agent_name}': {e}"
            logging.critical(error_msg, exc_info=True)
            self.progress_tracker.fail_step(current_task.task_id, error_msg)
            self.progress_tracker.finish_agent_progress(current_task.task_id, success=False)
            return AgentResponse(success=False, message=error_msg)
