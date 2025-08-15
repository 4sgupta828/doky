# orchestrator.py
import logging
import time
from typing import Dict, Optional, Any

# Foundational dependencies
from core.context import GlobalContext
from core.adaptive_engine import AdaptiveEngine
from core.models import TaskNode, AgentResponse
from agents import AGENT_REGISTRY, BaseAgent
from utils.logger import setup_logger
from interfaces.progress_tracker import ProgressTracker

# Configure the logger for this module
setup_logger()

class Orchestrator:
    """
    The heart of the system, responsible for driving the entire mission from start to finish.
    It manages the main execution loop, invokes agents to perform tasks, and coordinates
    with the AdaptiveEngine when failures occur. It acts purely as a conductor.
    """

    def __init__(self, workspace_path: Optional[str] = None, ui_interface: Any = None, global_context: Optional[GlobalContext] = None):
        """
        Initializes the Orchestrator and all core components of the agent collective.
        This setup phase is critical for establishing the mission's environment.
        
        Args:
            workspace_path: The directory path for the mission's workspace. If None,
                          auto-generates a timestamped directory in /Users/sgupta/
            ui_interface: The UI interface for user interaction (optional).
            global_context: Existing GlobalContext to use (for crash recovery), overrides workspace_path
        """
        self.ui_interface = ui_interface
        self.progress_tracker = ProgressTracker(ui_interface=ui_interface)
        
        # Use provided context or create new one
        if global_context:
            self.global_context = global_context
        
        # Load environment variables from .env file
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            # Manual loading if python-dotenv not available
            import os
            from pathlib import Path
            env_file = Path(".env")
            if env_file.exists():
                for line in env_file.read_text().strip().split('\n'):
                    if '=' in line and not line.startswith('#'):
                        key, value = line.split('=', 1)
                        os.environ[key.strip()] = value.strip()
        
        # Initialize real LLM client
        from real_llm_client import create_llm_client
        self.llm_client = create_llm_client()
        logging.info(f"Initialized LLM client: {self.llm_client.provider} with model {self.llm_client.model}")
        
        # Use provided context or create new one
        if not hasattr(self, 'global_context') or self.global_context is None:
            self.global_context = GlobalContext(workspace_path=workspace_path)
        self.adaptive_engine = AdaptiveEngine()
        self.agent_registry: Dict[str, BaseAgent] = self._load_agents()
        logging.info(f"Orchestrator initialized with agents: {list(self.agent_registry.keys())}")

    def _load_agents(self) -> Dict[str, BaseAgent]:
        """Loads agent instances from the central agent registry with real LLM client."""
        registry = {}
        # The planner needs to know about all other agents to create valid plans.
        all_capabilities = []
        for name, cls in AGENT_REGISTRY.items():
            if name != "PlannerAgent":
                temp_agent = cls()
                all_capabilities.append(temp_agent.get_capabilities())
        
        for agent_name, agent_class in AGENT_REGISTRY.items():
            try:
                if agent_name == "PlannerAgent":
                    registry[agent_name] = agent_class(
                        agent_capabilities=all_capabilities, 
                        llm_client=self.llm_client
                    )
                elif agent_name == "IntentValidationAgent":
                    # IntentValidationAgent needs both LLM client and UI interface
                    registry[agent_name] = agent_class(
                        llm_client=self.llm_client,
                        ui_interface=self.ui_interface
                    )
                elif agent_name == "SpecValidationAgent":
                    # SpecValidationAgent needs both LLM client and UI interface
                    registry[agent_name] = agent_class(
                        llm_client=self.llm_client,
                        ui_interface=self.ui_interface
                    )
                else:
                    # Try to inject LLM client into other agents
                    try:
                        registry[agent_name] = agent_class(llm_client=self.llm_client)
                    except TypeError:
                        # Some agents might not accept llm_client parameter
                        registry[agent_name] = agent_class()
                        
            except Exception as e:
                logging.error(f"Failed to load agent '{agent_name}': {e}", exc_info=True)
        
        # Second pass: Inject agent registry into agents that need it
        self._inject_agent_registry(registry)
        
        return registry
        
    def _inject_agent_registry(self, registry: Dict[str, BaseAgent]):
        """Second pass to inject agent registry into agents that need cross-agent capabilities."""
        # TestRunnerAgent needs access to DebuggingAgent
        if "TestRunnerAgent" in registry:
            test_runner = registry["TestRunnerAgent"]
            if hasattr(test_runner, 'agent_registry'):
                test_runner.agent_registry = registry
                logging.info("✅ Injected agent registry into TestRunnerAgent for debugging integration")
        
        # DebuggingAgent needs access to other agents too
        if "DebuggingAgent" in registry:
            debugging_agent = registry["DebuggingAgent"]
            if hasattr(debugging_agent, 'agent_registry'):
                debugging_agent.agent_registry = registry
                logging.info("✅ Injected agent registry into DebuggingAgent for cross-agent orchestration")

    def plan_mission(self, mission_goal: str) -> AgentResponse:
        """
        Invokes ONLY the PlannerAgent to create a new TaskGraph.
        """
        logging.info(f"Orchestrator received planning request for: '{mission_goal}'")
        self.global_context.task_graph.nodes.clear() # Clear any previous plan
        
        planning_task = TaskNode(goal=mission_goal, assigned_agent="PlannerAgent")
        return self._invoke_agent(planning_task)

    def refine_mission_plan(self, user_feedback: str) -> AgentResponse:
        """
        Invokes ONLY the PlanRefinementAgent to modify the current TaskGraph.
        """
        logging.info(f"Orchestrator received plan refinement request: '{user_feedback}'")
        
        if not self.global_context.task_graph.nodes:
            return AgentResponse(success=False, message="There is no active plan to refine.")

        refinement_task = TaskNode(
            goal=user_feedback, # The user's feedback is the goal for this agent
            assigned_agent="PlanRefinementAgent"
        )
        return self._invoke_agent(refinement_task)

    def execute_plan(self) -> str:
        """
        Executes the TaskGraph currently loaded in the GlobalContext.
        """
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
        Creates and executes a single, standalone TaskNode immediately.
        This is used for direct user commands that bypass the main plan.
        """
        if agent_name not in self.agent_registry:
            return AgentResponse(success=False, message=f"Agent '{agent_name}' not found.")

        # Create a one-off task for this command
        single_task = TaskNode(goal=goal, assigned_agent=agent_name)
        
        # We can reuse the core invocation logic
        response = self._invoke_agent(single_task)
        
        # Log the result for transparency
        status = "succeeded" if response.success else "failed"
        self.global_context.log_event(
            f"direct_command_{status}",
            {"agent": agent_name, "goal": goal, "message": response.message}
        )
        return response

    def _run_main_loop(self):
        """
        The core execution loop that processes the TaskGraph sequentially.
        Sequential execution preserves context and dependencies between tasks.
        Agents can implement internal parallelism for their specific operations.
        """
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
        """
        Scans the TaskGraph to find a task that is ready to be executed.
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
        """
        agent_name = current_task.assigned_agent
        agent_to_run = self.agent_registry.get(agent_name)

        if not agent_to_run:
            error_msg = f"Agent '{agent_name}' not found in registry."
            logging.error(error_msg)
            return AgentResponse(success=False, message=error_msg)

        logging.info(f"Invoking agent '{agent_name}' for task '{current_task.task_id}'...")
        
        # Enhanced progress feedback - show what's happening
        print(f"\n🎯 Starting: {agent_name}")
        print(f"   Task: {current_task.goal[:80]}{'...' if len(current_task.goal) > 80 else ''}")
        if current_task.dependencies:
            print(f"   Dependencies: {', '.join(current_task.dependencies)}")
        
        # Start progress tracking
        self.progress_tracker.start_agent_progress(
            agent_name, current_task.task_id, current_task.goal
        )
        
        try:
            # Inject progress tracker into agent if it supports it
            if hasattr(agent_to_run, 'set_progress_tracker'):
                agent_to_run.set_progress_tracker(self.progress_tracker, current_task.task_id)
            
            response = agent_to_run.execute(current_task.goal, self.global_context, current_task)
            
            # Enhanced completion feedback
            if response.success:
                print(f"✅ Completed: {agent_name}")
                if response.artifacts_generated:
                    print(f"   Generated: {', '.join(response.artifacts_generated)}")
                print(f"   Result: {response.message[:100]}{'...' if len(response.message) > 100 else ''}")
            else:
                print(f"❌ Failed: {agent_name}")
                print(f"   Error: {response.message[:100]}{'...' if len(response.message) > 100 else ''}")
            
            # Complete progress tracking
            self.progress_tracker.finish_agent_progress(current_task.task_id, success=response.success)
            
            return response
            
        except Exception as e:
            error_msg = f"Uncaught exception in agent '{agent_name}': {e}"
            logging.critical(error_msg, exc_info=True)
            
            # Report failure with troubleshooting steps
            troubleshooting_steps = [
                f"Check agent implementation for '{agent_name}'",
                "Review task inputs and dependencies",
                "Check logs for detailed error information",
                "Verify LLM client configuration if agent uses LLM"
            ]
            self.progress_tracker.fail_step(current_task.task_id, error_msg, troubleshooting_steps)
            self.progress_tracker.finish_agent_progress(current_task.task_id, success=False)
            
            return AgentResponse(success=False, message=error_msg)


# --- Self-Testing Block ---
if __name__ == "__main__":
    import unittest
    from unittest.mock import patch, MagicMock
    import shutil
    from pathlib import Path

    setup_logger(default_level=logging.INFO)

    class MockAgent(BaseAgent):
        def __init__(self, name="MockAgent", should_succeed=True):
            super().__init__(name, "A mock agent for testing.")
            self.should_succeed = should_succeed
        
        def execute(self, goal: str, context: GlobalContext, current_task: TaskNode) -> AgentResponse:
            if self.should_succeed:
                return AgentResponse(success=True, message="Mock task complete.")
            else:
                return AgentResponse(success=False, message="Mock task failed.")

    @patch('agents.AGENT_REGISTRY', {
        'PlannerAgent': lambda **kwargs: MockAgent('PlannerAgent'),
        'PlanRefinementAgent': lambda: MockAgent('PlanRefinementAgent'),
        'OtherAgent': lambda: MockAgent('OtherAgent')
    })
    class TestOrchestrator(unittest.TestCase):

        def setUp(self):
            self.test_workspace_path = "./temp_orchestrator_test_ws"
            if Path(self.test_workspace_path).exists():
                shutil.rmtree(self.test_workspace_path)
            self.orchestrator = Orchestrator(workspace_path=self.test_workspace_path)
            # We can mock the agents directly on the instance for fine-grained control
            self.orchestrator.agent_registry['PlannerAgent'] = MagicMock(spec=BaseAgent)
            self.orchestrator.agent_registry['PlanRefinementAgent'] = MagicMock(spec=BaseAgent)

        def tearDown(self):
            shutil.rmtree(self.test_workspace_path)
        
        def test_plan_mission(self):
            """Tests that `plan_mission` correctly invokes the planner."""
            print("\n--- [Test Case 1: Plan Mission] ---")
            mock_planner = self.orchestrator.agent_registry['PlannerAgent']
            mock_planner.execute.return_value = AgentResponse(success=True, message="Plan created.")
            
            response = self.orchestrator.plan_mission("test goal")
            
            self.assertTrue(response.success)
            mock_planner.execute.assert_called_once()
            self.assertEqual(mock_planner.execute.call_args[0][0], "test goal")
            logger.info("✅ test_plan_mission: PASSED")

        def test_refine_mission_plan(self):
            """Tests that `refine_mission_plan` correctly invokes the refiner."""
            print("\n--- [Test Case 2: Refine Mission Plan] ---")
            mock_refiner = self.orchestrator.agent_registry['PlanRefinementAgent']
            mock_refiner.execute.return_value = AgentResponse(success=True, message="Plan refined.")
            
            # A plan must exist to be refined
            self.orchestrator.global_context.task_graph.add_task(TaskNode(task_id="t1", goal="a", assigned_agent="b"))

            response = self.orchestrator.refine_mission_plan("refinement feedback")

            self.assertTrue(response.success)
            mock_refiner.execute.assert_called_once()
            self.assertEqual(mock_refiner.execute.call_args[0][0], "refinement feedback")
            logger.info("✅ test_refine_mission_plan: PASSED")
            
        def test_refine_mission_plan_fails_without_plan(self):
            """Tests that refinement fails if no plan exists."""
            print("\n--- [Test Case 3: Refine Mission Plan Fails Without Plan] ---")
            # Ensure the task graph is empty
            self.orchestrator.global_context.task_graph.nodes = {}
            
            response = self.orchestrator.refine_mission_plan("feedback")
            
            self.assertFalse(response.success)
            self.assertIn("no active plan to refine", response.message)
            logger.info("✅ test_refine_mission_plan_fails_without_plan: PASSED")

    unittest.main(argv=['first-arg-is-ignored'], exit=False)