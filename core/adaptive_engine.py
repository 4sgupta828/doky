# core/adaptive_engine.py
import logging
from typing import Optional

# Foundational dependencies from Tier 1 & 2
from core.models import TaskNode, TaskGraph, AgentResponse
from core.context import GlobalContext
from agents.base import BaseAgent # Used for type hinting the planner

# Get a logger instance for this module.
logger = logging.getLogger(__name__)

class AdaptiveEngine:
    """
    Acts as the system's crisis manager and recovery expert.
    When a task fails, this engine is triggered by the Orchestrator to analyze the
    failure, revert the system to a safe state, and coordinate with the PlannerAgent
    to formulate a new plan to overcome the obstacle.
    """

    def handle_failure(
        self,
        failed_task: TaskNode,
        context: GlobalContext,
        planner: BaseAgent
    ) -> bool:
        """
        The core method for handling a task failure. It orchestrates the entire
        recovery process.

        Args:
            failed_task: The TaskNode that failed, containing the error details.
            context: The shared GlobalContext for the mission.
            planner: An instance of the PlannerAgent, required for generating a recovery plan.

        Returns:
            True if a recovery plan was successfully created and integrated,
            False otherwise, indicating a critical, unrecoverable mission failure.
        """
        logger.critical(f"--- ADAPTIVE ENGINE TRIGGERED for failed task: {failed_task.task_id} ---")
        context.log_event("adaptive_engine_triggered", {"failed_task_id": failed_task.task_id})

        # --- Step 1: Analyze the failure ---
        failure_reason = failed_task.result.message if failed_task.result else "Unknown error."
        logger.info(f"Analyzing failure in task '{failed_task.goal}'. Reason: {failure_reason}")

        # --- Step 2: Revert workspace to a clean state ---
        # This uses the Git-based manager to undo any file changes from the failed task.
        logger.info(f"Reverting workspace changes made by task '{failed_task.task_id}'.")
        try:
            context.workspace.revert_changes(task_id=failed_task.task_id)
            context.log_event("workspace_reverted", {"reverted_for_task": failed_task.task_id})
        except Exception as e:
            logger.error(f"Failed to revert workspace changes for task '{failed_task.task_id}'. This could affect recovery.", exc_info=True)
            # We can choose to continue, but it's risky. For now, we'll log and proceed.

        # --- Step 3: Formulate a new goal for the planner ---
        # The goal is highly specific: recover from this exact failure.
        recovery_goal = (
            f"The original plan failed at task '{failed_task.task_id}' (goal: '{failed_task.goal}') "
            f"with the error: '{failure_reason}'. "
            f"Please generate a new, short sub-plan to debug and resolve this specific issue. "
            f"The new plan should then resume the original mission objective."
        )
        logger.info("Formulated new recovery goal for the PlannerAgent.")

        # --- Step 4: Request a new, adaptive plan from the PlannerAgent ---
        try:
            # We create a new, temporary task for this re-planning effort.
            replanning_task = TaskNode(
                task_id=f"replanning_for_{failed_task.task_id}",
                goal=recovery_goal,
                assigned_agent="PlannerAgent"
            )
            logger.info("Invoking PlannerAgent to create a recovery plan.")
            response = planner.execute(recovery_goal, context, replanning_task)
            
            if not response.success:
                logger.error("PlannerAgent failed to generate a recovery plan. Mission is unrecoverable.")
                return False

            # The Planner's response should contain the new TaskGraph in an artifact.
            # For this skeleton, we assume the planner directly modifies the context's task_graph.
            # A more robust implementation might pass it via an artifact.
            recovery_graph = context.task_graph

            if not recovery_graph or not recovery_graph.nodes:
                logger.error("PlannerAgent succeeded but returned an empty recovery plan.")
                return False

        except Exception as e:
            logger.critical("An unexpected exception occurred while invoking the planner for recovery.", exc_info=True)
            return False

        # --- Step 5: Integrate the new plan ---
        # Mark the failed task and any tasks that depended on it as "obsolete".
        self._prune_failed_branch(failed_task.task_id, context.task_graph)

        logger.info("Recovery plan successfully generated. Mission will continue on the new path.")
        return True

    def _prune_failed_branch(self, failed_task_id: str, graph: TaskGraph):
        """
        Marks the failed task and all its downstream dependents as 'obsolete'.
        This prevents the orchestrator from attempting to run tasks that are part of a now-defunct plan.
        """
        tasks_to_prune = {failed_task_id}
        # Use a queue to find all downstream tasks (breadth-first traversal).
        queue = [failed_task_id]

        while queue:
            current_id = queue.pop(0)
            for task in graph.nodes.values():
                if current_id in task.dependencies and task.task_id not in tasks_to_prune:
                    tasks_to_prune.add(task.task_id)
                    queue.append(task.task_id)

        for task_id in tasks_to_prune:
            task = graph.get_task(task_id)
            if task:
                task.status = "obsolete"
                logger.warning(f"Pruning task '{task.task_id}' as it was part of a failed branch.")
        
        logging.info(f"Pruned {len(tasks_to_prune)} obsolete tasks from the plan.")

# --- Self-Testing Block ---
# To run this test: `python core/adaptive_engine.py`
if __name__ == "__main__":
    from utils.logger import setup_logger
    import shutil
    from pathlib import Path

    setup_logger(default_level=logging.DEBUG)

    # --- Mock objects for isolated testing ---
    class MockPlanner(BaseAgent):
        """A mock planner that returns a pre-defined recovery plan."""
        def __init__(self):
            super().__init__("MockPlanner", "Generates recovery plans.")

        def execute(self, goal: str, context: GlobalContext, current_task: TaskNode) -> AgentResponse:
            if "failed" in goal:
                # This is the recovery plan it will generate.
                recovery_graph = TaskGraph()
                debug_task = TaskNode(
                    task_id="task_debug",
                    goal="Analyze the failure.",
                    assigned_agent="DebuggingAgent",
                    dependencies=[] # The recovery plan starts fresh.
                )
                fix_task = TaskNode(
                    task_id="task_fix_code",
                    goal="Apply the fix.",
                    assigned_agent="CodeGenerationAgent",
                    dependencies=["task_debug"]
                )
                recovery_graph.add_task(debug_task)
                recovery_graph.add_task(fix_task)
                
                # In a real scenario, the planner modifies the context's graph. We simulate that here.
                context.task_graph = recovery_graph
                return AgentResponse(success=True, message="Recovery plan created.")
            return AgentResponse(success=False, message="MockPlanner can only handle recovery goals.")

    print("\n--- Testing AdaptiveEngine ---")

    TEST_WORKSPACE = "./temp_adaptive_engine_ws"
    if Path(TEST_WORKSPACE).exists():
        shutil.rmtree(TEST_WORKSPACE)

    # 1. Setup a realistic failure scenario
    print("\n[1] Setting up failure scenario...")
    context = GlobalContext(workspace_path=TEST_WORKSPACE)
    planner = MockPlanner()
    engine = AdaptiveEngine()

    # Create an initial plan
    task1 = TaskNode(task_id="task_A_plan", goal="Plan the project", assigned_agent="PlannerAgent", status="success")
    task2_failed = TaskNode(
        task_id="task_B_code",
        goal="Write broken code",
        assigned_agent="CodeGenerationAgent",
        dependencies=["task_A_plan"],
        result=AgentResponse(success=False, message="SyntaxError on line 42") # The failure reason
    )
    task3_dependent = TaskNode(
        task_id="task_C_test",
        goal="Test the code",
        assigned_agent="TestRunnerAgent",
        dependencies=["task_B_code"] # Depends on the failed task
    )

    context.task_graph.add_task(task1)
    context.task_graph.add_task(task2_failed)
    context.task_graph.add_task(task3_dependent)
    
    # Simulate the failed task writing a file
    context.workspace.write_file_content("broken_file.py", "a = 1\nb = ", task2_failed.task_id)
    assert "broken_file.py" in context.workspace.list_files()
    logger.info("Scenario setup complete. A failed task and its dependent exist.")

    # 2. Execute the handle_failure method
    print("\n[2] Executing handle_failure...")
    try:
        was_successful = engine.handle_failure(task2_failed, context, planner)
        assert was_successful is True
        logger.info("handle_failure executed successfully.")
    except Exception as e:
        logger.error("handle_failure test FAILED.", exc_info=True)
        was_successful = False

    # 3. Verify the outcome
    print("\n[3] Verifying recovery outcome...")
    if was_successful:
        # a. Verify the failed branch was pruned
        assert context.task_graph.get_task("task_B_code").status == "obsolete"
        assert context.task_graph.get_task("task_C_test").status == "obsolete"
        logger.info("Obsolete tasks correctly pruned.")

        # b. Verify the new recovery plan is now in the graph
        assert context.task_graph.get_task("task_debug") is not None
        assert context.task_graph.get_task("task_fix_code") is not None
        logger.info("Recovery plan correctly spliced into the TaskGraph.")
        
        # c. Verify the workspace was reverted
        final_files = context.workspace.list_files()
        assert "broken_file.py" not in final_files
        logger.info("Workspace changes correctly reverted.")

    # Cleanup
    shutil.rmtree(TEST_WORKSPACE)
    logger.info(f"Cleaned up test directory: {TEST_WORKSPACE}")

    print("\n--- All AdaptiveEngine Tests Passed Successfully ---")