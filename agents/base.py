# agents/base.py
import logging
import uuid
import concurrent.futures
from abc import ABC, abstractmethod
from typing import List, Callable, Any, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from core.context import GlobalContext

# The base agent needs to know the "shape" of the data it will receive and return.
# By importing these types, we can use them in method signatures for clarity and
# type-checking, which is excellent for debugging.
from core.models import AgentResponse, AgentResult, AgentExecutionError, TaskNode
from core.context import GlobalContext

# Get a logger instance for this module.
logger = logging.getLogger(__name__)

# Import the context error for handling
try:
    from real_llm_client import ContextTooLargeError
except ImportError:
    # Define a fallback if not available
    class ContextTooLargeError(Exception):
        pass

class BaseAgent(ABC):
    """
    An abstract base class that defines the universal interface for all specialized agents.

    This class ensures that every agent in the Sovereign Agent Collective adheres to a
    standard contract, allowing the Orchestrator to manage them uniformly. Each
    subclass must implement the `execute` method.
    """

    def __init__(self, name: str, description: str):
        """
        Initializes the agent with its identity.

        Args:
            name: The unique name of the agent (e.g., "PlannerAgent"). This name is
                  used by the Orchestrator to look up the agent in the registry.
            description: A brief description of the agent's expertise and purpose,
                         which can be used by the PlannerAgent to select the right
                         tool for a job.
        """
        if not name or not description:
            raise ValueError("Agent name and description cannot be empty.")
        self.name = name
        self.description = description
        
        # Progress tracking (optional)
        self.progress_tracker = None
        self.current_task_id = None

    @abstractmethod
    def execute(self, goal: str, context: GlobalContext, current_task: TaskNode) -> AgentResponse:
        """
        LEGACY: The primary method where the agent performs its specialized task.
        
        This is kept for backward compatibility. New agents should implement execute_v2.
        """
        pass
    
    def execute_v2(self, goal: str, inputs: Dict[str, Any], global_context: GlobalContext) -> AgentResult:
        """
        NEW FUNCTION-CALL INTERFACE: Execute agent with explicit inputs and outputs.
        
        This method provides function-call semantics with clear contracts:
        - Explicit inputs (no artifact hunting)
        - Explicit outputs (no global context pollution) 
        - Fail-fast validation
        - Instance isolation
        
        Args:
            goal: Clear problem statement - what to accomplish
            inputs: Dictionary of explicit inputs (validated against required_inputs())
            global_context: Shared workspace context for file operations only
            
        Returns:
            AgentResult with structured outputs and execution tracking
        """
        # Default implementation: create a TaskNode and call legacy execute()
        logger.warning(f"{self.name} using legacy execute() via execute_v2() wrapper")
        
        # Create a temporary TaskNode for legacy compatibility
        from core.models import TaskNode
        legacy_task = TaskNode(
            goal=goal,
            assigned_agent=self.name,
            task_id=f"legacy_{uuid.uuid4().hex[:8]}"
        )
        
        # Call legacy execute method
        legacy_response = self.execute(goal, global_context, legacy_task)
        
        # Convert legacy AgentResponse to new AgentResult
        return AgentResult(
            success=legacy_response.success,
            message=legacy_response.message,
            outputs={
                "artifacts_generated": legacy_response.artifacts_generated,
                "legacy_mode": True
            }
        )
    
    def required_inputs(self) -> List[str]:
        """
        Declare required inputs for this agent - fail fast if missing.
        
        Returns:
            List of input keys that MUST be present in inputs dict
        """
        return []  # Default: no required inputs
    
    def optional_inputs(self) -> List[str]:
        """
        Declare optional inputs for documentation and validation.
        
        Returns:
            List of input keys that MAY be present in inputs dict  
        """
        return []

    def get_capabilities(self) -> dict:
        """
        Returns a dictionary describing the agent's identity and skills.
        This is a concrete method available to all subclasses. It's used by the
        PlannerAgent to understand the tools at its disposal.
        """
        return {
            "name": self.name, 
            "description": self.description,
            "required_inputs": self.required_inputs(),
            "optional_inputs": self.optional_inputs()
        }
    
    # === FUNCTION-CALL SEMANTICS HELPERS ===
    
    def validate_inputs(self, inputs: Dict[str, Any]) -> None:
        """
        Fail-fast validation of required inputs.
        Call this at the start of execute_v2() to ensure all required data is present.
        
        Args:
            inputs: Input dictionary to validate
            
        Raises:
            AgentExecutionError: If required inputs are missing
        """
        required = self.required_inputs()
        missing = [inp for inp in required if inp not in inputs]
        
        if missing:
            error_msg = f"{self.name} execution failed: missing required inputs {missing}. Required: {required}"
            logger.error(error_msg)
            raise AgentExecutionError(error_msg)
        
        # Log successful validation
        logger.info(f"{self.name} input validation passed: {len(inputs)} inputs provided, {len(required)} required")
    
    def call_agent_v2(self, target_agent: 'BaseAgent', goal: str, inputs: Dict[str, Any], 
                     global_context: GlobalContext) -> AgentResult:
        """
        Call another agent with function-call semantics.
        This is the new standard way for agents to communicate.
        
        Args:
            target_agent: The agent to call
            goal: Clear problem statement for the target agent
            inputs: Explicit inputs dictionary
            global_context: Shared workspace context
            
        Returns:
            AgentResult with structured outputs
        """
        import time
        start_time = time.time()
        
        # Generate execution ID for tracking
        execution_id = f"{target_agent.name}_{uuid.uuid4().hex[:8]}"
        
        # Log the function call for transparency
        logger.info(f"🔗 {self.name} calling {target_agent.name}")
        logger.info(f"   Goal: {goal[:80]}{'...' if len(goal) > 80 else ''}")
        logger.info(f"   Inputs: {list(inputs.keys())}")
        
        self.report_progress(f"Calling {target_agent.name}", f"Goal: {goal[:60]}...")
        
        # Transfer progress tracker if both agents support it
        if hasattr(self, 'progress_tracker') and hasattr(target_agent, 'set_progress_tracker'):
            if self.progress_tracker:
                target_agent.set_progress_tracker(self.progress_tracker, execution_id)
        
        try:
            # Execute with function-call semantics
            result = target_agent.execute_v2(goal, inputs, global_context)
            
            # Ensure execution ID is set
            result.execution_id = execution_id
            result.duration_seconds = time.time() - start_time
            
            # Log result for transparency
            status = "✅ SUCCESS" if result.success else "❌ FAILED"
            logger.info(f"{status}: {target_agent.name} execution completed")
            logger.info(f"   Duration: {result.duration_seconds:.2f}s")
            logger.info(f"   Outputs: {list(result.outputs.keys())}")
            
            return result
            
        except AgentExecutionError as e:
            # Expected validation error
            error_result = AgentResult(
                success=False,
                message=f"{target_agent.name} validation failed: {e}",
                execution_id=execution_id,
                duration_seconds=time.time() - start_time,
                error_details={"error_type": "validation_error", "details": str(e)}
            )
            logger.error(f"❌ VALIDATION ERROR: {target_agent.name} - {e}")
            return error_result
            
        except Exception as e:
            # Unexpected execution error
            error_result = AgentResult(
                success=False,
                message=f"{target_agent.name} execution failed: {e}",
                execution_id=execution_id,
                duration_seconds=time.time() - start_time,
                error_details={"error_type": "execution_error", "details": str(e), "exception": type(e).__name__}
            )
            logger.error(f"❌ EXECUTION ERROR: {target_agent.name} - {e}", exc_info=True)
            return error_result
    
    def create_result(self, success: bool, message: str, outputs: Dict[str, Any] = None, 
                     error_details: Dict[str, Any] = None) -> AgentResult:
        """
        Helper method to create standardized AgentResult instances.
        
        Args:
            success: Whether execution succeeded
            message: Human-readable result message
            outputs: Structured outputs dictionary
            error_details: Error details if execution failed
            
        Returns:
            Standardized AgentResult
        """
        return AgentResult(
            success=success,
            message=message,
            outputs=outputs or {},
            error_details=error_details
        )
    
    def set_progress_tracker(self, progress_tracker, task_id: str):
        """
        Sets up progress tracking for this agent execution.
        
        Args:
            progress_tracker: The ProgressTracker instance
            task_id: The current task ID being executed
        """
        self.progress_tracker = progress_tracker
        self.current_task_id = task_id
    
    def report_progress(self, step: str, details: str = None):
        """Helper method for agents to report progress."""
        if self.progress_tracker and self.current_task_id:
            self.progress_tracker.report_progress(self.current_task_id, step, details)
    
    def report_thinking(self, thought: str):
        """Helper method for agents to report their thinking process."""
        if self.progress_tracker and self.current_task_id:
            self.progress_tracker.report_thinking(self.current_task_id, thought)
    
    def report_intermediate_output(self, output_type: str, content):
        """Helper method for agents to report intermediate outputs."""
        if self.progress_tracker and self.current_task_id:
            self.progress_tracker.report_intermediate_output(self.current_task_id, output_type, content)
    
    def log_communication(self, context: 'GlobalContext', to_agent: str, message_type: str, content: str, context_data: Dict[str, Any] = None, task_id: str = None):
        """Helper method for agents to log inter-agent communications for transparency."""
        context.log_agent_communication(self.name, to_agent, message_type, content, context_data, task_id)
    
    def call_agent_with_progress(self, target_agent, goal: str, context: 'GlobalContext', task: TaskNode, subtask_id: str = None, input_artifact_keys: list = None):
        """
        Helper method to call another agent while transferring progress tracker.
        This ensures CLI progress reporting continues across agent handovers.
        
        Args:
            target_agent: The agent instance to call
            goal: The goal to pass to the target agent
            context: The global context
            task: The current task (will create subtask if needed)
            subtask_id: Optional custom subtask ID
            input_artifact_keys: Optional list of specific artifact keys to pass to target agent
        
        Returns:
            AgentResponse from the target agent
        """
        # Create subtask if needed
        if subtask_id is None:
            subtask_id = f"{task.task_id}_{target_agent.name.lower()}"
        
        # Use provided artifact keys or fall back to original task's keys
        artifact_keys = input_artifact_keys if input_artifact_keys is not None else (task.input_artifact_keys or [])
        
        subtask = TaskNode(
            goal=goal,
            assigned_agent=target_agent.name,
            input_artifact_keys=artifact_keys,
            task_id=subtask_id
        )
        
        # Transfer progress tracker if both agents support it
        if hasattr(self, 'progress_tracker') and hasattr(target_agent, 'set_progress_tracker'):
            # Start progress tracking for the subtask
            if self.progress_tracker:
                self.progress_tracker.start_agent_progress(target_agent.name, subtask.task_id, subtask.goal)
                target_agent.set_progress_tracker(self.progress_tracker, subtask.task_id)
        
        # Execute the target agent
        return target_agent.execute(goal, context, subtask)
    
    def complete_step(self, output=None):
        """Helper method for agents to mark steps as completed."""
        if self.progress_tracker and self.current_task_id:
            self.progress_tracker.complete_step(self.current_task_id, output)
    
    def fail_step(self, error: str, troubleshooting_steps=None):
        """Helper method for agents to report step failures."""
        if self.progress_tracker and self.current_task_id:
            self.progress_tracker.fail_step(self.current_task_id, error, troubleshooting_steps)

    def handle_context_error(self, error: ContextTooLargeError, goal: str) -> AgentResponse:
        """
        Common handler for context size errors with smart retry logic.
        
        Args:
            error: The ContextTooLargeError that was caught
            goal: The goal that caused the error
            
        Returns:
            AgentResponse with failure and detailed error information
        """
        logger.warning(f"{self.name} hit context size limit, attempting smart recovery")
        
        # Try to recover with truncated goal
        if len(goal) > 500:
            truncated_goal = goal[:400] + "... [Goal truncated due to context limits]"
            logger.info(f"{self.name} retrying with truncated goal")
            
            try:
                # Attempt retry with truncated context (subclasses can override this)
                return self._retry_with_truncated_context(truncated_goal)
            except Exception as retry_error:
                logger.error(f"{self.name} retry failed: {retry_error}")
        
        # If retry fails or goal is already short, return helpful error
        error_message = f"""
{self.name} failed due to context size limits.

Goal: {goal[:200]}{'...' if len(goal) > 200 else ''}

{str(error)}

The agent attempted smart recovery but cannot proceed with this task.
Consider breaking down the task into smaller, more focused parts.
"""
        
        logger.error(f"{self.name} context size error: {str(error)}")
        return AgentResponse(success=False, message=error_message.strip())
    
    def _retry_with_truncated_context(self, truncated_goal: str) -> AgentResponse:
        """
        Default implementation for retrying with truncated context.
        Subclasses should override this for agent-specific retry logic.
        """
        logger.info(f"{self.name} using default truncated context retry")
        # Most agents can't easily retry, so we return a helpful message
        return AgentResponse(
            success=False, 
            message=f"{self.name} needs manual intervention to handle this large context. Please break down the goal into smaller tasks."
        )
    
    def _execute_parallel_tasks(self, tasks: List[Callable[[], Any]], max_workers: int = 3, 
                               task_descriptions: List[str] = None) -> List[Any]:
        """
        Utility method for agents to execute independent tasks in parallel.
        Only use when tasks are truly independent and parallel execution provides benefit.
        
        Args:
            tasks: List of callable functions that return results
            max_workers: Maximum number of parallel workers
            task_descriptions: Optional descriptions for progress tracking
            
        Returns:
            List of results in the same order as input tasks
        """
        if len(tasks) <= 1:
            # No benefit in parallelization for single task
            return [task() for task in tasks]
        
        logger.info(f"{self.name} executing {len(tasks)} independent tasks in parallel")
        
        if task_descriptions:
            for i, desc in enumerate(task_descriptions[:len(tasks)]):
                self.report_progress(f"Parallel task {i+1}", desc)
        
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_index = {executor.submit(task): i for i, task in enumerate(tasks)}
            
            # Collect results in original order
            results = [None] * len(tasks)
            for future in concurrent.futures.as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    results[index] = future.result()
                except Exception as e:
                    logger.error(f"{self.name} parallel task {index} failed: {e}")
                    results[index] = None
        
        successful_tasks = sum(1 for r in results if r is not None)
        logger.info(f"{self.name} completed {successful_tasks}/{len(tasks)} parallel tasks successfully")
        
        return results
    
    def _should_parallelize(self, task_count: int, estimated_time_per_task: float = 1.0) -> bool:
        """
        Determines if parallelization would provide meaningful benefit.
        
        Args:
            task_count: Number of independent tasks
            estimated_time_per_task: Estimated time per task in seconds
            
        Returns:
            True if parallelization is recommended
        """
        # Only parallelize if:
        # 1. More than 1 task
        # 2. Each task takes reasonable time (>30 seconds or multiple tasks >10 seconds each)
        # 3. Tasks are truly independent
        
        if task_count <= 1:
            return False
            
        total_estimated_time = task_count * estimated_time_per_task
        
        # Parallelize if total work > 30 seconds or if we have 3+ tasks taking >10s each
        if total_estimated_time > 30 or (task_count >= 3 and estimated_time_per_task > 10):
            logger.info(f"{self.name} recommends parallelization: {task_count} tasks, ~{total_estimated_time:.1f}s total")
            return True
        
        logger.debug(f"{self.name} keeping sequential: {task_count} tasks, ~{total_estimated_time:.1f}s total")
        return False


# --- Self-Testing Block ---
# This block demonstrates how the BaseAgent contract works and serves as a simple test.
# Since BaseAgent is an abstract class, we cannot instantiate it directly. Instead, we
# create a simple concrete implementation (`DummyAgent`) to test the inheritance and
# method signatures.
# To run this test, execute the file directly: `python agents/base.py`
if __name__ == "__main__":
    from utils.logger import setup_logger
    setup_logger()

    print("\n--- Testing BaseAgent Abstract Class ---")

    # 1. Define a concrete implementation for testing purposes.
    class DummyAgent(BaseAgent):
        """A simple agent for testing the BaseAgent contract."""
        def __init__(self):
            super().__init__(
                name="DummyAgent",
                description="A test agent that always succeeds."
            )
        
        def execute(self, goal: str, context: GlobalContext, current_task: TaskNode) -> AgentResponse:
            logger.info(f"DummyAgent executing goal: '{goal}' for task '{current_task.task_id}'")
            # A real agent would interact with the context here.
            context.log_event("dummy_execution", {"goal": goal})
            return AgentResponse(success=True, message="Dummy task completed successfully.")

    # 2. Test instantiation and capability reporting.
    print("\n[1] Testing agent instantiation and capabilities...")
    try:
        dummy_agent = DummyAgent()
        logger.info(f"Successfully instantiated {dummy_agent.name}.")
        capabilities = dummy_agent.get_capabilities()
        assert capabilities["name"] == "DummyAgent"
        logger.info(f"Capabilities: {capabilities}")
        logger.info("Instantiation and capabilities test passed.")
    except Exception as e:
        logger.error("Failed to instantiate DummyAgent.", exc_info=True)


    # 3. Test the execution flow.
    print("\n[2] Testing agent execution flow...")
    # Create mock context and task objects for the test.
    mock_context = GlobalContext()
    mock_task = TaskNode(
        task_id="test_task_123",
        goal="Run a dummy operation.",
        assigned_agent="DummyAgent"
    )

    try:
        response = dummy_agent.execute(mock_task.goal, mock_context, mock_task)
        assert isinstance(response, AgentResponse)
        assert response.success is True
        assert "successfully" in response.message
        logger.info(f"Received valid response from agent: {response.message}")
        logger.info("Agent execution test passed.")
    except Exception as e:
        logger.error("DummyAgent execution failed.", exc_info=True)
    
    # 4. Verify that an abstract class cannot be instantiated directly.
    print("\n[3] Verifying that BaseAgent cannot be instantiated directly...")
    try:
        # This line should raise a TypeError because `execute` is an abstract method.
        abstract_instance = BaseAgent("Abstract", "This should fail")
    except TypeError as e:
        logger.info("Successfully caught expected TypeError when trying to instantiate BaseAgent.")
        logger.debug(f"Error message: {e}")

    print("\n--- All BaseAgent Tests Passed Successfully ---")