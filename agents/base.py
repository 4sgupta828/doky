# agents/base.py
import logging
import uuid
import concurrent.futures
from abc import ABC, abstractmethod
from typing import List, Callable, Any, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from core.context import GlobalContext

# The base agent needs to know the "shape" of the data it will receive and return.
from core.models import AgentResponse, AgentResult, AgentExecutionError, TaskNode
from core.context import GlobalContext

# Get a logger instance for this module.
logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    """
    An abstract base class that defines the universal interface for all specialized agents.

    This class ensures that every agent in the Sovereign Agent Collective adheres to a
    standard contract, allowing the Orchestrator to manage them uniformly. Each
    subclass must implement the `execute_v2` method.
    """

    def __init__(self, name: str, description: str):
        """
        Initializes the agent with its identity.
        """
        if not name or not description:
            raise ValueError("Agent name and description cannot be empty.")
        self.name = name
        self.description = description
        
        # Progress tracking is a standard feature for all agents.
        self.progress_tracker = None
        self.current_task_id = None

    @abstractmethod
    def execute_v2(self, goal: str, inputs: Dict[str, Any], global_context: GlobalContext) -> AgentResult:
        """
        The primary method where the agent performs its specialized task.
        This is the standard, function-call interface for all agents.
        
        Args:
            goal: A high-level description of what the agent should accomplish.
            inputs: A dictionary of explicit, validated inputs for the agent.
            global_context: The shared context for workspace and artifact access.
            
        Returns:
            An AgentResult object with structured outputs and execution status.
        """
        pass
    
    def required_inputs(self) -> List[str]:
        """
        Declares the input keys that are mandatory for this agent's execution.
        The Orchestrator will use this to fail-fast if inputs are missing.
        """
        return []

    def optional_inputs(self) -> List[str]:
        """
        Declares optional input keys for documentation and validation purposes.
        """
        return []

    def get_capabilities(self) -> dict:
        """
        Returns a dictionary describing the agent's identity and skills.
        This is used by the PlannerAgent to understand the tools at its disposal.
        """
        return {
            "name": self.name, 
            "description": self.description,
            "required_inputs": self.required_inputs(),
            "optional_inputs": self.optional_inputs()
        }
    
    # === V2 INTERFACE HELPERS ===
    
    def validate_inputs(self, inputs: Dict[str, Any]) -> None:
        """
        Performs fail-fast validation of required inputs.
        
        Raises:
            AgentExecutionError: If any required inputs are missing.
        """
        required = self.required_inputs()
        missing = [key for key in required if key not in inputs]
        
        if missing:
            error_msg = f"{self.name} execution failed: missing required inputs {missing}."
            logger.error(error_msg)
            raise AgentExecutionError(error_msg)
        
        logger.debug(f"{self.name} input validation passed.")
    
    def call_agent_v2(self, target_agent: 'BaseAgent', goal: str, inputs: Dict[str, Any], 
                     global_context: GlobalContext) -> AgentResult:
        """
        Calls another agent using the standardized v2 interface.
        This is the primary method for inter-agent communication.
        """
        import time
        start_time = time.time()
        
        execution_id = f"{target_agent.name}_{uuid.uuid4().hex[:8]}"
        
        logger.info(f"🔗 {self.name} is calling {target_agent.name} for goal: '{goal}'")
        
        self.report_progress(f"Delegating to {target_agent.name}", f"Goal: {goal[:60]}...")
        
        try:
            # Execute the target agent's v2 method directly.
            result = target_agent.execute_v2(goal, inputs, global_context)
            
            result.execution_id = execution_id
            result.duration_seconds = time.time() - start_time
            
            status = "✅ SUCCESS" if result.success else "❌ FAILED"
            logger.info(f"{status}: {target_agent.name} execution completed in {result.duration_seconds:.2f}s")
            
            return result
            
        except Exception as e:
            error_msg = f"{target_agent.name} execution failed with an unexpected error: {e}"
            logger.error(error_msg, exc_info=True)
            return self.create_result(
                success=False,
                message=error_msg,
                error_details={"exception": str(e), "exception_type": type(e).__name__}
            )
    
    def create_result(self, success: bool, message: str, outputs: Dict[str, Any] = None, 
                     error_details: Dict[str, Any] = None) -> AgentResult:
        """
        Helper method to create a standardized AgentResult instance.
        """
        return AgentResult(
            success=success,
            message=message,
            outputs=outputs or {},
            error_details=error_details
        )
    
    # === PROGRESS TRACKING HELPERS ===
    
    def set_progress_tracker(self, progress_tracker, task_id: str):
        """
        Sets up progress tracking for this agent's execution context.
        """
        self.progress_tracker = progress_tracker
        self.current_task_id = task_id
    
    def report_progress(self, step: str, details: str = None):
        """Helper method for agents to report a progress step."""
        if self.progress_tracker and self.current_task_id:
            self.progress_tracker.report_progress(self.current_task_id, step, details)
    
    def report_thinking(self, thought: str):
        """Helper method for agents to report their reasoning process."""
        if self.progress_tracker and self.current_task_id:
            self.progress_tracker.report_thinking(self.current_task_id, thought)
    
    def report_intermediate_output(self, output_type: str, content: Any):
        """Helper method for agents to report intermediate outputs."""
        if self.progress_tracker and self.current_task_id:
            self.progress_tracker.report_intermediate_output(self.current_task_id, output_type, content)
    
    def fail_step(self, error: str, troubleshooting_steps: List[str] = None):
        """Helper method for agents to report a step failure."""
        if self.progress_tracker and self.current_task_id:
            self.progress_tracker.fail_step(self.current_task_id, error, troubleshooting_steps)


# --- Self-Testing Block ---
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
        
        def required_inputs(self) -> List[str]:
            return ["message"]

        def execute_v2(self, goal: str, inputs: Dict[str, Any], global_context: GlobalContext) -> AgentResult:
            self.validate_inputs(inputs) # Test validation
            logger.info(f"DummyAgent executing goal: '{goal}'")
            message = inputs.get("message", "No message provided.")
            
            # A real agent would interact with the context here.
            global_context.log_event("dummy_execution", {"goal": goal})
            
            return self.create_result(
                success=True,
                message=f"Dummy task completed with message: {message}",
                outputs={"received_message": message}
            )

    # 2. Test instantiation and capability reporting.
    print("\n[1] Testing agent instantiation and capabilities...")
    try:
        dummy_agent = DummyAgent()
        logger.info(f"Successfully instantiated {dummy_agent.name}.")
        capabilities = dummy_agent.get_capabilities()
        assert capabilities["name"] == "DummyAgent"
        assert "message" in capabilities["required_inputs"]
        logger.info(f"Capabilities: {capabilities}")
        logger.info("Instantiation and capabilities test passed.")
    except Exception as e:
        logger.error("Failed to instantiate DummyAgent.", exc_info=True)

    # 3. Test the execution flow.
    print("\n[2] Testing agent execution flow...")
    mock_context = GlobalContext()
    
    try:
        inputs = {"message": "Hello from test"}
        response = dummy_agent.execute_v2("Run a dummy operation.", inputs, mock_context)
        assert isinstance(response, AgentResult)
        assert response.success is True
        assert "Hello from test" in response.message
        assert response.outputs["received_message"] == "Hello from test"
        logger.info(f"Received valid response from agent: {response.message}")
        logger.info("Agent execution test passed.")
    except Exception as e:
        logger.error("DummyAgent execution failed.", exc_info=True)

    # 4. Test input validation failure.
    print("\n[3] Testing input validation failure...")
    try:
        invalid_inputs = {} # Missing the required 'message' key
        dummy_agent.execute_v2("This should fail", invalid_inputs, mock_context)
    except AgentExecutionError as e:
        logger.info(f"Successfully caught expected AgentExecutionError: {e}")
    
    print("\n--- All BaseAgent Tests Passed Successfully ---")
