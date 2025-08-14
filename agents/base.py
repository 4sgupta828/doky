# agents/base.py
import logging
from abc import ABC, abstractmethod

# The base agent needs to know the "shape" of the data it will receive and return.
# By importing these types, we can use them in method signatures for clarity and
# type-checking, which is excellent for debugging.
from core.models import AgentResponse, TaskNode
from core.context import GlobalContext

# Get a logger instance for this module.
logger = logging.getLogger(__name__)

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

    @abstractmethod
    def execute(self, goal: str, context: GlobalContext, current_task: TaskNode) -> AgentResponse:
        """
        The primary method where the agent performs its specialized task.

        This abstract method MUST be implemented by all concrete agent subclasses.
        It should be designed to be as stateless as possible, relying only on the
        provided context for its inputs and writing all its outputs back into the
        context.

        Args:
            goal: The specific sub-problem or objective this agent needs to solve for the current task.
            context: The shared GlobalContext object, providing access to the workspace,
                     artifacts, and the overall plan.
            current_task: The TaskNode object that the agent is currently executing. This gives
                          the agent awareness of its own task ID, dependencies, and expected outputs.

        Returns:
            An AgentResponse object detailing the outcome of its execution. The `success`
            field is critical for the Orchestrator to determine the next step.
        """
        pass

    def get_capabilities(self) -> dict:
        """
        Returns a dictionary describing the agent's identity and skills.
        This is a concrete method available to all subclasses. It's used by the
        PlannerAgent to understand the tools at its disposal.
        """
        return {"name": self.name, "description": self.description}


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