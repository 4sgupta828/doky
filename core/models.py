# core/models.py
import uuid
import logging
from typing import List, Dict, Any, Literal, Optional

# Pydantic is used for its robust data validation and serialization capabilities.
# It ensures that data flowing through the system conforms to a strict schema.
from pydantic import BaseModel, Field, ValidationError

# Get a logger instance for this module to report on model-related events.
logger = logging.getLogger(__name__)

# --- Inter-Agent Communication Models ---

class AgentCommunication(BaseModel):
    """
    Represents a single communication between agents for transparency tracking.
    This allows users to see the chain of inter-agent communications.
    """
    message_id: str = Field(default_factory=lambda: f"msg_{uuid.uuid4().hex[:8]}", description="Unique identifier for this message.")
    from_agent: str = Field(..., description="Name of the agent sending the message.")
    to_agent: str = Field(..., description="Name of the agent receiving the message.")
    message_type: Literal["handover", "delegation", "response", "error"] = Field(..., description="Type of inter-agent communication.")
    content: str = Field(..., description="The actual message content or task description.")
    context_data: Dict[str, Any] = Field(default_factory=dict, description="Additional context data for the communication.")
    timestamp: str = Field(default_factory=lambda: str(uuid.uuid4().int), description="Timestamp when message was created.")
    task_id: Optional[str] = Field(None, description="Associated task ID if applicable.")

class InterAgentLog(BaseModel):
    """
    A log of all inter-agent communications for a session or task chain.
    This provides complete transparency into agent interactions.
    """
    session_id: str = Field(default_factory=lambda: f"session_{uuid.uuid4().hex[:8]}", description="Unique identifier for this communication session.")
    communications: List[AgentCommunication] = Field(default_factory=list, description="List of all communications in chronological order.")
    
    def add_communication(self, from_agent: str, to_agent: str, message_type: str, content: str, context_data: Dict[str, Any] = None, task_id: str = None):
        """Add a new communication to the log."""
        comm = AgentCommunication(
            from_agent=from_agent,
            to_agent=to_agent,
            message_type=message_type,
            content=content,
            context_data=context_data or {},
            task_id=task_id
        )
        self.communications.append(comm)
        logger.info(f"Inter-agent communication: {from_agent} -> {to_agent} ({message_type}): {content}")
    
    def get_communication_chain(self) -> List[str]:
        """Get a human-readable representation of the communication chain."""
        chain = []
        for comm in self.communications:
            chain.append(f"{comm.from_agent} -> {comm.to_agent} ({comm.message_type}): {comm.content}")
        return chain
    
    def get_formatted_summary(self) -> str:
        """Get a formatted summary for user display."""
        if not self.communications:
            return "No inter-agent communications recorded."
        
        summary = f"Inter-Agent Communication Chain (Session: {self.session_id}):\n"
        summary += "=" * 60 + "\n"
        
        for i, comm in enumerate(self.communications, 1):
            summary += f"{i}. {comm.from_agent} -> {comm.to_agent}\n"
            summary += f"   Type: {comm.message_type.upper()}\n"
            summary += f"   Message: {comm.content}\n"
            if comm.context_data:
                summary += f"   Context: {comm.context_data}\n"
            if comm.task_id:
                summary += f"   Task ID: {comm.task_id}\n"
            summary += "\n"
        
        return summary

# --- Core Data Models ---

class AgentResponse(BaseModel):
    """
    A standardized response object returned by every agent after execution.
    This structured response allows the Orchestrator to reliably understand the
    outcome of any task.
    """
    success: bool = Field(..., description="Indicates whether the agent's task was successful.")
    message: str = Field(..., description="A human-readable summary of the outcome.")
    
    # List of artifact keys that this agent's execution produced.
    artifacts_generated: List[str] = Field(default_factory=list, description="A list of keys for artifacts created by the agent.")

class TaskNode(BaseModel):
    """
    Represents a single task in the project plan (TaskGraph). Each node is a discrete
    unit of work to be performed by a specialized agent. The model ensures that every
    task is well-defined and its state is tracked explicitly.
    """
    # We use a default_factory to generate a unique ID for each task, ensuring traceability.
    task_id: str = Field(default_factory=lambda: f"task_{uuid.uuid4().hex[:8]}", description="A unique identifier for the task.")
    goal: str = Field(..., description="The specific, high-level objective for this task.")
    assigned_agent: str = Field(..., description="The name of the agent responsible for executing the task.")
    
    dependencies: List[str] = Field(default_factory=list, description="A list of task_ids that must be successfully completed before this task can start.")
    status: Literal["pending", "running", "success", "failed", "obsolete"] = Field("pending", description="The current execution status of the task.")
    
    # Explicitly defining data dependencies is crucial for debugging data flow.
    input_artifact_keys: List[str] = Field(default_factory=list, description="A list of artifact keys this task requires as input.")
    output_artifact_keys: List[str] = Field(default_factory=list, description="A list of artifact keys this task is expected to produce.")

    # The result is optional because it only exists after a task has been run.
    result: Optional[AgentResponse] = Field(None, description="The final response from the agent after execution.")

class TaskGraph(BaseModel):
    """
    A directed acyclic graph (DAG) representing the entire project plan. It is
    composed of TaskNodes and the dependencies between them. This model acts
    as the central "blueprint" for the mission.
    """
    nodes: Dict[str, TaskNode] = Field(default_factory=dict, description="A dictionary mapping task_ids to TaskNode objects.")
    
    def add_task(self, task: TaskNode):
        """Adds a new task to the graph, ensuring no ID collision."""
        if task.task_id in self.nodes:
            logger.warning(f"Task with ID '{task.task_id}' already exists in the graph. It will be overwritten.")
        self.nodes[task.task_id] = task
        logger.debug(f"Task '{task.task_id}' added to the graph.")

    def get_task(self, task_id: str) -> Optional[TaskNode]:
        """Safely retrieves a task by its ID."""
        return self.nodes.get(task_id)

    def to_json(self) -> str:
        """Serializes the entire task graph to a JSON string for logging or saving."""
        return self.model_dump_json(indent=2)


# --- Self-Testing Block ---
# This block serves as a unit test for our models. It verifies that:
# 1. Models can be instantiated correctly.
# 2. Default values (like unique IDs) are generated as expected.
# 3. Validation catches incorrect data types.
# 4. Serialization to and from dictionaries/JSON works correctly.
# To run this test, execute the file directly: `python core/models.py`
if __name__ == "__main__":
    from utils.logger import setup_logger
    setup_logger()

    print("\n--- Testing Core Data Models ---")

    # 1. Test AgentResponse model
    print("\n[1] Testing AgentResponse...")
    success_response = AgentResponse(success=True, message="Code generated.", artifacts_generated=["main.py.diff"])
    failure_response = AgentResponse(success=False, message="Test suite failed.")
    assert success_response.success is True
    assert failure_response.artifacts_generated == []
    logger.info("AgentResponse tests passed.")
    logger.info(f"Serialized success response: {success_response.model_dump_json(indent=2)}")

    # 2. Test TaskNode model
    print("\n[2] Testing TaskNode...")
    plan_task = TaskNode(goal="Create a plan", assigned_agent="PlannerAgent")
    code_task = TaskNode(
        goal="Implement the login endpoint",
        assigned_agent="CodeGenerationAgent",
        dependencies=[plan_task.task_id],
        input_artifact_keys=["technical_spec.md"]
    )
    assert plan_task.status == "pending"
    assert "task_" in code_task.task_id
    assert code_task.dependencies == [plan_task.task_id]
    logger.info("TaskNode tests passed.")
    logger.info(f"Planner Task ID: {plan_task.task_id}")
    logger.info(f"Coder Task ID: {code_task.task_id}")

    # 3. Test TaskGraph model
    print("\n[3] Testing TaskGraph...")
    mission_plan = TaskGraph()
    mission_plan.add_task(plan_task)
    mission_plan.add_task(code_task)
    assert len(mission_plan.nodes) == 2
    retrieved_task = mission_plan.get_task(plan_task.task_id)
    assert retrieved_task is not None and retrieved_task.goal == "Create a plan"
    logger.info("TaskGraph tests passed.")
    logger.info(f"Full TaskGraph JSON representation:\n{mission_plan.to_json()}")

    # 4. Test Pydantic Validation
    print("\n[4] Testing Pydantic Validation...")
    try:
        # This should fail because 'success' is a required field.
        invalid_response = AgentResponse(message="This will fail")
    except ValidationError as e:
        logger.info("Successfully caught expected validation error.")
        # Pydantic provides detailed, readable error messages.
        logger.debug(f"Validation error details:\n{e}")
    
    try:
        # This should fail because status must be one of the Literal values.
        invalid_task = TaskNode(
            goal="Invalid task",
            assigned_agent="TestAgent",
            status="in_progress" # This is not a valid status
        )
    except ValidationError as e:
        logger.info("Successfully caught expected validation error on Literal field.")
        logger.debug(f"Validation error details:\n{e}")

    print("\n--- All Model Tests Passed Successfully ---")