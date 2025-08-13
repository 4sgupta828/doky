# core/models.py
from dataclasses import dataclass, field
from typing import List, Dict, Any, Literal

# The use of dataclasses provides structure and type safety, which is vital for debugging.
# When an error occurs, we can be confident in the shape of our data.

@dataclass
class AgentResponse:
    """A standardized response object returned by every agent after execution."""
    success: bool
    message: str
    artifacts_generated: List[str] = field(default_factory=list)

@dataclass
class TaskNode:
    """
    Represents a single task in the project plan (TaskGraph). Each node is a discrete
    unit of work to be performed by a specialized agent.
    """
    task_id: str
    goal: str
    assigned_agent: str
    dependencies: List[str] = field(default_factory=list)
    status: Literal["pending", "running", "success", "failed", "obsolete"] = "pending"
    
    # These keys link the task to the data it needs and produces in the GlobalContext.
    # This explicit declaration is crucial for debugging data flow issues.
    input_artifact_keys: List[str] = field(default_factory=list)
    output_artifact_keys: List[str] = field(default_factory=list)

    # Stores the AgentResponse for post-mortem analysis.
    result: Optional[AgentResponse] = None

@dataclass
class TaskGraph:
    """A directed acyclic graph (DAG) representing the entire project plan."""
    nodes: Dict[str, TaskNode] = field(default_factory=dict)
    
    def add_task(self, task: TaskNode):
        """Adds a new task to the graph."""
        self.nodes[task.task_id] = task