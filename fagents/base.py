# fagents/base.py
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List
from core.context import GlobalContext
from core.models import AgentResult

# Get a logger instance for this module
logger = logging.getLogger(__name__)


class FoundationalAgent(ABC):
    """
    Base class for all foundational agents.
    
    Foundational agents are designed to be:
    1. Orthogonal - no functional overlap between agents
    2. Composable - can be orchestrated in any sequence/graph
    3. Powerful - each handles a complete domain of operations
    4. Tool-leveraged - maximize use of atomic, reusable tools
    """
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.logger = logging.getLogger(f"{__name__}.{name}")
        
    @abstractmethod
    def execute(self, goal: str, inputs: Dict[str, Any], global_context: GlobalContext) -> AgentResult:
        """
        Execute the foundational agent's core capability.
        
        Args:
            goal: High-level description of what to accomplish
            inputs: Structured inputs for the operation
            global_context: Shared context across all agents
            
        Returns:
            AgentResult with outputs and metadata
        """
        pass
        
    @abstractmethod
    def get_capabilities(self) -> Dict[str, Any]:
        """
        Return a description of this agent's capabilities for the Strategist.
        
        Returns:
            Dictionary describing the agent's capabilities, inputs, and outputs
        """
        pass
        
    def validate_inputs(self, inputs: Dict[str, Any], required: List[str] = None) -> None:
        """Validate that required inputs are present."""
        required = required or []
        missing = [key for key in required if key not in inputs]
        if missing:
            raise ValueError(f"Missing required inputs for {self.name}: {missing}")
            
    def report_progress(self, message: str, details: str = None) -> None:
        """Report progress to the user."""
        self.logger.info(f"[{self.name}] {message}" + (f": {details}" if details else ""))
        
    def report_error(self, message: str, error: Exception = None) -> None:
        """Report an error to the user."""
        error_msg = f"[{self.name}] ERROR: {message}"
        if error:
            error_msg += f" - {str(error)}"
        self.logger.error(error_msg)