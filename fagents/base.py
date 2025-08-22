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
    
    def __init__(self, name: str, description: str, ui_interface: Any = None):
        self.name = name
        self.description = description
        self.ui_interface = ui_interface
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
    
    def report_intermediate_output(self, output_type: str, data: Any) -> None:
        """Report intermediate output during processing."""
        self.logger.info(f"[{self.name}] {output_type}: {type(data).__name__} output generated")
    
    def report_llm_communication(self, prompt: str, response: str) -> None:
        """Report LLM communication for transparency."""
        if self.ui_interface and hasattr(self.ui_interface, 'display_llm_communication'):
            self.ui_interface.display_llm_communication(self.name, prompt, response)
        
        # Also log for debugging
        self.logger.debug(f"[{self.name}] LLM prompt: {prompt[:200]}...")
        self.logger.debug(f"[{self.name}] LLM response: {response[:200]}...")
    
    def report_agent_io(self, goal: str, inputs: Dict[str, Any], result: AgentResult) -> None:
        """Report agent input/output for transparency."""
        if self.ui_interface:
            if hasattr(self.ui_interface, 'display_agent_input'):
                self.ui_interface.display_agent_input(self.name, goal, inputs)
            if hasattr(self.ui_interface, 'display_agent_output'):
                self.ui_interface.display_agent_output(self.name, result.success, result.message, result.outputs)
        
    def create_result(self, success: bool, message: str, outputs: Dict[str, Any] = None) -> AgentResult:
        """Create a standardized AgentResult."""
        return AgentResult(
            success=success,
            message=message,
            outputs=outputs or {}
        )