# fagents/__init__.py
import logging
from typing import Dict, Type

# Foundational Agent Imports
from .base import FoundationalAgent
from .analyst import AnalystAgent
from .strategist import StrategistAgent

# Get a logger instance for this module
logger = logging.getLogger(__name__)

# Foundational Agent Registry
FOUNDATIONAL_AGENT_REGISTRY: Dict[str, Type[FoundationalAgent]] = {
    "AnalystAgent": AnalystAgent,
    "StrategistAgent": StrategistAgent,
}

# User-friendly aliases for foundational agents
FOUNDATIONAL_AGENT_ALIASES: Dict[str, str] = {
    "@analyst": "AnalystAgent",
    "@strategist": "StrategistAgent",
}

def get_foundational_agent(agent_name: str) -> FoundationalAgent:
    """
    Factory function to retrieve a foundational agent instance.
    
    Args:
        agent_name: The string name of the foundational agent to retrieve.
        
    Returns:
        An initialized instance of the requested foundational agent class.
        
    Raises:
        ValueError: If the requested agent name is not found in the registry.
    """
    agent_class = FOUNDATIONAL_AGENT_REGISTRY.get(agent_name)
    if not agent_class:
        error_msg = f"Unknown foundational agent: '{agent_name}'. Available: {list(FOUNDATIONAL_AGENT_REGISTRY.keys())}"
        logger.error(error_msg)
        raise ValueError(error_msg)
        
    logger.debug(f"Instantiating foundational agent: {agent_name}")
    return agent_class()