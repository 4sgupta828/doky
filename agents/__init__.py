# agents/__init__.py
from .base import BaseAgent
from .planner import PlannerAgent
from .coder import CodeGenerationAgent
from .tester import TestGenerationAgent, TestRunnerAgent

# The agent registry is a central dictionary that maps agent names to their
# class implementations. This allows the Orchestrator to dynamically load and
# invoke agents by name without hardcoding them.
AGENT_REGISTRY = {
    "PlannerAgent": PlannerAgent,
    "CodeGenerationAgent": CodeGenerationAgent,
    "TestGenerationAgent": TestGenerationAgent,
    "TestRunnerAgent": TestRunnerAgent,
    # Add other agents here as they are implemented...
}

def get_agent(agent_name: str) -> BaseAgent:
    """
    Factory function to retrieve an agent instance from the registry.

    Args:
        agent_name: The name of the agent to retrieve.

    Returns:
        An instance of the requested agent class.
    """
    agent_class = AGENT_REGISTRY.get(agent_name)
    if not agent_class:
        raise ValueError(f"Unknown agent: {agent_name}")
    return agent_class()