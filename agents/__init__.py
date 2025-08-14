# agents/__init__.py
import logging
from typing import Dict, Type, List, Any

# --- Agent Class Imports ---
# Import the base contract that all agents adhere to.
from .base import BaseAgent

# Import every concrete agent implementation from its respective module.
from .planner import PlannerAgent
from .clarifier import IntentClarificationAgent
from .spec_generator import SpecGenerationAgent
from .code_manifest import CodeManifestAgent
from .coder import CodeGenerationAgent
from .test_generator import TestGenerationAgent
from .test_runner import TestRunnerAgent
from .context_builder import ContextBuilderAgent
from .tooling import ToolingAgent
from .debugging import DebuggingAgent
from .quality_officer import ChiefQualityOfficerAgent
from .plan_refiner import PlanRefinementAgent

# Get a logger instance for this module.
logger = logging.getLogger(__name__)


# --- Central Agent Registry ---
# This registry maps a string name to the actual agent class.
# It is the single source of truth for all available agents in the system.
AGENT_REGISTRY: Dict[str, Type[BaseAgent]] = {
    # Foundational Agents
    "PlannerAgent": PlannerAgent,
    "PlanRefinementAgent": PlanRefinementAgent,     
    "IntentClarificationAgent": IntentClarificationAgent,

    # Architecture & Design Agents
    "SpecGenerationAgent": SpecGenerationAgent,
    "CodeManifestAgent": CodeManifestAgent,

    # Development & Testing Agents
    "CodeGenerationAgent": CodeGenerationAgent,
    "TestGenerationAgent": TestGenerationAgent,
    "TestRunnerAgent": TestRunnerAgent,

    # Research & Environment Agents
    "ContextBuilderAgent": ContextBuilderAgent,
    "ToolingAgent": ToolingAgent,

    # Diagnostics & Quality Agents
    "DebuggingAgent": DebuggingAgent,
    "ChiefQualityOfficerAgent": ChiefQualityOfficerAgent,
}


# --- Agent Factory Function ---
def get_agent(agent_name: str, agent_capabilities: List[Dict[str, Any]] = None) -> BaseAgent:
    """
    Factory function to retrieve an agent instance from the registry.

    This function acts as a controlled gateway for creating agent instances.
    It decouples the Orchestrator from the specifics of how each agent is
    instantiated, including special cases like the PlannerAgent.

    Args:
        agent_name: The string name of the agent to retrieve.
        agent_capabilities: A list of capabilities from all other agents.
                            This is specifically required by the PlannerAgent.

    Returns:
        An initialized instance of the requested agent class.

    Raises:
        ValueError: If the requested agent name is not found in the registry.
    """
    agent_class = AGENT_REGISTRY.get(agent_name)
    if not agent_class:
        error_msg = f"Unknown agent: '{agent_name}'. It is not registered in AGENT_REGISTRY."
        logger.error(error_msg)
        raise ValueError(error_msg)

    logger.debug(f"Instantiating agent: {agent_name}")

    # Special handling for the PlannerAgent, which needs to know about all
    # other agents to create effective plans.
    if agent_name == "PlannerAgent":
        return agent_class(agent_capabilities=agent_capabilities)

    # For all other agents, instantiate without special arguments.
    return agent_class()


# --- Self-Testing Block ---
# This block verifies that the registry is correctly populated and that the
# factory function works as expected for both standard and special cases.
if __name__ == "__main__":
    import unittest
    from utils.logger import setup_logger

    setup_logger(default_level=logging.INFO)

    class TestAgentRegistry(unittest.TestCase):

        def test_registry_population(self):
            """Tests if the registry contains all the expected agents."""
            print("\n--- [Test Case 1: Registry Population] ---")
            self.assertEqual(len(AGENT_REGISTRY), 11)
            self.assertIn("PlannerAgent", AGENT_REGISTRY)
            self.assertIn("CodeGenerationAgent", AGENT_REGISTRY)
            self.assertIn("TestRunnerAgent", AGENT_REGISTRY)
            logger.info(f"✅ Registry contains {len(AGENT_REGISTRY)} agents as expected.")

        def test_standard_agent_retrieval(self):
            """Tests successful retrieval and instantiation of a standard agent."""
            print("\n--- [Test Case 2: Standard Agent Retrieval] ---")
            try:
                coder_instance = get_agent("CodeGenerationAgent")
                self.assertIsInstance(coder_instance, BaseAgent)
                self.assertEqual(coder_instance.name, "CodeGenerationAgent")
                logger.info("✅ Successfully retrieved a standard agent instance.")
            except Exception as e:
                self.fail(f"Standard agent retrieval failed unexpectedly: {e}")

        def test_planner_agent_retrieval(self):
            """Tests the special case for retrieving the PlannerAgent with capabilities."""
            print("\n--- [Test Case 3: PlannerAgent Retrieval] ---")
            try:
                # Create a mock capabilities list for the test
                mock_caps = [{"name": "Coder", "description": "Writes code."}]
                planner_instance = get_agent("PlannerAgent", agent_capabilities=mock_caps)
                self.assertIsInstance(planner_instance, PlannerAgent)
                # Verify that the capabilities were passed to the instance
                self.assertEqual(planner_instance.agent_capabilities, mock_caps)
                logger.info("✅ Successfully retrieved PlannerAgent with capabilities.")
            except Exception as e:
                self.fail(f"PlannerAgent retrieval failed unexpectedly: {e}")

        def test_unknown_agent_failure(self):
            """Tests that retrieving an unregistered agent raises a ValueError."""
            print("\n--- [Test Case 4: Unknown Agent Failure] ---")
            with self.assertRaises(ValueError) as context:
                get_agent("NonExistentAgent")
            self.assertIn("Unknown agent", str(context.exception))
            logger.info("✅ Correctly raised ValueError for an unknown agent.")

    unittest.main(argv=['first-arg-is-ignored'], exit=False)