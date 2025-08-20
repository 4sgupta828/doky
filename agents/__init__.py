# agents/__init__.py
import logging
from typing import Dict, Type, List, Any

# --- Agent Class Imports ---
# Import the base contract that all agents adhere to.
from .base import BaseAgent

# Import every concrete agent implementation from its respective module.
from .planner import PlannerAgent
from .clarifier import ClarifierAgent
from .spec_generator import SpecGeneratorAgent
from .code_manifest import CodeManifestAgent
from .coder import CoderAgent
from .test_generator import TestGenerationAgent
from .test_runner import TestRunnerAgent
from .tooling import ToolingAgent
from .debugging import DebuggingAgent
from .script_executor import ScriptExecutorAgent
from .quality_officer import QualityOfficerAgent
from .plan_refiner import PlanRefinementAgent
from .requirements_manager import RequirementsManagerAgent
from .cli_test_generator import CLITestGeneratorAgent
from .execution_validator import ExecutionValidatorAgent

# Intelligence Layer Agents - Phase 1 Implementation
from .workflow_adapter import WorkflowAdapterAgent

# New Agent Architecture - Aligned with 21-Agent Plan
# Analysis Tier (Read-only)
from .code_analysis import CodeAnalysisAgent
from .environment_analysis import EnvironmentAnalysisAgent
from .problem_analysis import ProblemAnalysisAgent

# Specialized Tier (Write-only)
# Note: CodeModifierAgent was removed as it contained validation logic, not modification logic
from .documentation import DocumentationAgent

# Infrastructure Tier (System operations)
from .file_system import FileSystemAgent
from .environment_modifier import EnvironmentModifierAgent
from .dependency_modifier import DependencyModifierAgent
from .configuration_modifier import ConfigurationModifierAgent

# Coordination Tier (Orchestrators)
from .requirements_orchestrator import RequirementsOrchestratorAgent
from .development_orchestrator import DevelopmentOrchestratorAgent
from .debugging_orchestrator import DebuggingOrchestratorAgent

# Foundational Agents - New Architecture
from .executor import ExecutorAgent

# Get a logger instance for this module.
logger = logging.getLogger(__name__)


# --- Central Agent Registry ---
# This registry maps a string name to the actual agent class.
# It is the single source of truth for all available agents in the system.
AGENT_REGISTRY: Dict[str, Type[BaseAgent]] = {
    # Intelligence Layer - Phase 1 Implementation
    # "MasterIntelligenceAgent": MasterIntelligenceAgent,  # File does not exist
    "WorkflowAdapterAgent": WorkflowAdapterAgent,

    # Analysis Tier - Read-only analysis agents
    "CodeAnalysisAgent": CodeAnalysisAgent,
    "EnvironmentAnalysisAgent": EnvironmentAnalysisAgent,
    "ProblemAnalysisAgent": ProblemAnalysisAgent,
    
    # Specialized Tier - Write-only modification agents
    # Note: CodeModifierAgent removed - contained validation logic, not modification
    # "TestModifierAgent": TestModifierAgent,  # File does not exist
    "DocumentationAgent": DocumentationAgent,
    
    # Infrastructure Tier - System operation agents
    "FileSystemAgent": FileSystemAgent,
    "EnvironmentModifierAgent": EnvironmentModifierAgent,
    "DependencyModifierAgent": DependencyModifierAgent,
    "ConfigurationModifierAgent": ConfigurationModifierAgent,
    
    # Coordination Tier - Orchestration agents
    "RequirementsOrchestratorAgent": RequirementsOrchestratorAgent,
    "DevelopmentOrchestratorAgent": DevelopmentOrchestratorAgent,
    "DebuggingOrchestratorAgent": DebuggingOrchestratorAgent,

    # Foundational Agents - New Architecture
    "ExecutorAgent": ExecutorAgent,

    # Foundational Agents
    "PlannerAgent": PlannerAgent,
    "PlanRefinementAgent": PlanRefinementAgent,     
    "ClarifierAgent": ClarifierAgent,

    # Architecture & Design Agents
    "SpecGeneratorAgent": SpecGeneratorAgent,
    "CodeManifestAgent": CodeManifestAgent,

    # Development & Testing Agents
    "CoderAgent": CoderAgent,
    "TestGenerationAgent": TestGenerationAgent,
    "TestRunnerAgent": TestRunnerAgent,
    "RequirementsManagerAgent": RequirementsManagerAgent,
    "CLITestGeneratorAgent": CLITestGeneratorAgent,
    "ExecutionValidatorAgent": ExecutionValidatorAgent,

    # Research & Environment Agents
    # "ContextBuilderAgent": ContextBuilderAgent,  # File does not exist
    "ToolingAgent": ToolingAgent,

    # Diagnostics & Quality Agents
    "DebuggingAgent": DebuggingAgent,
    "ScriptExecutorAgent": ScriptExecutorAgent,
    "QualityOfficerAgent": QualityOfficerAgent,
}

# NEW: User-friendly aliases for direct invocation.
AGENT_ALIASES: Dict[str, str] = {
    "@help": "HelpAgent", # A virtual agent for the help command
    # Intelligence Layer
    # "@intelligence": "MasterIntelligenceAgent",  # File does not exist
    "@workflow": "WorkflowAdapterAgent",
    # Analysis Tier
    "@code-analysis": "CodeAnalysisAgent",
    "@env-analysis": "EnvironmentAnalysisAgent",
    "@problem-analysis": "ProblemAnalysisAgent",
    # Specialized Tier
    # Note: @code-modifier alias removed with CodeModifierAgent
    # "@test-modifier": "TestModifierAgent",  # File does not exist
    "@docs": "DocumentationAgent",
    # Infrastructure Tier
    "@filesystem": "FileSystemAgent",
    "@env-modifier": "EnvironmentModifierAgent",
    "@dependency": "DependencyModifierAgent",
    "@config": "ConfigurationModifierAgent",
    # Coordination Tier
    "@requirements-orchestrator": "RequirementsOrchestratorAgent",
    "@development-orchestrator": "DevelopmentOrchestratorAgent",
    "@debugging-orchestrator": "DebuggingOrchestratorAgent",
    # Foundational Agents - New Architecture
    "@executor": "ExecutorAgent",
    # Foundational & Planning
    "@planner": "PlannerAgent",
    "@refiner": "PlanRefinementAgent",
    "@clarify": "ClarifierAgent",
    "@spec": "SpecGeneratorAgent",
    "@manifest": "CodeManifestAgent",
    # Development & Testing
    "@coder": "CoderAgent",
    "@testgen": "TestGenerationAgent",
    "@tester": "TestRunnerAgent",
    "@requirements": "RequirementsManagerAgent",
    "@cli-test": "CLITestGeneratorAgent", 
    "@validate": "ExecutionValidatorAgent",
    # Environment & Diagnostics
    # "@context": "ContextBuilderAgent",  # File does not exist
    "@run": "ToolingAgent",
    "@debug": "DebuggingAgent",
    "@script": "ScriptExecutorAgent",
    "@audit": "QualityOfficerAgent",
}

def get_agent_help() -> str:
    """Generates a formatted help string of all available commands."""
    help_text = "Available commands:\n"
    # We need to get the description from an instance of each agent
    for alias, agent_name in AGENT_ALIASES.items():
        if agent_name in AGENT_REGISTRY:
            # Briefly instantiate the agent to get its description
            agent_instance = AGENT_REGISTRY[agent_name]()
            help_text += f"  {alias:<12} - {agent_instance.description}\n"
    return help_text

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
            self.assertEqual(len(AGENT_REGISTRY), 27)  # Updated: added ExecutorAgent
            self.assertIn("PlannerAgent", AGENT_REGISTRY)
            self.assertIn("CoderAgent", AGENT_REGISTRY)
            self.assertIn("TestRunnerAgent", AGENT_REGISTRY)
            # Check new intelligence agents
            # self.assertIn("MasterIntelligenceAgent", AGENT_REGISTRY)  # File does not exist
            self.assertIn("WorkflowAdapterAgent", AGENT_REGISTRY)
            # Check new tier agents
            self.assertIn("CodeAnalysisAgent", AGENT_REGISTRY)
            self.assertIn("EnvironmentAnalysisAgent", AGENT_REGISTRY)
            self.assertIn("ProblemAnalysisAgent", AGENT_REGISTRY)
            # self.assertIn("CodeModifierAgent", AGENT_REGISTRY)  # Removed
            # self.assertIn("TestModifierAgent", AGENT_REGISTRY)  # File does not exist
            self.assertIn("EnvironmentModifierAgent", AGENT_REGISTRY)
            self.assertIn("DependencyModifierAgent", AGENT_REGISTRY)
            self.assertIn("ConfigurationModifierAgent", AGENT_REGISTRY)
            logger.info(f"✅ Registry contains {len(AGENT_REGISTRY)} agents as expected.")

        def test_standard_agent_retrieval(self):
            """Tests successful retrieval and instantiation of a standard agent."""
            print("\n--- [Test Case 2: Standard Agent Retrieval] ---")
            try:
                coder_instance = get_agent("CoderAgent")
                self.assertIsInstance(coder_instance, BaseAgent)
                self.assertEqual(coder_instance.name, "CoderAgent")
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