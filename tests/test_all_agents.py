#!/usr/bin/env python3
"""
Comprehensive test for all agents in the Sovereign Agent Collective.
This test uses REAL LLM integration to verify all agents work end-to-end.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the agent system components
from agents import AGENT_REGISTRY, BaseAgent, get_agent
from core.context import GlobalContext
from core.models import TaskNode, AgentResponse
from utils.logger import setup_logger
from real_llm_client import create_llm_client

# Configure logging for the test
setup_logger(default_level=logging.INFO)
logger = logging.getLogger(__name__)


def check_llm_availability():
    """Check if LLM API keys are available."""
    import os
    
    # Try to load from .env file
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        # If python-dotenv not available, try manual loading
        env_file = Path(".env")
        if env_file.exists():
            for line in env_file.read_text().strip().split('\n'):
                if '=' in line and not line.startswith('#'):
                    key, value = line.split('=', 1)
                    os.environ[key.strip()] = value.strip()
    
    has_openai = bool(os.getenv("OPENAI_API_KEY"))
    has_anthropic = bool(os.getenv("ANTHROPIC_API_KEY"))
    
    if not (has_openai or has_anthropic):
        print("❌ No LLM API keys found!")
        print("   Set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable")
        return False
    
    provider = "OpenAI" if has_openai else "Anthropic"
    print(f"✅ Using {provider} for LLM integration")
    return True


def create_real_agent(agent_name: str, llm_client: Any) -> BaseAgent:
    """Create a real agent instance with LLM integration."""
    agent_class = AGENT_REGISTRY[agent_name]
    
    if agent_name == "PlannerAgent":
        # PlannerAgent needs capabilities of other agents
        all_capabilities = []
        for name, cls in AGENT_REGISTRY.items():
            if name != "PlannerAgent":
                temp_agent = cls()  # Create temp instance to get capabilities
                all_capabilities.append(temp_agent.get_capabilities())
        
        return agent_class(agent_capabilities=all_capabilities, llm_client=llm_client)
    else:
        # Most agents accept llm_client parameter
        try:
            return agent_class(llm_client=llm_client)
        except TypeError:
            # Some agents might not accept llm_client
            return agent_class()


def test_agent_instantiation_with_llm():
    """Test that all agents can be instantiated with real LLM clients."""
    print("\n=== Testing Real Agent Instantiation ===")
    
    try:
        llm_client = create_llm_client()
        print(f"Created {llm_client.provider} LLM client with model {llm_client.model}")
    except Exception as e:
        print(f"❌ Failed to create LLM client: {e}")
        return {}, [(agent_name, f"LLM client creation failed: {e}") for agent_name in AGENT_REGISTRY]
    
    instantiated_agents = {}
    failed_agents = []
    
    for agent_name in AGENT_REGISTRY:
        try:
            print(f"Instantiating {agent_name} with real LLM...")
            agent = create_real_agent(agent_name, llm_client)
            instantiated_agents[agent_name] = agent
            print(f"✅ {agent_name} instantiated successfully")
            
        except Exception as e:
            failed_agents.append((agent_name, str(e)))
            print(f"❌ {agent_name} failed to instantiate: {e}")
    
    return instantiated_agents, failed_agents


def test_agent_execution(agents: Dict[str, BaseAgent]):
    """Test that all agents can execute without crashing."""
    print("\n=== Testing Agent Execution ===")
    
    # Create test context and workspace
    test_context = GlobalContext()
    test_workspace = Path("./test_workspace_temp")
    test_workspace.mkdir(exist_ok=True)
    test_context.workspace.repo_path = test_workspace
    
    # Add some basic artifacts that agents might need
    test_context.add_artifact("technical_spec", "Build a simple Python calculator with add/subtract functions", "initial")
    test_context.add_artifact("user_requirements", "I need a calculator that can do basic math", "initial")
    
    # Create a simple test file for agents that need to read code
    test_file = test_workspace / "sample.py"
    test_file.write_text("""def add(a, b):
    return a + b

def subtract(a, b)
    return a - b  # Missing colon - syntax error!
""")
    test_context.add_artifact("sample_code", str(test_file), "test_file")
    
    execution_results = {}
    failed_executions = []
    
    # Define simple, realistic test goals for each agent type
    test_goals = {
        "PlannerAgent": "Create a simple todo app",
        "IntentClarificationAgent": "I want to build an app, but I'm not sure what kind",
        "SpecGenerationAgent": "Build a simple calculator application", 
        "CodeManifestAgent": "Python calculator with add/subtract functions",
        "CodeGenerationAgent": "Write a Python function that adds two numbers",
        "TestGenerationAgent": "Create tests for an add function",
        "TestRunnerAgent": "Analyze test results for a Python project",
        "ContextBuilderAgent": "Find relevant code examples for a calculator",
        "ToolingAgent": "Install required packages for a Python project",
        "DebuggingAgent": "Fix a syntax error in Python code",
        "ChiefQualityOfficerAgent": "Review code quality for a simple function"
    }
    
    for agent_name, agent in agents.items():
        try:
            print(f"Executing {agent_name}...")
            
            # Create a test task for this agent
            test_task = TaskNode(
                task_id=f"test_{agent_name.lower()}",
                goal=test_goals.get(agent_name, f"Test task for {agent_name}"),
                assigned_agent=agent_name
            )
            
            # Execute the agent
            response = agent.execute(
                goal=test_task.goal,
                context=test_context,
                current_task=test_task
            )
            
            # Verify the response is valid
            assert isinstance(response, AgentResponse), f"Invalid response type from {agent_name}"
            assert hasattr(response, 'success'), f"Response missing 'success' field from {agent_name}"
            assert hasattr(response, 'message'), f"Response missing 'message' field from {agent_name}"
            
            execution_results[agent_name] = response
            status = "✅" if response.success else "⚠️"
            print(f"{status} {agent_name} executed: {response.message[:100]}...")
            
        except Exception as e:
            failed_executions.append((agent_name, str(e)))
            print(f"❌ {agent_name} execution failed: {e}")
    
    # Cleanup
    import shutil
    if test_workspace.exists():
        shutil.rmtree(test_workspace)
    
    return execution_results, failed_executions


def test_agent_capabilities():
    """Test that all agents report their capabilities correctly."""
    print("\n=== Testing Agent Capabilities ===")
    
    capabilities_results = {}
    failed_capabilities = []
    
    for agent_name in AGENT_REGISTRY:
        try:
            print(f"Testing capabilities for {agent_name}...")
            
            if agent_name == "PlannerAgent":
                mock_capabilities = [
                    {"name": name, "description": f"Mock description for {name}"}
                    for name in AGENT_REGISTRY if name != "PlannerAgent"
                ]
                agent = get_agent(agent_name, agent_capabilities=mock_capabilities)
            else:
                agent = get_agent(agent_name)
            
            capabilities = agent.get_capabilities()
            
            # Verify capabilities structure
            assert isinstance(capabilities, dict), f"Capabilities not a dict for {agent_name}"
            assert 'name' in capabilities, f"Missing 'name' in capabilities for {agent_name}"
            assert 'description' in capabilities, f"Missing 'description' in capabilities for {agent_name}"
            assert capabilities['name'] == agent_name, f"Name mismatch in capabilities for {agent_name}"
            
            capabilities_results[agent_name] = capabilities
            print(f"✅ {agent_name} capabilities: {capabilities['description'][:60]}...")
            
        except Exception as e:
            failed_capabilities.append((agent_name, str(e)))
            print(f"❌ {agent_name} capabilities test failed: {e}")
    
    return capabilities_results, failed_capabilities


def main():
    """Run all agent tests with real LLM integration."""
    print("🚀 Starting comprehensive REAL agent testing...")
    print(f"Testing {len(AGENT_REGISTRY)} agents: {list(AGENT_REGISTRY.keys())}")
    
    # Check if we have LLM API keys available
    if not check_llm_availability():
        print("\n❌ Cannot proceed without LLM API keys")
        return 1
    
    # Test 1: Agent instantiation with real LLM
    agents, instantiation_failures = test_agent_instantiation_with_llm()
    
    # Test 2: Agent capabilities  
    capabilities, capability_failures = test_agent_capabilities()
    
    # Test 3: Agent execution with real LLM (only for successfully instantiated agents)
    execution_results, execution_failures = test_agent_execution(agents)
    
    # Print summary
    print("\n" + "="*60)
    print("🏁 TEST SUMMARY")
    print("="*60)
    
    total_agents = len(AGENT_REGISTRY)
    successful_instantiations = len(agents)
    successful_executions = len(execution_results)
    successful_capabilities = len(capabilities)
    
    print(f"📊 Agents tested: {total_agents}")
    print(f"✅ Successful instantiations: {successful_instantiations}/{total_agents}")
    print(f"✅ Successful capability tests: {successful_capabilities}/{total_agents}")
    print(f"✅ Successful executions: {successful_executions}/{successful_instantiations}")
    
    if instantiation_failures:
        print(f"\n❌ Instantiation failures ({len(instantiation_failures)}):")
        for agent_name, error in instantiation_failures:
            print(f"   - {agent_name}: {error}")
    
    if capability_failures:
        print(f"\n❌ Capability test failures ({len(capability_failures)}):")
        for agent_name, error in capability_failures:
            print(f"   - {agent_name}: {error}")
    
    if execution_failures:
        print(f"\n❌ Execution failures ({len(execution_failures)}):")
        for agent_name, error in execution_failures:
            print(f"   - {agent_name}: {error}")
    
    # Determine overall result
    overall_success = (len(instantiation_failures) == 0 and 
                      len(capability_failures) == 0 and 
                      len(execution_failures) == 0)
    
    if overall_success:
        print(f"\n🎉 ALL TESTS PASSED! All {total_agents} agents are working correctly.")
        return 0
    else:
        total_failures = len(instantiation_failures) + len(capability_failures) + len(execution_failures)
        print(f"\n⚠️  SOME TESTS FAILED! {total_failures} issues found.")
        return 1


if __name__ == "__main__":
    sys.exit(main())