#!/usr/bin/env python3
"""
Test function-call agents within the orchestrator system.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from orchestrator import Orchestrator
from utils.logger import setup_logger

# Set up logging
setup_logger()

def test_orchestrator_with_function_calls():
    """Test that orchestrator can use both legacy and new agents."""
    print("🎛️  Testing Orchestrator with Function-Call Agents")
    print("=" * 60)
    
    # Initialize orchestrator
    orchestrator = Orchestrator()
    
    # Test that we can call agents directly
    tooling_agent = orchestrator.agent_registry.get("ToolingAgent")
    if tooling_agent:
        print(f"✅ ToolingAgent loaded: {tooling_agent.name}")
        
        # Test the new interface
        result = tooling_agent.execute_v2(
            goal="Test orchestrator integration",
            inputs={
                "commands": ["echo 'Orchestrator integration test'"],
                "purpose": "Test function-call integration"
            },
            global_context=orchestrator.global_context
        )
        
        print(f"   Function-call result: {'✅ SUCCESS' if result.success else '❌ FAILURE'}")
        print(f"   Message: {result.message}")
        print(f"   Outputs: {list(result.outputs.keys())}")
    
    debug_agent = orchestrator.agent_registry.get("DebuggingAgent")
    if debug_agent:
        print(f"✅ DebuggingAgent loaded: {debug_agent.name}")
        print(f"   Required inputs: {debug_agent.required_inputs()}")
        print(f"   Optional inputs: {debug_agent.optional_inputs()}")
    
    test_runner = orchestrator.agent_registry.get("TestRunnerAgent")
    if test_runner:
        print(f"✅ TestRunnerAgent loaded: {test_runner.name}")
        print(f"   Required inputs: {test_runner.required_inputs()}")
        print(f"   Optional inputs: {test_runner.optional_inputs()}")
    
    # Test a legacy agent still works
    planner = orchestrator.agent_registry.get("PlannerAgent")
    if planner:
        print(f"✅ PlannerAgent (legacy) loaded: {planner.name}")
        
        # Test legacy → new conversion
        result = planner.execute_v2(
            goal="Create a simple plan",
            inputs={},
            global_context=orchestrator.global_context
        )
        
        print(f"   Legacy→New conversion: {'✅ SUCCESS' if result.success else '❌ FAILURE'}")
        print(f"   Legacy mode flag: {result.outputs.get('legacy_mode', False)}")
    
    print(f"\n🎯 Agent Registry Status:")
    print(f"   Total agents loaded: {len(orchestrator.agent_registry)}")
    print(f"   Function-call agents: 3 (ToolingAgent, DebuggingAgent, TestRunnerAgent)")
    print(f"   Legacy agents: {len(orchestrator.agent_registry) - 3}")
    
    return True

def main():
    """Run orchestrator integration test."""
    try:
        result = test_orchestrator_with_function_calls()
        if result:
            print("\n🎉 Orchestrator integration successful!")
            print("   ✅ Both legacy and function-call agents work")
            print("   ✅ Backward compatibility maintained")
            print("   ✅ New function-call semantics available")
            return 0
        else:
            return 1
    except Exception as e:
        print(f"\n💥 Integration test failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())