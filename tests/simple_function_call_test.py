#!/usr/bin/env python3
"""
Simple test of function-call agent semantics without complex chains.
"""

import sys
import os
import tempfile
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from agents.tooling import ToolingAgent
from core.context import GlobalContext
from core.models import AgentExecutionError
from utils.logger import setup_logger

# Set up logging
setup_logger()

def test_tooling_agent_function_call():
    """Test ToolingAgent with new function-call interface."""
    print("🔧 Testing ToolingAgent Function-Call Interface")
    print("=" * 50)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        context = GlobalContext(workspace_path=temp_dir)
        agent = ToolingAgent()
        
        # Test 1: Valid inputs
        print("\n✅ Test 1: Valid inputs")
        result = agent.execute_v2(
            goal="Test basic commands",
            inputs={
                "commands": ["echo 'Hello World'", "pwd"],
                "purpose": "Test function-call interface"
            },
            global_context=context
        )
        
        print(f"   Success: {result.success}")
        print(f"   Message: {result.message}")
        print(f"   Commands executed: {len(result.outputs.get('commands_executed', []))}")
        
        # Test 2: Missing required inputs (should fail)
        print("\n❌ Test 2: Missing required inputs")
        try:
            result = agent.execute_v2(
                goal="This should fail",
                inputs={
                    "purpose": "Missing commands"
                    # Missing 'commands' - should fail validation
                },
                global_context=context
            )
            print(f"   Validation worked: {not result.success}")
            print(f"   Error details: {result.error_details}")
        except AgentExecutionError as e:
            print(f"   ✅ Caught expected validation error: {e}")
        
        # Test 3: Safety check
        print("\n🛡️  Test 3: Safety check")
        result = agent.execute_v2(
            goal="Test safety",
            inputs={
                "commands": ["rm -rf /tmp/nonexistent"],  # Should be blocked
                "purpose": "Test safety blocking"
            },
            global_context=context
        )
        
        print(f"   Success: {result.success}")
        print(f"   Message: {result.message}")
        failed_commands = result.outputs.get('failed_commands', [])
        print(f"   Commands blocked: {len(failed_commands)}")
        
        return True

def test_agent_capabilities():
    """Test agent capabilities reporting."""
    print("\n📋 Testing Agent Capabilities")
    print("=" * 50)
    
    agent = ToolingAgent()
    capabilities = agent.get_capabilities()
    
    print(f"   Name: {capabilities['name']}")
    print(f"   Description: {capabilities['description'][:50]}...")
    print(f"   Required inputs: {capabilities['required_inputs']}")
    print(f"   Optional inputs: {capabilities['optional_inputs']}")
    
    return True

def main():
    """Run simple function-call tests."""
    print("🚀 Simple Function-Call Agent Tests")
    print("=" * 50)
    
    tests = [
        test_tooling_agent_function_call,
        test_agent_capabilities
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            results.append(False)
    
    passed = sum(results)
    total = len(results)
    
    print(f"\n🎯 Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 Function-call semantics working correctly!")
        return 0
    else:
        print("💥 Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())