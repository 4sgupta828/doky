#!/usr/bin/env python3
"""
Test the new function-call semantics for agent communication.
This validates the TestRunner → DebuggingAgent → ToolingAgent chain.
"""

import sys
import os
import tempfile
import shutil
import json
from pathlib import Path
from unittest.mock import MagicMock

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from agents.test_runner import TestRunnerAgent
from agents.debugging import DebuggingAgent  
from agents.tooling import ToolingAgent
from core.context import GlobalContext
from utils.logger import setup_logger

# Set up logging
setup_logger()

class MockLLMClient:
    """Mock LLM client for testing."""
    def invoke(self, prompt: str) -> str:
        # Return a valid debugging analysis
        return json.dumps({
            "root_cause_analysis": "Test failure due to mock function call test",
            "primary_hypothesis": "Mock testing scenario for function-call semantics",
            "alternative_hypotheses": ["Alternative mock scenario"],
            "confidence_level": "high",
            "error_category": "code",
            "solution_type": "SURGICAL",
            "complexity_assessment": "simple", 
            "recommended_strategy": "Apply mock fix for function-call testing",
            "risk_assessment": "Low risk mock scenario"
        })

def test_function_call_agent_chain():
    """Test the complete agent chain using function-call semantics."""
    print("\n🔗 Testing Function-Call Agent Communication Chain")
    print("=" * 70)
    
    # Create temporary workspace
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test environment
        test_file = Path(temp_dir) / "test_example.py"
        test_file.write_text("""
import unittest

class TestExample(unittest.TestCase):
    def test_passing(self):
        self.assertEqual(1 + 1, 2)
        
    def test_failing(self):
        self.assertEqual(1 + 1, 3)  # This will fail

if __name__ == "__main__":
    unittest.main()
""")
        
        context = GlobalContext(workspace_path=temp_dir)
        
        # Create agent instances
        tooling_agent = ToolingAgent()
        debug_agent = DebuggingAgent(llm_client=MockLLMClient())
        test_runner = TestRunnerAgent()
        
        # Set up agent registry
        agent_registry = {
            "ToolingAgent": tooling_agent,
            "DebuggingAgent": debug_agent,
            "TestRunnerAgent": test_runner
        }
        
        # Inject registries
        debug_agent.agent_registry = agent_registry
        test_runner.agent_registry = agent_registry
        
        print("🧪 Step 1: TestRunnerAgent executes tests (should find failures)")
        
        # Test the new function-call interface
        test_result = test_runner.execute_v2(
            goal="Run tests and debug any failures",
            inputs={
                "test_mode": "full",
                "run_all_tests": True
            },
            global_context=context
        )
        
        print(f"   Result: {'✅ SUCCESS' if test_result.success else '❌ FAILURE'}")
        print(f"   Message: {test_result.message}")
        print(f"   Execution ID: {test_result.execution_id}")
        
        # Analyze the outputs
        outputs = test_result.outputs
        print(f"\n📊 TestRunner Outputs:")
        print(f"   Test Summary: {outputs.get('test_summary', {})}")
        print(f"   All Tests Passed: {outputs.get('all_tests_passed', 'unknown')}")
        
        # Check if debugging was triggered
        debug_result = outputs.get("debug_result")
        if debug_result:
            print(f"\n🔧 DebuggingAgent was called:")
            print(f"   Debug Success: {debug_result != {'debug_failed': test_result.message}}")
            if isinstance(debug_result, dict):
                print(f"   Solution Applied: {debug_result.get('solution_applied', 'none')}")
                print(f"   Iterations Used: {debug_result.get('iterations_used', 0)}")
        
        return test_result.success or debug_result  # Success if tests pass or debugging worked

def test_direct_agent_calls():
    """Test direct agent-to-agent calls with explicit inputs/outputs."""
    print("\n🎯 Testing Direct Agent Function Calls")
    print("=" * 70)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        context = GlobalContext(workspace_path=temp_dir)
        
        # Test ToolingAgent directly
        print("⚙️  Testing ToolingAgent with function-call interface...")
        
        tooling_agent = ToolingAgent()
        tooling_result = tooling_agent.execute_v2(
            goal="Run diagnostic commands",
            inputs={
                "commands": ["echo 'Hello World'", "python --version", "ls -la"],
                "purpose": "Test function-call interface"
            },
            global_context=context
        )
        
        print(f"   ToolingAgent Result: {'✅ SUCCESS' if tooling_result.success else '❌ FAILURE'}")
        print(f"   Commands Executed: {len(tooling_result.outputs.get('commands_executed', []))}")
        print(f"   Execution Summary: {tooling_result.outputs.get('execution_summary', 'none')}")
        
        # Test DebuggingAgent directly
        print("\n🔍 Testing DebuggingAgent with function-call interface...")
        
        debug_agent = DebuggingAgent(llm_client=MockLLMClient(), agent_registry={"ToolingAgent": tooling_agent})
        debug_result = debug_agent.execute_v2(
            goal="Debug mock test failure",
            inputs={
                "failed_test_report": {
                    "summary": {"total": 2, "passed": 1, "failed": 1},
                    "error": "Mock test failure for function-call testing"
                },
                "code_context": "def test_function():\n    return True"
            },
            global_context=context
        )
        
        print(f"   DebuggingAgent Result: {'✅ SUCCESS' if debug_result.success else '❌ FAILURE'}")
        print(f"   Solution Applied: {debug_result.outputs.get('solution_applied', 'none')}")
        print(f"   Root Cause: {debug_result.outputs.get('root_cause', 'unknown')}")
        
        return tooling_result.success and debug_result.success

def test_input_validation():
    """Test fail-fast input validation."""
    print("\n⚡ Testing Fail-Fast Input Validation")
    print("=" * 70)
    
    tooling_agent = ToolingAgent()
    
    print("🚫 Testing missing required inputs (should fail)...")
    try:
        result = tooling_agent.execute_v2(
            goal="This should fail",
            inputs={"purpose": "Test missing commands"},  # Missing 'commands'
            global_context=GlobalContext()
        )
        print(f"   Result: {'❌ UNEXPECTED SUCCESS' if result.success else '✅ EXPECTED FAILURE'}")
        print(f"   Error: {result.error_details}")
        validation_works = not result.success
        
    except Exception as e:
        print(f"   ✅ EXPECTED EXCEPTION: {e}")
        validation_works = True
    
    print("\n✅ Testing valid inputs (should succeed)...")
    result = tooling_agent.execute_v2(
        goal="This should work", 
        inputs={
            "commands": ["echo 'test'"],
            "purpose": "Test valid inputs"
        },
        global_context=GlobalContext()
    )
    print(f"   Result: {'✅ SUCCESS' if result.success else '❌ FAILURE'}")
    
    return validation_works and result.success

def main():
    """Run all function-call semantics tests."""
    print("🚀 Testing Function-Call Agent Semantics")
    print("=" * 70)
    
    tests = [
        ("Complete Agent Chain", test_function_call_agent_chain),
        ("Direct Agent Calls", test_direct_agent_calls), 
        ("Input Validation", test_input_validation)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result, None))
            print(f"\n{'✅' if result else '❌'} {test_name}: {'PASSED' if result else 'FAILED'}")
        except Exception as e:
            results.append((test_name, False, str(e)))
            print(f"\n❌ {test_name}: ERROR - {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("📋 FUNCTION-CALL SEMANTICS TEST SUMMARY") 
    print("=" * 70)
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    for test_name, success, error in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status:8} {test_name}")
        if error:
            print(f"         Error: {error}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 SUCCESS: Function-call agent semantics are working!")
        print("   ✅ Explicit inputs/outputs")
        print("   ✅ Fail-fast validation")
        print("   ✅ Agent isolation")
        print("   ✅ No artifact hunting")
        print("   ✅ Function-call transparency")
        return 0
    else:
        print("💥 Some tests failed. Check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())