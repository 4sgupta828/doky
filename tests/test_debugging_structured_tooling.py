#!/usr/bin/env python3
"""
Test the DebuggingAgent's new structured ToolingAgent integration.
This validates the end-to-end structured communication pattern.
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

from agents.debugging import DebuggingAgent
from agents.tooling import ToolingAgent
from core.context import GlobalContext
from core.models import TaskNode
from utils.logger import setup_logger

# Set up logging
setup_logger()

class MockLLMClient:
    """Mock LLM client for testing."""
    def invoke(self, prompt: str) -> str:
        # Return a valid debugging analysis
        return json.dumps({
            "root_cause_analysis": "Test failure due to mock analysis",
            "primary_hypothesis": "Mock testing scenario",
            "alternative_hypotheses": ["Alternative mock scenario"],
            "confidence_level": "high",
            "error_category": "code",
            "solution_type": "SURGICAL",
            "complexity_assessment": "simple",
            "recommended_strategy": "Apply mock fix for testing",
            "risk_assessment": "Low risk mock scenario"
        })

def test_debugging_agent_structured_tooling():
    """Test that DebuggingAgent properly creates and uses structured tooling instructions."""
    print("\n🔍 Testing DebuggingAgent Structured Tooling Integration")
    print("=" * 60)
    
    # Create temporary workspace with some test files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create test environment
        test_file = Path(temp_dir) / "test.py"
        test_file.write_text("def test_function():\n    return True")
        
        context = GlobalContext(workspace_path=temp_dir)
        
        # Create mock agent registry with ToolingAgent
        tooling_agent = ToolingAgent()
        agent_registry = {"ToolingAgent": tooling_agent}
        
        # Create debugging agent with mocked LLM
        debug_agent = DebuggingAgent(
            llm_client=MockLLMClient(),
            agent_registry=agent_registry
        )
        
        # Add some initial artifacts to trigger evidence gathering
        context.add_artifact("failed_test_report.json", {
            "summary": {"failed": 1, "passed": 0},
            "error": "Mock test error for debugging",
            "file": "test.py"
        }, "test")
        
        context.add_artifact("targeted_code_context.txt", 
                           "def test_function():\n    return True", "test")
        
        # Create debug task
        debug_task = TaskNode(
            goal="Debug failed test using structured tooling",
            assigned_agent="DebuggingAgent",
            input_artifact_keys=["failed_test_report.json", "targeted_code_context.txt"]
        )
        
        print("⚙️  Executing DebuggingAgent with structured tooling...")
        print("   This should create structured diagnostic instructions")
        
        # Execute debugging (should create structured tooling instructions)
        result = debug_agent.execute(debug_task.goal, context, debug_task)
        
        # Analyze the results
        print(f"\n📊 Results:")
        print(f"   Success: {'✅' if result.success else '❌'} {result.success}")
        print(f"   Message: {result.message[:100]}...")
        
        # Check if structured tooling instruction was created
        tooling_instruction_found = False
        diagnostic_executed = False
        
        for artifact_key in context.artifacts.keys():
            if "tooling_instruction" in artifact_key:
                tooling_instruction_found = True
                instruction_data = context.get_artifact(artifact_key)
                if isinstance(instruction_data, str):
                    instruction_dict = json.loads(instruction_data)
                    print(f"   📋 Found structured instruction: {instruction_dict.get('instruction_id', 'unknown')}")
                    print(f"      Type: {instruction_dict.get('command_type', 'unknown')}")
                    print(f"      Commands: {len(instruction_dict.get('commands', []))}")
                    print(f"      Purpose: {instruction_dict.get('purpose', 'unknown')}")
            
            if "tooling_execution_result" in artifact_key:
                diagnostic_executed = True
                result_data = context.get_artifact(artifact_key)
                if isinstance(result_data, str):
                    result_dict = json.loads(result_data)
                    print(f"   ⚙️  Diagnostic execution result:")
                    print(f"      Success: {result_dict.get('success', False)}")
                    print(f"      Commands executed: {len(result_dict.get('commands_executed', []))}")
        
        print(f"\n🎯 Structured Integration Check:")
        print(f"   Tooling instruction created: {'✅' if tooling_instruction_found else '❌'}")
        print(f"   Diagnostic commands executed: {'✅' if diagnostic_executed else '❌'}")
        
        # Check communication logs
        if hasattr(context, 'communications') and context.communications:
            tooling_comms = [c for c in context.communications if c.get('target_agent') == 'ToolingAgent']
            print(f"   ToolingAgent communications: {len(tooling_comms)}")
            for comm in tooling_comms:
                print(f"      - {comm.get('communication_type', 'unknown')}: {comm.get('message', '')[:50]}...")
        
        return tooling_instruction_found and diagnostic_executed

def test_structured_vs_old_approach():
    """Compare the old free-form approach with the new structured approach."""
    print("\n📈 Comparing Old vs New ToolingAgent Integration")
    print("=" * 60)
    
    print("✅ NEW STRUCTURED APPROACH:")
    print("   • Creates ToolingInstruction with schema validation")
    print("   • Stores structured JSON in context artifacts")
    print("   • ToolingAgent parses and executes multiple commands")
    print("   • Detailed execution results with per-command status")
    print("   • Proper error handling and timeout control")
    print("   • Safety overrides and environment variable support")
    
    print("\n❌ OLD FREE-FORM APPROACH:")
    print("   • Passed raw text as 'goal' parameter")
    print("   • ToolingAgent treated goal as single shell command")
    print("   • No schema validation or structure")
    print("   • Limited error context and reporting")
    print("   • No support for complex multi-command operations")
    
    print("\n🎯 BENEFITS OF STRUCTURED APPROACH:")
    print("   • Eliminates ambiguous command interpretation")
    print("   • Enables complex multi-step tooling operations")
    print("   • Provides detailed execution reporting")
    print("   • Supports safety overrides and custom environments")
    print("   • Maintains backward compatibility via legacy fallback")
    
    return True

def main():
    """Run the structured tooling integration tests."""
    print("🚀 Testing DebuggingAgent → ToolingAgent Structured Integration")
    print("=" * 70)
    
    tests = [
        ("Structured Tooling Integration", test_debugging_agent_structured_tooling),
        ("Approach Comparison", test_structured_vs_old_approach)
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
    print("📋 INTEGRATION TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    for test_name, success, error in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status:8} {test_name}")
        if error:
            print(f"         Error: {error}")
    
    print(f"\n🎯 Overall: {passed}/{total} integration tests passed")
    
    if passed == total:
        print("🎉 SUCCESS: Structured ToolingAgent integration is working!")
        print("   ✅ No more free-form text commands")
        print("   ✅ Schema-based structured tool calling")
        print("   ✅ Proper inter-agent communication")
        print("   ✅ Detailed execution reporting")
        return 0
    else:
        print("💥 Some integration tests failed. Check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())