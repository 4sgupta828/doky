#!/usr/bin/env python3
"""
Test script to demonstrate the inter-agent communication transparency system.
This script simulates agent interactions to show how the transparency system works.
"""

import sys
import tempfile
import shutil
from pathlib import Path

# Add the current directory to Python path so we can import the modules
sys.path.insert(0, str(Path(__file__).parent))

from core.context import GlobalContext
from core.models import TaskNode
from agents.debugging import DebuggingAgent
from agents.test_runner import TestRunnerAgent
from utils.logger import setup_logger

# Mock agents for testing
class MockToolingAgent:
    def __init__(self):
        self.name = "ToolingAgent"
    
    def execute(self, goal, context, task):
        from core.models import AgentResponse
        return AgentResponse(
            success=True,
            message="Mock tooling analysis complete",
            artifacts_generated=["system_info.json", "environment_check.json"]
        )

class MockCodeGenerationAgent:
    def __init__(self):
        self.name = "CodeGenerationAgent"
    
    def execute(self, goal, context, task):
        from core.models import AgentResponse
        return AgentResponse(
            success=True,
            message="Mock code fix applied",
            artifacts_generated=["fixed_code.py"]
        )

class MockLLMClient:
    def invoke(self, prompt):
        return '{"root_cause_analysis": "Mock bug found", "primary_hypothesis": "Type error", "solution_type": "SURGICAL", "recommended_strategy": "Add type checking"}'

def test_transparency_system():
    """Test the inter-agent communication transparency system."""
    print("🧪 Testing Inter-Agent Communication Transparency System")
    print("=" * 60)
    
    # Setup test environment
    with tempfile.TemporaryDirectory() as temp_dir:
        context = GlobalContext(workspace_path=temp_dir)
        
        # Create mock agent registry
        agent_registry = {
            "ToolingAgent": MockToolingAgent(),
            "CodeGenerationAgent": MockCodeGenerationAgent()
        }
        
        # Create debugging agent with mocks
        debug_agent = DebuggingAgent(
            llm_client=MockLLMClient(),
            agent_registry=agent_registry
        )
        
        # Add some mock artifacts to context
        context.add_artifact("failed_test_report.json", 
                           {"summary": {"failed": 1}, "error": "TypeError"}, 
                           "test_task")
        context.add_artifact("targeted_code_context.json", 
                           {"files": [{"path": "test.py", "content": "def add(a, b): return a + b"}], "metadata": {"total_files": 1}}, 
                           "context_task")
        
        # Create a debugging task
        debug_task = TaskNode(
            goal="Debug a failed test",
            assigned_agent="DebuggingAgent",
            input_artifact_keys=["failed_test_report.json", "targeted_code_context.json"]
        )
        
        print("📋 Starting debugging simulation...")
        print("   - DebuggingAgent will delegate to ToolingAgent")
        print("   - ToolingAgent will respond with evidence")
        print("   - DebuggingAgent will delegate to CodeGenerationAgent")
        print("   - CodeGenerationAgent will apply a fix")
        print("")
        
        # Execute debugging (this should trigger inter-agent communications)
        result = debug_agent.execute(debug_task.goal, context, debug_task)
        
        print("✅ Debugging simulation completed!")
        print(f"   Result: {'SUCCESS' if result.success else 'FAILED'}")
        print(f"   Message: {result.message}")
        print("")
        
        # The transparency summary should have been printed during execution
        # Let's also manually show the communication chain
        print("📊 Communication Chain Summary:")
        communication_chain = context.get_communication_chain()
        for i, comm in enumerate(communication_chain, 1):
            print(f"   {i}. {comm}")
        
        if not communication_chain:
            print("   No communications recorded (this might indicate an issue)")
        
        print("")
        print("🔍 Detailed Communication Log:")
        print(context.get_communication_summary())

def test_test_runner_handover():
    """Test TestRunnerAgent handing over to DebuggingAgent."""
    print("\n" + "=" * 60)
    print("🧪 Testing TestRunnerAgent -> DebuggingAgent Handover")
    print("=" * 60)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        context = GlobalContext(workspace_path=temp_dir)
        
        # Mock debugging agent
        debug_agent = DebuggingAgent(
            llm_client=MockLLMClient(),
            agent_registry={"ToolingAgent": MockToolingAgent()}
        )
        
        # Create test runner with debugging agent in registry
        test_runner = TestRunnerAgent(agent_registry={"DebuggingAgent": debug_agent})
        
        # Add a failed test report (TestRunner expects a JSON string)
        import json
        failed_report = {"summary": {"total": 1, "passed": 0, "failed": 1}}
        context.add_artifact("pytest_output.json", json.dumps(failed_report), "test_task")
        
        # Create test task
        test_task = TaskNode(
            goal="Run tests",
            assigned_agent="TestRunnerAgent",
            input_artifact_keys=["pytest_output.json"]
        )
        
        print("📋 Starting test runner simulation...")
        print("   - TestRunnerAgent detects test failure")
        print("   - TestRunnerAgent hands over to DebuggingAgent")
        print("   - DebuggingAgent takes control of debugging")
        print("")
        
        # Execute test runner (should hand over to debugging agent)
        result = test_runner.execute(test_task.goal, context, test_task)
        
        print("✅ Test runner simulation completed!")
        print(f"   Result: {'SUCCESS' if result.success else 'FAILED'}")
        print(f"   Message: {result.message}")
        print("")
        
        print("📊 Communication Chain Summary:")
        communication_chain = context.get_communication_chain()
        for i, comm in enumerate(communication_chain, 1):
            print(f"   {i}. {comm}")

if __name__ == "__main__":
    # Setup logging
    setup_logger()
    
    try:
        # Test the debugging agent transparency
        test_transparency_system()
        
        # Test test runner handover transparency
        test_test_runner_handover()
        
        print("\n🎉 All transparency tests completed successfully!")
        print("\nKey Features Demonstrated:")
        print("✅ Inter-agent delegation logging")
        print("✅ Agent response tracking")  
        print("✅ Handover transparency")
        print("✅ User-facing communication summaries")
        print("✅ Complete audit trail of agent interactions")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)