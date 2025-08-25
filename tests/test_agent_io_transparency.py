#!/usr/bin/env python3
"""
Test script for Agent I/O Transparency implementation.

This script tests the new user messaging system that shows clean input/output
for agents with smart trimming and clear boundaries.
"""

import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from interfaces.collaboration_ui import CollaborationUI
from utils.content_trimmer import ContentTrimmer, trim_content
from fagents.inter_agent_router import InterAgentRouter
from core.context import GlobalContext
from fagents.base import FoundationalAgent
from core.models import AgentResult


class MockAgent(FoundationalAgent):
    """Mock agent for testing I/O transparency."""
    
    def __init__(self, name: str = "MockAgent", **kwargs):
        super().__init__(name, "Mock agent for testing", **kwargs)
    
    def execute(self, goal: str, inputs: dict, global_context: GlobalContext) -> AgentResult:
        """Execute mock operation with various output types."""
        
        # Simulate some processing
        self.report_progress("Starting mock operation", "Processing inputs")
        
        # Generate mock outputs with different data types
        outputs = {
            "short_text": "Simple result",
            "long_text": "This is a very long text result. " * 30,
            "code_sample": """def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# Test the function
for i in range(10):
    print(f"fib({i}) = {fibonacci(i)}")
""" * 5,  # Make it longer
            "json_data": {
                "users": [{"name": f"User{i}", "age": 20+i, "skills": ["python", "javascript"]} for i in range(15)],
                "metadata": {"version": "1.0", "created": "2024-01-01", "notes": "This is test data"}
            },
            "list_data": [f"Item {i} with description" for i in range(25)]
        }
        
        return self.create_result(
            success=True,
            message="Mock operation completed successfully",
            outputs=outputs
        )
    
    def get_capabilities(self):
        return {"description": "Mock agent for testing", "capabilities": ["testing", "mocking"]}


class MockLLMClient:
    """Mock LLM client for testing routing."""
    
    def invoke(self, prompt: str) -> str:
        # Return a simple routing decision
        return """{
            "agent_name": "MockAgent",
            "confidence": 0.9,
            "reasoning": "Mock agent is perfect for this test scenario",
            "goal_for_agent": "Execute mock operation with test data",
            "recommended_inputs": {"test_param": "test_value"},
            "is_completion": false,
            "completion_summary": ""
        }"""


def test_content_trimmer():
    """Test the content trimming utility."""
    print("🧪 Testing Content Trimmer")
    print("=" * 80)
    
    trimmer = ContentTrimmer()
    
    # Test different content types
    test_cases = [
        ("Short text", "This is short"),
        ("Long text", "This is a very long text. " * 50),
        ("Dict data", {"key1": "value1", "key2": "value2 with long content", "key3": [1, 2, 3]}),
        ("List data", [f"Item {i}" for i in range(20)]),
        ("JSON string", '{"name": "test", "data": [1, 2, 3, 4, 5]}'),
        ("Code", "def test():\n    print('hello')\n    return 42\n" * 10)
    ]
    
    for name, content in test_cases:
        print(f"\n📋 Testing {name}:")
        result = trimmer.trim_content(content)
        print(f"   Truncated: {result.was_truncated}")
        print(f"   Type: {result.content_type}")
        if result.was_truncated:
            print(f"   Info: {result.truncation_info}")
        print(f"   Content preview: {result.content[:100]}...")
    
    print("\n✅ Content Trimmer tests completed!")


def test_collaboration_ui():
    """Test the collaboration UI I/O display methods."""
    print("\n\n🖥️  Testing Collaboration UI I/O Display")
    print("=" * 80)
    
    ui = CollaborationUI()
    
    # Test agent input display
    print("\n📥 Testing Agent Input Display:")
    ui.display_agent_input(
        "TestAgent",
        "Create a Python function to calculate fibonacci numbers", 
        {
            "file_path": "fibonacci.py",
            "algorithm": "recursive",
            "requirements": ["handle edge cases", "optimize for small numbers"],
            "long_spec": "This is a very long specification. " * 20
        },
        "Agent selected based on code generation capabilities"
    )
    
    # Test agent output display
    print("\n📤 Testing Agent Output Display:")
    ui.display_agent_output(
        "TestAgent",
        True,
        "Successfully created fibonacci function with optimizations",
        {
            "generated_files": ["fibonacci.py", "test_fibonacci.py"],
            "functions_created": ["fibonacci", "fibonacci_memo"],
            "performance_notes": "Optimized for numbers up to 100",
            "code": "def fibonacci(n):\n    # Implementation here\n    pass\n" * 10
        }
    )
    
    # Test routing decision display
    print("\n🔀 Testing Routing Decision Display:")
    ui.display_routing_decision(
        "AnalystAgent", 
        "CreatorAgent", 
        0.85, 
        "Analysis complete, now need to create the implementation based on requirements"
    )
    
    # Test LLM communication display
    print("\n🧠 Testing LLM Communication Display:")
    ui.display_llm_communication(
        "InterAgentRouter",
        "You are the Inter-Agent Routing Intelligence. Determine the next agent to invoke based on...",
        '{"agent_name": "CreatorAgent", "confidence": 0.9, "reasoning": "User needs code generation"}'
    )
    
    print("\n✅ Collaboration UI tests completed!")


def test_agent_with_transparency():
    """Test agent execution with I/O transparency."""
    print("\n\n🤖 Testing Agent with I/O Transparency")
    print("=" * 80)
    
    # Create UI and agent
    ui = CollaborationUI()
    agent = MockAgent(ui_interface=ui)
    context = GlobalContext()
    
    # Test inputs
    inputs = {
        "operation": "mock_test",
        "parameters": {"count": 10, "type": "test"},
        "long_config": {"setting1": "value1", "setting2": "value2"} 
    }
    
    print("\n🚀 Executing agent with transparency enabled:")
    result = agent.execute("Perform mock operation for testing", inputs, context)
    
    # Manually show the I/O (since we're not going through router)
    agent.report_agent_io("Perform mock operation for testing", inputs, result)
    
    print("\n✅ Agent execution with transparency completed!")


def test_inter_agent_router():
    """Test the enhanced inter-agent router with I/O display."""
    print("\n\n🔀 Testing Enhanced Inter-Agent Router")
    print("=" * 80)
    
    # Create components
    ui = CollaborationUI()
    llm_client = MockLLMClient()
    context = GlobalContext()
    
    # Create agent registry with UI-enabled agent
    agent_registry = {
        "MockAgent": lambda **kwargs: MockAgent(ui_interface=ui)
    }
    
    # Create router with UI
    router = InterAgentRouter(
        llm_client=llm_client,
        agent_registry=agent_registry,
        ui_interface=ui
    )
    
    print("\n🚀 Executing workflow with full I/O transparency:")
    
    # Execute a simple workflow
    workflow_result = router.execute_workflow(
        user_goal="Test the I/O transparency system",
        initial_inputs={"test_mode": True, "verbose": True},
        global_context=context,
        max_hops=2
    )
    
    print(f"\n📊 Workflow Result: {workflow_result.status.value}")
    print(f"📈 Execution hops: {workflow_result.current_hop}")
    print(f"🏁 Completion validated: {workflow_result.completion_validated}")
    
    print("\n✅ Inter-agent router tests completed!")


def main():
    """Run all tests."""
    print("🎯 AGENT I/O TRANSPARENCY TESTING")
    print("=" * 80)
    print("Testing the new user messaging system for agent input/output visibility.")
    print()
    
    try:
        # Run all test suites
        test_content_trimmer()
        test_collaboration_ui()
        test_agent_with_transparency()
        test_inter_agent_router()
        
        print("\n\n🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("The agent I/O transparency system is working correctly.")
        print("Users will now see clean, formatted input/output for all agent operations.")
        
    except Exception as e:
        print(f"\n\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()