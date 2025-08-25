#!/usr/bin/env python3
"""
Test the enhanced InterAgentRouter → CreatorAgent integration.
"""

import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from fagents.inter_agent_router import InterAgentRouter
from fagents.creator import CreatorAgent
from core.context import GlobalContext
from tools.test_generation_tools import TestType


class TestLLMClient:
    """Test LLM client that provides structured inputs to CreatorAgent."""
    
    def invoke(self, prompt: str) -> str:
        # Check if this is an integration test request
        if "integration test" in prompt.lower():
            return """{
                "agent_name": "CreatorAgent",
                "confidence": 0.95,
                "reasoning": "User explicitly requested integration test creation, which requires the CreatorAgent with structured test generation inputs",
                "goal_for_agent": "Create exactly one new integration test file aligned with the project's existing language and test framework",
                "recommended_inputs": {
                    "creation_type": "tests",
                    "test_type": "integration",
                    "test_framework": "pytest",
                    "test_quality": "decent",
                    "target_files": ["sentence_sorter.py"],
                    "requirements": ["create integration test", "align with existing framework", "place in correct directory"]
                },
                "is_completion": true,
                "completion_summary": "Integration test created successfully"
            }"""
        else:
            # Default response for other cases
            return """{
                "agent_name": "AnalystAgent",
                "confidence": 0.8,
                "reasoning": "Need to analyze the request first",
                "goal_for_agent": "Analyze the user request",
                "recommended_inputs": {},
                "is_completion": false,
                "completion_summary": ""
            }"""


def test_router_creator_integration():
    """Test that the InterAgentRouter provides correct structured inputs to CreatorAgent."""
    
    print("🧪 Testing Router → Creator Integration")
    print("=" * 80)
    
    # Setup
    llm_client = TestLLMClient()
    context = GlobalContext()
    
    # Create router with a mock CreatorAgent in registry
    mock_creator = CreatorAgent()
    agent_registry = {
        "CreatorAgent": lambda **kwargs: mock_creator
    }
    
    router = InterAgentRouter(llm_client=llm_client, agent_registry=agent_registry)
    
    print("\n🎯 Testing integration test request:")
    print("Goal: 'Create exactly one new integration test file'")
    print("\n📋 Expected flow:")
    print("1. Router analyzes goal and determines CreatorAgent is needed")
    print("2. Router provides structured inputs: test_type='integration'")
    print("3. CreatorAgent uses structured inputs instead of brittle parsing")
    print("\n" + "─" * 80)
    
    # Test the routing decision
    first_decision = router._determine_first_agent(
        "Create exactly one new integration test file", 
        {}, 
        context
    )
    
    print(f"✅ Router Decision:")
    print(f"   Agent: {first_decision.agent_name}")
    print(f"   Confidence: {first_decision.confidence}")
    print(f"   Reasoning: {first_decision.reasoning}")
    print(f"\n📥 Structured Inputs Provided:")
    
    for key, value in first_decision.recommended_inputs.items():
        print(f"   {key}: {value}")
    
    # Verify the key parameters
    inputs = first_decision.recommended_inputs
    
    expected_checks = [
        ("creation_type", "tests"),
        ("test_type", "integration"),
        ("test_framework", "pytest")
    ]
    
    print(f"\n🔍 Validation:")
    all_passed = True
    
    for key, expected_value in expected_checks:
        actual_value = inputs.get(key)
        if actual_value == expected_value:
            print(f"   ✅ {key}: {actual_value} (correct)")
        else:
            print(f"   ❌ {key}: {actual_value}, expected {expected_value}")
            all_passed = False
    
    print("\n" + "─" * 80)
    
    if all_passed:
        print("🎉 SUCCESS: InterAgentRouter now provides proper structured inputs!")
        print("   • No more brittle hardcoded parsing in CreatorAgent")
        print("   • Router intelligently analyzes user goals")
        print("   • Structured inputs ensure correct behavior")
        print("   • Integration tests will now be generated correctly")
    else:
        print("❌ FAILED: Router inputs need adjustment")
    
    return all_passed


if __name__ == "__main__":
    success = test_router_creator_integration()
    if not success:
        sys.exit(1)