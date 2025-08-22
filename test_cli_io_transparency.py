#!/usr/bin/env python3
"""
Quick test to verify CLI I/O transparency integration.
"""

import sys
from pathlib import Path

# Add project root to Python path  
sys.path.insert(0, str(Path(__file__).parent))

from main_interactive_intelligent import IntelligentInteractiveSession
from mock_llm_client import MockLLMClient


class TestLLMClient:
    """Test LLM client that returns simple completion response."""
    
    def invoke(self, prompt: str) -> str:
        # Simple completion response to test I/O transparency
        return """{
            "agent_name": "AnalystAgent",
            "confidence": 0.9,
            "reasoning": "This is a test goal that requires analysis to verify I/O transparency is working",
            "goal_for_agent": "Analyze and validate the I/O transparency system",
            "recommended_inputs": {"test_mode": true},
            "is_completion": true,
            "completion_summary": "I/O transparency test completed successfully"
        }"""


def test_cli_integration():
    """Test that the CLI shows I/O transparency."""
    print("🧪 Testing CLI I/O Transparency Integration")
    print("=" * 80)
    
    # Create session with test LLM client
    llm_client = TestLLMClient()
    session = IntelligentInteractiveSession(llm_client=llm_client)
    
    print("\n🚀 Executing test goal through CLI system:")
    print("Goal: 'Test the I/O transparency system'")
    print("\nExpected output: Clean agent I/O display with boundaries and smart trimming")
    print("-" * 80)
    
    # Execute goal through the session
    try:
        session._execute_user_goal("Test the I/O transparency system")
        print("-" * 80)
        print("✅ CLI Integration test completed!")
        print("If you see formatted agent I/O above with boundaries (═══), the integration is working!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_cli_integration()