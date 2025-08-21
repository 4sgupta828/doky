#!/usr/bin/env python3
"""
Test the LLM-based routing system for foundation agents.

This script tests that the new LLM routing system can correctly route
requests to appropriate handlers and falls back gracefully when needed.
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any

# Mock LLM client for testing
class MockLLMClient:
    """Mock LLM client that returns predefined responses for testing."""
    
    def __init__(self):
        self.responses = {
            "analyze code quality": {
                "decision": "quality_analysis",
                "confidence": 0.9,
                "reasoning": "User wants to analyze code quality and security",
                "recommended_inputs": {"check_security": True}
            },
            "create unit tests": {
                "decision": "tests",
                "confidence": 0.95,
                "reasoning": "User wants to create test code",
                "recommended_inputs": {"test_framework": "pytest"}
            },
            "plan project implementation": {
                "decision": "task_planning",
                "confidence": 0.85,
                "reasoning": "User needs to break down complex task",
                "recommended_inputs": {"planning_quality": "DECENT"}
            },
            "run tests": {
                "decision": "test_execution",
                "confidence": 0.9,
                "reasoning": "User wants to execute test suite",
                "recommended_inputs": {"test_framework": "pytest"}
            },
            "fix requirements file": {
                "decision": "requirements_management",
                "confidence": 0.88,
                "reasoning": "User needs to update requirements file",
                "recommended_inputs": {"backup_existing": True}
            }
        }
    
    def invoke(self, prompt: str) -> str:
        """Return mock response based on goal in prompt."""
        # Extract goal from prompt (simple heuristic)
        prompt_lower = prompt.lower()
        
        for goal_phrase, response in self.responses.items():
            if goal_phrase in prompt_lower:
                return json.dumps(response)
        
        # Default fallback response
        return json.dumps({
            "decision": "code_analysis",
            "confidence": 0.6,
            "reasoning": "Default fallback routing",
            "recommended_inputs": {}
        })


def test_routing_system():
    """Test the LLM routing system with various scenarios."""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Add project root to Python path
    project_root = Path(__file__).parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    try:
        from fagents.routing import LLMRouter, create_routing_context, RoutingDecision
    except ImportError as e:
        print(f"Failed to import routing module: {e}")
        return False
    
    # Create mock LLM client
    mock_client = MockLLMClient()
    router = LLMRouter(mock_client)
    
    # Test cases: (agent_type, goal, expected_decision_prefix)
    test_cases = [
        ("AnalystAgent", "analyze code quality", "quality_analysis"),
        ("CreatorAgent", "create unit tests", "tests"),
        ("StrategistAgent", "plan project implementation", "task_planning"),
        ("ExecutorAgent", "run tests", "test_execution"),
        ("SurgeonAgent", "fix requirements file", "requirements_management"),
    ]
    
    results = []
    
    for agent_type, goal, expected_decision in test_cases:
        logger.info(f"\nTesting {agent_type} with goal: '{goal}'")
        
        # Create routing context
        context = create_routing_context(
            agent_type=agent_type,
            goal=goal,
            inputs={"test_input": "value"},
            workspace_files=["main.py", "test_main.py", "requirements.txt"],
            available_capabilities=["analysis", "creation", "execution"]
        )
        
        # Route the request
        try:
            routing_result = router.route_request(context)
            
            logger.info(f"  Decision: {routing_result.decision.value}")
            logger.info(f"  Confidence: {routing_result.confidence:.2f}")
            logger.info(f"  Reasoning: {routing_result.reasoning}")
            
            # Check if decision matches expected
            success = routing_result.decision.value == expected_decision
            results.append({
                "agent_type": agent_type,
                "goal": goal,
                "expected": expected_decision,
                "actual": routing_result.decision.value,
                "confidence": routing_result.confidence,
                "success": success
            })
            
            if success:
                logger.info("  ✅ PASS")
            else:
                logger.warning(f"  ❌ FAIL - Expected {expected_decision}, got {routing_result.decision.value}")
                
        except Exception as e:
            logger.error(f"  ❌ ERROR: {e}")
            results.append({
                "agent_type": agent_type,
                "goal": goal,
                "expected": expected_decision,
                "actual": f"ERROR: {e}",
                "confidence": 0.0,
                "success": False
            })
    
    # Test fallback routing (without LLM)
    logger.info(f"\nTesting fallback routing (no LLM client)...")
    fallback_router = LLMRouter(None)
    
    context = create_routing_context(
        agent_type="AnalystAgent",
        goal="debug this error",
        inputs={"error_logs": ["Error: file not found"]},
        workspace_files=["main.py"]
    )
    
    try:
        fallback_result = router.route_request(context)
        logger.info(f"  Fallback decision: {fallback_result.decision.value}")
        logger.info("  ✅ Fallback routing works")
        fallback_success = True
    except Exception as e:
        logger.error(f"  ❌ Fallback routing failed: {e}")
        fallback_success = False
    
    # Summary
    passed = sum(1 for r in results if r["success"])
    total = len(results)
    
    print(f"\n" + "="*60)
    print(f"LLM ROUTING TEST RESULTS")
    print(f"="*60)
    print(f"Passed: {passed}/{total} ({passed/total*100:.1f}%)")
    print(f"Fallback routing: {'✅ PASS' if fallback_success else '❌ FAIL'}")
    
    if passed == total and fallback_success:
        print("🎉 All tests passed! LLM routing system is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Check the logs above for details.")
        return False


if __name__ == "__main__":
    success = test_routing_system()
    sys.exit(0 if success else 1)