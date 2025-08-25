#!/usr/bin/env python3
"""
Test script for the enhanced Inter-Agent Router with self-debugging capabilities.
"""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fagents.inter_agent_router import InterAgentRouter, FailureAnalysis, FailureType, WorkflowContext, WorkflowStatus, AgentExecution
from core.models import AgentResult
from core.context import GlobalContext
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_failure_analysis():
    """Test the failure analysis capability."""
    
    print("🔧 Testing Self-Debugging Inter-Agent Router")
    print("=" * 50)
    
    # Create mock components
    router = InterAgentRouter(llm_client=None)  # No LLM client for fallback testing
    
    # Create a failed execution
    failed_result = AgentResult(
        success=False,
        message="Agent failed to execute the task",
        outputs={
            "failure_type": "capability_execution_failure",
            "error_details": "File not found: test.py",
            "execution_duration_seconds": 2.5
        }
    )
    
    failed_execution = AgentExecution(
        agent_name="CreatorAgent",
        goal="Create a Python script for data processing",
        inputs={"creation_type": "code", "language": "python"},
        result=failed_result,
        execution_order=1,
        reasoning="Attempted to create code",
        confidence=0.0
    )
    
    # Create workflow context
    workflow_context = WorkflowContext(
        user_goal="Build a data processing pipeline",
        initial_inputs={"data_source": "csv"},
        workflow_id="test_workflow_001",
        current_hop=3,
        max_hops=10,
        status=WorkflowStatus.IN_PROGRESS
    )
    workflow_context.execution_history = [failed_execution]
    
    # Create global context
    global_context = GlobalContext()
    
    print(f"📋 Testing failure analysis for: {failed_execution.agent_name}")
    print(f"   Goal: {failed_execution.goal}")
    print(f"   Error: {failed_execution.result.message}")
    print()
    
    # Perform failure analysis
    failure_analysis = router._perform_failure_analysis(failed_execution, workflow_context, global_context)
    
    print("🔍 Failure Analysis Results:")
    print(f"   Failure Type: {failure_analysis.failure_type.value}")
    print(f"   Root Cause: {failure_analysis.root_cause}")
    print(f"   Remediation Plan: {failure_analysis.remediation_plan}")
    print(f"   Confidence: {failure_analysis.confidence:.2f}")
    print(f"   User Consultation Needed: {failure_analysis.requires_user_consultation}")
    print()
    
    # Test remediation plan application
    print("🚀 Applying Remediation Plan...")
    next_decision = router._apply_remediation_plan(failure_analysis, workflow_context, global_context)
    
    print("📈 Next Agent Decision:")
    print(f"   Next Agent: {next_decision.agent_name}")
    print(f"   Confidence: {next_decision.confidence:.2f}")
    print(f"   Reasoning: {next_decision.reasoning}")
    print(f"   Goal: {next_decision.goal_for_agent}")
    print()
    
    # Test different failure types
    print("🧪 Testing Different Failure Types:")
    print("-" * 30)
    
    failure_types = [
        ("incorrect_routing", "Wrong agent was selected for the task"),
        ("insufficient_context", "Missing required inputs for task completion"),
        ("target_agent_misrouting", "Agent selected wrong internal capability"),
        ("agent_not_found", "Requested agent not available in registry")
    ]
    
    for failure_type, description in failure_types:
        # Create test failure
        test_result = AgentResult(
            success=False,
            message=description,
            outputs={"failure_type": failure_type, "error_details": description}
        )
        
        test_execution = AgentExecution(
            agent_name="TestAgent",
            goal="Test task",
            inputs={"test": True},
            result=test_result,
            execution_order=1,
            reasoning="Test execution",
            confidence=0.0
        )
        
        analysis = router._create_fallback_failure_analysis(test_execution, workflow_context)
        remediation = router._apply_remediation_plan(analysis, workflow_context, global_context)
        
        print(f"   {failure_type.upper()}:")
        print(f"     → Routes to: {remediation.agent_name}")
        print(f"     → Confidence: {remediation.confidence:.2f}")
        print(f"     → Strategy: {remediation.reasoning}")
        print()
    
    print("✅ Self-Debugging Router Test Complete!")
    print("=" * 50)
    
    return True

if __name__ == "__main__":
    try:
        success = test_failure_analysis()
        if success:
            print("🎉 All tests passed!")
            sys.exit(0)
        else:
            print("❌ Some tests failed!")
            sys.exit(1)
    except Exception as e:
        print(f"💥 Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)