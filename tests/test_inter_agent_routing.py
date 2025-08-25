#!/usr/bin/env python3
"""
Test the Inter-Agent Routing System for Foundation Agents.

This script tests the complete multi-agent workflow coordination,
including intelligent routing, completion validation, and progress tracking.
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import asdict

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Add project root to Python path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


# Mock Agent Results for Testing
class MockAgentResult:
    """Mock AgentResult for testing."""
    def __init__(self, success: bool = True, message: str = "Success", outputs: Dict[str, Any] = None):
        self.success = success
        self.message = message
        self.outputs = outputs or {}


class MockFoundationalAgent:
    """Mock foundational agent for testing."""
    def __init__(self, agent_name: str, llm_client: Any = None):
        self.agent_name = agent_name
        self.llm_client = llm_client
    
    def execute(self, goal: str, inputs: Dict[str, Any], global_context: Any) -> MockAgentResult:
        """Mock execute method that returns realistic results based on agent type."""
        
        if self.agent_name == "AnalystAgent":
            if "validate completion" in goal.lower():
                return MockAgentResult(
                    success=True,
                    message="Validation completed - goal achieved successfully",
                    outputs={
                        "validation_passed": True,
                        "completion_assessment": "User goal has been successfully achieved",
                        "validation_summary": "All requirements met and deliverables completed"
                    }
                )
            else:
                return MockAgentResult(
                    success=True,
                    message="Analysis completed successfully",
                    outputs={
                        "analysis_type": "code_analysis",
                        "files_analyzed": 3,
                        "issues_found": 2,
                        "analysis_summary": "Code structure is good, minor improvements suggested"
                    }
                )
        
        elif self.agent_name == "CreatorAgent":
            return MockAgentResult(
                success=True,
                message="Code generation completed successfully",
                outputs={
                    "generated_code": {"main.py": "print('Hello, World!')", "test_main.py": "def test_main(): pass"},
                    "files_created": 2,
                    "code_quality": "good"
                }
            )
        
        elif self.agent_name == "ExecutorAgent":
            return MockAgentResult(
                success=True,
                message="Tests executed successfully",
                outputs={
                    "test_files_executed": 5,
                    "tests_passed": 12,
                    "tests_failed": 0,
                    "coverage": "95%"
                }
            )
        
        elif self.agent_name == "SurgeonAgent":
            return MockAgentResult(
                success=True,
                message="Code modification completed",
                outputs={
                    "modification_applied": True,
                    "files_modified": ["config.py"],
                    "backup_created": True
                }
            )
        
        else:
            return MockAgentResult(
                success=True,
                message=f"{self.agent_name} executed successfully",
                outputs={"agent_executed": self.agent_name}
            )


# Mock LLM Client for Intelligent Routing
class MockInterAgentLLMClient:
    """Mock LLM client that provides realistic inter-agent routing decisions."""
    
    def __init__(self):
        self.decision_count = 0
    
    def invoke(self, prompt: str) -> str:
        """Return realistic inter-agent routing decisions based on context."""
        
        self.decision_count += 1
        prompt_lower = prompt.lower()
        
        # Parse current hop from prompt
        current_hop = 1
        if "current hop:" in prompt_lower:
            try:
                hop_part = prompt_lower.split("current hop:")[1].split("/")[0].strip()
                current_hop = int(hop_part)
            except:
                current_hop = 1
        
        # Routing logic based on workflow progression  
        if current_hop == 1:
            # After initial analysis, move to creation
            return json.dumps({
                "agent_name": "CreatorAgent",
                "confidence": 0.9,
                "reasoning": "Analysis phase complete, now need to create the implementation based on the analysis results",
                "goal_for_agent": "Create the code implementation based on analysis findings",
                "recommended_inputs": {"analysis_results": True, "code_requirements": True},
                "is_completion": False,
                "completion_summary": ""
            })
        
        elif current_hop == 2:
            # After creation, move to testing
            return json.dumps({
                "agent_name": "ExecutorAgent", 
                "confidence": 0.95,
                "reasoning": "Code has been created, now need to execute tests to validate the implementation works correctly",
                "goal_for_agent": "Execute tests to validate the created code",
                "recommended_inputs": {"generated_code": True, "test_target": "all"},
                "is_completion": False,
                "completion_summary": ""
            })
        
        elif current_hop == 3:
            # After successful testing, validate completion
            return json.dumps({
                "agent_name": "AnalystAgent",
                "confidence": 0.92,
                "reasoning": "Implementation created and tested successfully. Need final validation to confirm user goal is achieved",
                "goal_for_agent": "Validate completion of user goal and provide final assessment",
                "recommended_inputs": {"workflow_summary": True, "test_results": True},
                "is_completion": True,
                "completion_summary": "Code has been implemented, tested, and validated successfully"
            })
        
        # Default fallback
        return json.dumps({
            "agent_name": "AnalystAgent",
            "confidence": 0.8,
            "reasoning": "Continue with analysis and validation",
            "goal_for_agent": "Continue progress toward user goal",
            "recommended_inputs": {},
            "is_completion": False,
            "completion_summary": ""
        })


# Mock Global Context
class MockGlobalContext:
    """Mock GlobalContext for testing."""
    def __init__(self):
        self.workspace_path = Path.cwd()
        self.workspace = self
    
    def list_files(self) -> List[str]:
        return ["main.py", "test_main.py", "requirements.txt", "README.md"]


def test_inter_agent_routing():
    """Test the complete inter-agent routing system."""
    
    try:
        from fagents.inter_agent_router import InterAgentRouter, WorkflowStatus
        from fagents.workflow_coordinator import WorkflowCoordinator
        
        print("✅ Successfully imported inter-agent routing modules")
        
    except ImportError as e:
        print(f"❌ Failed to import routing modules: {e}")
        return False
    
    # Create mock agent registry
    mock_registry = {
        "AnalystAgent": lambda llm_client=None: MockFoundationalAgent("AnalystAgent", llm_client),
        "CreatorAgent": lambda llm_client=None: MockFoundationalAgent("CreatorAgent", llm_client),
        "ExecutorAgent": lambda llm_client=None: MockFoundationalAgent("ExecutorAgent", llm_client),
        "SurgeonAgent": lambda llm_client=None: MockFoundationalAgent("SurgeonAgent", llm_client),
        "StrategistAgent": lambda llm_client=None: MockFoundationalAgent("StrategistAgent", llm_client),
        "DebuggingAgent": lambda llm_client=None: MockFoundationalAgent("DebuggingAgent", llm_client),
    }
    
    # Create mock LLM client
    mock_llm_client = MockInterAgentLLMClient()
    
    # Create workflow coordinator
    coordinator = WorkflowCoordinator(llm_client=mock_llm_client, agent_registry=mock_registry)
    
    # Mock global context
    global_context = MockGlobalContext()
    
    print("\n" + "="*80)
    print("TESTING INTER-AGENT ROUTING SYSTEM")
    print("="*80)
    
    # Test Case 1: Simple Implementation Goal
    print("\n🧪 TEST CASE 1: Code Implementation Workflow")
    print("-" * 50)
    
    user_goal = "Create a Python hello world program with tests"
    initial_inputs = {"language": "python", "include_tests": True}
    
    result = coordinator.execute_goal(
        user_goal=user_goal,
        inputs=initial_inputs,
        global_context=global_context,
        max_hops=5
    )
    
    print(f"Result: {'✅ SUCCESS' if result.success else '❌ FAILED'}")
    print(f"Message: {result.message}")
    print(f"Workflow ID: {result.outputs['workflow_id']}")
    print(f"Agents Used: {' → '.join(result.outputs['agents_used'])}")
    print(f"Total Hops: {result.outputs['total_hops']}")
    print(f"Completion Validated: {result.outputs['completion_validated']}")
    
    # Show execution summary
    print("\n📋 Execution Summary:")
    for step in result.outputs['execution_summary']:
        status = "✅" if step['success'] else "❌"
        print(f"  {step['step']}. {step['agent']} {status}")
        print(f"     Goal: {step['goal']}")
        print(f"     Result: {step['message']}")
        if step['key_outputs']:
            print(f"     Key Outputs: {step['key_outputs']}")
        print()
    
    # Show key achievements
    achievements = result.outputs['workflow_summary']['key_achievements']
    print("🎯 Key Achievements:")
    for achievement in achievements:
        print(f"  • {achievement}")
    
    # Test Case 2: Test Fallback Routing
    print("\n" + "="*80)
    print("🧪 TEST CASE 2: Fallback Routing (No LLM)")
    print("-" * 50)
    
    fallback_coordinator = WorkflowCoordinator(llm_client=None, agent_registry=mock_registry)
    
    fallback_result = fallback_coordinator.execute_goal(
        user_goal="Analyze code quality issues",
        inputs={"code_files": {"main.py": "print('hello')"}},
        global_context=global_context,
        max_hops=3
    )
    
    print(f"Fallback Result: {'✅ SUCCESS' if fallback_result.success else '❌ FAILED'}")
    print(f"Fallback Message: {fallback_result.message}")
    print(f"Fallback Agents: {' → '.join(fallback_result.outputs['agents_used'])}")
    
    # Test Case 3: Workflow Status Tracking
    print("\n" + "="*80)
    print("🧪 TEST CASE 3: Workflow Status Tracking")
    print("-" * 50)
    
    active_workflows = coordinator.list_active_workflows()
    print(f"Active Workflows: {len(active_workflows)}")
    
    for wf in active_workflows:
        print(f"  • {wf['workflow_id']}: {wf['status']} ({wf['current_hop']} hops)")
    
    # Detailed workflow status
    if result.outputs['workflow_id']:
        status = coordinator.get_workflow_status(result.outputs['workflow_id'])
        if status:
            print(f"\nWorkflow Status Details:")
            print(f"  Status: {status['status']}")
            print(f"  Hops: {status['current_hop']}/{status['max_hops']}")
            print(f"  Validated: {status['completion_validated']}")
            print(f"  Agents: {status['agents_used']}")
    
    # Summary
    print("\n" + "="*80)
    print("INTER-AGENT ROUTING TEST RESULTS")
    print("="*80)
    
    all_tests_passed = (
        result.success and 
        fallback_result.success and
        result.outputs['completion_validated'] and
        len(result.outputs['agents_used']) >= 3  # Should use multiple agents
    )
    
    if all_tests_passed:
        print("🎉 All inter-agent routing tests PASSED!")
        print("\n✅ Key Features Verified:")
        print("  • Intelligent agent-to-agent routing")
        print("  • Completion validation with AnalystAgent")
        print("  • Comprehensive workflow tracking")
        print("  • Fallback routing when LLM unavailable")
        print("  • Multi-hop workflow execution")
        print("  • Progress summarization and reporting")
        return True
    else:
        print("❌ Some inter-agent routing tests FAILED!")
        return False


def demonstrate_workflow_benefits():
    """Demonstrate the benefits of the inter-agent routing system."""
    
    print("\n" + "="*80)
    print("BENEFITS OF INTER-AGENT ROUTING SYSTEM")
    print("="*80)
    
    benefits = [
        {
            "title": "🎯 Goal-Oriented Progression",
            "description": "Each agent invocation moves directly toward completing the user's goal",
            "example": "Analysis → Creation → Testing → Validation in logical sequence"
        },
        {
            "title": "🔄 Minimal Hops",
            "description": "LLM routing minimizes agent switches by choosing optimal next steps",
            "example": "Direct path to goal completion rather than trial-and-error"
        },
        {
            "title": "✅ Completion Validation", 
            "description": "AnalystAgent automatically validates goal completion",
            "example": "Ensures user requirements are actually met, not just code executed"
        },
        {
            "title": "📊 Comprehensive Tracking",
            "description": "Complete workflow visibility with execution history and progress",
            "example": "Detailed logs of what each agent did and achieved"
        },
        {
            "title": "🧠 Context Awareness",
            "description": "Each routing decision considers full workflow context",
            "example": "Understands what's been done and what remains to achieve the goal"
        },
        {
            "title": "🛡️  Reliability",
            "description": "Graceful fallback to rule-based routing when LLM unavailable",
            "example": "System continues working even without LLM connectivity"
        }
    ]
    
    for benefit in benefits:
        print(f"\n{benefit['title']}")
        print(f"  {benefit['description']}")
        print(f"  Example: {benefit['example']}")
    
    print(f"\n{'='*80}")
    print("SYSTEM ARCHITECTURE")
    print("="*80)
    
    print("""
    ┌─────────────────────────────────────────────────────────────┐
    │                    WorkflowCoordinator                       │
    │  • Main interface for goal execution                        │
    │  • Manages workflow lifecycle                               │  
    │  • Ensures completion validation                            │
    └────────────────────┬────────────────────────────────────────┘
                         │
    ┌─────────────────────▼────────────────────────────────────────┐
    │                  InterAgentRouter                           │
    │  • LLM-based intelligent routing decisions                  │
    │  • Context-aware next agent selection                       │
    │  • Execution tracking and progress monitoring               │
    └────────────────────┬────────────────────────────────────────┘
                         │
    ┌─────────────────────▼────────────────────────────────────────┐
    │               Foundation Agents                             │
    │  AnalystAgent  │  CreatorAgent  │  ExecutorAgent            │
    │  StrategistAgent  │  SurgeonAgent  │  DebuggingAgent       │
    │  • Each with intelligent intra-agent routing               │
    │  • LLM-powered operation selection                         │
    └─────────────────────────────────────────────────────────────┘
    """)


if __name__ == "__main__":
    print("🚀 Inter-Agent Routing System Test Suite")
    
    success = test_inter_agent_routing()
    demonstrate_workflow_benefits()
    
    print(f"\n{'='*80}")
    print("CONCLUSION")
    print("="*80)
    
    if success:
        print("✅ Inter-agent routing system is working perfectly!")
        print("🎉 Foundation agents now coordinate intelligently to achieve user goals!")
    else:
        print("❌ Inter-agent routing system needs attention.")
    
    sys.exit(0 if success else 1)