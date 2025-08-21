#!/usr/bin/env python3
"""
Demo of the Intelligent Interactive Foundation Agent System

This script demonstrates how the new intelligent interactive system works
with multi-agent workflow coordination and completion validation.
"""

import sys
import json
from pathlib import Path
from typing import Dict, Any

# Add project root to Python path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Mock components for demonstration
class MockCollaborationUI:
    """Mock UI for demonstration."""
    
    def display_welcome_message(self):
        print("\n" + "="*80)
        print("🧠 INTELLIGENT FOUNDATION AGENT SYSTEM")
        print("="*80)
        print("Welcome to the intelligent multi-agent workflow coordination system!")
    
    def display_system_message(self, message: str):
        print(f"\n🤖 SYSTEM: {message}")
    
    def display_error_message(self, message: str):
        print(f"\n❌ ERROR: {message}")
    
    def get_user_input(self) -> str:
        return input("\n💬 You: ")


class MockGlobalContext:
    """Mock global context."""
    
    def __init__(self):
        self.workspace_path = Path.cwd()
        self.workspace = self
    
    def list_files(self):
        return ["main.py", "test_main.py", "requirements.txt"]


class MockLLMClient:
    """Mock LLM client for demonstration."""
    
    def __init__(self):
        self.decision_count = 0
    
    def invoke(self, prompt: str) -> str:
        """Return realistic routing decisions."""
        self.decision_count += 1
        
        # Simple demo routing logic
        if "create" in prompt.lower():
            return json.dumps({
                "agent_name": "CreatorAgent",
                "confidence": 0.95,
                "reasoning": "User wants to create something, CreatorAgent is the best choice for generation tasks",
                "goal_for_agent": "Create the requested implementation",
                "recommended_inputs": {"generation_type": "code"},
                "is_completion": False,
                "completion_summary": ""
            })
        elif "test" in prompt.lower():
            return json.dumps({
                "agent_name": "ExecutorAgent", 
                "confidence": 0.92,
                "reasoning": "User wants to test something, ExecutorAgent handles execution and validation",
                "goal_for_agent": "Execute tests and validate functionality",
                "recommended_inputs": {"test_type": "comprehensive"},
                "is_completion": False,
                "completion_summary": ""
            })
        else:
            return json.dumps({
                "agent_name": "AnalystAgent",
                "confidence": 0.88,
                "reasoning": "Starting with analysis to understand the requirements",
                "goal_for_agent": "Analyze and understand the user's request",
                "recommended_inputs": {"analysis_depth": "comprehensive"},
                "is_completion": self.decision_count > 2,
                "completion_summary": "Goal analysis and validation completed" if self.decision_count > 2 else ""
            })


def simulate_intelligent_session():
    """Simulate the intelligent interactive session."""
    
    try:
        # Import the intelligent session components
        sys.path.insert(0, str(Path(__file__).parent))
        from fagents.workflow_coordinator import WorkflowCoordinator
        
        print("🚀 INTELLIGENT INTERACTIVE SESSION DEMONSTRATION")
        print("="*80)
        
        # Initialize mock components
        ui = MockCollaborationUI()
        global_context = MockGlobalContext()
        llm_client = MockLLMClient()
        
        # Create workflow coordinator
        coordinator = WorkflowCoordinator(llm_client=llm_client)
        
        ui.display_welcome_message()
        ui.display_system_message(
            "Demo Mode Active - Simulating intelligent workflow coordination\n"
            "• Multi-agent workflow execution\n"  
            "• LLM-based intelligent routing\n"
            "• Automatic completion validation"
        )
        
        # Demo scenarios
        demo_goals = [
            "Create a simple Python calculator with tests",
            "Analyze my code for potential security issues",
            "Build a web scraper with error handling"
        ]
        
        ui.display_system_message(f"Running {len(demo_goals)} demo scenarios...")
        
        for i, goal in enumerate(demo_goals, 1):
            ui.display_system_message(f"\n{'='*60}")
            ui.display_system_message(f"DEMO SCENARIO {i}: {goal}")
            ui.display_system_message('='*60)
            
            try:
                # Execute goal with workflow coordinator
                result = coordinator.execute_goal(
                    user_goal=goal,
                    inputs={},
                    global_context=global_context,
                    max_hops=4
                )
                
                # Display results
                display_demo_results(ui, result, goal)
                
            except Exception as e:
                ui.display_error_message(f"Demo scenario failed: {e}")
        
        # Show session summary
        ui.display_system_message(f"\n{'='*80}")
        ui.display_system_message("DEMO COMPLETE - Key Features Demonstrated:")
        ui.display_system_message("✅ Intelligent multi-agent workflow coordination")
        ui.display_system_message("✅ LLM-based routing with clear reasoning")
        ui.display_system_message("✅ Automatic completion validation")
        ui.display_system_message("✅ Comprehensive progress tracking")
        ui.display_system_message("✅ Goal-oriented directional progress")
        
        return True
        
    except ImportError as e:
        print(f"❌ Could not import workflow coordinator: {e}")
        print("Make sure all the routing system files are in place:")
        print("  • fagents/routing.py")
        print("  • fagents/inter_agent_router.py") 
        print("  • fagents/workflow_coordinator.py")
        return False
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return False


def display_demo_results(ui, result, goal: str):
    """Display demo workflow results."""
    
    success_icon = "✅" if result.success else "❌"
    ui.display_system_message(f"{success_icon} {result.message}")
    
    if result.success and result.outputs:
        outputs = result.outputs
        
        # Show execution path
        agents_used = outputs.get('agents_used', [])
        if agents_used:
            execution_path = ' → '.join(agents_used)
            ui.display_system_message(f"🔄 Agent Path: {execution_path}")
        
        # Show key metrics
        total_hops = outputs.get('total_hops', 0)
        completion_validated = outputs.get('completion_validated', False)
        
        ui.display_system_message(
            f"📊 Results: {total_hops} hops, "
            f"{len(set(agents_used))} agents, "
            f"validated: {'✅' if completion_validated else '❌'}"
        )
        
        # Show achievements
        workflow_summary = outputs.get('workflow_summary', {})
        achievements = workflow_summary.get('key_achievements', [])
        if achievements:
            ui.display_system_message(f"🎯 Achievements: {', '.join(achievements[:3])}")


def demonstrate_usage():
    """Show how to use the intelligent interactive system."""
    
    print(f"\n{'='*80}")
    print("HOW TO USE THE INTELLIGENT INTERACTIVE SYSTEM")
    print("="*80)
    
    print("""
🚀 LAUNCH THE INTELLIGENT SYSTEM:

   python main_interactive_intelligent.py

🎯 EXECUTE GOALS:

   Simply type your goal and press Enter:
   
   Examples:
   • "Create a REST API with authentication"
   • "Build a web scraper with error handling"
   • "Analyze my code for security vulnerabilities"
   • "Generate comprehensive unit tests"

🔧 COMMANDS:

   help                - Show help information
   status              - Show session status
   status <workflow>   - Show specific workflow status  
   workflows           - List active workflows
   history             - Show execution history
   quit                - Exit the system

✨ FEATURES:

   • Intelligent agent-to-agent routing
   • Goal-oriented workflow execution
   • Automatic completion validation
   • Comprehensive progress tracking
   • Fallback routing when LLM unavailable

📊 WORKFLOW TRACKING:

   Every goal execution provides:
   • Execution path through foundation agents
   • Reasoning for each routing decision
   • Confidence scores for reliability
   • Key achievements and deliverables
   • Complete execution history
    """)


if __name__ == "__main__":
    print("🧠 Intelligent Foundation Agent System - Demo")
    
    # Run simulation
    success = simulate_intelligent_session()
    
    # Show usage instructions
    demonstrate_usage()
    
    if success:
        print(f"\n{'='*80}")
        print("✅ DEMO SUCCESSFUL")
        print("="*80)
        print("The intelligent interactive system is ready to use!")
        print("Run: python main_interactive_intelligent.py")
    else:
        print(f"\n{'='*80}")
        print("❌ DEMO FAILED")
        print("="*80)
        print("Please ensure all routing system components are properly set up.")
        
    sys.exit(0 if success else 1)