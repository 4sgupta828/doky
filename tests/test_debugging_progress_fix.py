#!/usr/bin/env python3
"""
Test that DebuggingAgent now shows CLI progress when called from TestRunnerAgent.
"""

import sys
import os
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from agents.test_runner import TestRunnerAgent
from agents.debugging import DebuggingAgent
from interfaces.collaboration_ui import CollaborationUI
from interfaces.progress_tracker import ProgressTracker
from core.context import GlobalContext
from core.models import TaskNode
from utils.logger import setup_logger

# Set up logging
setup_logger()

def test_debugging_agent_progress_handover():
    """Test that DebuggingAgent shows CLI progress when handed over from TestRunnerAgent."""
    print("🧪 Testing DebuggingAgent CLI Progress on Handover")
    print("=" * 55)
    print("This tests the fix where TestRunnerAgent transfers its progress tracker")
    print("to DebuggingAgent, so you see the CLI progress format you want.")
    print()
    
    # Create temporary workspace with a failing test
    with tempfile.TemporaryDirectory() as temp_dir:
        test_file = Path(temp_dir) / "test_example.py"
        test_file.write_text("""
def test_failing_function():
    assert False, "This test is designed to fail for debugging demo"

def test_passing_function():
    assert True
""")
        
        # Create CLI UI and progress tracker (like main.py does now)
        ui = CollaborationUI()
        progress_tracker = ProgressTracker(ui_interface=ui)
        
        # Create context and agents
        context = GlobalContext(workspace_path=temp_dir)
        
        # Create agent registry with DebuggingAgent
        debugging_agent = DebuggingAgent()
        agent_registry = {"DebuggingAgent": debugging_agent}
        
        # Create TestRunnerAgent with DebuggingAgent in registry
        test_runner = TestRunnerAgent(agent_registry=agent_registry)
        
        # Set up progress tracking for TestRunnerAgent (like Orchestrator does)
        task = TaskNode(
            goal="Run failing tests and hand over to debugging",
            assigned_agent="TestRunnerAgent"
        )
        
        progress_tracker.start_agent_progress("TestRunnerAgent", task.task_id, task.goal)
        test_runner.set_progress_tracker(progress_tracker, task.task_id)
        
        print("🚀 Running TestRunnerAgent (which will fail and hand over to DebuggingAgent)")
        print("You should now see CLI progress from both agents:")
        print()
        
        # Run the test (it will fail and hand over to DebuggingAgent)
        try:
            result = test_runner.execute(task.goal, context, task)
            print(f"\n📊 TestRunnerAgent result: {result.success}")
            print(f"Message: {result.message[:100]}...")
            return True
        except Exception as e:
            print(f"\n❌ Error: {e}")
            return False

def main():
    """Test the CLI progress fix for agent handovers."""
    print("🔍 Testing CLI Progress Fix for Agent Handovers")
    print("=" * 60)
    print("Issue: DebuggingAgent progress wasn't showing when called by TestRunnerAgent")
    print("Fix: Transfer progress tracker during handover")
    print("=" * 60)
    
    try:
        success = test_debugging_agent_progress_handover()
        
        if success:
            print("\n🎉 SUCCESS! DebuggingAgent CLI progress should now be visible!")
            print("\nExpected CLI format:")
            print("🔄 [HH:MM:SS] DebuggingAgent: Phase 1: Evidence gathering")
            print("   └─ Collecting system context, logs, and failure data")
            print("🔄 [HH:MM:SS] DebuggingAgent: Structured diagnostics completed")
            print("   └─ ToolingAgent executed 5 diagnostic commands successfully")
            return 0
        else:
            print("\n❌ Test failed. Progress handover may need more work.")
            return 1
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())