#!/usr/bin/env python3
"""
Test that artifact keys are correctly passed between agents during handovers.
"""

import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from agents.test_runner import TestRunnerAgent
from agents.debugging import DebuggingAgent
from agents.tooling import ToolingAgent
from interfaces.collaboration_ui import CollaborationUI
from interfaces.progress_tracker import ProgressTracker
from core.context import GlobalContext
from core.models import TaskNode
from utils.logger import setup_logger

# Set up logging
setup_logger()

class MockAgent:
    """Mock agent to capture what artifact keys are passed."""
    
    def __init__(self, name):
        self.name = name
        self.received_artifact_keys = None
        self.received_task = None
        
    def set_progress_tracker(self, tracker, task_id):
        pass
        
    def execute(self, goal, context, task):
        self.received_artifact_keys = task.input_artifact_keys
        self.received_task = task
        return MagicMock(success=True, message="Mock success", artifacts_generated=[])

def test_artifact_key_passing():
    """Test that specific artifact keys are passed correctly between agents."""
    print("🧪 Testing Artifact Key Passing Between Agents")
    print("=" * 55)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create CLI UI and progress tracker
        ui = CollaborationUI()
        progress_tracker = ProgressTracker(ui_interface=ui)
        context = GlobalContext(workspace_path=temp_dir)
        
        # Create mock agents to capture artifact keys
        mock_debugging_agent = MockAgent("DebuggingAgent")
        mock_tooling_agent = MockAgent("ToolingAgent")
        mock_script_executor = MockAgent("ScriptExecutorAgent")
        
        # Create TestRunnerAgent with mock DebuggingAgent
        test_runner = TestRunnerAgent(agent_registry={"DebuggingAgent": mock_debugging_agent})
        
        # Create DebuggingAgent with mock sub-agents
        debugging_agent = DebuggingAgent(agent_registry={
            "ToolingAgent": mock_tooling_agent,
            "ScriptExecutorAgent": mock_script_executor
        })
        
        # Set up progress tracking
        main_task = TaskNode(
            goal="Test artifact key passing",
            assigned_agent="TestRunnerAgent",
            input_artifact_keys=["original_input.json"]  # Original task has different artifacts
        )
        
        progress_tracker.start_agent_progress("TestRunnerAgent", main_task.task_id, main_task.goal)
        test_runner.set_progress_tracker(progress_tracker, main_task.task_id)
        debugging_agent.set_progress_tracker(progress_tracker, "debug_task")
        
        print("🔄 Testing TestRunnerAgent → DebuggingAgent handover")
        
        # Create test failure artifacts (like TestRunnerAgent would)
        context.add_artifact("failed_test_report.json", {"failures": ["test1"]}, main_task.task_id)
        context.add_artifact("test_execution_report.json", {"summary": "tests failed"}, main_task.task_id)
        
        # Simulate TestRunnerAgent calling DebuggingAgent
        result = test_runner.call_agent_with_progress(
            mock_debugging_agent,
            "Debug test failures",
            context,
            main_task,
            "debug_task",
            input_artifact_keys=["failed_test_report.json", "test_execution_report.json"]
        )
        
        print(f"✅ TestRunnerAgent → DebuggingAgent:")
        print(f"   Expected artifacts: ['failed_test_report.json', 'test_execution_report.json']")
        print(f"   Received artifacts: {mock_debugging_agent.received_artifact_keys}")
        print(f"   Correct: {'✅' if mock_debugging_agent.received_artifact_keys == ['failed_test_report.json', 'test_execution_report.json'] else '❌'}")
        
        print("\n🔄 Testing DebuggingAgent → ToolingAgent call")
        
        # Create tooling instruction artifact (like DebuggingAgent would)
        context.add_artifact("tooling_instruction.json", {"commands": ["ls", "pwd"]}, "debug_task")
        
        # Simulate DebuggingAgent calling ToolingAgent
        debug_task = TaskNode(goal="Debug task", assigned_agent="DebuggingAgent", task_id="debug_task")
        debugging_agent.call_agent_with_progress(
            mock_tooling_agent,
            "Execute diagnostic instruction",
            context,
            debug_task,
            "tooling_subtask",
            input_artifact_keys=["tooling_instruction.json"]
        )
        
        print(f"✅ DebuggingAgent → ToolingAgent:")
        print(f"   Expected artifacts: ['tooling_instruction.json']")
        print(f"   Received artifacts: {mock_tooling_agent.received_artifact_keys}")
        print(f"   Correct: {'✅' if mock_tooling_agent.received_artifact_keys == ['tooling_instruction.json'] else '❌'}")
        
        print("\n🔄 Testing DebuggingAgent → ScriptExecutorAgent call")
        
        # Create script instruction artifact (like DebuggingAgent would)
        context.add_artifact("instruction_script.json", {"instructions": []}, "debug_task")
        
        # Simulate DebuggingAgent calling ScriptExecutorAgent
        debugging_agent.call_agent_with_progress(
            mock_script_executor,
            "Execute repair script",
            context,
            debug_task,
            "script_subtask",
            input_artifact_keys=["instruction_script.json"]
        )
        
        print(f"✅ DebuggingAgent → ScriptExecutorAgent:")
        print(f"   Expected artifacts: ['instruction_script.json']")
        print(f"   Received artifacts: {mock_script_executor.received_artifact_keys}")
        print(f"   Correct: {'✅' if mock_script_executor.received_artifact_keys == ['instruction_script.json'] else '❌'}")
        
        # Check that original task artifact keys didn't leak through
        original_leaked = any(
            agent.received_artifact_keys and "original_input.json" in agent.received_artifact_keys
            for agent in [mock_debugging_agent, mock_tooling_agent, mock_script_executor]
        )
        
        print(f"\n🛡️  Original task artifacts didn't leak: {'✅' if not original_leaked else '❌'}")
        
        return (
            mock_debugging_agent.received_artifact_keys == ["failed_test_report.json", "test_execution_report.json"] and
            mock_tooling_agent.received_artifact_keys == ["tooling_instruction.json"] and
            mock_script_executor.received_artifact_keys == ["instruction_script.json"] and
            not original_leaked
        )

def main():
    """Test artifact key passing implementation."""
    print("🔗 Testing Artifact Key Passing Implementation")
    print("=" * 60)
    
    try:
        success = test_artifact_key_passing()
        
        if success:
            print("\n🎉 SUCCESS! Artifact keys are passed correctly!")
            print("✅ TestRunnerAgent → DebuggingAgent gets specific test failure artifacts")
            print("✅ DebuggingAgent → ToolingAgent gets tooling instruction")
            print("✅ DebuggingAgent → ScriptExecutorAgent gets repair script")
            print("✅ No artifact key leakage between unrelated agents")
            return 0
        else:
            print("\n❌ Some artifact key tests failed.")
            return 1
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())