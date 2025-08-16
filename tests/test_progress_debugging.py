#!/usr/bin/env python3
"""
Debug why progress reporting isn't visible in our tests.
"""

import sys
import os
import tempfile
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from agents.tooling import ToolingAgent
from core.context import GlobalContext
from core.models import TaskNode
from core.instruction_schemas import create_diagnostic_instruction
from interfaces.progress_tracker import ProgressTracker
from utils.logger import setup_logger

# Set up logging with DEBUG level to see progress messages
setup_logger(default_level=logging.DEBUG)

class MockUI:
    """Mock UI interface that prints progress to console."""
    
    def display_agent_progress(self, agent_name: str, step: str, details: str = None):
        print(f"🔄 [{agent_name}] {step}")
        if details:
            print(f"   📋 {details}")
    
    def display_agent_thinking(self, agent_name: str, thought: str):
        print(f"🧠 [{agent_name}] {thought}")
    
    def display_intermediate_output(self, agent_name: str, output_type: str, content):
        print(f"📤 [{agent_name}] {output_type}: {str(content)[:100]}...")

def test_progress_reporting_with_ui():
    """Test progress reporting with a mock UI interface."""
    print("🧪 Testing Progress Reporting WITH UI Interface")
    print("=" * 55)
    
    # Create temporary workspace
    with tempfile.TemporaryDirectory() as temp_dir:
        context = GlobalContext(workspace_path=temp_dir)
        
        # Create progress tracker WITH UI interface
        mock_ui = MockUI()
        progress_tracker = ProgressTracker(ui_interface=mock_ui)
        
        # Create agent and set up progress tracking
        agent = ToolingAgent()
        task = TaskNode(
            goal="Test progress reporting",
            assigned_agent="ToolingAgent"
        )
        
        # Set up progress tracking (like Orchestrator does)
        progress_tracker.start_agent_progress("ToolingAgent", task.task_id, task.goal)
        agent.set_progress_tracker(progress_tracker, task.task_id)
        
        print("✅ Progress tracker set up. Testing progress reports...")
        
        # Test progress reporting directly
        agent.report_progress("Testing step 1", "This is a test step with details")
        agent.report_thinking("This is a test thought from the agent")
        agent.report_intermediate_output("test_output", {"key": "value", "data": "sample"})
        
        print("\n🎯 Progress reporting test complete!")
        return True

def test_progress_reporting_without_ui():
    """Test progress reporting without UI interface (current situation)."""
    print("\n🧪 Testing Progress Reporting WITHOUT UI Interface")
    print("=" * 55)
    
    # Create temporary workspace
    with tempfile.TemporaryDirectory() as temp_dir:
        context = GlobalContext(workspace_path=temp_dir)
        
        # Create progress tracker WITHOUT UI interface
        progress_tracker = ProgressTracker(ui_interface=None)
        
        # Create agent and set up progress tracking
        agent = ToolingAgent()
        task = TaskNode(
            goal="Test progress reporting",
            assigned_agent="ToolingAgent"
        )
        
        # Set up progress tracking
        progress_tracker.start_agent_progress("ToolingAgent", task.task_id, task.goal)
        agent.set_progress_tracker(progress_tracker, task.task_id)
        
        print("⚠️  Progress tracker set up WITHOUT UI. Reports only go to DEBUG log...")
        
        # Test progress reporting directly
        agent.report_progress("Testing step 1", "This should only appear in DEBUG logs")
        agent.report_thinking("This thought should only appear in DEBUG logs")
        agent.report_intermediate_output("test_output", {"key": "value", "data": "sample"})
        
        print("\n🎯 No visible progress (as expected without UI)")
        return True

def test_structured_tooling_with_progress():
    """Test structured tooling with proper progress reporting."""
    print("\n🧪 Testing Structured Tooling WITH Progress Reporting")
    print("=" * 55)
    
    # Create temporary workspace
    with tempfile.TemporaryDirectory() as temp_dir:
        context = GlobalContext(workspace_path=temp_dir)
        
        # Create progress tracker WITH UI interface
        mock_ui = MockUI()
        progress_tracker = ProgressTracker(ui_interface=mock_ui)
        
        # Create structured diagnostic instruction
        diagnostic_instruction = create_diagnostic_instruction(
            instruction_id="progress_test_001",
            commands=["echo 'Hello'", "date", "pwd"],
            purpose="Test structured tooling with progress reporting",
            timeout=30
        )
        
        # Store instruction in context
        context.add_artifact("tooling_instruction.json", diagnostic_instruction.model_dump_json(indent=2), "test")
        
        # Create agent and task
        agent = ToolingAgent()
        task = TaskNode(
            goal=f"Execute diagnostic instruction: {diagnostic_instruction.instruction_id}",
            assigned_agent="ToolingAgent",
            input_artifact_keys=["tooling_instruction.json"]
        )
        
        # Set up progress tracking (like Orchestrator does)
        progress_tracker.start_agent_progress("ToolingAgent", task.task_id, task.goal)
        agent.set_progress_tracker(progress_tracker, task.task_id)
        
        print("✅ Everything set up. Executing structured tooling with progress...")
        print()
        
        # Execute with progress reporting
        result = agent.execute(f"Execute diagnostic: {diagnostic_instruction.instruction_id}", context, task)
        
        print(f"\n🎯 Result: {'✅' if result.success else '❌'} {result.success}")
        return result.success

def main():
    """Test progress reporting in different scenarios."""
    print("🔍 Debugging Progress Reporting Visibility")
    print("=" * 60)
    
    tests = [
        ("Progress WITH UI", test_progress_reporting_with_ui),
        ("Progress WITHOUT UI", test_progress_reporting_without_ui),
        ("Structured Tooling + Progress", test_structured_tooling_with_progress)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result, None))
        except Exception as e:
            results.append((test_name, False, str(e)))
            print(f"\n❌ {test_name}: ERROR - {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 PROGRESS DEBUG SUMMARY")
    print("=" * 60)
    
    for test_name, success, error in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status:8} {test_name}")
        if error:
            print(f"         Error: {error}")
    
    print("\n🎯 SOLUTION:")
    print("   The issue is that progress reports need a UI interface to be visible.")
    print("   In tests, we either need to:")
    print("   1. Set up a MockUI interface (as shown above)")
    print("   2. Or use the Orchestrator which handles this automatically")
    print("   3. Or modify ProgressTracker to show reports even without UI")

if __name__ == "__main__":
    main()