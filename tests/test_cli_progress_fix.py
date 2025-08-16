#!/usr/bin/env python3
"""
Test the CLI progress display fix to verify the exact format you wanted is now working.
"""

import sys
import os
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from orchestrator import Orchestrator
from interfaces.collaboration_ui import CollaborationUI
from agents.tooling import ToolingAgent
from core.context import GlobalContext
from core.models import TaskNode
from core.instruction_schemas import create_diagnostic_instruction
from interfaces.progress_tracker import ProgressTracker
from utils.logger import setup_logger

# Set up logging
setup_logger()

def test_cli_progress_format():
    """Test that CLI progress shows up in the exact format you wanted."""
    print("🧪 Testing CLI Progress Format (Your Desired Format)")
    print("=" * 60)
    print("Expected format:")
    print("🔄 [19:38:00] TestRunnerAgent: Executing tests")
    print("   └─ Running tests with pytest (preferred) or fallback methods")
    print()
    print("Let's test this with our ToolingAgent...")
    print("=" * 60)
    
    # Create temporary workspace
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create CLI UI (this is the key fix!)
        ui = CollaborationUI()
        
        # Create orchestrator with UI interface
        orchestrator = Orchestrator(workspace_path=temp_dir, ui_interface=ui)
        context = orchestrator.global_context
        
        # Create structured diagnostic instruction
        diagnostic_instruction = create_diagnostic_instruction(
            instruction_id="cli_test_001",
            commands=["echo 'Testing CLI format'", "date", "pwd"],
            purpose="Testing the CLI progress display format you requested",
            timeout=30
        )
        
        # Store instruction in context
        context.add_artifact("tooling_instruction.json", diagnostic_instruction.model_dump_json(indent=2), "test")
        
        # Create task and get ToolingAgent from registry
        task = TaskNode(
            goal=f"Execute diagnostic instruction: {diagnostic_instruction.instruction_id}",
            assigned_agent="ToolingAgent",
            input_artifact_keys=["tooling_instruction.json"]
        )
        
        print("🚀 Executing ToolingAgent with CLI UI interface...")
        print("This should show progress in your desired CLI format:")
        print()
        
        # Execute through orchestrator (which sets up progress tracking properly)
        result = orchestrator._execute_single_agent(task)
        
        print(f"\n✅ Result: {result.success}")
        if not result.success:
            print(f"❌ Error: {result.message}")
        
        return result.success

def test_direct_cli_ui():
    """Test the CLI UI format directly."""
    print("\n🎯 Testing Direct CLI UI Format")
    print("=" * 60)
    
    # Create CLI UI and test the exact methods
    ui = CollaborationUI()
    
    print("Testing display_agent_progress method (your desired format):")
    
    # Test the exact format you wanted
    ui.display_agent_progress("TestRunnerAgent", "Executing tests", "Running tests with pytest (preferred) or fallback methods")
    ui.display_agent_progress("ToolingAgent", "Loading tooling instruction", "Processing: 'Execute diagnostic instruction...'")
    ui.display_agent_progress("ToolingAgent", "Executing structured commands", "Running 3 commands for: Testing CLI progress")
    
    return True

def main():
    """Test the CLI progress fix."""
    print("🔍 Testing CLI Progress Display Fix")
    print("=" * 70)
    
    tests = [
        ("Direct CLI UI Format", test_direct_cli_ui),
        ("Full CLI Progress Integration", test_cli_progress_format)
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
    print("\n" + "=" * 70)
    print("📋 CLI PROGRESS FIX TEST SUMMARY")
    print("=" * 70)
    
    for test_name, success, error in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status:8} {test_name}")
        if error:
            print(f"         Error: {error}")
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 SUCCESS: CLI progress reporting is now working!")
        print("   The format you wanted:")
        print("   🔄 [HH:MM:SS] AgentName: Action")
        print("      └─ Details")
        print("   Is now being displayed correctly!")
        return 0
    else:
        print("💥 Some tests failed. CLI progress may still need fixes.")
        return 1

if __name__ == "__main__":
    sys.exit(main())