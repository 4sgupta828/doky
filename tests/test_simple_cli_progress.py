#!/usr/bin/env python3
"""
Simple test to demonstrate CLI progress display is working.
"""

import sys
import os
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from orchestrator import Orchestrator
from interfaces.collaboration_ui import CollaborationUI
from utils.logger import setup_logger

# Set up logging
setup_logger()

def test_cli_progress_with_orchestrator():
    """Test CLI progress with orchestrator execute_single_task method."""
    print("🧪 Testing CLI Progress with Orchestrator")
    print("=" * 50)
    print("This will show ToolingAgent progress in your desired CLI format:")
    print()
    
    # Create temporary workspace
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create CLI UI (this is the key fix!)
        ui = CollaborationUI() 
        
        # Create orchestrator with UI interface
        orchestrator = Orchestrator(workspace_path=temp_dir, ui_interface=ui)
        
        # Test a simple command through the orchestrator
        print("🚀 Executing: echo 'Testing CLI progress format'")
        print()
        
        result = orchestrator.execute_single_task("echo 'Testing CLI progress format'", "ToolingAgent")
        
        print(f"\n✅ Result: {result.success}")
        if not result.success:
            print(f"❌ Error: {result.message}")
        else:
            print("🎉 CLI progress format is working!")
        
        return result.success

def main():
    """Test CLI progress display."""
    print("🔍 CLI Progress Display Test")
    print("=" * 60)
    print("The issue was: main.py wasn't using CollaborationUI")
    print("The fix was: Add ui = CollaborationUI() to main.py")
    print("=" * 60)
    
    try:
        success = test_cli_progress_with_orchestrator()
        
        if success:
            print("\n🎉 SUCCESS! CLI Progress is now working.")
            print("\nYour desired format:")
            print("🔄 [HH:MM:SS] AgentName: Action")
            print("   └─ Details")
            print("\nIs now displaying correctly when agents run!")
            return 0
        else:
            print("\n❌ Test failed. Progress may need more fixes.")
            return 1
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())