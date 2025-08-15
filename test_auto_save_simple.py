#!/usr/bin/env python3
"""
Simple test for auto-save and recovery functionality.
"""

import logging
import tempfile
import time
from pathlib import Path
from datetime import datetime, timedelta

from core.context import GlobalContext
from utils.logger import setup_logger

setup_logger(default_level=logging.INFO)

def test_complete_flow():
    """Test the complete auto-save and recovery flow."""
    print("🔍 Testing Auto-Save and Recovery Flow")
    
    # Use a real temporary directory that persists for the duration of the test
    temp_dir = tempfile.mkdtemp()
    print(f"Using workspace: {temp_dir}")
    
    try:
        # Step 1: Create context and trigger auto-save
        print("\n1. Creating context and adding artifact...")
        context = GlobalContext(workspace_path=temp_dir)
        context.auto_save_enabled = True
        context.last_auto_save = context.last_auto_save - timedelta(seconds=60)  # Force save
        
        # Add artifact to trigger save
        context.add_artifact("test_spec.md", "# Test Spec\nContent here", "task_1")
        
        # Check snapshots
        snapshots = GlobalContext.list_available_snapshots(temp_dir)
        if not snapshots:
            print("❌ No snapshots created")
            return False
            
        print(f"✅ Snapshot created: {Path(snapshots[0]).name}")
        
        # Step 2: Test recovery
        print("\n2. Testing recovery from snapshot...")
        
        try:
            recovered_context = GlobalContext.load_from_snapshot(snapshots[0])
            print("✅ Successfully loaded context from snapshot")
            print(f"   Workspace: {recovered_context.workspace_path}")
            print(f"   Log entries: {len(recovered_context.mission_log)}")
            return True
            
        except Exception as e:
            print(f"❌ Recovery failed: {e}")
            return False
            
    finally:
        # Clean up
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

if __name__ == "__main__":
    success = test_complete_flow()
    if success:
        print("\n🎉 Auto-save and recovery test PASSED!")
    else:
        print("\n⚠️ Auto-save and recovery test FAILED!")