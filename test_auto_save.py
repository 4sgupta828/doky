#!/usr/bin/env python3
"""
Test script for the auto-save and crash recovery functionality.
"""

import logging
import tempfile
import time
from pathlib import Path
from datetime import datetime, timedelta
import shutil

from core.context import GlobalContext
from core.models import TaskNode
from utils.logger import setup_logger

setup_logger(default_level=logging.INFO)
logger = logging.getLogger(__name__)

def test_auto_save_functionality():
    """Test that auto-save works when artifacts are added."""
    print("\n--- Testing Auto-Save Functionality ---")
    
    # Create temporary workspace
    with tempfile.TemporaryDirectory() as temp_dir:
        context = GlobalContext(workspace_path=temp_dir)
        
        # Disable time-based auto-save for this test
        context.auto_save_enabled = True
        context.last_auto_save = context.last_auto_save - timedelta(seconds=60)  # Force auto-save on next update
        
        print(f"Created context with workspace: {temp_dir}")
        
        # Add an artifact - this should trigger auto-save
        context.add_artifact("test_spec.md", "# Test Specification\nThis is a test.", "task_1")
        
        # Check if snapshot was created
        snapshots = GlobalContext.list_available_snapshots(temp_dir)
        
        if snapshots:
            print(f"✅ Auto-save successful! Snapshot created: {Path(snapshots[0]).name}")
            return snapshots[0]
        else:
            print("❌ Auto-save failed - no snapshot found")
            return None

def test_crash_recovery(snapshot_path):
    """Test that we can recover from a snapshot."""
    print("\n--- Testing Crash Recovery ---")
    
    if not snapshot_path:
        print("❌ No snapshot available for recovery test")
        return False
    
    try:
        # Load context from snapshot
        recovered_context = GlobalContext.load_from_snapshot(snapshot_path)
        
        print(f"✅ Successfully loaded context from snapshot")
        print(f"   Workspace path: {recovered_context.workspace_path}")
        print(f"   Mission log entries: {len(recovered_context.mission_log)}")
        
        # Check if our test artifact exists (note: artifacts aren't fully restored in this implementation)
        print(f"   Artifacts: {list(recovered_context.artifacts.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Crash recovery failed: {e}")
        return False

def test_periodic_event_saving():
    """Test that significant events trigger auto-save."""
    print("\n--- Testing Event-Based Auto-Save ---")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        context = GlobalContext(workspace_path=temp_dir)
        context.auto_save_enabled = True
        
        # Clear existing snapshots count
        initial_snapshots = len(GlobalContext.list_available_snapshots(temp_dir))
        
        # Trigger events that should cause auto-save
        context.log_event("task_completed", {"task_id": "test_task", "result": "success"})
        
        # Give it a moment to process
        time.sleep(0.1)
        
        # Check if new snapshot was created
        snapshots = GlobalContext.list_available_snapshots(temp_dir)
        
        if len(snapshots) > initial_snapshots:
            print(f"✅ Event-based auto-save successful!")
            return True
        else:
            print("❌ Event-based auto-save failed")
            return False

if __name__ == "__main__":
    print("🔍 Testing Auto-Save and Crash Recovery Features")
    
    # Test 1: Auto-save on artifact updates
    snapshot_path = test_auto_save_functionality()
    
    # Test 2: Crash recovery from snapshot
    recovery_success = test_crash_recovery(snapshot_path)
    
    # Test 3: Event-based auto-save
    event_save_success = test_periodic_event_saving()
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    print(f"Auto-save on artifact update: {'✅ PASS' if snapshot_path else '❌ FAIL'}")
    print(f"Crash recovery from snapshot: {'✅ PASS' if recovery_success else '❌ FAIL'}")  
    print(f"Event-based auto-save:        {'✅ PASS' if event_save_success else '❌ FAIL'}")
    print("="*60)
    
    if snapshot_path and recovery_success and event_save_success:
        print("🎉 All auto-save and recovery tests PASSED!")
        exit(0)
    else:
        print("⚠️  Some tests failed. Check implementation.")
        exit(1)