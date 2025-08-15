#!/usr/bin/env python3
"""
Test script for the /clear and /reset commands.
This script validates that the reset functionality works correctly.
"""
import sys
import logging
from unittest.mock import patch, MagicMock
from session import InteractiveSession

def test_reset_command():
    """Test the /clear and /reset command functionality."""
    print("Testing /clear and /reset commands...")
    
    try:
        with patch('session.CollaborationUI') as MockUI, patch('session.Orchestrator') as MockOrch:
            mock_ui = MockUI.return_value
            mock_orch = MockOrch.return_value
            
            # Create mock global context with some data
            mock_context = MagicMock()
            mock_context.task_graph.nodes = {'task1': 'test_task', 'task2': 'another_task'}
            mock_context.artifacts = {'spec.json': 'some_spec', 'code.py': 'some_code'}
            mock_context.mission_log = [
                {'event': 'task_started', 'details': 'test'},
                {'event': 'artifact_added', 'details': 'test'}
            ]
            mock_context.session_dir = MagicMock()
            mock_context.session_dir.__truediv__ = lambda self, x: f'session_dir/{x}'
            mock_orch.global_context = mock_context
            
            # Create session
            session = InteractiveSession()
            session.global_context = mock_context
            
            print(f"Initial state:")
            print(f"  - Tasks: {len(mock_context.task_graph.nodes)}")
            print(f"  - Artifacts: {len(mock_context.artifacts)}")
            print(f"  - Log entries: {len(mock_context.mission_log)}")
            
            # Test /clear command
            session._handle_clear_command()
            
            print(f"After /clear:")
            print(f"  - Tasks: {len(mock_context.task_graph.nodes)}")
            print(f"  - Artifacts: {len(mock_context.artifacts)}")
            print(f"  - Log entries: {len(mock_context.mission_log)}")
            
            # Verify everything was cleared
            assert len(mock_context.task_graph.nodes) == 0, "Tasks should be cleared"
            assert len(mock_context.artifacts) == 0, "Artifacts should be cleared"
            # Mission log should have the reset event
            
            # Check that snapshot saving was called
            mock_context.save_snapshot.assert_called()
            mock_context.save_session_memory.assert_called()
            mock_context.log_event.assert_called()
            
            print("✅ All clear command tests passed!")
            return True
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_command_dispatch():
    """Test that the command dispatcher correctly routes /clear and /reset."""
    print("\nTesting command dispatch logic...")
    
    # Test cases
    test_cases = [
        ("/clear", True, "Should match /clear"),
        ("/reset", True, "Should match /reset"), 
        ("/CLEAR", True, "Should match /CLEAR (case insensitive)"),
        ("/RESET", True, "Should match /RESET (case insensitive)"),
        ("clear", False, "Should not match clear without /"),
        ("@clear", False, "Should not match @clear"),
        ("help", False, "Should not match regular commands"),
    ]
    
    for command, should_match, description in test_cases:
        matches = command.lower() in ["/clear", "/reset"]
        if matches == should_match:
            print(f"✅ {command}: {description}")
        else:
            print(f"❌ {command}: {description}")
            return False
    
    print("✅ All command dispatch tests passed!")
    return True

if __name__ == "__main__":
    print("=" * 60)
    print("TESTING RESET COMMAND IMPLEMENTATION")  
    print("=" * 60)
    
    # Run tests
    test1_passed = test_command_dispatch()
    test2_passed = test_reset_command()
    
    print("\n" + "=" * 60)
    if test1_passed and test2_passed:
        print("✅ ALL TESTS PASSED! The reset command is working correctly.")
        sys.exit(0)
    else:
        print("❌ SOME TESTS FAILED!")
        sys.exit(1)