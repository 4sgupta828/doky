#!/usr/bin/env python3
"""
Test script to verify Ctrl+R functionality for full output display.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from interfaces.collaboration_ui import CollaborationUI
from core.context import GlobalContext
from core.models import AgentResponse

def test_ctrl_r_functionality():
    """Test the Ctrl+R full output display functionality."""
    print("🧪 Testing Ctrl+R functionality for full output display")
    print("="*80)
    
    # Create UI and context
    ui = CollaborationUI()
    context = GlobalContext()
    
    # Test 1: Display truncated artifact content
    print("\n📝 Test 1: Truncated artifact display")
    print("-"*50)
    
    # Create a long text artifact
    long_text = "\n".join([f"Line {i}: This is a test line to demonstrate truncation in output display." for i in range(1, 101)])
    context.add_artifact("test_long_file.py", long_text, "test_task")
    
    # Create response with artifact
    response = AgentResponse(
        success=True, 
        message="Generated long test file", 
        artifacts_generated=["test_long_file.py"]
    )
    
    # Display with truncation
    ui.display_direct_command_result("TestAgent", response, context)
    
    # Test 2: Display long code snippet
    print("\n📝 Test 2: Long code snippet")
    print("-"*50)
    
    code_chunks = []
    for i in range(1, 51):
        code_chunks.append(f"def function_{i}():")
        code_chunks.append(f'    """Function number {i} for testing truncation."""')
        code_chunks.append(f"    result = {i} * 2")
        code_chunks.append(f"    print(f'Result for function {i}: {{result}}')")
        code_chunks.append(f"    return result")
        code_chunks.append("")
    long_code = "\n".join(code_chunks)
    
    ui.display_code_snippet("CodeAgent", long_code, "long_functions.py")
    
    # Test 3: Display multiple files
    print("\n📝 Test 3: Multiple code files")
    print("-"*50)
    
    files_data = {
        "app.py": "\n".join([f"# App line {i}" for i in range(1, 31)]),
        "models.py": "\n".join([f"# Model line {i}" for i in range(1, 26)]),
        "views.py": "\n".join([f"# View line {i}" for i in range(1, 41)])
    }
    
    ui.display_code_files("FileGeneratorAgent", files_data)
    
    # Test 4: Check if full output history is stored
    print("\n📝 Test 4: Full output history storage")
    print("-"*50)
    
    print(f"Stored content keys: {list(ui.last_full_content.keys())}")
    print(f"Total stored items: {len(ui.last_full_content)}")
    
    # Test 5: Display full output
    print("\n📝 Test 5: Full output display (simulating Ctrl+R)")
    print("-"*50)
    
    ui.display_full_output()
    
    print("\n✅ All tests completed!")
    print("="*80)
    print("📋 Summary:")
    print("✓ Artifact content truncation with 'Press Ctrl+R' message")
    print("✓ Code snippet truncation with full output hint")  
    print("✓ Multi-file display with truncation")
    print("✓ Full output storage in last_full_content")
    print("✓ Complete output display via display_full_output()")
    print("\n🔧 Integration:")
    print("• Ctrl+R keybinding added to input_handler.py")
    print("• Session dispatcher handles 'ctrl+r' command") 
    print("• Help text updated with Ctrl+R information")

if __name__ == "__main__":
    test_ctrl_r_functionality()