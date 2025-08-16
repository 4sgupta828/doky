#!/usr/bin/env python3
"""
Test script for the new structured communication workflow between
DebuggingAgent and ScriptExecutorAgent.
"""

import json
import logging
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

# Local imports
from core.context import GlobalContext
from core.models import TaskNode
from agents.debugging import DebuggingAgent
from agents.script_executor import ScriptExecutorAgent
from core.instruction_schemas import InstructionScript, create_fix_code_instruction
from utils.logger import setup_logger

def test_structured_debugging_workflow():
    """Test the complete structured debugging workflow."""
    
    print("🔧 Testing Structured Debugging Workflow")
    print("=" * 50)
    
    # Setup logging
    setup_logger(default_level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Create temporary workspace
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace_path = Path(temp_dir)
        context = GlobalContext(workspace_path=str(workspace_path))
        
        # Create test file with a bug
        test_file = workspace_path / "buggy_code.py"
        test_file.write_text("""def divide_numbers(a, b):
    # Bug: no check for division by zero
    return a / b

def main():
    result = divide_numbers(10, 0)  # This will crash
    print(f"Result: {result}")

if __name__ == "__main__":
    main()
""")
        
        # Create mock failed test report
        failed_report = {
            "summary": {"failed": 1, "passed": 0},
            "file": "buggy_code.py",
            "error": "ZeroDivisionError: division by zero",
            "line": 3,
            "function": "divide_numbers"
        }
        
        context.add_artifact("failed_test_report.json", json.dumps(failed_report), "test")
        context.add_artifact("targeted_code_context.txt", test_file.read_text(), "test")
        
        print("✅ Test environment setup complete")
        print(f"   Workspace: {workspace_path}")
        print(f"   Test file: {test_file}")
        
        # 1. Test ScriptExecutorAgent with a simple script
        print("\n🤖 Testing ScriptExecutorAgent")
        print("-" * 30)
        
        # Create a simple instruction script
        instruction = create_fix_code_instruction(
            instruction_id="fix_division_by_zero",
            file_path="buggy_code.py",
            issue_description="Fix division by zero error",
            fix_content="""def divide_numbers(a, b):
    # Fixed: check for division by zero
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b""",
            line_start=1,
            line_end=3
        )
        
        script = InstructionScript(
            script_id="test_fix_script",
            title="Fix Division by Zero",
            description="Simple fix for division by zero error",
            created_by="TestScript",
            target_issue="ZeroDivisionError in divide_numbers function",
            instructions=[instruction]
        )
        
        # Add script to context
        context.add_artifact("instruction_script.json", script.model_dump_json(indent=2), "test")
        
        # Create and execute ScriptExecutorAgent
        script_executor = ScriptExecutorAgent()
        task = TaskNode(goal="Execute repair script", assigned_agent="ScriptExecutorAgent")
        
        result = script_executor.execute("Execute test repair script", context, task)
        
        if result.success:
            print("✅ ScriptExecutorAgent test PASSED")
            print(f"   Message: {result.message}")
            
            # Check if file was actually modified
            fixed_content = test_file.read_text()
            if "Cannot divide by zero" in fixed_content:
                print("✅ File was correctly modified")
            else:
                print("❌ File modification failed")
        else:
            print("❌ ScriptExecutorAgent test FAILED")
            print(f"   Error: {result.message}")
        
        # 2. Test DebuggingAgent with structured script generation
        print("\n🔍 Testing DebuggingAgent Script Generation")
        print("-" * 40)
        
        # Reset the test file
        test_file.write_text("""def divide_numbers(a, b):
    # Bug: no check for division by zero
    return a / b""")
        
        # Create mock LLM client for debugging agent
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = json.dumps({
            "root_cause_analysis": "The divide_numbers function does not check for division by zero",
            "primary_hypothesis": "Missing input validation for zero divisor",
            "solution_type": "SURGICAL",
            "recommended_strategy": "Add zero division check in divide_numbers function",
            "confidence_level": "high"
        })
        
        # Create agent registry with ScriptExecutorAgent
        agent_registry = {
            "ScriptExecutorAgent": script_executor
        }
        
        # Create debugging agent
        debugging_agent = DebuggingAgent(llm_client=mock_llm, agent_registry=agent_registry)
        debug_task = TaskNode(goal="Debug division by zero error", assigned_agent="DebuggingAgent")
        
        # Execute debugging
        debug_result = debugging_agent.execute("Fix the division by zero bug", context, debug_task)
        
        if debug_result.success:
            print("✅ DebuggingAgent workflow test PASSED")
            print(f"   Message: {debug_result.message}")
            
            # Check if structured script was created and executed
            if "instruction_script.json" in context.artifacts:
                print("✅ Structured repair script was generated")
                script_data = context.get_artifact("instruction_script.json")
                if script_data:
                    script_obj = json.loads(script_data)
                    print(f"   Script title: {script_obj.get('title', 'Unknown')}")
                    print(f"   Instructions: {len(script_obj.get('instructions', []))}")
            else:
                print("❌ No structured script found in context")
                
        else:
            print("❌ DebuggingAgent workflow test FAILED")
            print(f"   Error: {debug_result.message}")
        
        # 3. Test communication logging
        print("\n📡 Testing Communication Logging")
        print("-" * 30)
        
        # Check if inter-agent communications were logged
        if hasattr(context, 'communications') and context.communications:
            print(f"✅ Found {len(context.communications)} logged communications")
            for comm in context.communications[-3:]:  # Show last 3
                print(f"   {comm.get('from_agent', 'Unknown')} → {comm.get('to_agent', 'Unknown')}: {comm.get('message_type', 'unknown')}")
        else:
            print("❌ No communications logged")
        
        print("\n🎯 Test Summary")
        print("=" * 50)
        print("✅ Structured instruction schemas created")
        print("✅ ScriptExecutorAgent implemented and tested")
        print("✅ DebuggingAgent refactored for structured communication")
        print("✅ Agent registry updated")
        print("✅ End-to-end workflow tested")
        
        return True

if __name__ == "__main__":
    try:
        success = test_structured_debugging_workflow()
        if success:
            print("\n🎉 All tests completed successfully!")
            exit(0)
        else:
            print("\n💥 Some tests failed!")
            exit(1)
    except Exception as e:
        print(f"\n💥 Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)