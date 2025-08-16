#!/usr/bin/env python3
"""
Test script for the new structured ToolingAgent approach.
This validates that our schema-based tooling instructions work correctly.
"""

import sys
import os
import tempfile
import shutil
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from agents.tooling import ToolingAgent
from core.context import GlobalContext
from core.models import TaskNode
from core.instruction_schemas import create_diagnostic_instruction, create_test_instruction
from utils.logger import setup_logger

# Set up logging
setup_logger()

def test_structured_diagnostic_instruction():
    """Test that ToolingAgent can execute structured diagnostic instructions."""
    print("\n🧪 Testing Structured Diagnostic Instruction")
    print("=" * 50)
    
    # Create temporary workspace
    with tempfile.TemporaryDirectory() as temp_dir:
        context = GlobalContext(workspace_path=temp_dir)
        agent = ToolingAgent()
        
        # Create structured diagnostic instruction
        diagnostic_instruction = create_diagnostic_instruction(
            instruction_id="test_diagnostic_001",
            commands=[
                "python --version",
                "ls -la",
                "pwd"
            ],
            purpose="Test diagnostic commands for ToolingAgent validation",
            timeout=30
        )
        
        print(f"📋 Created instruction: {diagnostic_instruction.instruction_id}")
        print(f"   Commands: {len(diagnostic_instruction.commands)}")
        print(f"   Purpose: {diagnostic_instruction.purpose}")
        
        # Store instruction in context
        context.add_artifact("tooling_instruction.json", diagnostic_instruction.model_dump_json(indent=2), "test")
        
        # Create task
        task = TaskNode(
            goal=f"Execute diagnostic instruction: {diagnostic_instruction.instruction_id}",
            assigned_agent="ToolingAgent",
            input_artifact_keys=["tooling_instruction.json"]
        )
        
        # Execute
        print("\n⚙️  Executing structured instruction...")
        result = agent.execute(f"Execute diagnostic: {diagnostic_instruction.instruction_id}", context, task)
        
        # Validate results
        print(f"\n📊 Results:")
        print(f"   Success: {'✅' if result.success else '❌'} {result.success}")
        print(f"   Message: {result.message}")
        print(f"   Artifacts: {len(result.artifacts_generated or [])} generated")
        
        if result.artifacts_generated:
            for artifact in result.artifacts_generated:
                print(f"      - {artifact}")
                content = context.get_artifact(artifact)
                if content and isinstance(content, str):
                    print(f"        Size: {len(content)} chars")
        
        return result.success

def test_legacy_fallback():
    """Test that ToolingAgent falls back to legacy mode when no structured instruction is provided."""
    print("\n🔄 Testing Legacy Fallback Mode")
    print("=" * 50)
    
    # Create temporary workspace
    with tempfile.TemporaryDirectory() as temp_dir:
        context = GlobalContext(workspace_path=temp_dir)
        agent = ToolingAgent()
        
        # Create task WITHOUT structured instruction
        task = TaskNode(
            goal="echo 'Hello from legacy mode'",
            assigned_agent="ToolingAgent"
        )
        
        # Execute (should fall back to legacy mode)
        print("⚙️  Executing legacy command...")
        result = agent.execute("echo 'Hello from legacy mode'", context, task)
        
        # Validate results
        print(f"\n📊 Results:")
        print(f"   Success: {'✅' if result.success else '❌'} {result.success}")
        print(f"   Message: {result.message}")
        print(f"   Used Legacy Mode: {'✅' if 'legacy' in result.message.lower() else '❌'}")
        
        return result.success

def test_structured_vs_legacy_comparison():
    """Compare structured vs legacy approaches side by side."""
    print("\n⚖️  Structured vs Legacy Comparison")
    print("=" * 50)
    
    # Create temporary workspace
    with tempfile.TemporaryDirectory() as temp_dir:
        context = GlobalContext(workspace_path=temp_dir)
        agent = ToolingAgent()
        
        # Test 1: Structured approach
        print("\n1️⃣  Structured Approach:")
        diagnostic_instruction = create_diagnostic_instruction(
            instruction_id="comparison_test_001",
            commands=["echo 'Structured'", "date"],
            purpose="Comparison test using structured instructions",
            timeout=15
        )
        
        context.add_artifact("tooling_instruction.json", diagnostic_instruction.model_dump_json(indent=2), "test")
        
        task1 = TaskNode(
            goal="Execute structured comparison test",
            assigned_agent="ToolingAgent",
            input_artifact_keys=["tooling_instruction.json"]
        )
        
        result1 = agent.execute("Execute structured test", context, task1)
        print(f"   Result: {'✅' if result1.success else '❌'} - {result1.message[:80]}...")
        
        # Test 2: Legacy approach (clear the artifact first)
        print("\n2️⃣  Legacy Approach:")
        context.artifacts.clear()  # Remove structured instruction
        
        task2 = TaskNode(
            goal="echo 'Legacy'",
            assigned_agent="ToolingAgent"
        )
        
        result2 = agent.execute("echo 'Legacy'", context, task2)
        print(f"   Result: {'✅' if result2.success else '❌'} - {result2.message[:80]}...")
        
        print(f"\n🏆 Both approaches work: {'✅' if result1.success and result2.success else '❌'}")
        return result1.success and result2.success

def main():
    """Run all structured tooling tests."""
    print("🚀 Testing Structured ToolingAgent Implementation")
    print("=" * 60)
    
    tests = [
        ("Structured Diagnostic", test_structured_diagnostic_instruction),
        ("Legacy Fallback", test_legacy_fallback),
        ("Structured vs Legacy", test_structured_vs_legacy_comparison)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result, None))
            print(f"\n{'✅' if result else '❌'} {test_name}: {'PASSED' if result else 'FAILED'}")
        except Exception as e:
            results.append((test_name, False, str(e)))
            print(f"\n❌ {test_name}: ERROR - {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, success, _ in results if success)
    total = len(results)
    
    for test_name, success, error in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status:8} {test_name}")
        if error:
            print(f"         Error: {error}")
    
    print(f"\n🎯 Overall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Structured ToolingAgent is working correctly.")
        return 0
    else:
        print("💥 Some tests failed. Check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())