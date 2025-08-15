#!/usr/bin/env python3
"""
Summary test covering all key TestGenerationAgent scenarios.
"""

import json
import logging
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.test_generator import TestGenerationAgent, TestQuality, LLMClient
from core.context import GlobalContext
from core.models import TaskNode
from utils.logger import setup_logger

setup_logger(default_level=logging.ERROR)  # Minimal logging

class TestGenerationAgentSummaryTest(unittest.TestCase):
    """Summary test covering all key scenarios."""

    def test_all_key_scenarios(self):
        """Test all key scenarios for TestGenerationAgent."""
        print("\n" + "="*60)
        print(" TESTING ALL KEY SCENARIOS")
        print("="*60)
        
        results = {}
        
        # Scenario 1: With artifacts available
        print("\n[Scenario 1] With artifacts available...")
        try:
            test_workspace = tempfile.mkdtemp()
            context = GlobalContext(workspace_path=test_workspace)
            context.add_artifact("technical_spec.md", "API Spec: Calculator", "task_spec")
            context.add_artifact("file_manifest.json", {"files_to_create": ["calc.py"]}, "task_manifest")
            
            # Create file through workspace
            workspace_dir = Path(test_workspace)
            (workspace_dir / "calc.py").write_text("def add(a, b): return a + b")
            
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = json.dumps({"tests/test_calc.py": "def test_add(): pass"})
            
            agent = TestGenerationAgent(llm_client=mock_llm)
            task = TaskNode(goal="Generate tests", assigned_agent="TestGenerationAgent")
            response = agent.execute(task.goal, context, task)
            
            results["with_artifacts"] = response.success
            print(f"   ✅ SUCCESS" if response.success else f"   ❌ FAILED: {response.message}")
            
            shutil.rmtree(test_workspace, ignore_errors=True)
        except Exception as e:
            results["with_artifacts"] = False
            print(f"   ❌ ERROR: {e}")
        
        # Scenario 2: Auto-discovery (missing artifacts)
        print("\n[Scenario 2] Auto-discovery with missing artifacts...")
        try:
            test_workspace = tempfile.mkdtemp()
            context = GlobalContext(workspace_path=test_workspace)
            
            # Create files directly
            workspace_dir = Path(test_workspace)
            (workspace_dir / "main.py").write_text("def main(): pass")
            (workspace_dir / "utils.py").write_text("def helper(): pass")
            
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = json.dumps({
                "tests/test_main.py": "def test_main(): pass",
                "tests/test_utils.py": "def test_helper(): pass"
            })
            
            agent = TestGenerationAgent(llm_client=mock_llm)
            task = TaskNode(goal="Generate tests", assigned_agent="TestGenerationAgent")
            response = agent.execute(task.goal, context, task)
            
            results["auto_discovery"] = response.success and "2 unit test" in response.message
            print(f"   ✅ SUCCESS" if results["auto_discovery"] else f"   ❌ FAILED: {response.message}")
            
            shutil.rmtree(test_workspace, ignore_errors=True)
        except Exception as e:
            results["auto_discovery"] = False
            print(f"   ❌ ERROR: {e}")
        
        # Scenario 3: No files available
        print("\n[Scenario 3] No files available...")
        try:
            test_workspace = tempfile.mkdtemp()
            context = GlobalContext(workspace_path=test_workspace)
            
            mock_llm = MagicMock()
            agent = TestGenerationAgent(llm_client=mock_llm)
            task = TaskNode(goal="Generate tests", assigned_agent="TestGenerationAgent")
            response = agent.execute(task.goal, context, task)
            
            results["no_files"] = response.success and "No application code found" in response.message
            print(f"   ✅ SUCCESS" if results["no_files"] else f"   ❌ FAILED: {response.message}")
            
            shutil.rmtree(test_workspace, ignore_errors=True)
        except Exception as e:
            results["no_files"] = False
            print(f"   ❌ ERROR: {e}")
            
        # Scenario 4: Quality detection
        print("\n[Scenario 4] Quality level detection...")
        try:
            test_workspace = tempfile.mkdtemp()
            context = GlobalContext(workspace_path=test_workspace)
            
            workspace_dir = Path(test_workspace)
            (workspace_dir / "app.py").write_text("def func(): pass")
            
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = json.dumps({"tests/test_app.py": "def test_func(): pass"})
            
            agent = TestGenerationAgent(llm_client=mock_llm)
            task = TaskNode(goal="Generate comprehensive tests", assigned_agent="TestGenerationAgent")
            response = agent.execute(task.goal, context, task)
            
            # Check if DECENT quality was detected in the prompt
            prompt = mock_llm.invoke.call_args[0][0]
            results["quality_detection"] = "**Test Quality Level: DECENT**" in prompt
            print(f"   ✅ SUCCESS" if results["quality_detection"] else f"   ❌ FAILED: Quality not detected")
            
            shutil.rmtree(test_workspace, ignore_errors=True)
        except Exception as e:
            results["quality_detection"] = False
            print(f"   ❌ ERROR: {e}")
            
        # Scenario 5: File filtering (exclude test files)
        print("\n[Scenario 5] File filtering...")
        try:
            test_workspace = tempfile.mkdtemp()
            context = GlobalContext(workspace_path=test_workspace)
            
            workspace_dir = Path(test_workspace)
            (workspace_dir / "app.py").write_text("def main(): pass")
            (workspace_dir / "test_existing.py").write_text("def test_existing(): pass")
            (workspace_dir / "README.md").write_text("# Project")
            
            mock_llm = MagicMock()
            mock_llm.invoke.return_value = json.dumps({"tests/test_app.py": "def test_main(): pass"})
            
            agent = TestGenerationAgent(llm_client=mock_llm)
            task = TaskNode(goal="Generate tests", assigned_agent="TestGenerationAgent")
            response = agent.execute(task.goal, context, task)
            
            prompt = mock_llm.invoke.call_args[0][0]
            results["file_filtering"] = (
                "app.py" in prompt and 
                "test_existing.py" not in prompt and 
                "README.md" not in prompt
            )
            print(f"   ✅ SUCCESS" if results["file_filtering"] else f"   ❌ FAILED: File filtering not working")
            
            shutil.rmtree(test_workspace, ignore_errors=True)
        except Exception as e:
            results["file_filtering"] = False
            print(f"   ❌ ERROR: {e}")
            
        # Summary
        print("\n" + "="*60)
        print(" TEST SUMMARY")
        print("="*60)
        
        total_tests = len(results)
        passed_tests = sum(results.values())
        
        for scenario, passed in results.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{scenario:20} {status}")
            
        print(f"\nOverall: {passed_tests}/{total_tests} scenarios passed")
        
        if passed_tests == total_tests:
            print("🎉 ALL TESTS PASSED! TestGenerationAgent is working correctly.")
        else:
            print("⚠️  Some tests failed. Check the output above.")
            
        # Assert for unittest
        self.assertEqual(passed_tests, total_tests, f"Expected all {total_tests} scenarios to pass, but only {passed_tests} passed")

if __name__ == "__main__":
    unittest.main(verbosity=1, exit=False)