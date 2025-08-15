#!/usr/bin/env python3
"""
Comprehensive test suite for TestGenerationAgent covering all scenarios:
1. Normal case with artifacts available
2. Missing artifacts but existing Python files
3. No artifacts and no existing Python files
4. Quality level detection
5. Unit vs integration test type detection
"""

import json
import logging
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

# Setup proper imports
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.test_generator import TestGenerationAgent, TestQuality, LLMClient
from core.context import GlobalContext
from core.models import TaskNode
from utils.logger import setup_logger

# Setup logging
setup_logger(default_level=logging.INFO)
logger = logging.getLogger(__name__)

class ComprehensiveTestGenerationAgentTest(unittest.TestCase):
    """Comprehensive test suite for TestGenerationAgent covering all scenarios."""

    def setUp(self):
        """Set up test environment with temporary workspace."""
        self.test_workspace_path = tempfile.mkdtemp(prefix="test_gen_comprehensive_")
        logger.info(f"Created test workspace: {self.test_workspace_path}")
        
        self.mock_llm_client = MagicMock(spec=LLMClient)
        self.agent = TestGenerationAgent(llm_client=self.mock_llm_client)
        
    def tearDown(self):
        """Clean up test workspace."""
        shutil.rmtree(self.test_workspace_path, ignore_errors=True)
        logger.info(f"Cleaned up test workspace: {self.test_workspace_path}")

    def test_scenario_1_with_artifacts(self):
        """Test Case 1: Normal case with both technical_spec.md and file_manifest.json available."""
        print("\n--- [Test Case 1: With Both Artifacts Available] ---")
        
        context = GlobalContext(workspace_path=self.test_workspace_path)
        
        # Add required artifacts
        context.add_artifact("technical_spec.md", "Spec: Calculator API with add(a, b) function.", "task_spec")
        context.add_artifact("file_manifest.json", {"files_to_create": ["src/calculator.py", "src/utils.py"]}, "task_manifest")
        
        # Create the actual files in workspace
        context.workspace.write_file_content("src/calculator.py", "def add(a, b): return a + b", "task_code")
        context.workspace.write_file_content("src/utils.py", "def validate_number(n): return isinstance(n, (int, float))", "task_code")
        
        # Mock LLM response
        mock_test_code = json.dumps({
            "tests/test_calculator.py": "def test_add(): assert add(2, 3) == 5",
            "tests/test_utils.py": "def test_validate_number(): assert validate_number(5) is True"
        })
        self.mock_llm_client.invoke.return_value = mock_test_code
        
        task = TaskNode(goal="Generate unit tests for calculator", assigned_agent="TestGenerationAgent")
        response = self.agent.execute(task.goal, context, task)
        
        self.assertTrue(response.success, f"Expected success but got: {response.message}")
        self.assertIn("2 unit test", response.message)
        self.mock_llm_client.invoke.assert_called_once()
        
        # Verify prompt contains both spec and both source files
        prompt = self.mock_llm_client.invoke.call_args[0][0]
        self.assertIn("Calculator API", prompt)
        self.assertIn("src/calculator.py", prompt)
        self.assertIn("src/utils.py", prompt)
        
        logger.info("✅ test_scenario_1_with_artifacts: PASSED")

    def test_scenario_2_missing_artifacts_with_existing_files(self):
        """Test Case 2: Missing artifacts but existing Python files should be auto-discovered."""
        print("\n--- [Test Case 2: Missing Artifacts, Auto-Discover Files] ---")
        
        context = GlobalContext(workspace_path=self.test_workspace_path)
        
        # Create Python files in workspace without adding artifacts
        context.workspace.write_file_content("main.py", "def hello(): return 'Hello, World!'", "task_code")
        context.workspace.write_file_content("stats_utils.py", "def mean(numbers): return sum(numbers) / len(numbers)", "task_code")
        context.workspace.write_file_content("config.py", "DATABASE_URL = 'sqlite:///test.db'", "task_code")
        
        # Mock LLM response
        mock_test_code = json.dumps({
            "tests/test_main.py": "def test_hello(): assert hello() == 'Hello, World!'",
            "tests/test_stats.py": "def test_mean(): assert mean([1, 2, 3]) == 2.0",
            "tests/test_config.py": "def test_config(): assert DATABASE_URL is not None"
        })
        self.mock_llm_client.invoke.return_value = mock_test_code
        
        task = TaskNode(goal="Generate unit tests for the existing codebase", assigned_agent="TestGenerationAgent")
        response = self.agent.execute(task.goal, context, task)
        
        self.assertTrue(response.success, f"Expected success but got: {response.message}")
        self.assertIn("3 unit test", response.message)
        
        # Verify prompt contains auto-discovered files and uses goal as spec
        prompt = self.mock_llm_client.invoke.call_args[0][0]
        self.assertIn("User Request:", prompt)  # Fallback spec
        self.assertIn("main.py", prompt)
        self.assertIn("stats_utils.py", prompt) 
        self.assertIn("config.py", prompt)
        
        logger.info("✅ test_scenario_2_missing_artifacts_with_existing_files: PASSED")

    def test_scenario_3_no_artifacts_no_files(self):
        """Test Case 3: No artifacts and no existing Python files."""
        print("\n--- [Test Case 3: No Artifacts, No Files] ---")
        
        context = GlobalContext(workspace_path=self.test_workspace_path)
        # No artifacts added, no files created
        
        task = TaskNode(goal="Generate unit tests", assigned_agent="TestGenerationAgent")
        response = self.agent.execute(task.goal, context, task)
        
        self.assertTrue(response.success)
        self.assertEqual("No application code found to test.", response.message)
        
        # LLM should not be called when no code is found
        self.mock_llm_client.invoke.assert_not_called()
        
        logger.info("✅ test_scenario_3_no_artifacts_no_files: PASSED")

    def test_scenario_4_quality_level_detection(self):
        """Test Case 4: Quality level detection from goal keywords."""
        print("\n--- [Test Case 4: Quality Level Detection] ---")
        
        test_cases = [
            ("Generate quick tests", TestQuality.FAST),
            ("Generate comprehensive tests", TestQuality.DECENT), 
            ("Generate production-ready tests", TestQuality.PRODUCTION),
            ("Generate tests", TestQuality.FAST)  # Default
        ]
        
        for goal, expected_quality in test_cases:
            with self.subTest(goal=goal, expected=expected_quality):
                context = GlobalContext(workspace_path=self.test_workspace_path)
                context.workspace.write_file_content("app.py", "def func(): pass", "task_code")
                
                mock_test_code = json.dumps({"tests/test_app.py": "def test_func(): pass"})
                self.mock_llm_client.invoke.return_value = mock_test_code
                
                task = TaskNode(goal=goal, assigned_agent="TestGenerationAgent")
                response = self.agent.execute(task.goal, context, task)
                
                self.assertTrue(response.success)
                
                # Check that the correct quality level was used in the prompt
                prompt = self.mock_llm_client.invoke.call_args[0][0]
                self.assertIn(f"**Test Quality Level: {expected_quality.value.upper()}**", prompt)
                
                # Reset mock for next iteration
                self.mock_llm_client.reset_mock()
        
        logger.info("✅ test_scenario_4_quality_level_detection: PASSED")

    def test_scenario_5_test_type_detection(self):
        """Test Case 5: Unit vs integration test type detection."""
        print("\n--- [Test Case 5: Test Type Detection] ---")
        
        test_cases = [
            ("Generate unit tests", "unit"),
            ("Generate integration tests", "integration"),
            ("Generate tests", "unit"),  # Default
            ("Create integration test suite", "integration")
        ]
        
        for goal, expected_type in test_cases:
            with self.subTest(goal=goal, expected=expected_type):
                context = GlobalContext(workspace_path=self.test_workspace_path)
                context.workspace.write_file_content("service.py", "def service_func(): pass", "task_code")
                
                mock_test_code = json.dumps({"tests/test_service.py": "def test_service(): pass"})
                self.mock_llm_client.invoke.return_value = mock_test_code
                
                task = TaskNode(goal=goal, assigned_agent="TestGenerationAgent")
                response = self.agent.execute(task.goal, context, task)
                
                self.assertTrue(response.success)
                self.assertIn(f"{expected_type} test", response.message)
                
                # Check that the correct test type instructions were used in the prompt
                prompt = self.mock_llm_client.invoke.call_args[0][0]
                self.assertIn(f"Instructions for {expected_type.title()} Tests", prompt)
                
                # Reset mock for next iteration
                self.mock_llm_client.reset_mock()
        
        logger.info("✅ test_scenario_5_test_type_detection: PASSED")

    def test_scenario_6_mixed_file_discovery(self):
        """Test Case 6: Ensure test files and non-Python files are excluded from discovery."""
        print("\n--- [Test Case 6: Mixed File Discovery] ---")
        
        context = GlobalContext(workspace_path=self.test_workspace_path)
        
        # Create various file types
        context.workspace.write_file_content("app.py", "def main(): pass", "task_code")
        context.workspace.write_file_content("tests/test_existing.py", "def test_existing(): pass", "task_code")  # Should be excluded
        context.workspace.write_file_content("test_standalone.py", "def test_standalone(): pass", "task_code")  # Should be excluded
        context.workspace.write_file_content("README.md", "# My Project", "task_code")  # Should be excluded
        context.workspace.write_file_content("config.json", '{"key": "value"}', "task_code")  # Should be excluded
        context.workspace.write_file_content("utils.py", "def helper(): pass", "task_code")  # Should be included
        
        mock_test_code = json.dumps({
            "tests/test_app.py": "def test_main(): pass",
            "tests/test_utils.py": "def test_helper(): pass"
        })
        self.mock_llm_client.invoke.return_value = mock_test_code
        
        task = TaskNode(goal="Generate tests", assigned_agent="TestGenerationAgent")
        response = self.agent.execute(task.goal, context, task)
        
        self.assertTrue(response.success)
        
        # Verify only non-test Python files were included
        prompt = self.mock_llm_client.invoke.call_args[0][0]
        self.assertIn("app.py", prompt)
        self.assertIn("utils.py", prompt)
        self.assertNotIn("test_existing.py", prompt)
        self.assertNotIn("test_standalone.py", prompt)
        self.assertNotIn("README.md", prompt)
        self.assertNotIn("config.json", prompt)
        
        logger.info("✅ test_scenario_6_mixed_file_discovery: PASSED")

if __name__ == "__main__":
    print("\n" + "="*60)
    print(" COMPREHENSIVE TEST SUITE FOR TEST GENERATION AGENT")
    print("="*60)
    
    unittest.main(verbosity=2, exit=False)
    
    print("\n" + "="*60)
    print(" ALL COMPREHENSIVE TESTS COMPLETED")
    print("="*60)