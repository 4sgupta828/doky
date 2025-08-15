#!/usr/bin/env python3
"""
Simple test to verify TestGenerationAgent auto-discovery functionality.
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
setup_logger(default_level=logging.WARNING)  # Reduce log noise
logger = logging.getLogger(__name__)

class SimpleTestGenerationAgentTest(unittest.TestCase):
    """Simple test to verify key functionality works."""

    def test_auto_discovery_functionality(self):
        """Test that TestGenerationAgent can auto-discover Python files when artifacts are missing."""
        print("\n--- [Simple Test: Auto-Discovery Functionality] ---")
        
        # Create temporary workspace
        test_workspace_path = tempfile.mkdtemp(prefix="simple_test_gen_")
        
        try:
            context = GlobalContext(workspace_path=test_workspace_path)
            
            # Create Python files directly in the workspace directory (not through git)
            workspace_dir = Path(test_workspace_path)
            (workspace_dir / "main.py").write_text("def hello(): return 'Hello, World!'")
            (workspace_dir / "utils.py").write_text("def add(a, b): return a + b")
            (workspace_dir / "test_existing.py").write_text("def test_existing(): pass")  # Should be ignored
            (workspace_dir / "config.json").write_text('{"key": "value"}')  # Should be ignored
            
            # Setup mock LLM
            mock_llm_client = MagicMock(spec=LLMClient)
            mock_test_code = json.dumps({
                "tests/test_main.py": "def test_hello(): assert hello() == 'Hello, World!'",
                "tests/test_utils.py": "def test_add(): assert add(2, 3) == 5"
            })
            mock_llm_client.invoke.return_value = mock_test_code
            
            agent = TestGenerationAgent(llm_client=mock_llm_client)
            task = TaskNode(goal="Generate unit tests for existing code", assigned_agent="TestGenerationAgent")
            
            # Execute
            response = agent.execute(task.goal, context, task)
            
            # Verify success
            self.assertTrue(response.success, f"Expected success but got: {response.message}")
            self.assertIn("2 unit test", response.message)
            
            # Verify LLM was called with correct content
            mock_llm_client.invoke.assert_called_once()
            prompt = mock_llm_client.invoke.call_args[0][0]
            
            # Should include discovered Python files
            self.assertIn("main.py", prompt)
            self.assertIn("utils.py", prompt)
            
            # Should NOT include test files or non-Python files
            self.assertNotIn("test_existing.py", prompt)
            self.assertNotIn("config.json", prompt)
            
            # Should use fallback spec
            self.assertIn("User Request:", prompt)
            
            print("✅ Auto-discovery test PASSED")
            
        finally:
            # Cleanup
            shutil.rmtree(test_workspace_path, ignore_errors=True)

if __name__ == "__main__":
    print("="*60)
    print(" SIMPLE TEST FOR TEST GENERATION AGENT AUTO-DISCOVERY")
    print("="*60)
    
    unittest.main(verbosity=2, exit=False)
    
    print("\n" + "="*60)
    print(" SIMPLE TEST COMPLETED")
    print("="*60)