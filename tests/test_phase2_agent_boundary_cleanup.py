# tests/test_phase2_agent_boundary_cleanup.py
import unittest
import logging
from unittest.mock import Mock, MagicMock
from typing import Dict, Any

# Test dependencies
from core.context import GlobalContext
from core.models import AgentResult
from agents.environment_modifier import EnvironmentModifierAgent
from agents.code_analysis import CodeAnalysisAgent
from agents.test_analysis import TestAnalysisAgent
from agents.process_executor import ProcessExecutorAgent
from utils.logger import setup_logger

# Set up logging
setup_logger(default_level=logging.INFO)
logger = logging.getLogger(__name__)


class TestPhase2AgentBoundaryCleanup(unittest.TestCase):
    """
    Test suite for Phase 2 Agent Boundary Cleanup.
    
    Tests the new specialized agents and their clean separation of concerns.
    """
    
    def setUp(self):
        """Set up test environment"""
        self.context = GlobalContext(workspace_path="./temp_phase2_test")
        
        # Create new tier agents
        self.env_modifier = EnvironmentModifierAgent()
        self.code_analysis = CodeAnalysisAgent()
        self.test_analysis = TestAnalysisAgent()
        self.process_executor = ProcessExecutorAgent()
    
    def test_environment_manager_agent(self):
        """Test EnvironmentModifierAgent functionality"""
        print("\n--- [Test Case 1: EnvironmentModifierAgent] ---")
        
        inputs = {
            "workspace_path": str(self.context.workspace_path),
            "additional_packages": ["pytest"],
            "venv_name": "test_venv"
        }
        
        result = self.env_modifier.execute_v2(
            goal="Set up test environment",
            inputs=inputs,
            global_context=self.context
        )
        
        # Environment setup might fail due to system constraints, but should handle gracefully
        self.assertIsNotNone(result)
        self.assertIn("venv_path", result.outputs if result.success else {})
        
        logger.info(f"✅ Environment manager result: {result.success} - {result.message}")
    
    def test_code_validator_agent(self):
        """Test CodeAnalysisAgent functionality"""
        print("\n--- [Test Case 2: CodeAnalysisAgent] ---")
        
        # Test with valid Python code
        code_files = {
            "valid.py": "def hello():\n    return 'Hello, World!'",
            "also_valid.py": "import os\nprint('Valid code')"
        }
        
        inputs = {
            "code_files": code_files,
            "validation_level": "standard",
            "check_imports": True
        }
        
        result = self.code_analysis.execute_v2(
            goal="Validate Python code",
            inputs=inputs,
            global_context=self.context
        )
        
        self.assertTrue(result.success)
        self.assertIn("syntax_validation", result.outputs)
        self.assertEqual(result.outputs["files_processed"], 2)
        
        logger.info("✅ test_code_validator_agent: PASSED")
    
    def test_code_validator_agent_with_syntax_errors(self):
        """Test CodeAnalysisAgent with syntax errors"""
        print("\n--- [Test Case 3: CodeAnalysisAgent with Syntax Errors] ---")
        
        # Test with invalid Python code
        code_files = {
            "invalid.py": "def hello(\n    return 'Missing closing paren'",
            "valid.py": "def world():\n    return 'Valid code'"
        }
        
        inputs = {
            "code_files": code_files,
            "validation_level": "strict",
            "check_imports": True
        }
        
        result = self.code_analysis.execute_v2(
            goal="Validate Python code with errors",
            inputs=inputs,
            global_context=self.context
        )
        
        self.assertFalse(result.success)  # Should fail due to syntax error
        self.assertIn("syntax_validation", result.outputs)
        self.assertGreater(len(result.outputs["syntax_validation"]["errors"]), 0)
        
        logger.info("✅ test_code_validator_agent_with_syntax_errors: PASSED")
    
    def test_test_executor_agent(self):
        """Test TestAnalysisAgent functionality"""
        print("\n--- [Test Case 4: TestAnalysisAgent] ---")
        
        inputs = {
            "test_target": str(self.context.workspace_path),
            "test_framework": "pytest",
            "timeout_seconds": 30
        }
        
        result = self.test_analysis.execute_v2(
            goal="Execute tests in workspace",
            inputs=inputs,
            global_context=self.context
        )
        
        # Test execution might fail due to no tests, but should handle gracefully
        self.assertIsNotNone(result)
        self.assertIn("test_framework", result.outputs if result.success else {})
        
        logger.info(f"✅ Test executor result: {result.success} - {result.message}")
    
    def test_file_system_agent_discovery(self):
        """Test ProcessExecutorAgent file discovery"""
        print("\n--- [Test Case 5: ProcessExecutorAgent Discovery] ---")
        
        inputs = {
            "operation": "discover",
            "target_path": ".",
            "patterns": ["*.py"],
            "file_types": ["py"],
            "recursive": True
        }
        
        result = self.process_executor.execute_v2(
            goal="Discover Python files",
            inputs=inputs,
            global_context=self.context
        )
        
        self.assertTrue(result.success)
        self.assertIn("discovered_files", result.outputs)
        self.assertIsInstance(result.outputs["discovered_files"], list)
        
        logger.info("✅ test_file_system_agent_discovery: PASSED")
    
    def test_file_system_agent_read_write(self):
        """Test ProcessExecutorAgent read/write operations"""
        print("\n--- [Test Case 6: ProcessExecutorAgent Read/Write] ---")
        
        # Test write operation
        test_content = "# Test file\nprint('Hello from test file')\n"
        
        write_inputs = {
            "operation": "write",
            "target_path": "test_file.py",
            "content": test_content
        }
        
        write_result = self.process_executor.execute_v2(
            goal="Write test file",
            inputs=write_inputs,
            global_context=self.context
        )
        
        self.assertTrue(write_result.success)
        self.assertIn("file_path", write_result.outputs)
        
        # Test read operation
        read_inputs = {
            "operation": "read",
            "target_path": write_result.outputs["file_path"]
        }
        
        read_result = self.process_executor.execute_v2(
            goal="Read test file",
            inputs=read_inputs,
            global_context=self.context
        )
        
        self.assertTrue(read_result.success)
        self.assertEqual(read_result.outputs["content"], test_content)
        
        logger.info("✅ test_file_system_agent_read_write: PASSED")
    
    def test_agent_input_validation(self):
        """Test input validation across specialized agents"""
        print("\n--- [Test Case 7: Agent Input Validation] ---")
        
        # Test EnvironmentModifierAgent missing required input
        result = self.env_modifier.execute_v2(
            goal="Test validation",
            inputs={},  # Missing workspace_path
            global_context=self.context
        )
        
        self.assertFalse(result.success)
        self.assertIn("missing required inputs", result.message)
        
        # Test CodeAnalysisAgent missing required input
        result = self.code_analysis.execute_v2(
            goal="Test validation",
            inputs={},  # Missing code_files
            global_context=self.context
        )
        
        self.assertFalse(result.success)
        self.assertIn("missing required inputs", result.message)
        
        # Test TestAnalysisAgent missing required input
        result = self.test_analysis.execute_v2(
            goal="Test validation",
            inputs={},  # Missing test_target
            global_context=self.context
        )
        
        self.assertFalse(result.success)
        self.assertIn("missing required inputs", result.message)
        
        # Test ProcessExecutorAgent missing required input
        result = self.process_executor.execute_v2(
            goal="Test validation",
            inputs={},  # Missing operation
            global_context=self.context
        )
        
        self.assertFalse(result.success)
        self.assertIn("missing required inputs", result.message)
        
        logger.info("✅ test_agent_input_validation: PASSED")
    
    def test_agent_separation_of_concerns(self):
        """Test that agents have proper separation of concerns"""
        print("\n--- [Test Case 8: Agent Separation of Concerns] ---")
        
        # Each agent should have distinct responsibilities
        
        # EnvironmentModifierAgent - only environment setup
        env_inputs = self.env_modifier.required_inputs() + self.env_modifier.optional_inputs()
        self.assertIn("workspace_path", env_inputs)
        self.assertIn("venv_name", env_inputs)
        
        # CodeAnalysisAgent - only code validation
        code_inputs = self.code_analysis.required_inputs() + self.code_analysis.optional_inputs()
        self.assertIn("code_files", code_inputs)
        self.assertIn("validation_level", code_inputs)
        
        # TestAnalysisAgent - only test execution
        test_inputs = self.test_analysis.required_inputs() + self.test_analysis.optional_inputs()
        self.assertIn("test_target", test_inputs)
        self.assertIn("test_framework", test_inputs)
        
        # ProcessExecutorAgent - only file operations
        file_inputs = self.process_executor.required_inputs() + self.process_executor.optional_inputs()
        self.assertIn("operation", file_inputs)
        self.assertIn("target_path", file_inputs)
        
        # Verify no overlap in core functionality
        self.assertNotIn("code_files", env_inputs)  # Env doesn't handle code
        self.assertNotIn("test_target", code_inputs)  # Code validator doesn't run tests
        self.assertNotIn("workspace_path", test_inputs)  # Test executor gets working_directory
        
        logger.info("✅ test_agent_separation_of_concerns: PASSED")
    
    def test_backward_compatibility(self):
        """Test that agents maintain backward compatibility"""
        print("\n--- [Test Case 9: Backward Compatibility] ---")
        
        # All specialized agents should have legacy execute methods
        self.assertTrue(hasattr(self.env_modifier, 'execute'))
        self.assertTrue(hasattr(self.code_analysis, 'execute'))
        self.assertTrue(hasattr(self.test_analysis, 'execute'))
        self.assertTrue(hasattr(self.process_executor, 'execute'))
        
        # Legacy methods should work (even if they use fallback logic)
        try:
            from core.models import TaskNode
            task = TaskNode(goal="legacy test", assigned_agent="TestAgent")
            
            # These might not succeed due to missing context, but should not crash
            env_response = self.env_modifier.execute("Setup environment", self.context, task)
            self.assertIsNotNone(env_response)
            
            code_response = self.code_analysis.execute("Validate code", self.context, task)
            self.assertIsNotNone(code_response)
            
            test_response = self.test_analysis.execute("Run tests", self.context, task)
            self.assertIsNotNone(test_response)
            
            file_response = self.process_executor.execute("Find files", self.context, task)
            self.assertIsNotNone(file_response)
            
        except Exception as e:
            self.fail(f"Legacy compatibility failed: {e}")
        
        logger.info("✅ test_backward_compatibility: PASSED")


class TestAgentRegistryIntegration(unittest.TestCase):
    """Test that specialized agents are properly integrated in the agent registry."""
    
    def test_specialized_agents_in_registry(self):
        """Test that all specialized agents are registered"""
        print("\n--- [Test Case 10: Agent Registry Integration] ---")
        
        from agents import AGENT_REGISTRY
        
        # Check that specialized agents are in registry
        self.assertIn("EnvironmentModifierAgent", AGENT_REGISTRY)
        self.assertIn("CodeAnalysisAgent", AGENT_REGISTRY)
        self.assertIn("TestAnalysisAgent", AGENT_REGISTRY)
        self.assertIn("ProcessExecutorAgent", AGENT_REGISTRY)
        
        # Verify they can be instantiated
        env_manager = AGENT_REGISTRY["EnvironmentModifierAgent"]()
        self.assertIsNotNone(env_manager)
        
        code_validator = AGENT_REGISTRY["CodeAnalysisAgent"]()
        self.assertIsNotNone(code_validator)
        
        test_executor = AGENT_REGISTRY["TestAnalysisAgent"]()
        self.assertIsNotNone(test_executor)
        
        file_system = AGENT_REGISTRY["ProcessExecutorAgent"]()
        self.assertIsNotNone(file_system)
        
        logger.info("✅ test_specialized_agents_in_registry: PASSED")
    
    def test_agent_aliases(self):
        """Test that specialized agents have proper aliases"""
        print("\n--- [Test Case 11: Agent Aliases] ---")
        
        from agents import AGENT_ALIASES
        
        # Check that new tier agents have aliases
        self.assertEqual(AGENT_ALIASES["@code-analysis"], "CodeAnalysisAgent")
        self.assertEqual(AGENT_ALIASES["@test-analysis"], "TestAnalysisAgent")
        self.assertEqual(AGENT_ALIASES["@env-modifier"], "EnvironmentModifierAgent")
        self.assertEqual(AGENT_ALIASES["@process"], "ProcessExecutorAgent")
        
        logger.info("✅ test_agent_aliases: PASSED")


if __name__ == "__main__":
    # Run the tests
    unittest.main(argv=['first-arg-is-ignored'], exit=False)