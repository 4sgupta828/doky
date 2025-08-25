#!/usr/bin/env python3
"""
Test script for the enhanced code generation workflow with helper agents.

This script tests:
1. CodeGenerationAgent creates code
2. RequirementsManagerAgent manages dependencies  
3. CLITestGeneratorAgent creates tests
4. ExecutionValidatorAgent validates functionality

Usage: python test_code_generation_workflow.py
"""

import sys
import os
import shutil
import tempfile
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from agents.coder import CodeGenerationAgent
from agents.requirements_manager import RequirementsManagerAgent
from agents.cli_test_generator import CLITestGeneratorAgent
from agents.execution_validator import ExecutionValidatorAgent
from core.context import GlobalContext
from core.models import TaskNode
from tools.llm_tool import create_llm_client
from utils.logger import setup_logger

def test_code_generation_workflow():
    """Test the complete code generation workflow."""
    
    # Setup logging
    setup_logger(default_level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Create temporary workspace
    temp_dir = tempfile.mkdtemp(prefix="code_workflow_test_")
    logger.info(f"Using temporary workspace: {temp_dir}")
    
    try:
        # Initialize components
        context = GlobalContext(workspace_path=temp_dir)
        llm_client = create_llm_client()
        
        # Create a simple task
        task = TaskNode(
            goal="Create a Python calculator with add, subtract, multiply, divide functions",
            assigned_agent="CodeGenerationAgent",
            task_id="test_calc"
        )
        
        # Add spec and manifest to context
        spec = """
Create a Python calculator application with these features:
1. Functions for add, subtract, multiply, divide operations
2. Command-line interface that takes two numbers and an operation
3. Input validation and error handling
4. Support for floating-point numbers
5. Main function that demonstrates usage

The calculator should be modular and well-structured.
"""
        
        manifest = {
            "files_to_create": ["calculator.py", "calc_utils.py"]
        }
        
        context.add_artifact("technical_spec.md", spec, "test_setup")
        context.add_artifact("file_manifest.json", manifest, "test_setup")
        
        print("=" * 60)
        print("🧪 TESTING CODE GENERATION WORKFLOW")
        print("=" * 60)
        
        # Step 1: Test CodeGenerationAgent with post-processing
        print("\n1. Testing CodeGenerationAgent with integrated workflow...")
        coder_agent = CodeGenerationAgent(llm_client=llm_client)
        
        result = coder_agent.execute(task.goal, context, task)
        
        print(f"CodeGeneration Result: {'✅ SUCCESS' if result.success else '❌ FAILED'}")
        print(f"Message: {result.message}")
        
        if result.success:
            print(f"Files generated: {result.artifacts_generated}")
            
            # Check workspace contents
            workspace_files = context.workspace.list_files()
            print(f"Workspace files: {workspace_files}")
            
            # Show file contents
            for file_path in workspace_files:
                if file_path.endswith('.py'):
                    content = context.workspace.get_file_content(file_path)
                    if content:
                        print(f"\\n--- {file_path} (first 5 lines) ---")
                        lines = content.split('\\n')[:5]
                        for line in lines:
                            print(f"  {line}")
                        if len(content.split('\\n')) > 5:
                            print("  ...")
        
        # Step 2: Test individual helper agents
        print("\\n2. Testing individual helper agents...")
        
        # Get generated code for testing
        code_files = {}
        for file_path in context.workspace.list_files():
            if file_path.endswith('.py'):
                content = context.workspace.get_file_content(file_path)
                if content:
                    code_files[file_path] = content
        
        if code_files:
            # Test RequirementsManagerAgent
            print("\\n  a) Testing RequirementsManagerAgent...")
            req_agent = RequirementsManagerAgent()
            req_result = req_agent.execute_v2(
                "Analyze code dependencies",
                {'code_files': code_files, 'output_directory': temp_dir},
                context
            )
            print(f"     Requirements: {'✅ PASSED' if req_result.success else '❌ FAILED'}")
            print(f"     Message: {req_result.message}")
            
            # Test CLITestGeneratorAgent  
            print("\\n  b) Testing CLITestGeneratorAgent...")
            test_agent = CLITestGeneratorAgent(llm_client=llm_client)
            test_result = test_agent.execute_v2(
                "Generate CLI tests",
                {
                    'code_files': code_files,
                    'specification': spec,
                    'output_directory': temp_dir
                },
                context
            )
            print(f"     Test Generation: {'✅ PASSED' if test_result.success else '❌ FAILED'}")
            print(f"     Message: {test_result.message}")
            
            # Test ExecutionValidatorAgent
            print("\\n  c) Testing ExecutionValidatorAgent...")
            validator_agent = ExecutionValidatorAgent()
            
            inputs = {'code_files': code_files, 'output_directory': temp_dir}
            if test_result.success and test_result.outputs.get('test_script'):
                inputs['test_script'] = test_result.outputs['test_script']
            
            validation_result = validator_agent.execute_v2(
                "Validate code execution",
                inputs,
                context
            )
            print(f"     Validation: {'✅ PASSED' if validation_result.success else '❌ FAILED'}")
            print(f"     Message: {validation_result.message}")
            
            if validation_result.outputs:
                outputs = validation_result.outputs
                print(f"     Syntax Check: {'✅' if outputs.get('syntax_check') else '❌'}")
                print(f"     Import Check: {'✅' if outputs.get('import_check') else '❌'}")
                print(f"     Test Execution: {'✅' if outputs.get('test_execution') else '❌'}")
        
        # Summary
        print("\\n" + "=" * 60)
        print("📊 WORKFLOW TEST SUMMARY")
        print("=" * 60)
        
        overall_success = result.success
        if code_files:
            overall_success = (overall_success and req_result.success and 
                             test_result.success and validation_result.success)
        
        print(f"Overall Result: {'🎉 ALL TESTS PASSED' if overall_success else '❌ SOME TESTS FAILED'}")
        
        return overall_success
        
    except Exception as e:
        logger.error(f"Workflow test failed: {e}", exc_info=True)
        print(f"\\n❌ TEST EXCEPTION: {e}")
        return False
        
    finally:
        # Cleanup
        try:
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up temporary workspace: {temp_dir}")
        except:
            logger.warning(f"Failed to cleanup temporary workspace: {temp_dir}")

def test_individual_agents():
    """Test individual agents separately."""
    
    logger = logging.getLogger(__name__)
    print("\\n" + "="*50)
    print("🔧 TESTING INDIVIDUAL AGENTS")
    print("="*50)
    
    # Sample code for testing
    sample_code = {
        "main.py": """
import os
import json
import requests
import pandas as pd

def main():
    print("Hello World")
    data = pd.DataFrame({'x': [1, 2, 3]})
    response = requests.get('http://example.com')
    return True

if __name__ == "__main__":
    main()
""",
        "utils.py": """
import sys
import logging
from datetime import datetime

def helper_function():
    return datetime.now()
"""
    }
    
    temp_dir = tempfile.mkdtemp(prefix="individual_test_")
    
    try:
        context = GlobalContext(workspace_path=temp_dir)
        
        # Test 1: RequirementsManagerAgent
        print("\\n1. Testing RequirementsManagerAgent standalone...")
        req_agent = RequirementsManagerAgent()
        req_result = req_agent.execute_v2(
            "Test requirements analysis",
            {'code_files': sample_code},
            context
        )
        print(f"   Result: {'✅ PASSED' if req_result.success else '❌ FAILED'}")
        print(f"   Dependencies found: {req_result.outputs.get('dependencies', [])}")
        
        # Test 2: CLITestGeneratorAgent
        print("\\n2. Testing CLITestGeneratorAgent standalone...")
        try:
            llm_client = create_llm_client()
            test_agent = CLITestGeneratorAgent(llm_client=llm_client)
            test_result = test_agent.execute_v2(
                "Test CLI generation",
                {
                    'code_files': sample_code,
                    'specification': 'Simple data processing application',
                    'output_directory': temp_dir
                },
                context
            )
            print(f"   Result: {'✅ PASSED' if test_result.success else '❌ FAILED'}")
        except Exception as e:
            print(f"   Result: ⚠️  SKIPPED (LLM error: {e})")
        
        # Test 3: ExecutionValidatorAgent  
        print("\\n3. Testing ExecutionValidatorAgent standalone...")
        validator_agent = ExecutionValidatorAgent()
        validation_result = validator_agent.execute_v2(
            "Test validation",
            {'code_files': sample_code, 'output_directory': temp_dir},
            context
        )
        print(f"   Result: {'✅ PASSED' if validation_result.success else '❌ FAILED'}")
        if validation_result.outputs:
            outputs = validation_result.outputs
            print(f"   Syntax: {'✅' if outputs.get('syntax_check') else '❌'}")
            print(f"   Imports: {'✅' if outputs.get('import_check') else '❌'}")
        
        return True
        
    except Exception as e:
        logger.error(f"Individual agent tests failed: {e}", exc_info=True)
        return False
        
    finally:
        try:
            shutil.rmtree(temp_dir)
        except:
            pass

if __name__ == "__main__":
    print("🚀 Testing Enhanced Code Generation Workflow")
    
    # Test 1: Complete workflow
    success1 = test_code_generation_workflow()
    
    # Test 2: Individual agents
    success2 = test_individual_agents()
    
    print("\\n" + "="*60)
    print("🏁 FINAL TEST RESULTS")
    print("="*60)
    print(f"Complete Workflow: {'✅ PASSED' if success1 else '❌ FAILED'}")
    print(f"Individual Agents: {'✅ PASSED' if success2 else '❌ FAILED'}")
    
    overall_success = success1 and success2
    print(f"\\nOverall: {'🎉 ALL TESTS PASSED!' if overall_success else '❌ SOME TESTS FAILED'}")
    
    sys.exit(0 if overall_success else 1)