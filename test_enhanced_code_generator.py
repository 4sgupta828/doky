#!/usr/bin/env python3
"""
Test script for the Enhanced Code Generator Agent workflow.

This script demonstrates the complete workflow:
1. Code generation from specification
2. Automatic dependency management  
3. CLI test creation
4. Execution validation

Usage: python test_enhanced_code_generator.py
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

from agents.enhanced_code_generator import EnhancedCodeGeneratorAgent
from core.context import GlobalContext
from tools.llm_tool import create_llm_client
from utils.logger import setup_logger

def test_enhanced_code_generation():
    """Test the complete enhanced code generation workflow."""
    
    # Setup logging
    setup_logger(default_level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Create temporary workspace
    temp_dir = tempfile.mkdtemp(prefix="enhanced_coder_test_")
    logger.info(f"Using temporary workspace: {temp_dir}")
    
    try:
        # Initialize components
        context = GlobalContext(workspace_path=temp_dir)
        llm_client = create_llm_client()
        agent = EnhancedCodeGeneratorAgent(llm_client=llm_client)
        
        # Test specification: Simple calculator application
        specification = """
Create a Python calculator application with the following features:
1. Basic arithmetic operations (add, subtract, multiply, divide)
2. Command-line interface
3. Input validation and error handling
4. Support for floating-point numbers

The application should be modular with separate functions for each operation.
Include a main function that provides a user-friendly CLI interface.
"""
        
        # Test inputs
        inputs = {
            'specification': specification,
            'output_directory': temp_dir,
            'quality_level': 'decent'
        }
        
        logger.info("=== Starting Enhanced Code Generation Test ===")
        logger.info(f"Specification: {specification[:100]}...")
        
        # Execute the enhanced code generator
        result = agent.execute_v2(
            goal="Create a calculator application with validation and testing",
            inputs=inputs,
            global_context=context
        )
        
        # Check results
        print("\n" + "="*60)
        print("ENHANCED CODE GENERATION RESULTS")
        print("="*60)
        
        if result.success:
            print(f"✅ SUCCESS: {result.message}")
            
            # Display outputs
            outputs = result.outputs
            print(f"\n📂 Generated Files: {len(outputs.get('generated_files', []))}")
            for file_path in outputs.get('generated_files', []):
                print(f"   - {file_path}")
                
                # Show file content (first 10 lines)
                full_path = os.path.join(temp_dir, file_path)
                if os.path.exists(full_path):
                    with open(full_path, 'r') as f:
                        lines = f.readlines()[:10]
                        print(f"     Preview: {lines[0].strip()}")
            
            if outputs.get('requirements_file'):
                print(f"\n📋 Requirements File: {outputs['requirements_file']}")
                
            if outputs.get('test_script'):
                print(f"\n🧪 Test Script: {outputs['test_script']}")
                
            validation = outputs.get('validation_results', {})
            if validation:
                print(f"\n✔️  Validation Results:")
                print(f"   - Syntax Check: {'✅' if validation.get('syntax_check') else '❌'}")
                print(f"   - Import Check: {'✅' if validation.get('import_check') else '❌'}")
                print(f"   - Test Execution: {'✅' if validation.get('test_execution') else '❌'}")
                
                for detail in validation.get('details', []):
                    print(f"   - {detail}")
            
            print(f"\n📁 All Files Created: {len(outputs.get('all_files_created', []))}")
            
            # Show workspace contents
            print(f"\n📂 Workspace Contents ({temp_dir}):")
            for item in os.listdir(temp_dir):
                item_path = os.path.join(temp_dir, item)
                if os.path.isfile(item_path):
                    size = os.path.getsize(item_path)
                    print(f"   - {item} ({size} bytes)")
            
            return True
            
        else:
            print(f"❌ FAILED: {result.message}")
            if result.error_details:
                print(f"Error Details: {result.error_details}")
            return False
            
    except Exception as e:
        logger.error(f"Test failed with exception: {e}", exc_info=True)
        print(f"\n❌ TEST EXCEPTION: {e}")
        return False
        
    finally:
        # Cleanup
        try:
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up temporary workspace: {temp_dir}")
        except:
            logger.warning(f"Failed to cleanup temporary workspace: {temp_dir}")

def test_requirements_only():
    """Test just the requirements management functionality."""
    
    logger = logging.getLogger(__name__)
    temp_dir = tempfile.mkdtemp(prefix="req_test_")
    
    try:
        from agents.enhanced_code_generator import RequirementsManagerAgent
        
        agent = RequirementsManagerAgent()
        context = GlobalContext(workspace_path=temp_dir)
        
        # Test code with various imports
        code_files = {
            "main.py": """
import os
import json
import requests
import pandas as pd
from flask import Flask
from pathlib import Path
import numpy as np
""",
            "utils.py": """
import sys
import logging
from datetime import datetime
import sqlite3
"""
        }
        
        inputs = {
            'code_files': code_files,
            'output_directory': temp_dir
        }
        
        result = agent.execute_v2(
            "Analyze imports and create requirements.txt",
            inputs,
            context
        )
        
        print(f"\n🔍 Requirements Analysis: {'✅' if result.success else '❌'}")
        print(f"   Message: {result.message}")
        
        if result.success and result.outputs.get('requirements_file'):
            req_file = result.outputs['requirements_file']
            if os.path.exists(req_file):
                with open(req_file, 'r') as f:
                    content = f.read()
                    print(f"   Requirements Content:\n{content}")
            
            dependencies = result.outputs.get('dependencies', [])
            print(f"   Dependencies Found: {dependencies}")
        
        return result.success
        
    except Exception as e:
        logger.error(f"Requirements test failed: {e}", exc_info=True)
        return False
        
    finally:
        try:
            shutil.rmtree(temp_dir)
        except:
            pass

if __name__ == "__main__":
    print("🚀 Testing Enhanced Code Generator Agent")
    print("=" * 50)
    
    # Test 1: Full workflow
    print("\n1. Testing Complete Enhanced Code Generation Workflow...")
    success1 = test_enhanced_code_generation()
    
    # Test 2: Requirements management only
    print("\n2. Testing Requirements Management...")
    success2 = test_requirements_only()
    
    # Summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    print(f"Complete Workflow: {'✅ PASSED' if success1 else '❌ FAILED'}")
    print(f"Requirements Management: {'✅ PASSED' if success2 else '❌ FAILED'}")
    
    overall_success = success1 and success2
    print(f"\nOverall Result: {'🎉 ALL TESTS PASSED' if overall_success else '❌ SOME TESTS FAILED'}")
    
    sys.exit(0 if overall_success else 1)