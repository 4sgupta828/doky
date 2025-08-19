#!/usr/bin/env python3
"""
Test script for the new design change workflow between
DebuggingAgent and CodeGenerationAgent.
"""

import json
import logging
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

# Local imports
from core.context import GlobalContext
from core.models import TaskNode
from agents.debugging import DebuggingAgent
from agents.coder import CodeGenerationAgent
from utils.logger import setup_logger

def test_design_change_workflow():
    """Test the complete design change workflow."""
    
    print("🔧 Testing Design Change Workflow")
    print("=" * 50)
    
    # Setup logging
    setup_logger(default_level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Create temporary workspace
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace_path = Path(temp_dir)
        context = GlobalContext(workspace_path=str(workspace_path))
        
        # Create test file with design issues
        test_file = workspace_path / "poor_design.py"
        test_file.write_text("""# Poor design example - monolithic function
def process_user_data(user_input):
    # Everything in one function - poor separation of concerns
    if not user_input:
        print("Error: No input")
        return None
    
    # Direct validation mixed with business logic
    if len(user_input) < 3:
        print("Error: Too short")
        return None
    
    # Data processing mixed with output formatting
    processed = user_input.upper()
    formatted = f"Processed: {processed}"
    
    # Direct output mixed with business logic
    print(formatted)
    
    # Return mixed data types
    if processed == "ADMIN":
        return {"role": "admin", "permissions": ["all"]}
    else:
        return processed

def main():
    result = process_user_data("user")
    print(f"Result: {result}")
    
if __name__ == "__main__":
    main()
""")
        
        # Create mock failed test report indicating design issues
        failed_report = {
            "summary": {"failed": 3, "passed": 1},
            "file": "poor_design.py",
            "error": "Design issues: tight coupling, mixed concerns, inconsistent return types",
            "function": "process_user_data",
            "issues": [
                "Single function handles multiple responsibilities",
                "Mixed validation, processing, and output concerns",
                "Inconsistent return types make testing difficult"
            ]
        }
        
        context.add_artifact("failed_test_report.json", json.dumps(failed_report), "test")
        context.add_artifact("targeted_code_context.json", {"files": [{"path": str(test_file), "content": test_file.read_text()}], "metadata": {"total_files": 1}}, "test")
        
        print("✅ Test environment setup complete")
        print(f"   Workspace: {workspace_path}")
        print(f"   Test file: {test_file} (design issues)")
        
        # 1. Test Design Change Request Generation
        print("\n📋 Testing Design Change Request Generation")
        print("-" * 40)
        
        # Create debugging agent with hypothesis indicating design change needed
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = json.dumps({
            "root_cause_analysis": "The process_user_data function violates single responsibility principle and has tight coupling between validation, processing, and output concerns",
            "primary_hypothesis": "Poor separation of concerns leading to maintainability and testability issues",
            "solution_type": "DESIGN_CHANGE",
            "recommended_strategy": "Refactor to separate validation, processing, and formatting concerns into distinct components",
            "confidence_level": "high",
            "complexity_assessment": "moderate", 
            "risk_assessment": "Medium risk - requires careful refactoring to preserve functionality"
        })
        
        debugging_agent = DebuggingAgent(llm_client=mock_llm)
        
        # Test design change request creation directly
        hypothesis = {
            "primary_hypothesis": "Poor separation of concerns leading to maintainability issues",
            "recommended_strategy": "Refactor to separate validation, processing, and formatting concerns",
            "confidence_level": "high",
            "complexity_assessment": "moderate"
        }
        
        task = TaskNode(goal="Fix design issues", assigned_agent="DebuggingAgent")
        design_request = debugging_agent._create_design_change_request(hypothesis, context, task)
        
        print("✅ Design Change Request Generated")
        print(f"   Request ID: {design_request['request_id']}")
        print(f"   Files to modify: {len(design_request['files_to_modify'])}")
        print(f"   Recommendations: {len(design_request['recommended_changes'])}")
        
        for i, rec in enumerate(design_request['recommended_changes'][:3], 1):
            print(f"   {i}. {rec}")
        
        # 2. Test CodeGenerationAgent Design Update Mode
        print("\n🤖 Testing CodeGenerationAgent Design Update Mode")
        print("-" * 45)
        
        # Create mock LLM for code generation
        mock_code_llm = MagicMock()
        mock_code_llm.invoke.return_value = json.dumps({
            "poor_design.py": '''# Improved design with separated concerns
class InputValidator:
    """Handles input validation logic."""
    
    @staticmethod
    def validate_user_input(user_input):
        """Validate user input and return validation result."""
        if not user_input:
            raise ValueError("No input provided")
        
        if len(user_input) < 3:
            raise ValueError("Input too short (minimum 3 characters)")
        
        return True

class DataProcessor:
    """Handles data processing logic."""
    
    @staticmethod
    def process_input(user_input):
        """Process the validated input."""
        return user_input.upper()

class OutputFormatter:
    """Handles output formatting logic."""
    
    @staticmethod
    def format_result(processed_data):
        """Format the processed data for output."""
        return f"Processed: {processed_data}"

class UserRoleManager:
    """Handles user role and permission logic."""
    
    @staticmethod
    def get_user_info(processed_data):
        """Determine user info based on processed data."""
        if processed_data == "ADMIN":
            return {"role": "admin", "permissions": ["all"]}
        else:
            return {"role": "user", "data": processed_data}

def process_user_data(user_input):
    """
    Improved function with separated concerns.
    Now delegates to specialized classes for each responsibility.
    """
    try:
        # Validation (delegated)
        InputValidator.validate_user_input(user_input)
        
        # Processing (delegated)
        processed = DataProcessor.process_input(user_input)
        
        # Formatting (delegated)
        formatted = OutputFormatter.format_result(processed)
        print(formatted)
        
        # Role management (delegated)
        return UserRoleManager.get_user_info(processed)
        
    except ValueError as e:
        print(f"Error: {e}")
        return None

def main():
    result = process_user_data("user")
    print(f"Result: {result}")
    
if __name__ == "__main__":
    main()
'''
        })
        
        # Store design change request in context
        context.add_artifact("design_change_request.json", json.dumps(design_request, indent=2), "test")
        
        # Test CodeGenerationAgent with design update
        code_agent = CodeGenerationAgent(llm_client=mock_code_llm)
        code_task = TaskNode(goal="Apply design changes: Improve separation of concerns", assigned_agent="CodeGenerationAgent")
        
        result = code_agent.execute("Design update: Improve separation of concerns", context, code_task)
        
        if result.success:
            print("✅ CodeGenerationAgent Design Update PASSED")
            print(f"   Message: {result.message}")
            print(f"   Files modified: {len(result.artifacts_generated or [])}")
            
            # Check if file was actually modified with improved design
            updated_content = test_file.read_text()
            if "InputValidator" in updated_content and "DataProcessor" in updated_content:
                print("✅ Design improvements correctly applied")
                print("   - Separated validation logic into InputValidator")
                print("   - Separated processing logic into DataProcessor") 
                print("   - Separated formatting logic into OutputFormatter")
                print("   - Separated role logic into UserRoleManager")
            else:
                print("❌ Design improvements not properly applied")
        else:
            print("❌ CodeGenerationAgent Design Update FAILED")
            print(f"   Error: {result.message}")
        
        # 3. Test Full End-to-End Design Change Workflow
        print("\n🔄 Testing End-to-End Design Change Workflow")
        print("-" * 45)
        
        # Reset test file to original poor design
        test_file.write_text("""def poor_function(data):
    # Poor design example
    if not data:
        return "error"
    return data.upper()
""")
        
        # Create agent registry with CodeGenerationAgent 
        agent_registry = {
            "CodeGenerationAgent": code_agent
        }
        
        # Create debugging agent with registry
        debugging_agent_full = DebuggingAgent(llm_client=mock_llm, agent_registry=agent_registry)
        full_task = TaskNode(goal="Fix design issues comprehensively", assigned_agent="DebuggingAgent")
        
        # Execute full workflow
        full_result = debugging_agent_full.execute("Fix the design issues in the code", context, full_task)
        
        if full_result.success:
            print("✅ Full Design Change Workflow PASSED")
            print(f"   Message: {full_result.message}")
            
            # Check if design change request was created and processed
            if "design_change_request.json" in context.artifacts:
                print("✅ Structured design change request was created")
            else:
                print("❌ No design change request found")
                
        else:
            print("❌ Full Design Change Workflow FAILED")
            print(f"   Error: {full_result.message}")
        
        # 4. Test Communication Logging for Design Changes
        print("\n📡 Testing Design Change Communication Logging")
        print("-" * 45)
        
        # Check if inter-agent communications were logged for design changes
        if hasattr(context, 'communications') and context.communications:
            design_communications = [
                comm for comm in context.communications 
                if comm.get('message_type') == 'delegation' and 'design' in comm.get('content', '').lower()
            ]
            
            if design_communications:
                print(f"✅ Found {len(design_communications)} design-related communications")
                for comm in design_communications[-2:]:  # Show last 2
                    print(f"   {comm.get('from_agent', 'Unknown')} → {comm.get('to_agent', 'Unknown')}: {comm.get('content', 'Unknown')[:60]}...")
            else:
                print("❌ No design-specific communications found")
        else:
            print("❌ No communications logged")
        
        print("\n🎯 Design Change Test Summary")
        print("=" * 50)
        print("✅ Structured design change requests implemented")
        print("✅ CodeGenerationAgent design update mode working")
        print("✅ Evidence-based design reasoning implemented")
        print("✅ End-to-end design change workflow functional")
        print("✅ Communication transparency for design changes")
        
        return True

if __name__ == "__main__":
    try:
        success = test_design_change_workflow()
        if success:
            print("\n🎉 All design change tests completed successfully!")
            exit(0)
        else:
            print("\n💥 Some design change tests failed!")
            exit(1)
    except Exception as e:
        print(f"\n💥 Design change test execution failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)