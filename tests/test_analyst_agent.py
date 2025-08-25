#!/usr/bin/env python3

import sys
sys.path.append('.')

import logging
from fagents.analyst import AnalystAgent
from core.context import GlobalContext
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_analyst_agent():
    """Test the foundational AnalystAgent"""
    
    print("=" * 60)
    print("TESTING FOUNDATIONAL ANALYST AGENT")  
    print("=" * 60)
    
    # Initialize the agent
    analyst = AnalystAgent()
    
    # Test 1: Environment Analysis
    print("\n" + "─" * 40)
    print("TEST 1: Environment Analysis")
    print("─" * 40)
    
    global_context = GlobalContext(workspace_path=Path.cwd())
    
    result = analyst.execute(
        goal="Analyze development environment",
        inputs={"detailed_analysis": True},
        global_context=global_context
    )
    
    print(f"Success: {result.success}")
    print(f"Message: {result.message}")
    print(f"Analysis Type: {result.outputs.get('analysis_type', 'Unknown')}")
    
    if result.success:
        health = result.outputs.get('health_assessment', {})
        print(f"Environment Health: {health.get('summary', 'Unknown')}")
    
    # Test 2: Code Analysis (if Python files exist)
    print("\n" + "─" * 40)
    print("TEST 2: Code Analysis") 
    print("─" * 40)
    
    result = analyst.execute(
        goal="Analyze code quality and syntax",
        inputs={"validation_level": "standard", "check_imports": True},
        global_context=global_context
    )
    
    print(f"Success: {result.success}")
    print(f"Message: {result.message}")
    print(f"Files Analyzed: {result.outputs.get('files_analyzed', 0)}")
    print(f"Files Passed: {result.outputs.get('files_passed', 0)}")
    
    # Test 3: Problem Analysis
    print("\n" + "─" * 40)
    print("TEST 3: Problem Analysis")
    print("─" * 40)
    
    result = analyst.execute(
        goal="Diagnose import error problem",
        inputs={
            "problem_data": "ImportError: No module named 'nonexistent_module'",
            "error_logs": [
                "ImportError: No module named 'nonexistent_module'",
                "  File 'main.py', line 5, in <module>",
                "    import nonexistent_module"
            ],
            "context_information": {"python_version": "3.9"}
        },
        global_context=global_context
    )
    
    print(f"Success: {result.success}")
    print(f"Message: {result.message}")
    
    if result.success:
        problem_summary = result.outputs.get('problem_summary', {})
        print(f"Problem Category: {problem_summary.get('category', 'Unknown')}")
        print(f"Severity: {problem_summary.get('severity', 'Unknown')}")
        
        analysis = result.outputs.get('analysis_results', {})
        recommendations = analysis.get('recommendations', {})
        immediate_actions = recommendations.get('immediate_actions', [])
        if immediate_actions:
            print(f"Immediate Actions: {', '.join(immediate_actions[:2])}")
    
    # Test 4: Show capabilities
    print("\n" + "─" * 40)
    print("TEST 4: Agent Capabilities")
    print("─" * 40)
    
    capabilities = analyst.get_capabilities()
    print(f"Agent: {capabilities['name']}")
    print(f"Description: {capabilities['description']}")
    print(f"Analysis Modes: {len(capabilities['analysis_modes'])}")
    print(f"Primary Functions: {len(capabilities['primary_functions'])}")
    
    print("\n" + "=" * 60)
    print("FOUNDATIONAL ANALYST AGENT TEST COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    test_analyst_agent()