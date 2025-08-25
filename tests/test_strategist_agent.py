#!/usr/bin/env python3

import sys
sys.path.append('.')

import logging
from fagents.strategist import StrategistAgent
from fagents.analyst import AnalystAgent
from core.context import GlobalContext
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_strategist_agent():
    """Test the foundational StrategistAgent"""
    
    print("=" * 60)
    print("TESTING FOUNDATIONAL STRATEGIST AGENT")  
    print("=" * 60)
    
    # Create mock agent registry
    agent_registry = {
        "AnalystAgent": AnalystAgent,
        "StrategistAgent": StrategistAgent,
        "CreatorAgent": type('MockCreatorAgent', (), {'execute': lambda *args: None}),
        "ExecutorAgent": type('MockExecutorAgent', (), {'execute': lambda *args: None}),
        "SurgeonAgent": type('MockSurgeonAgent', (), {'execute': lambda *args: None})
    }
    
    # Initialize the agent
    strategist = StrategistAgent(agent_registry=agent_registry)
    
    # Test 1: Task Planning
    print("\n" + "─" * 40)
    print("TEST 1: Task Planning")
    print("─" * 40)
    
    global_context = GlobalContext(workspace_path=Path.cwd())
    
    result = strategist.execute(
        goal="Plan development of a Python web application",
        inputs={
            "requirements": "Build a simple web app with authentication",
            "quality_level": "decent",
            "constraints": {"timeline": "2 weeks"}
        },
        global_context=global_context
    )
    
    print(f"Success: {result.success}")
    print(f"Message: {result.message}")
    print(f"Strategy Type: {result.outputs.get('strategy_type', 'Unknown')}")
    
    if result.success:
        task_graph = result.outputs.get('task_graph', {})
        estimates = result.outputs.get('estimates', {})
        print(f"Tasks Generated: {estimates.get('total_tasks', 0)}")
        print(f"Estimated Duration: {estimates.get('total_duration_minutes', 0)} minutes")
        print(f"Critical Path Length: {estimates.get('critical_path_length', 0)}")
    
    # Test 2: Plan Refinement
    print("\n" + "─" * 40)
    print("TEST 2: Plan Refinement") 
    print("─" * 40)
    
    if result.success and 'task_graph' in result.outputs:
        refinement_result = strategist.execute(
            goal="Add comprehensive testing to the plan",
            inputs={
                "existing_plan": result.outputs['task_graph'],
                "refinement_request": "Add more testing steps and quality checks",
                "refinement_context": {"focus": "testing"}
            },
            global_context=global_context
        )
        
        print(f"Success: {refinement_result.success}")
        print(f"Message: {refinement_result.message}")
        
        if refinement_result.success:
            refinement_summary = refinement_result.outputs.get('refinement_summary', {})
            print(f"Original Tasks: {refinement_summary.get('original_task_count', 0)}")
            print(f"Refined Tasks: {refinement_summary.get('refined_task_count', 0)}")
            print(f"Tasks Added: {refinement_summary.get('tasks_added', 0)}")
    
    # Test 3: Resource Optimization
    print("\n" + "─" * 40)
    print("TEST 3: Resource Optimization")
    print("─" * 40)
    
    if result.success and 'task_graph' in result.outputs:
        optimization_result = strategist.execute(
            goal="Optimize workflow for parallel execution",
            inputs={
                "task_graph": result.outputs['task_graph'],
                "optimization_constraints": {"max_parallel": 4}
            },
            global_context=global_context
        )
        
        print(f"Success: {optimization_result.success}")
        print(f"Message: {optimization_result.message}")
        
        if optimization_result.success:
            optimization_summary = optimization_result.outputs.get('optimization_summary', {})
            print(f"Parallel Groups: {optimization_summary.get('parallel_groups', 0)}")
            print(f"Parallelization Opportunities: {optimization_summary.get('parallelization_opportunities', 0)}")
            
            recommendations = optimization_result.outputs.get('optimization_recommendations', [])
            if recommendations:
                print(f"Top Recommendation: {recommendations[0]}")
    
    # Test 4: Show capabilities
    print("\n" + "─" * 40)
    print("TEST 4: Agent Capabilities")
    print("─" * 40)
    
    capabilities = strategist.get_capabilities()
    print(f"Agent: {capabilities['name']}")
    print(f"Description: {capabilities['description']}")
    print(f"Strategy Modes: {len(capabilities['strategy_modes'])}")
    print(f"Orchestration Modes: {capabilities['orchestration_modes']}")
    print(f"Primary Functions: {len(capabilities['primary_functions'])}")
    
    # Test 5: Mock Workflow Orchestration (without actual execution)
    print("\n" + "─" * 40)
    print("TEST 5: Workflow Orchestration Planning")
    print("─" * 40)
    
    # This tests the orchestration setup without actually running agents
    if result.success and 'task_graph' in result.outputs:
        print("Workflow orchestration would coordinate:")
        task_graph = result.outputs['task_graph']
        if 'nodes' in task_graph:
            for task_id, task in task_graph['nodes'].items():
                agent_name = task.get('assigned_agent', 'Unknown')
                print(f"  - {task_id}: {agent_name}")
    
    print("\n" + "=" * 60)
    print("FOUNDATIONAL STRATEGIST AGENT TEST COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    test_strategist_agent()