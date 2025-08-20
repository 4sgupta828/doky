# tools/planning/task_planning_tools.py
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timedelta

# Core dependencies
from core.models import TaskGraph, TaskNode

# Get a logger instance for this module
logger = logging.getLogger(__name__)


class PlanningQuality(Enum):
    """Defines different planning quality levels for speed vs detail trade-offs."""
    FAST = "fast"
    DECENT = "decent"
    PRODUCTION = "production"


class WorkflowType(Enum):
    """Different types of workflows that can be orchestrated."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    DEBUGGING = "debugging"
    DEPLOYMENT = "deployment"
    RESEARCH = "research"
    MAINTENANCE = "maintenance"


@dataclass
class PlanningContext:
    """Context information for planning operations."""
    goal: str
    user_intent: str
    quality_level: PlanningQuality
    available_agents: List[Dict[str, Any]]
    workspace_files: List[str]
    constraints: Dict[str, Any]
    preferences: Dict[str, Any]


@dataclass 
class WorkflowStep:
    """Represents a single step in a workflow."""
    step_id: str
    step_name: str
    agent_name: str
    inputs: Dict[str, Any]
    dependencies: List[str]
    estimated_duration: Optional[int] = None
    parallel_group: Optional[str] = None


@dataclass
class WorkflowPlan:
    """Complete workflow plan with steps and metadata."""
    workflow_id: str
    workflow_type: WorkflowType
    goal: str
    steps: List[WorkflowStep]
    total_estimated_duration: int
    critical_path: List[str]
    parallel_groups: Dict[str, List[str]]
    quality_level: PlanningQuality
    created_at: datetime


def analyze_user_intent(goal: str, context_summary: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Analyze user intent and determine planning requirements.
    
    Args:
        goal: The user's high-level goal
        context_summary: Context information about workspace, files, etc.
        
    Returns:
        Dictionary with analyzed intent, planning quality, and requirements
    """
    context_summary = context_summary or {}
    
    # Simple intent analysis (in real implementation, this would use LLM)
    intent_analysis = {
        "user_intent": goal,
        "planning_quality": determine_planning_quality(goal),
        "complexity_level": assess_goal_complexity(goal),
        "domain": identify_domain(goal),
        "requirements": extract_requirements(goal),
        "constraints": extract_constraints(goal, context_summary),
        "success_criteria": define_success_criteria(goal)
    }
    
    logger.info(f"Intent analysis complete: {intent_analysis['complexity_level']} complexity, {intent_analysis['planning_quality']} quality")
    
    return intent_analysis


def determine_planning_quality(goal: str) -> PlanningQuality:
    """Determine appropriate planning quality level based on goal."""
    goal_lower = goal.lower()
    
    # Quality indicators
    production_indicators = [
        "production", "enterprise", "deploy", "release", "secure", "scalable", 
        "maintainable", "comprehensive", "robust", "enterprise-grade"
    ]
    
    fast_indicators = [
        "quick", "fast", "prototype", "poc", "demo", "experiment", 
        "test", "try", "simple", "basic"
    ]
    
    if any(indicator in goal_lower for indicator in production_indicators):
        return PlanningQuality.PRODUCTION
    elif any(indicator in goal_lower for indicator in fast_indicators):
        return PlanningQuality.FAST
    else:
        return PlanningQuality.DECENT


def assess_goal_complexity(goal: str) -> str:
    """Assess the complexity level of a goal."""
    goal_lower = goal.lower()
    
    high_complexity_indicators = [
        "system", "architecture", "multiple", "integration", "complex", "advanced",
        "enterprise", "distributed", "scalable", "production"
    ]
    
    low_complexity_indicators = [
        "simple", "basic", "single", "quick", "small", "prototype", "demo"
    ]
    
    if any(indicator in goal_lower for indicator in high_complexity_indicators):
        return "high"
    elif any(indicator in goal_lower for indicator in low_complexity_indicators):
        return "low"
    else:
        return "medium"


def identify_domain(goal: str) -> str:
    """Identify the domain or area of the goal."""
    goal_lower = goal.lower()
    
    domains = {
        "development": ["develop", "code", "implement", "build", "create", "write"],
        "testing": ["test", "validate", "verify", "check", "quality"],
        "debugging": ["debug", "fix", "error", "problem", "issue", "bug"],
        "deployment": ["deploy", "release", "publish", "launch", "production"],
        "research": ["research", "analyze", "investigate", "explore", "study"],
        "maintenance": ["maintain", "update", "refactor", "optimize", "improve"]
    }
    
    for domain, keywords in domains.items():
        if any(keyword in goal_lower for keyword in keywords):
            return domain
            
    return "general"


def extract_requirements(goal: str) -> List[str]:
    """Extract functional and non-functional requirements from goal."""
    requirements = []
    goal_lower = goal.lower()
    
    # Common requirement patterns
    if "secure" in goal_lower or "security" in goal_lower:
        requirements.append("Security requirements must be addressed")
    
    if "test" in goal_lower:
        requirements.append("Testing coverage is required")
        
    if "performance" in goal_lower or "fast" in goal_lower:
        requirements.append("Performance optimization is required")
        
    if "documentation" in goal_lower or "document" in goal_lower:
        requirements.append("Documentation must be created")
        
    if "scalable" in goal_lower or "scale" in goal_lower:
        requirements.append("Solution must be scalable")
        
    if not requirements:
        requirements.append("Basic functionality implementation")
        
    return requirements


def extract_constraints(goal: str, context_summary: Dict[str, Any]) -> Dict[str, Any]:
    """Extract constraints from goal and context."""
    constraints = {
        "time_constraints": [],
        "resource_constraints": [],
        "technology_constraints": [],
        "quality_constraints": []
    }
    
    goal_lower = goal.lower()
    
    # Time constraints
    if any(word in goal_lower for word in ["quick", "fast", "urgent", "asap"]):
        constraints["time_constraints"].append("Time-sensitive delivery")
        
    # Technology constraints from context
    if context_summary.get("files_in_workspace"):
        files = context_summary["files_in_workspace"]
        if any(f.endswith('.py') for f in files):
            constraints["technology_constraints"].append("Python-based solution")
        if any(f.endswith('.js') for f in files):
            constraints["technology_constraints"].append("JavaScript-based solution")
            
    return constraints


def define_success_criteria(goal: str) -> List[str]:
    """Define success criteria for the goal."""
    criteria = []
    goal_lower = goal.lower()
    
    # Basic success criteria
    criteria.append("Implementation matches requirements")
    
    if "test" in goal_lower:
        criteria.append("All tests pass")
        
    if "deploy" in goal_lower:
        criteria.append("Successfully deployed to target environment")
        
    if "fix" in goal_lower or "debug" in goal_lower:
        criteria.append("Problem is resolved and verified")
        
    if "performance" in goal_lower:
        criteria.append("Performance targets are met")
        
    return criteria


def generate_task_graph(planning_context: PlanningContext) -> TaskGraph:
    """
    Generate a complete task graph based on planning context.
    
    Args:
        planning_context: Complete context for planning
        
    Returns:
        TaskGraph with all tasks and dependencies
    """
    logger.info(f"Generating task graph for: {planning_context.goal}")
    
    # Determine workflow type
    workflow_type = determine_workflow_type(planning_context.goal)
    
    # Generate workflow steps based on type and quality
    workflow_steps = generate_workflow_steps(workflow_type, planning_context)
    
    # Convert to TaskGraph format
    task_graph = convert_to_task_graph(workflow_steps, planning_context)
    
    logger.info(f"Generated task graph with {len(task_graph.nodes)} tasks")
    
    return task_graph


def determine_workflow_type(goal: str) -> WorkflowType:
    """Determine the type of workflow based on the goal."""
    goal_lower = goal.lower()
    
    workflow_keywords = {
        WorkflowType.DEVELOPMENT: ["develop", "build", "create", "implement", "code", "write"],
        WorkflowType.TESTING: ["test", "validate", "verify", "check"],
        WorkflowType.DEBUGGING: ["debug", "fix", "error", "problem", "issue"],
        WorkflowType.DEPLOYMENT: ["deploy", "release", "publish", "launch"],
        WorkflowType.RESEARCH: ["research", "analyze", "investigate", "explore"],
        WorkflowType.MAINTENANCE: ["maintain", "update", "refactor", "optimize"]
    }
    
    for workflow_type, keywords in workflow_keywords.items():
        if any(keyword in goal_lower for keyword in keywords):
            return workflow_type
            
    return WorkflowType.DEVELOPMENT


def generate_workflow_steps(workflow_type: WorkflowType, context: PlanningContext) -> List[WorkflowStep]:
    """Generate workflow steps based on type and context."""
    
    if workflow_type == WorkflowType.DEVELOPMENT:
        return generate_development_workflow(context)
    elif workflow_type == WorkflowType.TESTING:
        return generate_testing_workflow(context)
    elif workflow_type == WorkflowType.DEBUGGING:
        return generate_debugging_workflow(context)
    elif workflow_type == WorkflowType.DEPLOYMENT:
        return generate_deployment_workflow(context)
    elif workflow_type == WorkflowType.RESEARCH:
        return generate_research_workflow(context)
    elif workflow_type == WorkflowType.MAINTENANCE:
        return generate_maintenance_workflow(context)
    else:
        return generate_generic_workflow(context)


def generate_development_workflow(context: PlanningContext) -> List[WorkflowStep]:
    """Generate development workflow steps."""
    steps = []
    
    # Analysis phase
    steps.append(WorkflowStep(
        step_id="analysis_001",
        step_name="Analyze Requirements",
        agent_name="AnalystAgent",
        inputs={
            "analysis_type": "requirements_analysis",
            "goal": context.goal,
            "context": context.workspace_files
        },
        dependencies=[]
    ))
    
    # Planning phase
    if context.quality_level in [PlanningQuality.DECENT, PlanningQuality.PRODUCTION]:
        steps.append(WorkflowStep(
            step_id="planning_001", 
            step_name="Create Technical Specification",
            agent_name="StrategistAgent",
            inputs={
                "planning_type": "technical_specification",
                "requirements_analysis": "analysis_001"
            },
            dependencies=["analysis_001"]
        ))
    
    # Implementation phase
    steps.append(WorkflowStep(
        step_id="implementation_001",
        step_name="Implement Core Functionality",
        agent_name="CreatorAgent",
        inputs={
            "creation_type": "code_implementation",
            "specifications": "planning_001" if context.quality_level != PlanningQuality.FAST else "analysis_001"
        },
        dependencies=["planning_001"] if context.quality_level != PlanningQuality.FAST else ["analysis_001"]
    ))
    
    # Testing phase (for DECENT and PRODUCTION quality)
    if context.quality_level in [PlanningQuality.DECENT, PlanningQuality.PRODUCTION]:
        steps.append(WorkflowStep(
            step_id="testing_001",
            step_name="Create and Run Tests",
            agent_name="ExecutorAgent",
            inputs={
                "execution_type": "test_execution",
                "code_artifacts": "implementation_001"
            },
            dependencies=["implementation_001"]
        ))
        
        steps.append(WorkflowStep(
            step_id="validation_001",
            step_name="Validate Implementation",
            agent_name="AnalystAgent", 
            inputs={
                "analysis_type": "quality_validation",
                "implementation": "implementation_001",
                "test_results": "testing_001"
            },
            dependencies=["implementation_001", "testing_001"],
            parallel_group="validation"
        ))
    
    # Quality assurance (for PRODUCTION quality)
    if context.quality_level == PlanningQuality.PRODUCTION:
        steps.append(WorkflowStep(
            step_id="quality_001",
            step_name="Security and Quality Audit",
            agent_name="AnalystAgent",
            inputs={
                "analysis_type": "comprehensive_quality_audit",
                "implementation": "implementation_001"
            },
            dependencies=["implementation_001"],
            parallel_group="validation"
        ))
    
    return steps


def generate_testing_workflow(context: PlanningContext) -> List[WorkflowStep]:
    """Generate testing-specific workflow steps."""
    return [
        WorkflowStep(
            step_id="test_analysis_001",
            step_name="Analyze Testing Requirements",
            agent_name="AnalystAgent",
            inputs={"analysis_type": "test_requirements", "goal": context.goal},
            dependencies=[]
        ),
        WorkflowStep(
            step_id="test_creation_001",
            step_name="Create Test Suite",
            agent_name="CreatorAgent",
            inputs={"creation_type": "test_suite", "requirements": "test_analysis_001"},
            dependencies=["test_analysis_001"]
        ),
        WorkflowStep(
            step_id="test_execution_001",
            step_name="Execute Tests",
            agent_name="ExecutorAgent",
            inputs={"execution_type": "test_execution", "test_suite": "test_creation_001"},
            dependencies=["test_creation_001"]
        )
    ]


def generate_debugging_workflow(context: PlanningContext) -> List[WorkflowStep]:
    """Generate debugging-specific workflow steps."""
    return [
        WorkflowStep(
            step_id="problem_analysis_001",
            step_name="Analyze Problem",
            agent_name="AnalystAgent",
            inputs={"analysis_type": "problem_diagnosis", "problem_data": context.goal},
            dependencies=[]
        ),
        WorkflowStep(
            step_id="solution_planning_001",
            step_name="Plan Solution Strategy",
            agent_name="StrategistAgent",
            inputs={"planning_type": "solution_strategy", "problem_analysis": "problem_analysis_001"},
            dependencies=["problem_analysis_001"]
        ),
        WorkflowStep(
            step_id="fix_implementation_001",
            step_name="Implement Fix",
            agent_name="SurgeonAgent",
            inputs={"modification_type": "bug_fix", "solution_plan": "solution_planning_001"},
            dependencies=["solution_planning_001"]
        ),
        WorkflowStep(
            step_id="fix_validation_001",
            step_name="Validate Fix",
            agent_name="ExecutorAgent",
            inputs={"execution_type": "fix_validation", "fix": "fix_implementation_001"},
            dependencies=["fix_implementation_001"]
        )
    ]


def generate_deployment_workflow(context: PlanningContext) -> List[WorkflowStep]:
    """Generate deployment-specific workflow steps."""
    return [
        WorkflowStep(
            step_id="deploy_prep_001",
            step_name="Prepare for Deployment",
            agent_name="AnalystAgent",
            inputs={"analysis_type": "deployment_readiness", "goal": context.goal},
            dependencies=[]
        ),
        WorkflowStep(
            step_id="deploy_execute_001",
            step_name="Execute Deployment",
            agent_name="ExecutorAgent", 
            inputs={"execution_type": "deployment", "deployment_plan": "deploy_prep_001"},
            dependencies=["deploy_prep_001"]
        )
    ]


def generate_research_workflow(context: PlanningContext) -> List[WorkflowStep]:
    """Generate research-specific workflow steps."""
    return [
        WorkflowStep(
            step_id="research_001",
            step_name="Research and Analysis",
            agent_name="AnalystAgent",
            inputs={"analysis_type": "research", "research_goal": context.goal},
            dependencies=[]
        ),
        WorkflowStep(
            step_id="synthesis_001",
            step_name="Synthesize Findings",
            agent_name="CreatorAgent",
            inputs={"creation_type": "research_report", "research_data": "research_001"},
            dependencies=["research_001"]
        )
    ]


def generate_maintenance_workflow(context: PlanningContext) -> List[WorkflowStep]:
    """Generate maintenance-specific workflow steps."""
    return [
        WorkflowStep(
            step_id="maintenance_analysis_001",
            step_name="Analyze Maintenance Requirements",
            agent_name="AnalystAgent",
            inputs={"analysis_type": "maintenance_analysis", "goal": context.goal},
            dependencies=[]
        ),
        WorkflowStep(
            step_id="maintenance_plan_001",
            step_name="Plan Maintenance Actions",
            agent_name="StrategistAgent",
            inputs={"planning_type": "maintenance_plan", "analysis": "maintenance_analysis_001"},
            dependencies=["maintenance_analysis_001"]
        ),
        WorkflowStep(
            step_id="maintenance_execute_001",
            step_name="Execute Maintenance",
            agent_name="SurgeonAgent",
            inputs={"modification_type": "maintenance", "plan": "maintenance_plan_001"},
            dependencies=["maintenance_plan_001"]
        )
    ]


def generate_generic_workflow(context: PlanningContext) -> List[WorkflowStep]:
    """Generate generic workflow steps."""
    return [
        WorkflowStep(
            step_id="generic_analysis_001",
            step_name="Analyze Task",
            agent_name="AnalystAgent",
            inputs={"analysis_type": "general", "goal": context.goal},
            dependencies=[]
        ),
        WorkflowStep(
            step_id="generic_execute_001",
            step_name="Execute Task",
            agent_name="CreatorAgent",
            inputs={"creation_type": "general", "analysis": "generic_analysis_001"},
            dependencies=["generic_analysis_001"]
        )
    ]


def convert_to_task_graph(workflow_steps: List[WorkflowStep], context: PlanningContext) -> TaskGraph:
    """Convert workflow steps to TaskGraph format."""
    task_nodes = {}
    
    for step in workflow_steps:
        task_node = TaskNode(
            task_id=step.step_id,
            goal=step.step_name,
            assigned_agent=step.agent_name,
            status="pending",
            dependencies=step.dependencies,
            input_artifact_keys=[],
            output_artifact_keys=[]
        )
        task_nodes[step.step_id] = task_node
    
    return TaskGraph(nodes=task_nodes)


def refine_task_graph(task_graph: TaskGraph, refinement_request: str, context: Dict[str, Any] = None) -> TaskGraph:
    """
    Refine an existing task graph based on feedback or new requirements.
    
    Args:
        task_graph: Existing task graph to refine
        refinement_request: Description of how to refine the graph
        context: Additional context for refinement
        
    Returns:
        Refined TaskGraph
    """
    context = context or {}
    logger.info(f"Refining task graph: {refinement_request}")
    
    refined_nodes = dict(task_graph.nodes)
    
    # Simple refinement logic (in real implementation, this would use LLM)
    refinement_lower = refinement_request.lower()
    
    if "add" in refinement_lower and "test" in refinement_lower:
        # Add testing steps
        test_step = TaskNode(
            task_id="test_refinement_001",
            goal="Additional Testing Step",
            assigned_agent="ExecutorAgent",
            status="pending",
            dependencies=list(refined_nodes.keys()),
            input_artifact_keys=[],
            output_artifact_keys=[]
        )
        refined_nodes["test_refinement_001"] = test_step
        
    elif "remove" in refinement_lower:
        # Remove optional steps (simplified example)
        # In this simplified version, we don't remove anything
        pass
                
    elif "optimize" in refinement_lower or "parallel" in refinement_lower:
        # For optimization, we can't modify metadata since TaskNode doesn't have it
        # In practice, this would be handled differently
        pass
    
    logger.info(f"Task graph refined: {len(refined_nodes)} tasks")
    
    return TaskGraph(nodes=refined_nodes)


def estimate_workflow_duration(workflow_steps: List[WorkflowStep]) -> int:
    """Estimate total workflow duration in minutes."""
    base_durations = {
        "AnalystAgent": 15,  # Analysis tasks
        "StrategistAgent": 10,  # Planning tasks
        "CreatorAgent": 30,  # Creation tasks
        "SurgeonAgent": 20,  # Modification tasks
        "ExecutorAgent": 25   # Execution tasks
    }
    
    total_duration = 0
    parallel_groups = {}
    
    for step in workflow_steps:
        duration = step.estimated_duration or base_durations.get(step.agent_name, 20)
        
        if step.parallel_group:
            if step.parallel_group not in parallel_groups:
                parallel_groups[step.parallel_group] = 0
            parallel_groups[step.parallel_group] = max(parallel_groups[step.parallel_group], duration)
        else:
            total_duration += duration
    
    # Add parallel group durations
    total_duration += sum(parallel_groups.values())
    
    return total_duration


def identify_critical_path(workflow_steps: List[WorkflowStep]) -> List[str]:
    """Identify the critical path through workflow steps."""
    # Simple critical path identification
    # In practice, this would use proper critical path method (CPM)
    
    dependency_chains = []
    current_chain = []
    
    # Find longest dependency chain
    for step in workflow_steps:
        if not step.dependencies:
            current_chain = [step.step_id]
        else:
            # Extend chain with dependent steps
            current_chain.append(step.step_id)
            
        dependency_chains.append(current_chain.copy())
    
    # Return longest chain as critical path
    critical_path = max(dependency_chains, key=len) if dependency_chains else []
    
    logger.info(f"Critical path identified: {len(critical_path)} steps")
    
    return critical_path