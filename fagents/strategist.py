# fagents/strategist.py
import json
import logging
from typing import Dict, Any, List, Optional
import uuid

# Foundational base
from .base import FoundationalAgent
from core.context import GlobalContext
from core.models import AgentResult, TaskGraph

# Planning tools - atomic and reusable
from tools.task_planning_tools import (
    analyze_user_intent, generate_task_graph, refine_task_graph,
    PlanningContext, PlanningQuality, WorkflowType, 
    estimate_workflow_duration, identify_critical_path
)
from tools.workflow_orchestration_tools import (
    orchestrate_workflow, create_orchestration_context,
    optimize_workflow_execution, monitor_workflow_progress,
    OrchestrationMode, OrchestrationResult
)

# Get a logger instance for this module
logger = logging.getLogger(__name__)


class StrategistAgent(FoundationalAgent):
    """
    FOUNDATIONAL AGENT 2: STRATEGIST
    
    The Intelligence That Plans - Task decomposition and intelligent orchestration.
    
    Core Capability: Complex task decomposition and multi-agent workflow orchestration.
    
    Powers:
    - Complex task decomposition with dependency analysis
    - Multi-agent workflow orchestration and optimization
    - Solution architecture and design pattern selection
    - Risk assessment and mitigation strategy formulation
    - Resource estimation and timeline planning
    - Adaptive planning with feedback loops and error recovery
    - Cross-cutting concern management (security, performance, maintainability)
    
    Unique Value: Can decompose ANY complex problem into executable workflows
    """
    
    def __init__(self, agent_registry: Dict[str, Any] = None, llm_client: Any = None):
        super().__init__(
            name="StrategistAgent",
            description="Task decomposition and intelligent orchestration agent that plans and coordinates complex workflows."
        )
        self.agent_registry = agent_registry or {}
        self._llm_client = llm_client
    
    def execute(self, goal: str, inputs: Dict[str, Any], global_context: GlobalContext) -> AgentResult:
        """
        Execute strategic planning and orchestration based on the goal and inputs.
        
        Supports multiple strategy types:
        - task_planning: Decompose goals into executable task graphs
        - workflow_orchestration: Execute complete workflows with multiple agents
        - plan_refinement: Refine existing plans based on feedback
        - resource_optimization: Optimize workflows for parallel execution
        - progress_monitoring: Monitor and report on workflow progress
        """
        self.report_progress("Starting strategic planning", f"Goal: {goal}")
        
        try:
            # Determine strategy type from goal and inputs
            strategy_type = self._determine_strategy_type(goal, inputs)
            self.report_progress("Strategy type determined", strategy_type)
            
            # Execute the appropriate strategy
            if strategy_type == "task_planning":
                return self._execute_task_planning(goal, inputs, global_context)
            elif strategy_type == "workflow_orchestration":
                return self._execute_workflow_orchestration(goal, inputs, global_context)
            elif strategy_type == "plan_refinement":
                return self._execute_plan_refinement(goal, inputs, global_context)
            elif strategy_type == "resource_optimization":
                return self._execute_resource_optimization(goal, inputs, global_context)
            elif strategy_type == "progress_monitoring":
                return self._execute_progress_monitoring(goal, inputs, global_context)
            else:
                # Auto-detect based on available inputs
                return self._execute_auto_strategy(goal, inputs, global_context)
                
        except Exception as e:
            self.report_error(f"Strategic planning failed: {e}", e)
            return AgentResult(
                success=False,
                message=f"Strategic planning failed: {e}",
                outputs={},
                error_details={"exception": str(e)}
            )
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Return comprehensive capabilities description for other agents."""
        return {
            "name": "StrategistAgent",
            "description": "Task decomposition and intelligent orchestration of complex workflows",
            "primary_functions": [
                "Complex task decomposition",
                "Multi-agent workflow orchestration",
                "Solution architecture planning",
                "Risk assessment and mitigation",
                "Resource estimation and optimization",
                "Adaptive planning with feedback loops"
            ],
            "input_types": [
                "goal", "requirements", "existing_plan", "workflow_constraints",
                "agent_capabilities", "resource_limits", "timeline", "quality_requirements"
            ],
            "output_types": [
                "task_graph", "workflow_plan", "orchestration_result",
                "resource_allocation", "timeline_estimate", "risk_assessment",
                "execution_strategy", "optimization_recommendations"
            ],
            "strategy_modes": [
                "task_planning", "workflow_orchestration", "plan_refinement",
                "resource_optimization", "progress_monitoring", "auto_strategy"
            ],
            "orchestration_modes": ["sequential", "parallel", "adaptive", "batch"],
            "complexity_handling": "Can handle simple tasks to complex multi-agent system orchestration"
        }
    
    def _determine_strategy_type(self, goal: str, inputs: Dict[str, Any]) -> str:
        """Determine the type of strategy to execute based on goal and inputs."""
        goal_lower = goal.lower()
        
        # Explicit strategy type requests
        if any(word in goal_lower for word in ["plan", "decompose", "break down", "strategy"]):
            return "task_planning"
        elif any(word in goal_lower for word in ["orchestrate", "execute", "run", "coordinate"]):
            return "workflow_orchestration"
        elif any(word in goal_lower for word in ["refine", "improve", "update", "modify"]):
            return "plan_refinement"
        elif any(word in goal_lower for word in ["optimize", "parallel", "efficient", "resource"]):
            return "resource_optimization"
        elif any(word in goal_lower for word in ["monitor", "progress", "status", "track"]):
            return "progress_monitoring"
            
        # Auto-detect based on inputs
        if "task_graph" in inputs and "agent_registry" in inputs:
            return "workflow_orchestration"
        elif "existing_plan" in inputs or "task_graph" in inputs:
            return "plan_refinement"
        elif inputs.get("workflow_id"):
            return "progress_monitoring"
        else:
            return "task_planning"
    
    def _execute_task_planning(self, goal: str, inputs: Dict[str, Any], global_context: GlobalContext) -> AgentResult:
        """Execute task planning and decomposition."""
        self.report_progress("Task planning", "Analyzing intent and creating task graph")
        
        # Analyze user intent
        context_summary = {
            "files_in_workspace": global_context.workspace.list_files() if global_context.workspace else [],
            "workspace_path": str(global_context.workspace_path) if global_context.workspace_path else ""
        }
        
        intent_analysis = analyze_user_intent(goal, context_summary)
        
        # Create planning context
        planning_context = PlanningContext(
            goal=goal,
            user_intent=intent_analysis["user_intent"],
            quality_level=PlanningQuality(intent_analysis["planning_quality"]),
            available_agents=self._get_available_agent_capabilities(),
            workspace_files=context_summary["files_in_workspace"],
            constraints=inputs.get("constraints", {}),
            preferences=inputs.get("preferences", {})
        )
        
        # Generate task graph using original PlannerAgent LLM prompts or fallback
        if self._llm_client:
            task_graph = self._generate_task_graph_with_llm(planning_context, intent_analysis, global_context)
        else:
            # Fallback to rule-based generation
            task_graph = generate_task_graph(planning_context)
        
        # Estimate duration and identify critical path
        workflow_steps = self._convert_task_graph_to_steps(task_graph)
        total_duration = estimate_workflow_duration(workflow_steps)
        critical_path = identify_critical_path(workflow_steps)
        
        # Store in global context
        if global_context:
            global_context.task_graph.nodes.update(task_graph.nodes)
            global_context.log_event("plan_generated", {
                "task_count": len(task_graph.nodes),
                "quality": planning_context.quality_level.value,
                "estimated_duration": total_duration
            })
        
        success_message = f"Task planning complete: {len(task_graph.nodes)} tasks, {total_duration}min estimated"
        self.report_progress("Task planning complete", success_message)
        
        return AgentResult(
            success=True,
            message=success_message,
            outputs={
                "strategy_type": "task_planning",
                "task_graph": task_graph.model_dump(),
                "intent_analysis": intent_analysis,
                "planning_context": {
                    "quality_level": planning_context.quality_level.value,
                    "complexity": intent_analysis["complexity_level"],
                    "domain": intent_analysis["domain"]
                },
                "estimates": {
                    "total_duration_minutes": total_duration,
                    "total_tasks": len(task_graph.nodes),
                    "critical_path_length": len(critical_path)
                },
                "critical_path": critical_path
            }
        )
    
    def _execute_workflow_orchestration(self, goal: str, inputs: Dict[str, Any], global_context: GlobalContext) -> AgentResult:
        """Execute complete workflow orchestration."""
        self.report_progress("Workflow orchestration", "Orchestrating multi-agent workflow")
        
        # Get task graph
        task_graph = inputs.get("task_graph")
        if isinstance(task_graph, dict):
            task_graph = TaskGraph(**task_graph)
        elif not isinstance(task_graph, TaskGraph):
            # Create simple task graph from goal
            return self._execute_task_planning(goal, inputs, global_context)
        
        # Create orchestration context
        workflow_id = inputs.get("workflow_id", f"workflow_{uuid.uuid4().hex[:8]}")
        orchestration_mode = inputs.get("orchestration_mode", "adaptive")
        
        context = create_orchestration_context(
            workflow_id=workflow_id,
            orchestration_mode=orchestration_mode,
            max_parallel_tasks=inputs.get("max_parallel_tasks", 3),
            retry_attempts=inputs.get("retry_attempts", 2),
            timeout_minutes=inputs.get("timeout_minutes", 60),
            error_handling=inputs.get("error_handling", "continue"),
            global_context=global_context
        )
        
        # Optimize workflow if requested
        if inputs.get("optimize_execution", True):
            task_graph = optimize_workflow_execution(task_graph, inputs.get("optimization_constraints", {}))
        
        # Execute orchestration
        orchestration_result = orchestrate_workflow(task_graph, self.agent_registry, context)
        
        success_message = f"Workflow orchestration {'completed' if orchestration_result.success else 'failed'}: "
        success_message += f"{orchestration_result.completed_steps}/{orchestration_result.total_steps} tasks completed"
        
        self.report_progress("Workflow orchestration complete", success_message)
        
        return AgentResult(
            success=orchestration_result.success,
            message=success_message,
            outputs={
                "strategy_type": "workflow_orchestration",
                "orchestration_result": {
                    "workflow_id": orchestration_result.workflow_id,
                    "success": orchestration_result.success,
                    "total_steps": orchestration_result.total_steps,
                    "completed_steps": orchestration_result.completed_steps,
                    "failed_steps": orchestration_result.failed_steps,
                    "total_duration_seconds": orchestration_result.total_duration_seconds,
                    "error_summary": orchestration_result.error_summary
                },
                "final_outputs": orchestration_result.final_outputs,
                "workflow_id": workflow_id,
                "orchestration_mode": orchestration_mode
            }
        )
    
    def _execute_plan_refinement(self, goal: str, inputs: Dict[str, Any], global_context: GlobalContext) -> AgentResult:
        """Execute plan refinement based on feedback."""
        self.report_progress("Plan refinement", "Refining existing task plan")
        
        # Get existing plan
        existing_plan = inputs.get("existing_plan") or inputs.get("task_graph")
        if isinstance(existing_plan, dict):
            task_graph = TaskGraph(**existing_plan)
        elif isinstance(existing_plan, TaskGraph):
            task_graph = existing_plan
        else:
            return AgentResult(
                success=False,
                message="No existing plan provided for refinement",
                outputs={}
            )
        
        # Get refinement request
        refinement_request = inputs.get("refinement_request", goal)
        refinement_context = inputs.get("refinement_context", {})
        
        # Refine the task graph
        refined_graph = refine_task_graph(task_graph, refinement_request, refinement_context)
        
        # Calculate changes
        original_task_count = len(task_graph.nodes)
        refined_task_count = len(refined_graph.nodes)
        
        # Update global context if available
        if global_context:
            global_context.task_graph.nodes.update(refined_graph.nodes)
            global_context.log_event("plan_refined", {
                "original_tasks": original_task_count,
                "refined_tasks": refined_task_count,
                "refinement": refinement_request
            })
        
        success_message = f"Plan refinement complete: {original_task_count} → {refined_task_count} tasks"
        self.report_progress("Plan refinement complete", success_message)
        
        return AgentResult(
            success=True,
            message=success_message,
            outputs={
                "strategy_type": "plan_refinement",
                "refined_task_graph": refined_graph.model_dump(),
                "refinement_summary": {
                    "original_task_count": original_task_count,
                    "refined_task_count": refined_task_count,
                    "tasks_added": max(0, refined_task_count - original_task_count),
                    "tasks_removed": max(0, original_task_count - refined_task_count),
                    "refinement_request": refinement_request
                }
            }
        )
    
    def _execute_resource_optimization(self, goal: str, inputs: Dict[str, Any], global_context: GlobalContext) -> AgentResult:
        """Execute resource optimization for workflows."""
        self.report_progress("Resource optimization", "Optimizing workflow execution")
        
        # Get task graph
        task_graph = inputs.get("task_graph")
        if isinstance(task_graph, dict):
            task_graph = TaskGraph(**task_graph)
        elif not isinstance(task_graph, TaskGraph):
            return AgentResult(
                success=False,
                message="No task graph provided for optimization",
                outputs={}
            )
        
        # Get optimization constraints
        constraints = inputs.get("optimization_constraints", {})
        
        # Optimize the workflow
        optimized_graph = optimize_workflow_execution(task_graph, constraints)
        
        # Calculate optimization benefits
        workflow_steps = self._convert_task_graph_to_steps(optimized_graph)
        optimized_duration = estimate_workflow_duration(workflow_steps)
        
        # Identify parallelization opportunities
        parallel_groups = self._identify_parallel_groups(optimized_graph)
        
        success_message = f"Resource optimization complete: {len(parallel_groups)} parallel groups identified"
        self.report_progress("Resource optimization complete", success_message)
        
        return AgentResult(
            success=True,
            message=success_message,
            outputs={
                "strategy_type": "resource_optimization",
                "optimized_task_graph": optimized_graph.model_dump(),
                "optimization_summary": {
                    "parallel_groups": len(parallel_groups),
                    "estimated_duration_minutes": optimized_duration,
                    "parallelization_opportunities": len(parallel_groups)  # Use groups instead of metadata
                },
                "optimization_recommendations": self._generate_optimization_recommendations(optimized_graph)
            }
        )
    
    def _execute_progress_monitoring(self, goal: str, inputs: Dict[str, Any], global_context: GlobalContext) -> AgentResult:
        """Execute workflow progress monitoring."""
        self.report_progress("Progress monitoring", "Monitoring workflow execution")
        
        # Get orchestration result or workflow ID
        orchestration_result = inputs.get("orchestration_result")
        workflow_id = inputs.get("workflow_id")
        
        if not orchestration_result and not workflow_id:
            return AgentResult(
                success=False,
                message="No orchestration result or workflow ID provided for monitoring",
                outputs={}
            )
        
        # Create mock orchestration result if only workflow ID provided
        if not orchestration_result:
            # In practice, this would query a workflow tracking system
            orchestration_result = OrchestrationResult(
                workflow_id=workflow_id,
                success=True,
                total_steps=5,
                completed_steps=3,
                failed_steps=0,
                total_duration_seconds=300
            )
        
        # Monitor progress
        progress_info = monitor_workflow_progress(orchestration_result)
        
        success_message = f"Progress monitoring complete: {progress_info['status']} ({progress_info['progress_percentage']:.1f}%)"
        self.report_progress("Progress monitoring complete", success_message)
        
        return AgentResult(
            success=True,
            message=success_message,
            outputs={
                "strategy_type": "progress_monitoring",
                "progress_info": progress_info,
                "monitoring_summary": {
                    "workflow_status": progress_info["status"],
                    "progress_percentage": progress_info["progress_percentage"],
                    "completed_tasks": progress_info["completed_tasks"],
                    "total_tasks": progress_info["total_tasks"],
                    "current_errors": progress_info.get("current_errors", [])
                }
            }
        )
    
    def _execute_auto_strategy(self, goal: str, inputs: Dict[str, Any], global_context: GlobalContext) -> AgentResult:
        """Auto-detect and execute the most appropriate strategy."""
        self.report_progress("Auto strategy", "Auto-detecting strategy from available data")
        
        # Default to task planning for most cases
        return self._execute_task_planning(goal, inputs, global_context)
    
    def _get_available_agent_capabilities(self) -> List[Dict[str, Any]]:
        """Get capabilities of all available agents."""
        capabilities = []
        
        for agent_name, agent_class in self.agent_registry.items():
            try:
                # Try to get capabilities from agent instance
                if hasattr(agent_class, 'get_capabilities'):
                    agent_instance = agent_class()
                    agent_capabilities = agent_instance.get_capabilities()
                    capabilities.append(agent_capabilities)
                else:
                    # Fallback to basic info
                    capabilities.append({
                        "name": agent_name,
                        "description": f"Agent: {agent_name}",
                        "primary_functions": ["General agent functionality"]
                    })
            except Exception as e:
                self.logger.warning(f"Could not get capabilities for {agent_name}: {e}")
        
        return capabilities
    
    def _convert_task_graph_to_steps(self, task_graph: TaskGraph) -> List[Any]:
        """Convert TaskGraph to workflow steps for estimation."""
        # This is a simplified conversion
        # In practice, this would create proper WorkflowStep objects
        steps = []
        
        for task_id, task in task_graph.nodes.items():
            step = type('WorkflowStep', (), {
                'step_id': task_id,
                'agent_name': task.assigned_agent,
                'dependencies': task.dependencies,
                'estimated_duration': None,  # Default estimation
                'parallel_group': None
            })()
            steps.append(step)
        
        return steps
    
    def _generate_task_graph_with_llm(self, planning_context: PlanningContext, intent_analysis: Dict[str, Any], global_context: GlobalContext) -> TaskGraph:
        """
        Generate task graph using original PlannerAgent LLM prompts.
        Preserved exactly as in the original agent.
        """
        try:
            # First use original intent analysis prompt
            context_summary = self._build_context_summary(global_context)
            intent_prompt = self._build_intent_analysis_prompt(planning_context.goal, context_summary)
            
            intent_response_str = self._llm_client.invoke(intent_prompt)
            import json
            intent_data = json.loads(intent_response_str)
            user_intent = intent_data.get("intent", planning_context.user_intent)
            quality_str = intent_data.get("planning_quality", planning_context.quality_level.value)
            
            # Get agent capabilities
            agent_capabilities = self._get_agent_capabilities()
            
            # Get quality instructions
            quality_instructions = self._get_quality_instructions(planning_context.quality_level)
            
            # Build planning prompt using original format
            planning_prompt = self._build_planning_prompt(
                intent=user_intent,
                context_summary=context_summary,
                quality=planning_context.quality_level,
                quality_instructions=quality_instructions,
                agent_capabilities=agent_capabilities
            )
            
            plan_response_str = self._llm_client.invoke(planning_prompt)
            plan_data = json.loads(plan_response_str)
            
            # Convert to TaskGraph
            task_graph = TaskGraph(**plan_data)
            
            if not task_graph.nodes:
                raise ValueError("LLM failed to generate any tasks for the plan.")
                
            return task_graph
            
        except Exception as e:
            logger.error(f"LLM task graph generation failed: {e}")
            # Fallback to rule-based generation
            return generate_task_graph(planning_context)
    
    def _build_intent_analysis_prompt(self, goal: str, context_summary: Dict) -> str:
        """Original PlannerAgent intent analysis prompt - preserved exactly."""
        return f"""
        You are an expert software development analyst. Analyze the user's request to understand their true intent and determine the appropriate planning quality.

        USER GOAL: "{goal}"
        
        PROJECT CONTEXT:
        {json.dumps(context_summary, indent=2)}
        
        Analyze the goal and context to determine:
        1.  **intent**: A concise, one-sentence summary of the user's core objective.
        2.  **planning_quality**: The appropriate quality level for the plan. Choose one of: "FAST", "DECENT", "PRODUCTION".
            - Use "FAST" for simple requests, prototypes, or quick fixes.
            - Use "DECENT" for standard feature development.
            - Use "PRODUCTION" for complex, critical, or enterprise-grade features.

        Your response must be a single JSON object with two keys: "intent" and "planning_quality".
        
        Example:
        {{
            "intent": "Refactor the existing user authentication module to use a more secure password hashing algorithm and add integration tests.",
            "planning_quality": "PRODUCTION"
        }}
        """
    
    def _build_planning_prompt(self, intent: str, context_summary: Dict[str, Any], quality: PlanningQuality, quality_instructions: Dict, agent_capabilities: List[Dict]) -> str:
        """Original PlannerAgent planning prompt - preserved exactly."""
        quality_instructions_str = "\n        ".join([f"- {inst}" for inst in quality_instructions["instructions"]])

        return f"""
        You are the PlannerAgent, a master strategist for an AI agent collective.
        Your mission is to decompose a user's intent into a detailed TaskGraph in JSON format.
        
        **User Intent:**
        {intent}

        **Planning Quality Level: {quality.value.upper()}**
        **Instructions for this quality level:**
        {quality_instructions_str}

        **Available Agents (Your Tools):**
        {json.dumps(agent_capabilities, indent=2)}

        **Current Workspace Context:**
        {json.dumps(context_summary, indent=2)}

        **General Instructions:**
        1.  Create a 'TaskNode' for each step with a unique `task_id`.
        2.  Assign the most appropriate agent from the list of available agents.
        3.  Define `dependencies` for each task using the `task_id` of prerequisite tasks.
        4.  Define `input_artifact_keys` and `output_artifact_keys` for data flow.
        5.  CRITICAL RULE: After any task that modifies code (e.g., CoderAgent), you MUST add a subsequent task to verify the change (e.g., TestRunnerAgent or CodeAnalysisAgent).
        6.  Your output MUST be a valid JSON object representing the TaskGraph.

        **JSON Output Format:**
        {{
            "nodes": {{
                "task_id_1": {{ "task_id": "...", "goal": "...", ... }},
                "task_id_2": {{ "task_id": "...", "goal": "...", ... }}
            }}
        }}

        Now, generate the TaskGraph JSON for the provided user intent.
        """
    
    def _build_context_summary(self, global_context: GlobalContext) -> Dict[str, Any]:
        """Build context summary for prompt."""
        context_summary = {
            "workspace_files": [],
            "recent_activities": [],
            "current_state": "active"
        }
        
        if global_context and global_context.workspace_path:
            try:
                # Get basic workspace info
                from pathlib import Path
                workspace = Path(global_context.workspace_path)
                if workspace.exists():
                    context_summary["workspace_files"] = [str(f.relative_to(workspace)) for f in workspace.rglob("*.py")][:20]  # First 20 files
            except Exception as e:
                logger.warning(f"Could not build context summary: {e}")
        
        return context_summary
    
    def _get_agent_capabilities(self) -> List[Dict]:
        """Get agent capabilities in the format expected by PlannerAgent."""
        capabilities = []
        
        # Default capabilities if registry not available
        default_agents = [
            {"name": "CoderAgent", "description": "Writes code based on specifications"},
            {"name": "TestRunnerAgent", "description": "Executes tests and validates code"},
            {"name": "AnalystAgent", "description": "Analyzes code quality and security"},
            {"name": "CreatorAgent", "description": "Creates new code components and tests"},
            {"name": "SurgeonAgent", "description": "Makes precise code modifications"},
            {"name": "ExecutorAgent", "description": "Executes system operations and validations"}
        ]
        
        if self.agent_registry:
            for agent_name, agent_class in self.agent_registry.items():
                try:
                    # Try to get description from agent instance
                    if hasattr(agent_class, 'description'):
                        description = agent_class.description
                    else:
                        description = f"Agent: {agent_name}"
                    capabilities.append({"name": agent_name, "description": description})
                except Exception as e:
                    logger.warning(f"Could not get capabilities for {agent_name}: {e}")
        else:
            capabilities = default_agents
            
        return capabilities
    
    def _get_quality_instructions(self, quality_level: PlanningQuality) -> Dict[str, List[str]]:
        """Get quality-specific instructions matching original PlannerAgent."""
        quality_configs = {
            PlanningQuality.FAST: {
                "description": "quick and minimal plans",
                "instructions": [
                    "Focus on speed over completeness",
                    "Use minimal validation steps", 
                    "Prefer simple, direct approaches",
                    "Skip comprehensive testing unless critical"
                ]
            },
            PlanningQuality.DECENT: {
                "description": "balanced plans with moderate detail",
                "instructions": [
                    "Balance speed and quality",
                    "Include basic validation and testing",
                    "Use established patterns and practices",
                    "Add reasonable error handling"
                ]
            },
            PlanningQuality.PRODUCTION: {
                "description": "comprehensive, enterprise-ready plans",
                "instructions": [
                    "Prioritize correctness and maintainability",
                    "Include comprehensive testing strategy",
                    "Add extensive validation and error handling",
                    "Consider scalability and performance",
                    "Include security considerations",
                    "Document all major decisions"
                ]
            }
        }
        
        return quality_configs.get(quality_level, quality_configs[PlanningQuality.DECENT])
    
    def _identify_parallel_groups(self, task_graph: TaskGraph) -> List[List[str]]:
        """Identify parallel execution groups in task graph."""
        # Simple parallel grouping based on dependencies
        groups = []
        remaining_tasks = set(task_graph.nodes.keys())
        
        while remaining_tasks:
            # Find tasks with no dependencies or all dependencies satisfied
            current_group = []
            for task_id in list(remaining_tasks):
                task = task_graph.nodes[task_id]
                if all(dep_id not in remaining_tasks for dep_id in task.dependencies):
                    current_group.append(task_id)
            
            if not current_group:
                # Handle circular dependencies or other issues
                current_group = [remaining_tasks.pop()]
            else:
                for task_id in current_group:
                    remaining_tasks.remove(task_id)
            
            groups.append(current_group)
        
        return groups
    
    def _generate_optimization_recommendations(self, optimized_graph: TaskGraph) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        # Analyze task dependencies for parallelization opportunities
        parallel_groups = self._identify_parallel_groups(optimized_graph)
        total_tasks = len(optimized_graph.nodes)
        
        if len(parallel_groups) < total_tasks:
            parallelizable_tasks = sum(len(group) for group in parallel_groups if len(group) > 1)
            if parallelizable_tasks > 0:
                recommendations.append(f"Execute {parallelizable_tasks} tasks in parallel to reduce overall execution time")
        
        # Agent-based recommendations
        agent_counts = {}
        for task in optimized_graph.nodes.values():
            agent_name = task.assigned_agent
            agent_counts[agent_name] = agent_counts.get(agent_name, 0) + 1
        
        # Resource allocation recommendations
        if agent_counts.get("AnalystAgent", 0) > 2:
            recommendations.append("Consider CPU resource allocation for analysis-intensive tasks")
        
        if agent_counts.get("CreatorAgent", 0) > 1:
            recommendations.append("Ensure adequate memory allocation for code generation tasks")
        
        if len(parallel_groups) > 1:
            recommendations.append("Workflow can benefit from parallel execution scheduling")
        
        if not recommendations:
            recommendations.append("Workflow is already well-optimized for current constraints")
        
        return recommendations