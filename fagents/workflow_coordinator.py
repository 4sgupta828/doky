# fagents/workflow_coordinator.py
"""
Intelligent Workflow Coordinator for Foundation Agents.

This module provides the main interface for executing multi-agent workflows
with intelligent routing, completion validation, and comprehensive reporting.
It ensures directional progress toward user goals with minimal agent hops.
"""

import logging
from typing import Dict, Any, List, Optional

from core.context import GlobalContext
from core.models import AgentResult
from .inter_agent_router import (
    InterAgentRouter, WorkflowContext, WorkflowStatus
)

logger = logging.getLogger(__name__)


class WorkflowCoordinator:
    """
    Main coordinator for intelligent multi-agent workflows.
    
    This coordinator:
    1. Executes multi-agent workflows with intelligent routing
    2. Ensures completion validation with AnalystAgent
    3. Provides comprehensive workflow summaries
    4. Maintains execution history and progress tracking
    """
    
    def __init__(self, llm_client: Any = None, agent_registry: Dict[str, Any] = None):
        self.llm_client = llm_client
        self.router = InterAgentRouter(llm_client, agent_registry)
        self.active_workflows: Dict[str, WorkflowContext] = {}
    
    def execute_goal(self, user_goal: str, inputs: Dict[str, Any] = None, 
                    global_context: GlobalContext = None, max_hops: int = 10) -> AgentResult:
        """
        Execute a complete workflow to achieve the user's goal.
        
        This is the main entry point for goal execution with intelligent routing.
        
        Args:
            user_goal: The user's high-level goal
            inputs: Initial inputs (optional)
            global_context: Global execution context
            max_hops: Maximum number of agent invocations
            
        Returns:
            AgentResult with workflow execution summary and final outputs
        """
        
        # Prepare inputs
        if inputs is None:
            inputs = {}
        
        # Create global context if not provided
        if global_context is None:
            from pathlib import Path
            global_context = GlobalContext(workspace_path=Path.cwd())
        
        logger.info(f"Starting workflow execution for goal: {user_goal}")
        
        try:
            # Execute multi-agent workflow
            workflow_context = self.router.execute_workflow(
                user_goal=user_goal,
                initial_inputs=inputs,
                global_context=global_context,
                max_hops=max_hops
            )
            
            # Store workflow for tracking
            self.active_workflows[workflow_context.workflow_id] = workflow_context
            
            # Ensure completion validation
            if workflow_context.status == WorkflowStatus.COMPLETED and not workflow_context.completion_validated:
                workflow_context = self._ensure_completion_validation(workflow_context, global_context)
            
            # Generate final result
            return self._create_workflow_result(workflow_context)
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}", exc_info=True)
            return AgentResult(
                success=False,
                message=f"Workflow execution failed: {str(e)}",
                outputs={"error": str(e)}
            )
    
    def _ensure_completion_validation(self, workflow_context: WorkflowContext, 
                                    global_context: GlobalContext) -> WorkflowContext:
        """
        Ensure the workflow completion is validated by AnalystAgent.
        
        If the last agent wasn't AnalystAgent, invoke it for final validation.
        """
        
        # Check if the last execution was AnalystAgent doing validation
        last_execution = workflow_context.execution_history[-1] if workflow_context.execution_history else None
        
        needs_validation = True
        if last_execution and last_execution.agent_name == "AnalystAgent":
            # Check if the goal was about validation/completion
            goal_lower = last_execution.goal.lower()
            if any(word in goal_lower for word in ["validate", "completion", "final", "summary", "complete"]):
                needs_validation = False
        
        if needs_validation and workflow_context.current_hop < workflow_context.max_hops:
            logger.info("Ensuring completion validation with AnalystAgent")
            
            # Prepare validation inputs
            validation_inputs = {
                "workflow_summary": self._create_workflow_summary(workflow_context),
                "accumulated_outputs": workflow_context.accumulated_outputs,
                "user_goal": workflow_context.user_goal,
                "execution_history": [
                    {
                        "agent": exec.agent_name,
                        "goal": exec.goal,
                        "success": exec.result.success,
                        "message": exec.result.message
                    }
                    for exec in workflow_context.execution_history
                ]
            }
            
            validation_goal = f"Validate completion of user goal: {workflow_context.user_goal}"
            
            # Execute AnalystAgent for validation
            validation_execution = self.router._execute_agent(
                "AnalystAgent", validation_goal, validation_inputs, global_context, workflow_context
            )
            
            if validation_execution:
                workflow_context.execution_history.append(validation_execution)
                workflow_context.current_hop += 1
                workflow_context.completion_validated = True
                
                # Update accumulated outputs with validation results
                if validation_execution.result.success and validation_execution.result.outputs:
                    workflow_context.accumulated_outputs.update(validation_execution.result.outputs)
                
                logger.info("Completion validation completed")
            else:
                logger.warning("Completion validation failed")
        
        return workflow_context
    
    def _create_workflow_result(self, workflow_context: WorkflowContext) -> AgentResult:
        """Create the final AgentResult from the workflow context."""
        
        # Determine overall success
        success = (workflow_context.status == WorkflowStatus.COMPLETED and 
                  workflow_context.completion_validated)
        
        # Create comprehensive message
        message = self._create_final_message(workflow_context)
        
        # Prepare final outputs
        outputs = {
            "workflow_id": workflow_context.workflow_id,
            "status": workflow_context.status.value,
            "total_hops": workflow_context.current_hop,
            "agents_used": [exec.agent_name for exec in workflow_context.execution_history],
            "completion_validated": workflow_context.completion_validated,
            "execution_summary": self._create_execution_summary(workflow_context),
            "final_outputs": workflow_context.accumulated_outputs,
            "workflow_summary": self._create_workflow_summary(workflow_context)
        }
        
        return AgentResult(
            success=success,
            message=message,
            outputs=outputs
        )
    
    def _create_final_message(self, workflow_context: WorkflowContext) -> str:
        """Create the final message describing workflow results."""
        
        agents_used = [exec.agent_name for exec in workflow_context.execution_history]
        unique_agents = list(dict.fromkeys(agents_used))  # Preserve order, remove duplicates
        
        if workflow_context.status == WorkflowStatus.COMPLETED:
            if workflow_context.completion_validated:
                return (f"✅ Goal completed successfully in {workflow_context.current_hop} hops using "
                       f"{len(unique_agents)} agents: {' → '.join(unique_agents)}")
            else:
                return (f"⚠️  Goal completed but validation pending in {workflow_context.current_hop} hops using "
                       f"{len(unique_agents)} agents: {' → '.join(unique_agents)}")
        
        elif workflow_context.status == WorkflowStatus.FAILED:
            return (f"❌ Workflow failed after {workflow_context.current_hop} hops using "
                   f"{len(unique_agents)} agents: {' → '.join(unique_agents)}")
        
        elif workflow_context.status == WorkflowStatus.MAX_HOPS_EXCEEDED:
            return (f"⏰ Workflow exceeded maximum hops ({workflow_context.max_hops}) using "
                   f"{len(unique_agents)} agents: {' → '.join(unique_agents)}")
        
        else:
            return f"🔄 Workflow in progress: {workflow_context.current_hop} hops completed"
    
    def _create_execution_summary(self, workflow_context: WorkflowContext) -> List[Dict[str, Any]]:
        """Create a detailed execution summary."""
        
        summary = []
        for i, execution in enumerate(workflow_context.execution_history, 1):
            summary.append({
                "step": i,
                "agent": execution.agent_name,
                "goal": execution.goal,
                "success": execution.result.success,
                "message": execution.result.message,
                "outputs_count": len(execution.result.outputs) if execution.result.outputs else 0,
                "key_outputs": list(execution.result.outputs.keys())[:5] if execution.result.outputs else [],
                "timestamp": execution.timestamp
            })
        
        return summary
    
    def _create_workflow_summary(self, workflow_context: WorkflowContext) -> Dict[str, Any]:
        """Create a comprehensive workflow summary."""
        
        # Calculate success rate
        successful_executions = sum(1 for exec in workflow_context.execution_history if exec.result.success)
        total_executions = len(workflow_context.execution_history)
        success_rate = (successful_executions / total_executions * 100) if total_executions > 0 else 0
        
        # Get unique agents used
        agents_used = [exec.agent_name for exec in workflow_context.execution_history]
        unique_agents = list(dict.fromkeys(agents_used))
        agent_usage = {agent: agents_used.count(agent) for agent in unique_agents}
        
        return {
            "user_goal": workflow_context.user_goal,
            "workflow_id": workflow_context.workflow_id,
            "status": workflow_context.status.value,
            "total_hops": workflow_context.current_hop,
            "max_hops": workflow_context.max_hops,
            "success_rate": f"{success_rate:.1f}%",
            "completion_validated": workflow_context.completion_validated,
            "agents_used": unique_agents,
            "agent_usage_count": agent_usage,
            "execution_path": " → ".join(agents_used),
            "total_outputs": len(workflow_context.accumulated_outputs),
            "key_achievements": self._extract_key_achievements(workflow_context)
        }
    
    def _extract_key_achievements(self, workflow_context: WorkflowContext) -> List[str]:
        """Extract key achievements from the workflow execution."""
        
        achievements = []
        
        for execution in workflow_context.execution_history:
            if execution.result.success and execution.result.outputs:
                # Extract meaningful achievements based on agent type and outputs
                agent = execution.agent_name
                outputs = execution.result.outputs
                
                if agent == "AnalystAgent":
                    if "analysis_type" in outputs:
                        analysis_type = outputs["analysis_type"].replace("_", " ").title()
                        achievements.append(f"Completed {analysis_type}")
                
                elif agent == "CreatorAgent":
                    if "generated_code" in outputs:
                        achievements.append("Generated code implementation")
                    if "generated_tests" in outputs:
                        achievements.append("Created test suite")
                    if "generated_documentation" in outputs:
                        achievements.append("Created documentation")
                
                elif agent == "ExecutorAgent":
                    if outputs.get("test_files_executed", 0) > 0:
                        count = outputs["test_files_executed"]
                        achievements.append(f"Executed {count} test files")
                    if "validation_results" in outputs:
                        achievements.append("Validated code execution")
                
                elif agent == "SurgeonAgent":
                    if outputs.get("modification_applied"):
                        achievements.append("Applied code modifications")
                    if outputs.get("requirements_updated"):
                        achievements.append("Updated project requirements")
                
                elif agent == "DebuggingAgent":
                    if outputs.get("final_result") == "success":
                        achievements.append("Successfully debugged issue")
        
        # Add completion achievement
        if workflow_context.status == WorkflowStatus.COMPLETED:
            achievements.append(f"Achieved goal: {workflow_context.user_goal}")
        
        return achievements[:10]  # Limit to top 10 achievements
    
    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get the current status of a workflow."""
        
        workflow = self.active_workflows.get(workflow_id)
        if not workflow:
            return None
        
        return {
            "workflow_id": workflow_id,
            "status": workflow.status.value,
            "current_hop": workflow.current_hop,
            "max_hops": workflow.max_hops,
            "completion_validated": workflow.completion_validated,
            "agents_used": [exec.agent_name for exec in workflow.execution_history]
        }
    
    def list_active_workflows(self) -> List[Dict[str, Any]]:
        """List all active workflows."""
        
        return [
            {
                "workflow_id": wf_id,
                "user_goal": workflow.user_goal,
                "status": workflow.status.value,
                "current_hop": workflow.current_hop,
                "agents_used": len(set(exec.agent_name for exec in workflow.execution_history))
            }
            for wf_id, workflow in self.active_workflows.items()
        ]


# Convenience function for simple goal execution
def execute_user_goal(user_goal: str, inputs: Dict[str, Any] = None,
                     llm_client: Any = None, max_hops: int = 10) -> AgentResult:
    """
    Execute a user goal with intelligent multi-agent coordination.
    
    This is the simplest way to achieve a goal using the foundational agents.
    
    Args:
        user_goal: What the user wants to achieve
        inputs: Optional initial inputs
        llm_client: LLM client for intelligent routing
        max_hops: Maximum agent invocations allowed
        
    Returns:
        AgentResult with comprehensive workflow results
    """
    coordinator = WorkflowCoordinator(llm_client=llm_client)
    return coordinator.execute_goal(user_goal, inputs, max_hops=max_hops)