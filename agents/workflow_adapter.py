# agents/workflow_adapter.py
import json
import logging
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor, Future

# Foundational dependencies
from .base import BaseAgent
from core.context import GlobalContext
from core.models import AgentResponse, AgentResult, TaskNode

# Import types from master intelligence
from .master_intelligence import StrategicPlan, UserIntent, AgentStep, ApproachType, WorkflowType

# Get a logger instance for this module
logger = logging.getLogger(__name__)


# --- Execution Models ---

class ExecutionState(Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING" 
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    ADAPTING = "ADAPTING"


class AdaptationTrigger(Enum):
    AGENT_FAILURE = "AGENT_FAILURE"
    UNEXPECTED_OUTPUT = "UNEXPECTED_OUTPUT"
    BETTER_APPROACH_FOUND = "BETTER_APPROACH_FOUND"
    USER_INPUT_CHANGE = "USER_INPUT_CHANGE"
    ENVIRONMENT_CHANGE = "ENVIRONMENT_CHANGE"
    PERFORMANCE_OPTIMIZATION = "PERFORMANCE_OPTIMIZATION"
    RESOURCE_CONSTRAINT = "RESOURCE_CONSTRAINT"


@dataclass
class AgentExecution:
    """Tracks the execution of a single agent step"""
    step: AgentStep
    state: ExecutionState
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    result: Optional[AgentResult] = None
    error_message: Optional[str] = None
    retry_count: int = 0
    adaptation_applied: bool = False


@dataclass
class ParallelGroup:
    """Group of agents that can execute in parallel"""
    agents: List[str]
    dependencies_met: bool = False
    all_completed: bool = False
    any_failed: bool = False


@dataclass
class AdaptationDecision:
    """Decision made during real-time adaptation"""
    trigger: AdaptationTrigger
    decision: str
    actions: List[str]
    confidence: float
    reasoning: str
    timestamp: datetime


@dataclass
class WorkflowExecution:
    """Complete state of workflow execution"""
    plan: StrategicPlan
    current_step: int
    agent_executions: List[AgentExecution]
    parallel_groups: List[ParallelGroup]
    adaptations: List[AdaptationDecision]
    start_time: datetime
    state: ExecutionState
    context_snapshots: Dict[int, Dict[str, Any]]


@dataclass 
class ExecutionResult:
    """Final result of workflow execution"""
    success: bool
    message: str
    execution_path: List[AgentExecution]
    total_duration: timedelta
    adaptations_made: List[AdaptationDecision]
    lessons_learned: List[str]
    artifacts_generated: List[str]
    final_state: Dict[str, Any]


# --- LLM Integration ---

class LLMClient:
    """Placeholder for LLM integration"""
    def invoke(self, prompt: str) -> str:
        raise NotImplementedError("LLMClient.invoke must be implemented by a concrete class.")


# --- Agent Implementation ---

class WorkflowAdapterAgent(BaseAgent):
    """
    Real-time execution coordinator that manages dynamic workflows and handles all failure scenarios.
    
    This agent takes strategic plans from MasterIntelligenceAgent and executes them with
    full adaptability, failure recovery, and real-time learning.
    """

    def __init__(self, llm_client: Any = None, agent_registry: Dict[str, Any] = None):
        super().__init__(
            name="WorkflowAdapterAgent",
            description="Real-time execution coordinator that manages dynamic workflows with failure recovery and adaptation."
        )
        self.llm_client = llm_client or LLMClient()
        self.agent_registry = agent_registry or {}
        
        # Execution management
        self.max_retries = 3
        self.max_adaptations = 5
        self.parallel_executor = ThreadPoolExecutor(max_workers=4)
        
        # Real-time learning storage
        self.adaptation_patterns = {}
        self.performance_metrics = {}

    def required_inputs(self) -> List[str]:
        """Required inputs for WorkflowAdapterAgent execution."""
        return ["strategic_plan", "user_intent"]

    def optional_inputs(self) -> List[str]:
        """Optional inputs for WorkflowAdapterAgent execution."""
        return ["execution_constraints", "priority_overrides", "resource_limits"]

    def execute_v2(self, goal: str, inputs: Dict[str, Any], global_context: GlobalContext) -> AgentResult:
        """
        NEW INTERFACE: Execute strategic plan with full adaptability.
        """
        logger.info(f"WorkflowAdapterAgent executing strategic plan for: '{goal}'")
        
        # Validate inputs with graceful handling
        try:
            self.validate_inputs(inputs)
        except Exception as validation_error:
            return self.create_result(
                success=False,
                message=str(validation_error),
                error_details={"validation_error": str(validation_error)}
            )
        
        # Deserialize inputs
        strategic_plan = self._deserialize_plan(inputs["strategic_plan"])
        user_intent = self._deserialize_intent(inputs["user_intent"])
        
        try:
            self.report_progress("Initializing workflow execution", 
                               f"{strategic_plan.approach.value} approach with {len(strategic_plan.agent_sequence)} steps")
            
            # Execute the strategic plan
            execution_result = self.execute_strategic_plan(strategic_plan, user_intent, global_context)
            
            # Learn from the execution
            self.learn_from_execution(execution_result, strategic_plan, user_intent)
            
            return self.create_result(
                success=execution_result.success,
                message=execution_result.message,
                outputs={
                    "execution_result": self._serialize_execution_result(execution_result),
                    "lessons_learned": execution_result.lessons_learned,
                    "adaptations_made": len(execution_result.adaptations_made)
                }
            )
            
        except Exception as e:
            error_msg = f"WorkflowAdapterAgent execution failed: {e}"
            logger.error(error_msg, exc_info=True)
            return self.create_result(
                success=False,
                message=error_msg,
                error_details={"exception": str(e)}
            )

    def execute_strategic_plan(self, plan: StrategicPlan, intent: UserIntent, 
                             context: GlobalContext) -> ExecutionResult:
        """
        Execute strategic plan with full adaptability and failure recovery.
        """
        # Initialize execution tracking
        execution = WorkflowExecution(
            plan=plan,
            current_step=0,
            agent_executions=[],
            parallel_groups=[],
            adaptations=[],
            start_time=datetime.now(),
            state=ExecutionState.RUNNING,
            context_snapshots={}
        )
        
        self.report_thinking("Starting workflow execution with dynamic adaptation capabilities")
        
        try:
            if plan.workflow_type == WorkflowType.LINEAR:
                return self._execute_linear_workflow(execution, context)
            elif plan.workflow_type == WorkflowType.PARALLEL:
                return self._execute_parallel_workflow(execution, context)
            elif plan.workflow_type == WorkflowType.ITERATIVE:
                return self._execute_iterative_workflow(execution, context)
            elif plan.workflow_type == WorkflowType.ADAPTIVE:
                return self._execute_adaptive_workflow(execution, context)
            elif plan.workflow_type == WorkflowType.BRANCHING:
                return self._execute_branching_workflow(execution, context)
            else:
                # Fallback to linear
                return self._execute_linear_workflow(execution, context)
                
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            execution.state = ExecutionState.FAILED
            
            return ExecutionResult(
                success=False,
                message=f"Workflow execution failed: {e}",
                execution_path=execution.agent_executions,
                total_duration=datetime.now() - execution.start_time,
                adaptations_made=execution.adaptations,
                lessons_learned=[f"Workflow failed due to: {e}"],
                artifacts_generated=[],
                final_state={"error": str(e)}
            )

    def _execute_linear_workflow(self, execution: WorkflowExecution, context: GlobalContext) -> ExecutionResult:
        """Execute agents in linear sequence with adaptation support."""
        self.report_thinking("Executing linear workflow with real-time monitoring and adaptation")
        
        for i, step in enumerate(execution.plan.agent_sequence):
            execution.current_step = i
            
            # Create agent execution tracker
            agent_execution = AgentExecution(
                step=step,
                state=ExecutionState.PENDING,
                start_time=datetime.now()
            )
            execution.agent_executions.append(agent_execution)
            
            # Execute agent with retry and adaptation
            success = self._execute_agent_with_adaptation(agent_execution, execution, context)
            
            if not success:
                # Try adaptation before failing
                adaptation_result = self._attempt_adaptation(
                    trigger=AdaptationTrigger.AGENT_FAILURE,
                    execution=execution,
                    context=context,
                    failed_step=agent_execution
                )
                
                if not adaptation_result:
                    # Workflow failed - return partial results
                    return self._create_failure_result(execution, f"Agent {step.agent_name} failed")
            
            # Take context snapshot after each step
            execution.context_snapshots[i] = self._capture_context_snapshot(context)
            
            self.report_progress(f"Step {i+1}/{len(execution.plan.agent_sequence)} complete",
                               f"{step.agent_name} completed successfully")

        # All steps completed successfully
        execution.state = ExecutionState.COMPLETED
        return self._create_success_result(execution)

    def _execute_parallel_workflow(self, execution: WorkflowExecution, context: GlobalContext) -> ExecutionResult:
        """Execute agents in parallel groups with dependency management."""
        self.report_thinking("Executing parallel workflow with dependency coordination")
        
        # Group agents by parallel opportunities
        parallel_groups = self._create_parallel_groups(execution.plan)
        execution.parallel_groups = parallel_groups
        
        for group_idx, group in enumerate(parallel_groups):
            self.report_progress(f"Executing parallel group {group_idx + 1}", 
                               f"Running {len(group.agents)} agents simultaneously")
            
            # Execute group in parallel
            group_success = self._execute_parallel_group(group, execution, context)
            
            if not group_success:
                return self._create_failure_result(execution, f"Parallel group {group_idx + 1} failed")
        
        execution.state = ExecutionState.COMPLETED
        return self._create_success_result(execution)

    def _execute_iterative_workflow(self, execution: WorkflowExecution, context: GlobalContext) -> ExecutionResult:
        """Execute workflow with iterations until success criteria met."""
        self.report_thinking("Executing iterative workflow with success criteria validation")
        
        max_iterations = 5
        iteration = 0
        
        while iteration < max_iterations:
            self.report_progress(f"Iteration {iteration + 1}/{max_iterations}", "Running workflow cycle")
            
            # Execute one iteration
            iteration_success = self._execute_single_iteration(execution, context, iteration)
            
            if iteration_success:
                # Check if success criteria are met
                criteria_met = self._validate_success_criteria(execution.plan.success_criteria, context)
                if criteria_met:
                    execution.state = ExecutionState.COMPLETED
                    return self._create_success_result(execution)
            
            iteration += 1
        
        # Max iterations reached without success
        return self._create_failure_result(execution, f"Max iterations ({max_iterations}) reached")

    def _execute_adaptive_workflow(self, execution: WorkflowExecution, context: GlobalContext) -> ExecutionResult:
        """Execute workflow with continuous adaptation based on results."""
        self.report_thinking("Executing adaptive workflow with continuous learning and adjustment")
        
        step_idx = 0
        adaptation_count = 0
        
        while step_idx < len(execution.plan.agent_sequence) and adaptation_count < self.max_adaptations:
            step = execution.plan.agent_sequence[step_idx]
            
            # Execute current step
            agent_execution = AgentExecution(step=step, state=ExecutionState.PENDING, start_time=datetime.now())
            execution.agent_executions.append(agent_execution)
            
            success = self._execute_agent_with_adaptation(agent_execution, execution, context)
            
            if success:
                # Check if we should adapt the remaining workflow
                should_adapt = self._should_adapt_workflow(agent_execution, execution, context)
                
                if should_adapt:
                    adaptation_decision = self._adapt_remaining_workflow(execution, context, step_idx)
                    execution.adaptations.append(adaptation_decision)
                    adaptation_count += 1
                    
                    # Workflow might have changed - continue with adapted plan
                    continue
                
                step_idx += 1
            else:
                # Failure - try adaptation
                adaptation_result = self._attempt_adaptation(
                    trigger=AdaptationTrigger.AGENT_FAILURE,
                    execution=execution,
                    context=context,
                    failed_step=agent_execution
                )
                
                if not adaptation_result:
                    return self._create_failure_result(execution, f"Adaptive workflow failed at step {step_idx}")
                
                adaptation_count += 1

        execution.state = ExecutionState.COMPLETED
        return self._create_success_result(execution)

    def _execute_branching_workflow(self, execution: WorkflowExecution, context: GlobalContext) -> ExecutionResult:
        """Execute workflow with decision points and multiple paths."""
        self.report_thinking("Executing branching workflow with dynamic path selection")
        
        for i, step in enumerate(execution.plan.agent_sequence):
            # Check if this is a decision point
            if self._is_decision_point(step):
                # Use LLM to decide which branch to take
                branch_decision = self._make_branch_decision(step, execution, context)
                self.report_progress("Branch decision made", f"Taking path: {branch_decision}")
                
                # Execute chosen branch
                branch_success = self._execute_branch(branch_decision, execution, context)
                if not branch_success:
                    return self._create_failure_result(execution, f"Branch execution failed: {branch_decision}")
            else:
                # Regular step execution
                agent_execution = AgentExecution(step=step, state=ExecutionState.PENDING, start_time=datetime.now())
                execution.agent_executions.append(agent_execution)
                
                success = self._execute_agent_with_adaptation(agent_execution, execution, context)
                if not success:
                    return self._create_failure_result(execution, f"Step {i} failed")
        
        execution.state = ExecutionState.COMPLETED
        return self._create_success_result(execution)

    def _execute_agent_with_adaptation(self, agent_execution: AgentExecution, 
                                     workflow_execution: WorkflowExecution, 
                                     context: GlobalContext) -> bool:
        """
        Execute a single agent with retry logic and adaptation support.
        """
        step = agent_execution.step
        agent_name = step.agent_name
        
        # Check if agent exists
        if agent_name not in self.agent_registry:
            logger.error(f"Agent {agent_name} not found in registry")
            agent_execution.state = ExecutionState.FAILED
            agent_execution.error_message = f"Agent {agent_name} not available"
            return False
        
        agent = self.agent_registry[agent_name]
        agent_execution.state = ExecutionState.RUNNING
        
        self.report_progress(f"Executing {agent_name}", f"Goal: {step.goal}")
        
        for retry in range(self.max_retries):
            try:
                # Prepare inputs for the agent
                agent_inputs = self._prepare_agent_inputs(step, context, workflow_execution)
                
                # Execute agent
                if hasattr(agent, 'execute_v2'):
                    result = agent.execute_v2(step.goal, agent_inputs, context)
                else:
                    # Fallback to legacy execute method
                    task = TaskNode(goal=step.goal, assigned_agent=agent_name)
                    legacy_result = agent.execute(step.goal, context, task)
                    result = self._convert_legacy_result(legacy_result)
                
                agent_execution.result = result
                agent_execution.end_time = datetime.now()
                
                if result.success:
                    agent_execution.state = ExecutionState.COMPLETED
                    self.report_progress(f"{agent_name} completed", result.message[:60] + "...")
                    return True
                else:
                    agent_execution.retry_count = retry + 1
                    logger.warning(f"Agent {agent_name} failed (attempt {retry + 1}): {result.message}")
                    
                    if retry < self.max_retries - 1:
                        self.report_progress(f"{agent_name} retry {retry + 1}", "Attempting retry with same inputs")
                        continue
                    
            except Exception as e:
                logger.error(f"Agent {agent_name} execution error (attempt {retry + 1}): {e}")
                agent_execution.error_message = str(e)
                agent_execution.retry_count = retry + 1
                
                if retry < self.max_retries - 1:
                    continue
        
        # All retries exhausted
        agent_execution.state = ExecutionState.FAILED
        agent_execution.end_time = datetime.now()
        return False

    def _attempt_adaptation(self, trigger: AdaptationTrigger, execution: WorkflowExecution,
                          context: GlobalContext, failed_step: Optional[AgentExecution] = None) -> bool:
        """
        Attempt to adapt workflow when failures or better approaches are detected.
        """
        self.report_thinking(f"Attempting workflow adaptation due to: {trigger.value}")
        
        try:
            # Use LLM to decide on adaptation
            adaptation_prompt = f"""
            You are a workflow adaptation expert. A workflow execution needs adaptation.

            SITUATION:
            Trigger: {trigger.value}
            Current Plan: {execution.plan.approach.value} {execution.plan.workflow_type.value}
            Failed Step: {failed_step.step.agent_name if failed_step else 'None'} - {failed_step.error_message if failed_step else 'N/A'}
            Completed Steps: {len([e for e in execution.agent_executions if e.state == ExecutionState.COMPLETED])}
            Remaining Steps: {len(execution.plan.agent_sequence) - execution.current_step - 1}
            
            CONTEXT:
            Available Agents: {list(self.agent_registry.keys())}
            Execution History: {[e.step.agent_name + ':' + e.state.value for e in execution.agent_executions]}
            
            Decide on adaptation strategy. Return JSON:
            {{
                "should_adapt": true/false,
                "adaptation_type": "retry_with_different_agent|skip_step|insert_new_step|change_approach|abort_workflow",
                "actions": ["specific action 1", "specific action 2"],
                "reasoning": "Why this adaptation makes sense",
                "confidence": 0.85
            }}
            
            Focus on practical, executable adaptations.
            """
            
            response = self.llm_client.invoke(adaptation_prompt)
            adaptation_data = json.loads(response)
            
            if not adaptation_data.get("should_adapt", False):
                return False
            
            # Apply the adaptation
            adaptation_decision = AdaptationDecision(
                trigger=trigger,
                decision=adaptation_data["adaptation_type"],
                actions=adaptation_data["actions"],
                confidence=adaptation_data["confidence"],
                reasoning=adaptation_data["reasoning"],
                timestamp=datetime.now()
            )
            
            execution.adaptations.append(adaptation_decision)
            
            # Execute adaptation actions
            success = self._apply_adaptation_actions(adaptation_decision, execution, context, failed_step)
            
            self.report_progress("Workflow adapted", f"{adaptation_decision.decision}: {adaptation_decision.reasoning[:50]}...")
            
            return success
            
        except Exception as e:
            logger.error(f"Adaptation failed: {e}")
            return False

    def _apply_adaptation_actions(self, decision: AdaptationDecision, execution: WorkflowExecution,
                                context: GlobalContext, failed_step: Optional[AgentExecution]) -> bool:
        """Apply specific adaptation actions to the workflow."""
        
        if decision.decision == "retry_with_different_agent":
            # Find alternative agent and retry
            if failed_step:
                alternative_agent = self._find_alternative_agent(failed_step.step.agent_name)
                if alternative_agent:
                    failed_step.step.agent_name = alternative_agent
                    failed_step.state = ExecutionState.PENDING
                    failed_step.retry_count = 0
                    failed_step.error_message = None
                    return self._execute_agent_with_adaptation(failed_step, execution, context)
        
        elif decision.decision == "skip_step":
            # Mark step as completed and continue
            if failed_step:
                failed_step.state = ExecutionState.COMPLETED
                failed_step.adaptation_applied = True
                return True
        
        elif decision.decision == "insert_new_step":
            # Insert new step to handle the problem
            new_step = self._create_recovery_step(failed_step, execution, context)
            if new_step:
                execution.plan.agent_sequence.insert(execution.current_step + 1, new_step)
                return True
        
        elif decision.decision == "change_approach":
            # Change workflow approach (fast → thorough, etc.)
            return self._change_workflow_approach(execution, decision.actions)
        
        return False

    def _prepare_agent_inputs(self, step: AgentStep, context: GlobalContext, 
                            execution: WorkflowExecution) -> Dict[str, Any]:
        """Prepare inputs for agent execution based on step requirements and context."""
        inputs = step.inputs.copy()
        
        # Add outputs from previous agents as inputs
        for dep in step.dependencies:
            for prev_execution in execution.agent_executions:
                if (prev_execution.step.agent_name == dep and 
                    prev_execution.result and 
                    prev_execution.result.success):
                    # Merge outputs from dependent agent
                    if hasattr(prev_execution.result, 'outputs'):
                        inputs.update(prev_execution.result.outputs)
        
        return inputs

    def _validate_success_criteria(self, criteria: List, context: GlobalContext) -> bool:
        """Validate if success criteria are met."""
        for criterion in criteria:
            if criterion.required:
                # Simple validation - in production this would be more sophisticated
                if "test" in criterion.validation_method.lower():
                    # Check if tests are passing by looking for test results
                    test_artifacts = [key for key in context.list_artifacts() if "test" in key.lower()]
                    if not test_artifacts:
                        return False
        return True

    def _create_success_result(self, execution: WorkflowExecution) -> ExecutionResult:
        """Create successful execution result."""
        total_duration = datetime.now() - execution.start_time
        
        lessons_learned = []
        lessons_learned.append(f"Successfully executed {execution.plan.approach.value} workflow")
        lessons_learned.extend([a.reasoning for a in execution.adaptations])
        
        artifacts = []
        for agent_exec in execution.agent_executions:
            if agent_exec.result and hasattr(agent_exec.result, 'outputs'):
                artifacts.extend(agent_exec.result.outputs.get('artifacts_generated', []))
        
        return ExecutionResult(
            success=True,
            message=f"Workflow completed successfully in {total_duration.total_seconds():.1f}s",
            execution_path=execution.agent_executions,
            total_duration=total_duration,
            adaptations_made=execution.adaptations,
            lessons_learned=lessons_learned,
            artifacts_generated=artifacts,
            final_state={"completed_steps": len(execution.agent_executions)}
        )

    def _create_failure_result(self, execution: WorkflowExecution, reason: str) -> ExecutionResult:
        """Create failed execution result."""
        total_duration = datetime.now() - execution.start_time
        
        lessons_learned = [f"Workflow failed: {reason}"]
        lessons_learned.extend([a.reasoning for a in execution.adaptations])
        
        return ExecutionResult(
            success=False,
            message=f"Workflow failed: {reason}",
            execution_path=execution.agent_executions,
            total_duration=total_duration,
            adaptations_made=execution.adaptations,
            lessons_learned=lessons_learned,
            artifacts_generated=[],
            final_state={"error": reason}
        )

    # Helper methods for workflow execution
    def _create_parallel_groups(self, plan: StrategicPlan) -> List[ParallelGroup]:
        """Create parallel groups from plan opportunities."""
        groups = []
        for opportunity in plan.parallel_opportunities:
            groups.append(ParallelGroup(agents=opportunity))
        return groups

    def _find_alternative_agent(self, failed_agent: str) -> Optional[str]:
        """Find alternative agent that might handle the same task."""
        # Simple heuristic - in production this would be more sophisticated
        alternatives = {
            "CoderAgent": "ScriptExecutorAgent",
            "ScriptExecutorAgent": "CoderAgent",
            "TestRunnerAgent": "TestGeneratorAgent"
        }
        return alternatives.get(failed_agent)

    def _deserialize_plan(self, plan_data: Dict[str, Any]) -> StrategicPlan:
        """Deserialize strategic plan from dictionary."""
        # Import here to avoid circular imports
        from .master_intelligence import StrategicPlan, AgentStep, ApproachType, WorkflowType, SuccessCriterion, LearningGoal
        
        return StrategicPlan(
            approach=ApproachType(plan_data["approach"]),
            workflow_type=WorkflowType(plan_data["workflow_type"]),
            agent_sequence=[
                AgentStep(
                    agent_name=step["agent_name"],
                    goal=step["goal"],
                    inputs=step["inputs"],
                    expected_outputs=step["expected_outputs"],
                    dependencies=step["dependencies"],
                    optional=step.get("optional", False),
                    confidence_threshold=step.get("confidence_threshold", 0.7)
                ) for step in plan_data["agent_sequence"]
            ],
            parallel_opportunities=plan_data.get("parallel_opportunities", []),
            success_criteria=[
                SuccessCriterion(
                    criterion=sc["criterion"],
                    validation_method=sc["validation_method"],
                    required=sc.get("required", True)
                ) for sc in plan_data.get("success_criteria", [])
            ],
            learning_objectives=[
                LearningGoal(
                    goal=lg["goal"],
                    success_metric=lg["success_metric"],
                    pattern_to_capture=lg["pattern_to_capture"]
                ) for lg in plan_data.get("learning_objectives", [])
            ]
        )

    def _deserialize_intent(self, intent_data: Dict[str, Any]) -> UserIntent:
        """Deserialize user intent from dictionary.""" 
        from .master_intelligence import UserIntent, IntentType, Specificity, Urgency, Scope, Domain, UserExperience
        
        return UserIntent(
            intent_type=IntentType(intent_data["intent_type"]),
            specificity=Specificity(intent_data["specificity"]),
            urgency=Urgency(intent_data["urgency"]),
            scope=Scope(intent_data["scope"]),
            domain=Domain(intent_data["domain"]),
            user_experience_level=UserExperience(intent_data["user_experience_level"]),
            requires_clarification=intent_data["requires_clarification"],
            extracted_entities=intent_data["extracted_entities"],
            confidence_score=intent_data["confidence_score"],
            original_input=intent_data["original_input"],
            processing_timestamp=datetime.fromisoformat(intent_data["processing_timestamp"])
        )

    def learn_from_execution(self, result: ExecutionResult, plan: StrategicPlan, intent: UserIntent):
        """Learn from execution outcomes to improve future adaptations."""
        pattern_key = f"{plan.approach.value}_{plan.workflow_type.value}_{intent.intent_type.value}"
        
        if pattern_key not in self.adaptation_patterns:
            self.adaptation_patterns[pattern_key] = []
        
        self.adaptation_patterns[pattern_key].append({
            "success": result.success,
            "duration": result.total_duration.total_seconds(),
            "adaptations_needed": len(result.adaptations_made),
            "timestamp": datetime.now().isoformat()
        })
        
        logger.info(f"Learned execution pattern: {pattern_key} → {result.success}")

    # Placeholder methods for complex operations
    def _execute_parallel_group(self, group: ParallelGroup, execution: WorkflowExecution, context: GlobalContext) -> bool:
        """Execute agents in parallel group - placeholder for full implementation."""
        # For now, execute sequentially
        for agent_name in group.agents:
            # Find step for this agent
            step = next((s for s in execution.plan.agent_sequence if s.agent_name == agent_name), None)
            if step:
                agent_execution = AgentExecution(step=step, state=ExecutionState.PENDING, start_time=datetime.now())
                execution.agent_executions.append(agent_execution)
                success = self._execute_agent_with_adaptation(agent_execution, execution, context)
                if not success:
                    return False
        return True

    def _execute_single_iteration(self, execution: WorkflowExecution, context: GlobalContext, iteration: int) -> bool:
        """Execute single iteration of iterative workflow."""
        # Simple implementation - execute all steps once
        for step in execution.plan.agent_sequence:
            agent_execution = AgentExecution(step=step, state=ExecutionState.PENDING, start_time=datetime.now())
            execution.agent_executions.append(agent_execution)
            success = self._execute_agent_with_adaptation(agent_execution, execution, context)
            if not success:
                return False
        return True

    def _convert_legacy_result(self, legacy_result: AgentResponse) -> AgentResult:
        """Convert legacy AgentResponse to new AgentResult format."""
        return AgentResult(
            success=legacy_result.success,
            message=legacy_result.message,
            outputs={"artifacts_generated": legacy_result.artifacts_generated}
        )

    def _serialize_execution_result(self, result: ExecutionResult) -> Dict[str, Any]:
        """Serialize execution result for output."""
        return {
            "success": result.success,
            "message": result.message,
            "total_duration_seconds": result.total_duration.total_seconds(),
            "adaptations_made": len(result.adaptations_made),
            "lessons_learned": result.lessons_learned,
            "artifacts_generated": result.artifacts_generated
        }

    # Placeholder methods that would be implemented for full functionality
    def _should_adapt_workflow(self, agent_execution: AgentExecution, execution: WorkflowExecution, context: GlobalContext) -> bool:
        return False  # Placeholder
    
    def _adapt_remaining_workflow(self, execution: WorkflowExecution, context: GlobalContext, step_idx: int) -> AdaptationDecision:
        return AdaptationDecision(AdaptationTrigger.PERFORMANCE_OPTIMIZATION, "No adaptation", [], 0.5, "Placeholder", datetime.now())
    
    def _is_decision_point(self, step: AgentStep) -> bool:
        return False  # Placeholder
    
    def _make_branch_decision(self, step: AgentStep, execution: WorkflowExecution, context: GlobalContext) -> str:
        return "default_branch"  # Placeholder
    
    def _execute_branch(self, branch: str, execution: WorkflowExecution, context: GlobalContext) -> bool:
        return True  # Placeholder
    
    def _create_recovery_step(self, failed_step: AgentExecution, execution: WorkflowExecution, context: GlobalContext) -> Optional[AgentStep]:
        return None  # Placeholder
    
    def _change_workflow_approach(self, execution: WorkflowExecution, actions: List[str]) -> bool:
        return True  # Placeholder
    
    def _capture_context_snapshot(self, context: GlobalContext) -> Dict[str, Any]:
        return {"timestamp": datetime.now().isoformat()}  # Placeholder

    # Legacy execute method for compatibility
    def execute(self, goal: str, context: GlobalContext, current_task: TaskNode) -> AgentResponse:
        """Legacy execute method for backward compatibility."""
        # This shouldn't typically be called directly
        return AgentResponse(
            success=False,
            message="WorkflowAdapterAgent requires strategic plan input via execute_v2",
            artifacts_generated=[]
        )