# tools/planning/workflow_orchestration_tools.py
import logging
from typing import Dict, Any, List, Optional, Callable
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, Future

# Core dependencies
from core.models import TaskGraph, TaskNode, AgentResult

# Get a logger instance for this module
logger = logging.getLogger(__name__)


class OrchestrationMode(Enum):
    """Different orchestration modes for workflow execution."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    ADAPTIVE = "adaptive"
    BATCH = "batch"


class ExecutionState(Enum):
    """Execution states for workflow steps."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


@dataclass
class OrchestrationContext:
    """Context for workflow orchestration."""
    workflow_id: str
    orchestration_mode: OrchestrationMode
    max_parallel_tasks: int = 3
    retry_attempts: int = 2
    timeout_minutes: int = 60
    error_handling: str = "continue"  # continue, stop, rollback
    progress_callback: Optional[Callable] = None
    global_context: Any = None


@dataclass
class ExecutionResult:
    """Result of executing a workflow step."""
    step_id: str
    agent_name: str
    state: ExecutionState
    result: Optional[AgentResult] = None
    error: Optional[Exception] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    retry_count: int = 0


@dataclass
class OrchestrationResult:
    """Complete result of workflow orchestration."""
    workflow_id: str
    success: bool
    total_steps: int
    completed_steps: int
    failed_steps: int
    total_duration_seconds: float
    step_results: Dict[str, ExecutionResult] = field(default_factory=dict)
    final_outputs: Dict[str, Any] = field(default_factory=dict)
    error_summary: List[str] = field(default_factory=list)


def orchestrate_workflow(
    task_graph: TaskGraph, 
    agent_registry: Dict[str, Any],
    context: OrchestrationContext
) -> OrchestrationResult:
    """
    Orchestrate the execution of a complete workflow.
    
    Args:
        task_graph: The task graph to execute
        agent_registry: Registry of available agents
        context: Orchestration context and settings
        
    Returns:
        Complete orchestration result with execution details
    """
    logger.info(f"Starting workflow orchestration: {context.workflow_id}")
    
    start_time = datetime.now()
    orchestrator = WorkflowOrchestrator(task_graph, agent_registry, context)
    
    try:
        result = orchestrator.execute()
        
        end_time = datetime.now()
        result.total_duration_seconds = (end_time - start_time).total_seconds()
        
        logger.info(f"Workflow orchestration complete: {result.success}")
        return result
        
    except Exception as e:
        logger.error(f"Workflow orchestration failed: {e}")
        
        return OrchestrationResult(
            workflow_id=context.workflow_id,
            success=False,
            total_steps=len(task_graph.nodes),
            completed_steps=0,
            failed_steps=len(task_graph.nodes),
            total_duration_seconds=(datetime.now() - start_time).total_seconds(),
            error_summary=[f"Orchestration failed: {e}"]
        )


class WorkflowOrchestrator:
    """Main orchestrator class for executing workflows."""
    
    def __init__(self, task_graph: TaskGraph, agent_registry: Dict[str, Any], context: OrchestrationContext):
        self.task_graph = task_graph
        self.agent_registry = agent_registry
        self.context = context
        self.step_results: Dict[str, ExecutionResult] = {}
        self.execution_queue: List[str] = []
        self.running_tasks: Dict[str, Future] = {}
        self.executor = ThreadPoolExecutor(max_workers=context.max_parallel_tasks)
    
    def execute(self) -> OrchestrationResult:
        """Execute the complete workflow."""
        logger.info(f"Executing workflow with {len(self.task_graph.nodes)} tasks")
        
        # Initialize execution results
        for task_id in self.task_graph.nodes:
            self.step_results[task_id] = ExecutionResult(
                step_id=task_id,
                agent_name=self.task_graph.nodes[task_id].assigned_agent,
                state=ExecutionState.PENDING
            )
        
        # Execute based on orchestration mode
        if self.context.orchestration_mode == OrchestrationMode.SEQUENTIAL:
            return self._execute_sequential()
        elif self.context.orchestration_mode == OrchestrationMode.PARALLEL:
            return self._execute_parallel()
        elif self.context.orchestration_mode == OrchestrationMode.ADAPTIVE:
            return self._execute_adaptive()
        else:
            return self._execute_sequential()
    
    def _execute_sequential(self) -> OrchestrationResult:
        """Execute tasks in sequential order respecting dependencies."""
        logger.info("Executing workflow sequentially")
        
        # Topological sort to determine execution order
        execution_order = self._topological_sort()
        
        for task_id in execution_order:
            if not self._should_execute_task(task_id):
                continue
                
            result = self._execute_single_task(task_id)
            self.step_results[task_id] = result
            
            # Report progress
            if self.context.progress_callback:
                self.context.progress_callback(task_id, result)
            
            # Handle failures with detailed logging
            if result.state == ExecutionState.FAILED:
                # Log detailed failure information
                task = self.task_graph.nodes[task_id]
                error_details = self._get_detailed_error_info(result, task)
                logger.error(f"Task {task_id} failed with detailed info: {error_details}")
                
                if self.context.error_handling == "stop":
                    logger.error(f"Stopping workflow due to failed task: {task_id}")
                    break
                elif self.context.error_handling == "rollback":
                    logger.info(f"Rolling back workflow due to failed task: {task_id}")
                    self._rollback_completed_tasks()
                    break
        
        return self._create_orchestration_result()
    
    def _execute_parallel(self) -> OrchestrationResult:
        """Execute tasks in parallel where possible."""
        logger.info("Executing workflow with parallelization")
        
        # Find tasks that can be executed in parallel
        parallel_groups = self._identify_parallel_groups()
        
        for group in parallel_groups:
            # Submit parallel tasks
            futures = {}
            for task_id in group:
                if self._should_execute_task(task_id):
                    future = self.executor.submit(self._execute_single_task, task_id)
                    futures[task_id] = future
            
            # Wait for completion
            for task_id, future in futures.items():
                try:
                    result = future.result(timeout=self.context.timeout_minutes * 60)
                    self.step_results[task_id] = result
                except Exception as e:
                    self.step_results[task_id] = ExecutionResult(
                        step_id=task_id,
                        agent_name=self.task_graph.nodes[task_id].assigned_agent,
                        state=ExecutionState.FAILED,
                        error=e
                    )
        
        return self._create_orchestration_result()
    
    def _execute_adaptive(self) -> OrchestrationResult:
        """Execute with adaptive scheduling based on task characteristics."""
        logger.info("Executing workflow with adaptive orchestration")
        
        # Start with ready tasks
        ready_tasks = self._get_ready_tasks()
        
        while ready_tasks or self.running_tasks:
            # Start new tasks if slots available
            while ready_tasks and len(self.running_tasks) < self.context.max_parallel_tasks:
                task_id = ready_tasks.pop(0)
                future = self.executor.submit(self._execute_single_task, task_id)
                self.running_tasks[task_id] = future
            
            # Check for completed tasks
            completed_tasks = []
            for task_id, future in self.running_tasks.items():
                if future.done():
                    try:
                        result = future.result()
                        self.step_results[task_id] = result
                    except Exception as e:
                        self.step_results[task_id] = ExecutionResult(
                            step_id=task_id,
                            agent_name=self.task_graph.nodes[task_id].assigned_agent,
                            state=ExecutionState.FAILED,
                            error=e
                        )
                    completed_tasks.append(task_id)
            
            # Remove completed tasks and find newly ready tasks
            for task_id in completed_tasks:
                del self.running_tasks[task_id]
                
            new_ready_tasks = self._get_ready_tasks()
            ready_tasks.extend(new_ready_tasks)
        
        return self._create_orchestration_result()
    
    def _execute_single_task(self, task_id: str) -> ExecutionResult:
        """Execute a single task."""
        task = self.task_graph.nodes[task_id]
        start_time = datetime.now()
        
        logger.info(f"Executing task: {task_id} with {task.assigned_agent}")
        
        try:
            # Get agent instance
            agent = self._get_agent_instance(task.assigned_agent)
            if not agent:
                raise ValueError(f"Agent not found: {task.assigned_agent}")
            
            # Prepare inputs with dependency outputs
            inputs = self._prepare_task_inputs(task)
            
            # Execute the task
            result = agent.execute(task.goal, inputs, self.context.global_context)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            execution_state = ExecutionState.COMPLETED if result.success else ExecutionState.FAILED
            
            execution_result = ExecutionResult(
                step_id=task_id,
                agent_name=task.assigned_agent,
                state=execution_state,
                result=result,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration
            )
            
            # Log execution details
            if execution_state == ExecutionState.FAILED:
                error_details = self._get_detailed_error_info(execution_result, task)
                logger.error(f"Task {task_id} failed: {error_details}")
            else:
                logger.info(f"Task {task_id} completed successfully in {duration:.2f}s")
            
            return execution_result
            
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            logger.error(f"Task execution failed: {task_id} - {e}")
            
            failed_result = ExecutionResult(
                step_id=task_id,
                agent_name=task.assigned_agent,
                state=ExecutionState.FAILED,
                error=e,
                start_time=start_time,
                end_time=end_time,
                duration_seconds=duration
            )
            
            # Log detailed failure information
            task = self.task_graph.nodes[task_id]
            error_details = self._get_detailed_error_info(failed_result, task)
            logger.error(f"Detailed task failure info for {task_id}: {error_details}")
            
            return failed_result
    
    def _should_execute_task(self, task_id: str) -> bool:
        """Check if a task should be executed based on dependencies."""
        task = self.task_graph.nodes[task_id]
        
        # Check if already executed
        if self.step_results[task_id].state in [ExecutionState.COMPLETED, ExecutionState.FAILED]:
            return False
        
        # Check dependencies
        for dep_id in task.dependencies:
            if dep_id not in self.step_results:
                return False
            if self.step_results[dep_id].state != ExecutionState.COMPLETED:
                return False
        
        return True
    
    def _get_ready_tasks(self) -> List[str]:
        """Get tasks that are ready to execute."""
        ready_tasks = []
        
        for task_id in self.task_graph.nodes:
            if (self.step_results[task_id].state == ExecutionState.PENDING and 
                self._should_execute_task(task_id) and
                task_id not in self.running_tasks):
                ready_tasks.append(task_id)
        
        return ready_tasks
    
    def _topological_sort(self) -> List[str]:
        """Perform topological sort of tasks based on dependencies."""
        visited = set()
        temp_visited = set()
        result = []
        
        def visit(task_id: str):
            if task_id in temp_visited:
                raise ValueError(f"Circular dependency detected involving task: {task_id}")
            if task_id in visited:
                return
                
            temp_visited.add(task_id)
            
            # Visit dependencies first
            task = self.task_graph.nodes[task_id]
            for dep_id in task.dependencies:
                if dep_id in self.task_graph.nodes:
                    visit(dep_id)
            
            temp_visited.remove(task_id)
            visited.add(task_id)
            result.append(task_id)
        
        for task_id in self.task_graph.nodes:
            if task_id not in visited:
                visit(task_id)
        
        return result
    
    def _identify_parallel_groups(self) -> List[List[str]]:
        """Identify groups of tasks that can be executed in parallel."""
        groups = []
        remaining_tasks = set(self.task_graph.nodes.keys())
        
        while remaining_tasks:
            # Find tasks with no dependencies or all dependencies satisfied
            current_group = []
            for task_id in list(remaining_tasks):
                task = self.task_graph.nodes[task_id]
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
    
    def _prepare_task_inputs(self, task: TaskNode) -> Dict[str, Any]:
        """Prepare inputs for task execution including dependency outputs."""
        inputs = {}  # TaskNode doesn't have inputs field, start empty
        
        # Extract command from goal for shell execution tasks
        if task.assigned_agent == "ShellExecutor" and task.goal.startswith("Execute: "):
            command = task.goal[9:]  # Remove "Execute: " prefix
            inputs["commands"] = [command]
            logger.debug(f"Extracted command for {task.task_id}: {command}")
        
        # Add outputs from dependency tasks
        for dep_id in task.dependencies:
            if dep_id in self.step_results and self.step_results[dep_id].result:
                dep_result = self.step_results[dep_id].result
                inputs[f"dependency_{dep_id}"] = dep_result.outputs
        
        return inputs
    
    def _get_agent_instance(self, agent_name: str):
        """Get agent instance from registry."""
        return self.agent_registry.get(agent_name)
    
    def _rollback_completed_tasks(self):
        """Rollback completed tasks (placeholder for actual rollback logic)."""
        logger.info("Rolling back completed tasks")
        # In practice, this would undo changes made by completed tasks
        for task_id, result in self.step_results.items():
            if result.state == ExecutionState.COMPLETED:
                result.state = ExecutionState.CANCELLED
                logger.info(f"Rolled back task: {task_id}")
    
    def _get_detailed_error_info(self, execution_result: ExecutionResult, task: TaskNode) -> str:
        """Get detailed error information for a failed task."""
        details = []
        
        # Basic task info
        details.append(f"Task: {execution_result.step_id}")
        details.append(f"Goal: {task.goal[:100]}..." if len(task.goal) > 100 else f"Goal: {task.goal}")
        details.append(f"Agent: {execution_result.agent_name}")
        
        # Execution details
        if execution_result.duration_seconds:
            details.append(f"Duration: {execution_result.duration_seconds:.2f}s")
        
        # Error information
        if execution_result.error:
            details.append(f"Exception: {execution_result.error}")
        
        # Agent result details
        if execution_result.result:
            result = execution_result.result
            details.append(f"Agent success: {result.success}")
            details.append(f"Agent message: {result.message}")
            
            # Check for shell command outputs
            if hasattr(result, 'outputs') and result.outputs:
                outputs = result.outputs
                
                # Look for command results
                if 'command_results' in outputs:
                    cmd_results = outputs['command_results']
                    if isinstance(cmd_results, list) and cmd_results:
                        for i, cmd_result in enumerate(cmd_results):
                            if isinstance(cmd_result, dict):
                                cmd = cmd_result.get('command', 'unknown')
                                exit_code = cmd_result.get('exit_code', 'unknown')
                                stdout = cmd_result.get('stdout', '').strip()[:200]
                                stderr = cmd_result.get('stderr', '').strip()[:200]
                                details.append(f"Command[{i}]: {cmd}")
                                details.append(f"Exit code[{i}]: {exit_code}")
                                if stdout:
                                    details.append(f"STDOUT[{i}]: {stdout}")
                                if stderr:
                                    details.append(f"STDERR[{i}]: {stderr}")
                
                # Look for failed commands
                if 'failed_commands' in outputs:
                    failed_cmds = outputs['failed_commands']
                    details.append(f"Failed commands: {failed_cmds}")
                
                # Look for error details
                if 'error_details' in outputs:
                    error_details = outputs['error_details']
                    details.append(f"Error details: {error_details}")
        
        return "; ".join(details)
    
    def _create_orchestration_result(self) -> OrchestrationResult:
        """Create final orchestration result."""
        completed_steps = sum(1 for r in self.step_results.values() if r.state == ExecutionState.COMPLETED)
        failed_steps = sum(1 for r in self.step_results.values() if r.state == ExecutionState.FAILED)
        
        # Collect final outputs from last tasks
        final_outputs = {}
        error_summary = []
        
        for task_id, result in self.step_results.items():
            if result.state == ExecutionState.COMPLETED and result.result:
                final_outputs[task_id] = result.result.outputs
            elif result.state == ExecutionState.FAILED:
                error_msg = f"Task {task_id} failed"
                if result.error:
                    error_msg += f": {result.error}"
                error_summary.append(error_msg)
        
        result = OrchestrationResult(
            workflow_id=self.context.workflow_id,
            success=failed_steps == 0,
            total_steps=len(self.step_results),
            completed_steps=completed_steps,
            failed_steps=failed_steps,
            total_duration_seconds=0,  # Will be set by caller
            step_results=self.step_results,
            final_outputs=final_outputs,
            error_summary=error_summary
        )
        
        # Log summary of workflow execution
        if failed_steps > 0:
            logger.error(f"Workflow {self.context.workflow_id} completed with {failed_steps} failures out of {len(self.step_results)} tasks")
            for task_id, step_result in self.step_results.items():
                if step_result.state == ExecutionState.FAILED:
                    task = self.task_graph.nodes[task_id]
                    error_details = self._get_detailed_error_info(step_result, task)
                    logger.error(f"Failed task {task_id} details: {error_details}")
        else:
            logger.info(f"Workflow {self.context.workflow_id} completed successfully with {completed_steps} tasks")
        
        return result


def create_orchestration_context(
    workflow_id: str,
    orchestration_mode: str = "adaptive",
    max_parallel_tasks: int = 3,
    retry_attempts: int = 2,
    timeout_minutes: int = 60,
    error_handling: str = "continue",
    global_context: Any = None
) -> OrchestrationContext:
    """Create orchestration context with default values."""
    
    return OrchestrationContext(
        workflow_id=workflow_id,
        orchestration_mode=OrchestrationMode(orchestration_mode),
        max_parallel_tasks=max_parallel_tasks,
        retry_attempts=retry_attempts,
        timeout_minutes=timeout_minutes,
        error_handling=error_handling,
        global_context=global_context
    )


def optimize_workflow_execution(task_graph: TaskGraph, constraints: Dict[str, Any] = None) -> TaskGraph:
    """
    Optimize workflow execution by identifying parallelization opportunities.
    
    Args:
        task_graph: Original task graph
        constraints: Execution constraints and preferences
        
    Returns:
        Optimized TaskGraph with parallel execution hints
    """
    constraints = constraints or {}
    logger.info("Optimizing workflow for parallel execution")
    
    # For now, return the original task graph since TaskNode doesn't have metadata
    # In a real implementation, we would create an enhanced TaskNode or use a different approach
    logger.info(f"Workflow optimization complete: {len(task_graph.nodes)} tasks processed")
    
    return task_graph


def monitor_workflow_progress(orchestration_result: OrchestrationResult) -> Dict[str, Any]:
    """Monitor and report workflow progress."""
    
    total_tasks = orchestration_result.total_steps
    completed_tasks = orchestration_result.completed_steps
    failed_tasks = orchestration_result.failed_steps
    
    progress_percentage = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
    
    status = "completed" if orchestration_result.success else "failed"
    if completed_tasks > 0 and completed_tasks < total_tasks:
        status = "in_progress"
    
    return {
        "workflow_id": orchestration_result.workflow_id,
        "status": status,
        "progress_percentage": progress_percentage,
        "completed_tasks": completed_tasks,
        "failed_tasks": failed_tasks,
        "total_tasks": total_tasks,
        "duration_seconds": orchestration_result.total_duration_seconds,
        "current_errors": orchestration_result.error_summary
    }