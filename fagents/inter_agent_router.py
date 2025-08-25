# fagents/inter_agent_router.py
"""
LLM-based Inter-Agent Routing System for Foundation Agents.

This module provides intelligent coordination between foundational agents,
dynamically determining the next agent to invoke based on current progress
toward the user's goal. It ensures minimal hops while making directional
progress and includes completion validation.
"""

import json
import logging
import re
from typing import Dict, Any, List, Optional
from enum import Enum
from dataclasses import dataclass, field

from core.context import GlobalContext
from core.models import AgentResult

logger = logging.getLogger(__name__)


class WorkflowStatus(Enum):
    """Status of the multi-agent workflow."""
    STARTING = "starting"
    IN_PROGRESS = "in_progress"
    VALIDATION_NEEDED = "validation_needed"
    COMPLETED = "completed"
    FAILED = "failed"
    MAX_HOPS_EXCEEDED = "max_hops_exceeded"
    SELF_DEBUGGING = "self_debugging"
    USER_CONSULTATION_NEEDED = "user_consultation_needed"


class FailureType(Enum):
    """Types of routing/agent execution failures."""
    INCORRECT_ROUTING = "incorrect_routing"
    INSUFFICIENT_CONTEXT = "insufficient_context"
    TARGET_AGENT_MISROUTING = "target_agent_misrouting"
    CAPABILITY_EXECUTION_FAILURE = "capability_execution_failure"
    AGENT_NOT_FOUND = "agent_not_found"
    EXECUTION_EXCEPTION = "execution_exception"


@dataclass
class AgentExecution:
    """Record of an agent execution in the workflow."""
    agent_name: str
    goal: str
    inputs: Dict[str, Any]
    result: AgentResult
    execution_order: int
    reasoning: str
    confidence: float
    timestamp: str = field(default_factory=lambda: str(__import__('datetime').datetime.now()))


@dataclass
class WorkflowContext:
    """Context for the entire multi-agent workflow."""
    user_goal: str
    initial_inputs: Dict[str, Any]
    workflow_id: str
    max_hops: int = 10
    current_hop: int = 0
    status: WorkflowStatus = WorkflowStatus.STARTING
    execution_history: List[AgentExecution] = field(default_factory=list)
    accumulated_outputs: Dict[str, Any] = field(default_factory=dict)
    completion_validated: bool = False


@dataclass
class FailureAnalysis:
    """Analysis of an agent execution failure."""
    failure_type: FailureType
    failed_agent: str
    failure_description: str
    root_cause: str
    remediation_plan: str
    confidence: float
    requires_user_consultation: bool = False
    consultation_prompt: str = ""
    timestamp: str = field(default_factory=lambda: str(__import__('datetime').datetime.now()))


@dataclass
class NextAgentDecision:
    """Decision about which agent to invoke next."""
    agent_name: str
    confidence: float
    reasoning: str
    recommended_inputs: Dict[str, Any]
    goal_for_agent: str
    is_completion: bool = False
    completion_summary: str = ""
    failure_analysis: Optional[FailureAnalysis] = None


class InterAgentRouter:
    """
    LLM-based intelligent router for coordinating foundational agents.
    
    This router makes sequential decisions about which foundational agent
    to invoke next, ensuring minimal hops while making directional progress
    toward the user's goal.
    """
    
    def __init__(self, llm_client: Any = None, agent_registry: Dict[str, Any] = None, ui_interface: Any = None):
        self.llm_client = llm_client
        self.agent_registry = agent_registry or {}
        self.ui_interface = ui_interface
        
        # Load foundational agent registry if not provided
        if not self.agent_registry:
            try:
                from . import FOUNDATIONAL_AGENT_REGISTRY
                self.agent_registry = FOUNDATIONAL_AGENT_REGISTRY
            except ImportError:
                logger.warning("Could not load foundational agent registry")
    
    def execute_workflow(self, user_goal: str, initial_inputs: Dict[str, Any], 
                        global_context: GlobalContext, max_hops: int = 10) -> WorkflowContext:
        """
        Execute a complete multi-agent workflow to achieve the user's goal.
        
        Args:
            user_goal: The user's high-level goal
            initial_inputs: Initial inputs provided by the user
            global_context: Global execution context
            max_hops: Maximum number of agent invocations allowed
            
        Returns:
            WorkflowContext with complete execution history and results
        """
        
        # Initialize workflow context
        import uuid
        workflow_context = WorkflowContext(
            user_goal=user_goal,
            initial_inputs=initial_inputs,
            workflow_id=f"workflow_{uuid.uuid4().hex[:8]}",
            max_hops=max_hops,
            status=WorkflowStatus.IN_PROGRESS
        )
        
        logger.info(f"Starting inter-agent workflow: {workflow_context.workflow_id}")
        logger.info(f"User goal: {user_goal}")
        
        try:
            # Intelligently determine the first agent based on user goal
            first_agent_decision = self._determine_first_agent(user_goal, initial_inputs, global_context)
            current_agent = first_agent_decision.agent_name
            current_inputs = first_agent_decision.recommended_inputs
            current_goal = first_agent_decision.goal_for_agent
            
            # Display initial routing decision to user
            if self.ui_interface and hasattr(self.ui_interface, 'display_routing_decision'):
                self.ui_interface.display_routing_decision(
                    "USER REQUEST", current_agent, first_agent_decision.confidence, first_agent_decision.reasoning
                )
            
            logger.info(f"Initial routing decision: {current_agent} (confidence: {first_agent_decision.confidence:.2f})")
            logger.info(f"Reasoning: {first_agent_decision.reasoning}")
            
            while (workflow_context.current_hop < max_hops and 
                   workflow_context.status == WorkflowStatus.IN_PROGRESS):
                
                workflow_context.current_hop += 1
                logger.info(f"Hop {workflow_context.current_hop}: Invoking {current_agent}")
                
                # Execute current agent
                execution_result = self._execute_agent(
                    current_agent, current_goal, current_inputs, global_context, workflow_context
                )
                
                if not execution_result:
                    workflow_context.status = WorkflowStatus.FAILED
                    break
                
                # Record execution
                workflow_context.execution_history.append(execution_result)
                
                # Update accumulated outputs
                if execution_result.result.success and execution_result.result.outputs:
                    workflow_context.accumulated_outputs.update(execution_result.result.outputs)
                
                # Determine next agent (with failure analysis if needed)
                next_decision = self._determine_next_agent(workflow_context, global_context)
                
                # If agent failed, perform self-debugging analysis
                if not execution_result.result.success:
                    workflow_context.status = WorkflowStatus.SELF_DEBUGGING
                    failure_analysis = self._perform_failure_analysis(execution_result, workflow_context, global_context)
                    
                    if failure_analysis.requires_user_consultation:
                        workflow_context.status = WorkflowStatus.USER_CONSULTATION_NEEDED
                        logger.info(f"User consultation required: {failure_analysis.consultation_prompt}")
                        # Display consultation prompt to user if UI is available
                        if self.ui_interface and hasattr(self.ui_interface, 'request_user_consultation'):
                            self.ui_interface.request_user_consultation(failure_analysis)
                        break
                    else:
                        # Apply remediation plan
                        next_decision = self._apply_remediation_plan(failure_analysis, workflow_context, global_context)
                        workflow_context.status = WorkflowStatus.IN_PROGRESS
                
                if next_decision.is_completion:
                    # Workflow is complete
                    workflow_context.status = WorkflowStatus.COMPLETED
                    workflow_context.completion_validated = True
                    logger.info(f"Workflow completed: {next_decision.completion_summary}")
                    break
                
                # Prepare for next iteration
                previous_agent = execution_result.agent_name
                current_agent = next_decision.agent_name
                current_goal = next_decision.goal_for_agent
                current_inputs = next_decision.recommended_inputs
                
                # Display routing decision to user
                if self.ui_interface and hasattr(self.ui_interface, 'display_routing_decision'):
                    self.ui_interface.display_routing_decision(
                        previous_agent, current_agent, next_decision.confidence, next_decision.reasoning
                    )
                
                logger.info(f"Next agent decision: {current_agent} (confidence: {next_decision.confidence:.2f})")
                logger.info(f"Reasoning: {next_decision.reasoning}")
            
            # Check if max hops exceeded
            if workflow_context.current_hop >= max_hops and workflow_context.status == WorkflowStatus.IN_PROGRESS:
                workflow_context.status = WorkflowStatus.MAX_HOPS_EXCEEDED
                logger.warning(f"Workflow exceeded max hops ({max_hops})")
            
            return workflow_context
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}", exc_info=True)
            workflow_context.status = WorkflowStatus.FAILED
            return workflow_context
    
    def _execute_agent(self, agent_name: str, goal: str, inputs: Dict[str, Any], 
                      global_context: GlobalContext, workflow_context: WorkflowContext) -> Optional[AgentExecution]:
        """Execute a single agent and return the execution record with enhanced failure context."""
        
        execution_start_time = __import__('datetime').datetime.now()
        
        try:
            # Display agent input to user if UI is available
            if self.ui_interface and hasattr(self.ui_interface, 'display_agent_input'):
                self.ui_interface.display_agent_input(agent_name, goal, inputs)
            
            # Get agent class from registry
            agent_class = self.agent_registry.get(agent_name)
            if not agent_class:
                logger.error(f"Agent not found in registry: {agent_name}")
                # Create failure result for analysis
                failure_result = AgentResult(
                    success=False,
                    message=f"Agent '{agent_name}' not found in registry",
                    outputs={"failure_type": "agent_not_found", "available_agents": list(self.agent_registry.keys())}
                )
                return self._create_execution_record(agent_name, goal, inputs, failure_result, workflow_context, execution_start_time)
            
            # Instantiate agent with LLM client
            try:
                if hasattr(agent_class, '__init__'):
                    try:
                        # Try to pass llm_client if agent accepts it
                        agent_instance = agent_class(llm_client=self.llm_client)
                    except TypeError:
                        # Fallback to default constructor
                        agent_instance = agent_class()
                else:
                    agent_instance = agent_class()
            except Exception as instantiation_error:
                logger.error(f"Failed to instantiate agent {agent_name}: {instantiation_error}")
                failure_result = AgentResult(
                    success=False,
                    message=f"Failed to instantiate {agent_name}: {str(instantiation_error)}",
                    outputs={"failure_type": "instantiation_error", "error_details": str(instantiation_error)}
                )
                return self._create_execution_record(agent_name, goal, inputs, failure_result, workflow_context, execution_start_time)
            
            # Execute agent with enhanced error context
            try:
                result = agent_instance.execute(goal, inputs, global_context)
            except Exception as execution_error:
                logger.error(f"Agent {agent_name} execution failed: {execution_error}", exc_info=True)
                failure_result = AgentResult(
                    success=False,
                    message=f"Execution exception in {agent_name}: {str(execution_error)}",
                    outputs={
                        "failure_type": "execution_exception",
                        "error_details": str(execution_error),
                        "error_type": type(execution_error).__name__
                    }
                )
                return self._create_execution_record(agent_name, goal, inputs, failure_result, workflow_context, execution_start_time)
            
            # Display agent output to user if UI is available
            if self.ui_interface and hasattr(self.ui_interface, 'display_agent_output'):
                self.ui_interface.display_agent_output(agent_name, result.success, result.message, result.outputs)
            
            execution_record = self._create_execution_record(agent_name, goal, inputs, result, workflow_context, execution_start_time)
            
            logger.info(f"Agent {agent_name} executed: {'SUCCESS' if result.success else 'FAILED'}")
            if not result.success:
                logger.warning(f"Agent failure: {result.message}")
            
            return execution_record
            
        except Exception as e:
            logger.error(f"Unexpected error executing agent {agent_name}: {e}", exc_info=True)
            failure_result = AgentResult(
                success=False,
                message=f"Unexpected execution error: {str(e)}",
                outputs={"failure_type": "unexpected_error", "error_details": str(e)}
            )
            return self._create_execution_record(agent_name, goal, inputs, failure_result, workflow_context, execution_start_time)
    
    def _create_execution_record(self, agent_name: str, goal: str, inputs: Dict[str, Any], 
                               result: AgentResult, workflow_context: WorkflowContext,
                               start_time: Any) -> AgentExecution:
        """Create an execution record with timing and enhanced context."""
        end_time = __import__('datetime').datetime.now()
        execution_duration = (end_time - start_time).total_seconds()
        
        # Add execution metadata to result outputs
        if result.outputs is None:
            result.outputs = {}
        result.outputs.update({
            "execution_duration_seconds": execution_duration,
            "execution_start_time": start_time.isoformat(),
            "execution_end_time": end_time.isoformat()
        })
        
        return AgentExecution(
            agent_name=agent_name,
            goal=goal,
            inputs=inputs,
            result=result,
            execution_order=len(workflow_context.execution_history) + 1,
            reasoning=f"Executed {agent_name} for: {goal}",
            confidence=1.0 if result.success else 0.0  # Failed executions have 0 confidence
        )
    
    def _determine_next_agent(self, workflow_context: WorkflowContext, 
                             global_context: GlobalContext) -> NextAgentDecision:
        """Determine which agent to invoke next using LLM reasoning."""
        
        if not self.llm_client:
            raise RuntimeError(
                "LLM client is required for inter-agent routing decisions. "
                "No fallback routing available - please ensure OpenAI LLM Tool is properly configured."
            )
        
        
        try:
            # Build context for LLM decision
            prompt = self._build_next_agent_prompt(workflow_context, global_context)
            
            # Get LLM response
            response_str = self.llm_client.invoke(prompt)
            
            # Sanitize and parse JSON response
            try:
                sanitized_response = self._sanitize_json_response(response_str)
                response_data = json.loads(sanitized_response)
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing failed even after sanitization. Error: {e}")
                logger.error(f"Original response (first 500 chars): {response_str[:500]}")
                logger.error(f"Sanitized response (first 500 chars): {sanitized_response[:500] if 'sanitized_response' in locals() else 'N/A'}")
                raise
            
            # Parse and validate response
            decision = self._parse_next_agent_decision(response_data, workflow_context)
            
            # ANTI-LOOP PROTECTION: Check for consecutive same-agent calls
            decision = self._apply_anti_loop_protection(decision, workflow_context)
            
            return decision
            
        except Exception as e:
            logger.error(f"LLM-based next agent decision failed: {e}")
            raise RuntimeError(f"Inter-agent routing failed: {e}. No fallback available - please check LLM configuration.")
    
    def _build_next_agent_prompt(self, workflow_context: WorkflowContext, 
                                global_context: GlobalContext) -> str:
        """Build the LLM prompt for determining the next agent."""
        
        # Get agent capabilities
        agent_capabilities = self._get_agent_capabilities()
        
        # Format execution history
        execution_summary = self._format_execution_history(workflow_context.execution_history)
        
        # Format accumulated outputs
        outputs_summary = self._format_accumulated_outputs(workflow_context.accumulated_outputs)
        
        # Get workspace context
        workspace_files = global_context.workspace.list_files() if global_context.workspace else []
        workspace_context = f"Workspace files: {len(workspace_files)} files" if workspace_files else "No workspace"
        
        return f"""
        You are the Inter-Agent Routing Intelligence for a foundational agent system.
        
        Your task is to determine which foundational agent should be invoked next to make 
        the most DIRECTIONAL PROGRESS toward completing the user's goal with MINIMAL HOPS.
        
        **USER'S GOAL:** "{workflow_context.user_goal}"
        
        **CURRENT PROGRESS:**
        - Workflow ID: {workflow_context.workflow_id}
        - Current Hop: {workflow_context.current_hop}/{workflow_context.max_hops}
        - Status: {workflow_context.status.value}
        
        **EXECUTION HISTORY:**
        {execution_summary}
        
        **ACCUMULATED OUTPUTS:**
        {outputs_summary}
        
        **WORKSPACE CONTEXT:**
        {workspace_context}
        
        **AVAILABLE FOUNDATIONAL AGENTS:**
        {json.dumps(agent_capabilities, indent=2)}
        
        **ROUTING PRINCIPLES:**
        1. **Directional Progress**: Choose the agent that moves closest to COMPLETING the user's goal
        2. **Minimal Hops**: Favor agents that can accomplish multiple sub-tasks in one execution
        3. **Logical Sequence**: Follow natural development workflow (analyze → create → test → validate)
        4. **Avoid Redundancy**: Don't repeat work already done by previous agents
        5. **Completion Detection**: If the goal appears completed, route to AnalystAgent for validation
        6. **NO FAILURE WORKAROUNDS**: If an agent failed, don't route the same task to a different agent. Instead:
           - For technical errors/failures: Route to DebuggingAgent to fix root cause
           - For capability mismatches: Route to AnalystAgent to determine correct approach
           - Only retry the same agent AFTER the underlying issue is fixed
        
        **AGENT SELECTION CRITERIA:**
        - **AnalystAgent**: Use for initial analysis, final validation, or when understanding is needed
        - **StrategistAgent**: Use for complex planning, task decomposition, or orchestration
        - **CreatorAgent**: Use when something new needs to be created (code, tests, docs, specs)  
        - **SurgeonAgent**: Use for precise modifications, fixes, or surgical operations on existing code
        - **ExecutorAgent**: Use for running tests, validation, builds, or system operations
        - **DebuggingAgent**: Use for systematic debugging and troubleshooting failures
        
        **FOR FAILURES**: 
        - Any technical errors/failures → DebuggingAgent to fix root cause
        - Wrong agent selection → AnalystAgent to reassess approach
        
        **RECOMMENDED_INPUTS GUIDANCE:**
        When routing to specific agents, provide structured inputs in recommended_inputs:
        
        - **ExecutorAgent**: For shell operations, provide:
          {{"commands": ["command1", "command2"], "working_directory": "/path", "purpose": "description", "orchestration_required": true/false}}
          
          **CRITICAL**: Set "orchestration_required" based on workflow complexity:
          - true: Multi-step workflows requiring environment setup → dependencies → execution → verification
          - false: Simple single-purpose execution (e.g., run one test, execute one script)
          
          Example complex workflow: {{"commands": ["python -m venv venv", "pip install deps", "python script.py", "verify output"], "orchestration_required": true}}
          Example simple execution: {{"commands": ["python test.py"], "orchestration_required": false}}
        
        - **CreatorAgent**: For content creation, provide structured inputs based on creation type:
          
          **CRITICAL**: Always provide "creation_type" to guide CreatorAgent's execution strategy:
          
          For **test creation**:
          {{"creation_type": "tests", "test_type": "unit|integration|cli|api|performance", "test_framework": "pytest|unittest|jest", "test_quality": "fast|decent|production", "target_files": ["file1.py"], "requirements": ["test requirement 1", "test requirement 2"]}}
          
          For **simple code creation** (single scripts, utilities, small programs):
          {{"creation_type": "code", "language": "python|javascript|java", "code_type": "script|function|class|module", "file_path": "path/to/file", "requirements": ["code requirement 1", "code requirement 2"]}}
          
          For **documentation creation**:
          {{"creation_type": "documentation", "doc_type": "readme|api_doc|user_guide|technical_spec", "format": "markdown|html|rst", "requirements": ["doc requirement 1", "doc requirement 2"]}}
          
          For **specification creation**:
          {{"creation_type": "specification", "spec_type": "technical|functional|api", "requirements": ["requirement 1", "requirement 2"]}}
          
          For **full project creation** (multi-file applications with structure, config, tests, docs):
          {{"creation_type": "full_project", "project_type": "library|cli_app|web_app|api", "language": "python|javascript|java", "structure": ["component1", "component2"], "requirements": ["project requirement 1", "project requirement 2"]}}
          
          **GUIDANCE**: Use "code" creation for simple requests like "write a program/script to X". Use "full_project" only when explicitly requiring structured applications with multiple components.
        
        - **SurgeonAgent**: For modifications, provide:
          {{"target_files": ["file1.py", "file2.py"], "operation": "fix|refactor|update", "specific_changes": ["change description"]}}
        
        - **DebuggingAgent**: For systematic debugging, provide:
          {{"problem_description": "clear description of the issue", "workspace_path": "/path/to/workspace", "error_observed": "specific error details", "failed_test_report": "test failure details if available", "debugging_mode": "evidence_gathering|hypothesis_formation|fix_validation|full_debugging"}}
        
        - **Other agents**: Provide relevant context and parameters as appropriate
        
        **COMPLETION DETECTION:**
        If you believe the user's goal has been substantially completed based on the execution history
        and accumulated outputs, set "is_completion" to true and route to AnalystAgent for final validation.
        
        **ANTI-LOOP PROTECTION:**
        - If the same agent has been called multiple times in a row with similar goals, consider routing to a different agent
        - If ExecutorAgent has already been successful but the goal is still incomplete, consider routing to AnalystAgent for validation or completion assessment
        - Avoid routing to the same agent type more than 2 times consecutively unless there is clear progress and different sub-tasks
        
        **CRITICAL INSTRUCTIONS:**
        - Analyze what has been done vs. what remains to achieve the user's goal
        - Choose the agent that makes the most direct progress toward completion
        - Provide specific, actionable goals for the next agent
        - If multiple agents could help, choose the one that advances the workflow most significantly
        - **FAILURE ANALYSIS**: If the previous agent FAILED, look at WHY:
          * Technical errors (git, files, dependencies, environment, etc.) → Route to DebuggingAgent to fix
          * Wrong agent chosen for the task → Route to AnalystAgent to reassess  
          * DO NOT route the failed task to a different agent as a "workaround"
        - **AGENT BOUNDARIES**: Only route tasks that match the agent's core capabilities
        
        **RESPONSE FORMAT:**
        Your response must be a single JSON object with these exact keys:
        {{
            "agent_name": "AnalystAgent|StrategistAgent|CreatorAgent|SurgeonAgent|ExecutorAgent|DebuggingAgent",
            "confidence": 0.9,
            "reasoning": "Detailed explanation of why this agent makes the most directional progress",
            "goal_for_agent": "Specific, actionable goal for the chosen agent",
            "recommended_inputs": {{"structured_inputs_based_on_agent_type": "see_guidance_above"}},
            "is_completion": false,
            "completion_summary": "Summary if workflow is complete, empty string otherwise"
        }}
        
        **IMPORTANT**: For ExecutorAgent, always provide specific commands in recommended_inputs when shell operations are needed.
        
        Analyze the current state and determine the next agent that will make the most directional progress.
        """
    
    def _get_agent_capabilities(self) -> Dict[str, Dict[str, Any]]:
        """Get capabilities of all foundational agents."""
        
        agent_descriptions = {
            "AnalystAgent": {
                "description": "Deep comprehension and diagnosis agent - analyzes code, environments, problems, and quality",
                "strengths": ["Code analysis", "Problem diagnosis", "Quality assessment", "Environment analysis", "Validation"],
                "use_cases": ["Initial analysis", "Final validation", "Understanding complex problems", "Code quality checks"]
            },
            "StrategistAgent": {
                "description": "Task decomposition and orchestration agent - plans and coordinates complex workflows",
                "strengths": ["Task planning", "Workflow orchestration", "Resource optimization", "Complex problem breakdown"],
                "use_cases": ["Planning complex projects", "Coordinating multiple operations", "Breaking down big tasks"]
            },
            "CreatorAgent": {
                "description": "Content generation agent - creates code, tests, documentation, specifications, and projects",
                "strengths": ["Code generation", "Test creation", "Documentation writing", "Specification creation", "Project scaffolding"],
                "use_cases": ["Building new functionality", "Creating test suites", "Writing documentation", "Project setup"]
            },
            "SurgeonAgent": {
                "description": "Precise modification agent - surgical operations, fixes, and system maintenance",
                "strengths": ["Code modification", "Dependency management", "Configuration updates", "Precise repairs"],
                "use_cases": ["Bug fixes", "Dependency updates", "Configuration changes", "Surgical code edits"]
            },
            "ExecutorAgent": {
                "description": "Execution specialist - runs tests, validates code, executes commands, and manages systems",
                "strengths": ["Test execution", "Code validation", "Shell operations", "Environment setup", "File management"],
                "use_cases": ["Running test suites", "Validating implementations", "System operations", "Build processes"]
            },
            "DebuggingAgent": {
                "description": "Systematic debugging specialist - diagnoses, reproduces, and fixes complex issues",
                "strengths": ["Bug reproduction", "Root cause analysis", "Systematic debugging", "Issue resolution"],
                "use_cases": ["Complex bug investigation", "Production issues", "Systematic troubleshooting"]
            }
        }
        
        return agent_descriptions
    
    def _format_execution_history(self, executions: List[AgentExecution]) -> str:
        """Format execution history for the prompt."""
        if not executions:
            return "No previous executions"
        
        history_lines = []
        for i, execution in enumerate(executions, 1):
            status = "✅ SUCCESS" if execution.result.success else "❌ FAILED"
            history_lines.append(f"{i}. {execution.agent_name} ({status})")
            history_lines.append(f"   Goal: {execution.goal}")
            history_lines.append(f"   Result: {execution.result.message}")
            if execution.result.outputs:
                key_outputs = list(execution.result.outputs.keys())[:3]  # First 3 keys
                history_lines.append(f"   Key Outputs: {key_outputs}")
            history_lines.append("")
        
        return "\n".join(history_lines)
    
    def _format_accumulated_outputs(self, outputs: Dict[str, Any]) -> str:
        """Format accumulated outputs for the prompt."""
        if not outputs:
            return "No accumulated outputs"
        
        # Summarize key information
        summary_lines = []
        for key, value in list(outputs.items())[:10]:  # Limit to first 10 keys
            if isinstance(value, (dict, list)) and len(str(value)) > 100:
                summary_lines.append(f"- {key}: {type(value).__name__} (length: {len(value)})")
            else:
                summary_lines.append(f"- {key}: {str(value)[:100]}...")
        
        if len(outputs) > 10:
            summary_lines.append(f"... and {len(outputs) - 10} more outputs")
        
        return "\n".join(summary_lines)
    
    def _parse_next_agent_decision(self, response_data: Dict[str, Any], 
                                  workflow_context: WorkflowContext) -> NextAgentDecision:
        """Parse and validate the LLM response for next agent decision."""
        
        agent_name = response_data.get("agent_name", "AnalystAgent")
        
        # Validate agent exists
        if agent_name not in self.agent_registry:
            logger.warning(f"Invalid agent name: {agent_name} for workflow {workflow_context.workflow_id}. Defaulting to AnalystAgent.")
            agent_name = "AnalystAgent"
        
        return NextAgentDecision(
            agent_name=agent_name,
            confidence=float(response_data.get("confidence", 0.5)),
            reasoning=response_data.get("reasoning", "LLM routing decision"),
            recommended_inputs=response_data.get("recommended_inputs", {}),
            goal_for_agent=response_data.get("goal_for_agent", "Continue toward user's goal"),
            is_completion=bool(response_data.get("is_completion", False)),
            completion_summary=response_data.get("completion_summary", "")
        )

    def _apply_anti_loop_protection(self, decision: NextAgentDecision, 
                                   workflow_context: WorkflowContext) -> NextAgentDecision:
        """Apply anti-loop protection to prevent infinite agent loops with context awareness."""
        
        if not workflow_context.execution_history:
            return decision  # No history to check
        
        # Check for consecutive same-agent executions
        recent_executions = workflow_context.execution_history[-3:]  # Last 3 executions
        consecutive_count = 0
        consecutive_executions = []
        
        # Count consecutive occurrences of the proposed agent from the end
        for execution in reversed(recent_executions):
            if execution.agent_name == decision.agent_name:
                consecutive_count += 1
                consecutive_executions.insert(0, execution)  # Insert at beginning to maintain order
            else:
                break
        
        # If the same agent has been called 2+ times consecutively, check for legitimate variation
        if consecutive_count >= 2:
            # Check if this is a legitimate workflow progression (not a loop)
            if self._is_legitimate_progression(decision, consecutive_executions, workflow_context):
                logger.info(
                    f"ANTI-LOOP CHECK: {decision.agent_name} called {consecutive_count} times consecutively, "
                    f"but detected legitimate workflow progression. Allowing continuation."
                )
                return decision  # Allow legitimate progression
            
            logger.warning(
                f"ANTI-LOOP PROTECTION: {decision.agent_name} called {consecutive_count} times consecutively "
                f"with similar goals/contexts. Routing to AnalystAgent for validation."
            )
            
            # Override decision to route to AnalystAgent for validation
            return NextAgentDecision(
                agent_name="AnalystAgent",
                confidence=0.8,
                reasoning=(
                    f"Anti-loop protection triggered: {decision.agent_name} was called {consecutive_count} "
                    f"times consecutively with similar goals. Routing to AnalystAgent to assess progress "
                    f"and determine if the user's goal has been completed or needs a different approach."
                ),
                recommended_inputs={
                    "validation_type": "workflow_progress",
                    "previous_agent": decision.agent_name,
                    "consecutive_calls": consecutive_count,
                    "user_goal": workflow_context.user_goal
                },
                goal_for_agent=(
                    f"Assess workflow progress and determine if '{workflow_context.user_goal}' has been "
                    f"completed or needs a different approach after {consecutive_count} consecutive "
                    f"{decision.agent_name} executions"
                ),
                is_completion=False,
                completion_summary=""
            )
        
        return decision  # No protection needed
    
    def _is_legitimate_progression(self, decision: NextAgentDecision, 
                                  consecutive_executions: List[AgentExecution],
                                  workflow_context: WorkflowContext) -> bool:
        """
        Determine if consecutive agent calls represent legitimate workflow progression.
        Uses a general approach based on input/output differences rather than agent-specific logic.
        """
        if not consecutive_executions:
            return True  # No previous executions to compare
        
        # Strategy 1: Check if inputs are substantially different
        # Compare current decision with the most recent execution
        recent_execution_signature = self._get_inputs_signature([consecutive_executions[-1]])
        current_inputs_signature = self._get_decision_inputs_signature(decision)
        
        if recent_execution_signature != current_inputs_signature:
            logger.info(f"Different input signature detected for {decision.agent_name} - allowing progression")
            return True
        
        # Strategy 2: Check if outputs show actual progress (different work being done)
        if len(consecutive_executions) >= 2:
            outputs_are_progressing = self._outputs_show_progression(consecutive_executions)
            if outputs_are_progressing:
                logger.info(f"Output progression detected for {decision.agent_name} - allowing continuation")
                return True
        
        # Strategy 3: Check goal semantic difference using simple heuristics
        goal_similarity = self._calculate_goal_similarity(consecutive_executions, decision.goal_for_agent)
        if goal_similarity < 0.3:  # Less than 30% similar = different enough
            logger.info(f"Goals sufficiently different (similarity: {goal_similarity:.2f}) - allowing progression")
            return True
        
        # If all checks fail, it's likely a genuine loop
        return False
    
    def _get_inputs_signature(self, executions: List[AgentExecution]) -> str:
        """Get a signature representing the key inputs across executions."""
        signatures = []
        for execution in executions:
            # Extract key differentiating fields from inputs
            key_fields = []
            inputs = execution.inputs
            
            # Common differentiating fields across all agents
            for field in ["creation_type", "commands", "target_files", "operation", "analysis_type", "test_type"]:
                if field in inputs:
                    value = inputs[field]
                    if isinstance(value, list):
                        key_fields.append(f"{field}:{','.join(sorted(str(v) for v in value))}")
                    else:
                        key_fields.append(f"{field}:{value}")
            
            signatures.append("|".join(sorted(key_fields)))
        
        return "::".join(signatures)
    
    def _get_decision_inputs_signature(self, decision: NextAgentDecision) -> str:
        """Get signature for the proposed decision's inputs."""
        inputs = decision.recommended_inputs
        key_fields = []
        
        # Same logic as _get_inputs_signature but for decision inputs
        for field in ["creation_type", "commands", "target_files", "operation", "analysis_type", "test_type"]:
            if field in inputs:
                value = inputs[field]
                if isinstance(value, list):
                    key_fields.append(f"{field}:{','.join(sorted(str(v) for v in value))}")
                else:
                    key_fields.append(f"{field}:{value}")
        
        return "|".join(sorted(key_fields))
    
    def _outputs_show_progression(self, executions: List[AgentExecution]) -> bool:
        """Check if outputs across executions show actual progression/different work."""
        if len(executions) < 2:
            return False
        
        # Compare first and last execution outputs
        first_outputs = executions[0].result.outputs or {}
        last_outputs = executions[-1].result.outputs or {}
        
        # Check for different output structure or values
        first_keys = set(first_outputs.keys())
        last_keys = set(last_outputs.keys())
        
        # Different keys = different work
        if first_keys != last_keys:
            return True
        
        # Same keys but different values = progression
        for key in first_keys:
            if first_outputs.get(key) != last_outputs.get(key):
                return True
        
        return False
    
    def _calculate_goal_similarity(self, executions: List[AgentExecution], current_goal: str) -> float:
        """Calculate similarity between previous goals and current goal."""
        if not executions:
            return 0.0
        
        # Combine all previous goals
        prev_goal_words = set()
        for execution in executions:
            prev_goal_words.update(execution.goal.lower().split())
        
        current_goal_words = set(current_goal.lower().split())
        
        if not prev_goal_words or not current_goal_words:
            return 0.0
        
        # Calculate Jaccard similarity
        intersection = len(prev_goal_words.intersection(current_goal_words))
        union = len(prev_goal_words.union(current_goal_words))
        
        return intersection / union if union > 0 else 0.0
    
    def _determine_first_agent(self, user_goal: str, initial_inputs: Dict[str, Any], 
                              global_context: GlobalContext) -> NextAgentDecision:
        """Determine which agent should handle the user's request initially."""
        
        if not self.llm_client:
            raise RuntimeError(
                "LLM client is required for first agent routing. "
                "No fallback routing available - please ensure OpenAI LLM Tool is properly configured."
            )
        
        try:
            # Build context for initial routing decision
            prompt = self._build_first_agent_prompt(user_goal, initial_inputs, global_context)
            
            # Get LLM response
            response_str = self.llm_client.invoke(prompt)
            
            # Sanitize and parse JSON response
            try:
                sanitized_response = self._sanitize_json_response(response_str)
                response_data = json.loads(sanitized_response)
            except json.JSONDecodeError as e:
                logger.error(f"JSON parsing failed for initial routing. Error: {e}")
                logger.error(f"Original response (first 500 chars): {response_str[:500]}")
                logger.error(f"Sanitized response (first 500 chars): {sanitized_response[:500] if 'sanitized_response' in locals() else 'N/A'}")
                raise
            
            # Parse and validate response
            return self._parse_next_agent_decision(response_data, None)
            
        except Exception as e:
            logger.error(f"First agent routing failed: {e}")
            # Fallback to AnalystAgent if routing fails
            logger.warning("Falling back to AnalystAgent for initial routing")
            return NextAgentDecision(
                agent_name="AnalystAgent",
                confidence=0.3,
                reasoning=f"Fallback routing due to error: {e}",
                recommended_inputs=initial_inputs,
                goal_for_agent=f"Analyze and understand: {user_goal}",
                is_completion=False
            )
    
    def _build_first_agent_prompt(self, user_goal: str, initial_inputs: Dict[str, Any], 
                                 global_context: GlobalContext) -> str:
        """Build the LLM prompt for determining the first agent to invoke."""
        
        # Get agent capabilities
        agent_capabilities = self._get_agent_capabilities()
        
        # Get workspace context
        workspace_files = global_context.workspace.list_files() if global_context.workspace else []
        workspace_context = f"Workspace files: {len(workspace_files)} files" if workspace_files else "No workspace"
        
        # Format initial inputs
        inputs_summary = self._format_inputs_summary(initial_inputs)
        
        return f"""
        You are the Initial Routing Intelligence for a foundational agent system.
        
        Your task is to determine which foundational agent should handle the user's request directly,
        making the most EFFICIENT routing decision without unnecessary analysis steps.
        
        **USER'S REQUEST:** "{user_goal}"
        
        **AVAILABLE INPUTS:**
        {inputs_summary}
        
        **WORKSPACE CONTEXT:**
        {workspace_context}
        
        **AVAILABLE FOUNDATIONAL AGENTS:**
        {json.dumps(agent_capabilities, indent=2)}
        
        **EFFICIENT ROUTING PRINCIPLES:**
        1. **Direct Action**: If the request is clear and actionable, route directly to the appropriate agent
        2. **Minimal Hops**: Avoid analysis unless the request is genuinely unclear or complex
        3. **Intent Recognition**: Recognize clear intent patterns and route accordingly
        4. **Analysis Only When Needed**: Only route to AnalystAgent if understanding is actually required
        
        **ROUTING PATTERNS:**
        - **"write/create/build/implement"** → CreatorAgent (for code, tests, docs, projects)
        - **"fix/debug/troubleshoot"** → DebuggingAgent or SurgeonAgent  
        - **"run/test/execute/validate"** → ExecutorAgent
        - **"plan/organize/coordinate"** → StrategistAgent
        - **"analyze/understand/explain"** → AnalystAgent
        - **"modify/update/change"** → SurgeonAgent
        
        **ANALYSIS ROUTING CRITERIA** (use AnalystAgent only if):
        - Request is genuinely unclear or ambiguous
        - Need to understand existing complex codebase before acting
        - Debugging complex system issues requiring investigation
        - User explicitly asks for analysis/understanding
        
        **TARGET AGENT HINTS:**
        Provide specific hints to the target agent about what type of work is expected:
        - For CreatorAgent: specify what to create (code/tests/docs) and quality level
        - For ExecutorAgent: specify what to run/execute/validate
        - For DebuggingAgent: specify what issue to debug
        - For SurgeonAgent: specify what to modify/fix
        - For StrategistAgent: specify what to plan/coordinate
        - For AnalystAgent: specify what to analyze and why
        
        **RECOMMENDED_INPUTS GUIDANCE:**
        Provide structured inputs in recommended_inputs based on target agent:
        
        - **ExecutorAgent**: For shell operations, provide:
          {{"commands": ["command1", "command2"], "working_directory": "/path", "purpose": "description", "orchestration_required": true/false}}
          
          **CRITICAL**: Set "orchestration_required" based on workflow complexity:
          - true: Multi-step workflows requiring environment setup → dependencies → execution → verification  
          - false: Simple single-purpose execution
        - **CreatorAgent**: For content creation, provide structured inputs based on creation type:
          
          **CRITICAL**: Always provide "creation_type" to guide CreatorAgent's execution strategy:
          
          For **test creation**:
          {{"creation_type": "tests", "test_type": "unit|integration|cli|api|performance", "test_framework": "pytest|unittest|jest", "test_quality": "fast|decent|production", "target_files": ["file1.py"], "requirements": ["test requirement 1", "test requirement 2"]}}
          
          For **simple code creation** (single scripts, utilities, small programs):
          {{"creation_type": "code", "language": "python|javascript|java", "code_type": "script|function|class|module", "file_path": "path/to/file", "requirements": ["code requirement 1", "code requirement 2"]}}
          
          For **documentation creation**:
          {{"creation_type": "documentation", "doc_type": "readme|api_doc|user_guide|technical_spec", "format": "markdown|html|rst", "requirements": ["doc requirement 1", "doc requirement 2"]}}
          
          For **specification creation**:
          {{"creation_type": "specification", "spec_type": "technical|functional|api", "requirements": ["requirement 1", "requirement 2"]}}
          
          For **full project creation** (multi-file applications with structure, config, tests, docs):
          {{"creation_type": "full_project", "project_type": "library|cli_app|web_app|api", "language": "python|javascript|java", "structure": ["component1", "component2"], "requirements": ["project requirement 1", "project requirement 2"]}}
          
          **GUIDANCE**: Use "code" creation for simple requests like "write a program/script to X". Use "full_project" only when explicitly requiring structured applications with multiple components.
        - **SurgeonAgent**: For modifications, provide:
          {{"target_files": ["file1.py"], "operation": "fix|refactor|update", "specific_changes": ["change description"]}}
        - **DebuggingAgent**: For systematic debugging, provide:
          {{"problem_description": "clear description of the issue", "workspace_path": "/path/to/workspace", "error_observed": "specific error details", "failed_test_report": "test failure details if available", "debugging_mode": "evidence_gathering|hypothesis_formation|fix_validation|full_debugging"}}
        - **Other agents**: Provide relevant context and parameters
        
        **RESPONSE FORMAT:**
        Your response must be a single JSON object with these exact keys:
        {{
            "agent_name": "AnalystAgent|StrategistAgent|CreatorAgent|SurgeonAgent|ExecutorAgent|DebuggingAgent",
            "confidence": 0.9,
            "reasoning": "Why this agent can directly handle the user's request",
            "goal_for_agent": "Specific, actionable goal with hints about what work is expected",
            "recommended_inputs": {{"structured_inputs_based_on_agent_type": "see_guidance_above"}},
            "is_completion": false,
            "completion_summary": ""
        }}
        
        **IMPORTANT**: For ExecutorAgent, always provide specific commands in recommended_inputs when shell operations are needed.
        
        Route efficiently to the agent that can directly accomplish the user's goal.
        """
    
    def _format_inputs_summary(self, inputs: Dict[str, Any]) -> str:
        """Format inputs for prompt context."""
        if not inputs:
            return "No inputs provided"
        
        summary_lines = []
        for key, value in inputs.items():
            if isinstance(value, (dict, list)) and len(str(value)) > 100:
                summary_lines.append(f"- {key}: {type(value).__name__} (length: {len(value)})")
            else:
                summary_lines.append(f"- {key}: {value}")
        
        return "\n".join(summary_lines)
    
    def _sanitize_json_response(self, response_str: str) -> str:
        """
        Sanitize LLM response to fix common JSON parsing issues.
        
        Args:
            response_str: Raw response string from LLM
            
        Returns:
            Sanitized JSON string
        """
        if not response_str:
            return response_str
        
        # Remove control characters that break JSON parsing
        # Keep only printable ASCII characters, spaces, tabs, and newlines
        sanitized = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', response_str)
        
        # Find JSON content between first { and last }
        start_brace = sanitized.find('{')
        end_brace = sanitized.rfind('}')
        
        if start_brace != -1 and end_brace != -1 and end_brace > start_brace:
            json_content = sanitized[start_brace:end_brace + 1]
        else:
            # If no braces found, return sanitized string as-is
            json_content = sanitized
        
        # Additional cleanup for common issues
        # Don't modify properly escaped quotes, only fix mangled ones
        # This is more conservative to avoid breaking valid JSON
        
        # Remove trailing commas before closing braces/brackets
        json_content = re.sub(r',(\s*[}\]])', r'\1', json_content)
        
        return json_content
    
    def _perform_failure_analysis(self, failed_execution: AgentExecution, 
                                workflow_context: WorkflowContext, 
                                global_context: GlobalContext) -> FailureAnalysis:
        """Perform comprehensive self-analysis of agent execution failure."""
        
        logger.info(f"Performing failure analysis for {failed_execution.agent_name}")
        
        if not self.llm_client:
            # Fallback analysis without LLM
            return self._create_fallback_failure_analysis(failed_execution, workflow_context)
        
        try:
            # Build self-analysis prompt
            analysis_prompt = self._build_failure_analysis_prompt(failed_execution, workflow_context, global_context)
            
            # Get LLM analysis
            response_str = self.llm_client.invoke(analysis_prompt)
            
            # Parse response
            sanitized_response = self._sanitize_json_response(response_str)
            response_data = json.loads(sanitized_response)
            
            # Create failure analysis from response
            return self._parse_failure_analysis(response_data, failed_execution)
            
        except Exception as e:
            logger.error(f"LLM failure analysis failed: {e}")
            return self._create_fallback_failure_analysis(failed_execution, workflow_context)
    
    def _build_failure_analysis_prompt(self, failed_execution: AgentExecution,
                                     workflow_context: WorkflowContext,
                                     global_context: GlobalContext) -> str:
        """Build comprehensive failure analysis prompt for LLM."""
        
        # Get recent execution history for context
        recent_history = workflow_context.execution_history[-5:] if workflow_context.execution_history else []
        history_summary = self._format_execution_history(recent_history)
        
        # Get workspace context
        workspace_files = global_context.workspace.list_files() if global_context.workspace else []
        workspace_context = f"Workspace files: {len(workspace_files)} files" if workspace_files else "No workspace"
        
        # Extract failure details
        failure_outputs = failed_execution.result.outputs or {}
        failure_type_hint = failure_outputs.get("failure_type", "unknown")
        error_details = failure_outputs.get("error_details", "")
        
        return f"""
        You are the Self-Debugging Intelligence for the Inter-Agent Router. 
        An agent execution has FAILED and you must perform thorough failure analysis.
        
        **YOUR TASK:** Analyze the failure, identify the root cause, and create a remediation plan.
        
        **USER'S ORIGINAL GOAL:** "{workflow_context.user_goal}"
        
        **FAILED EXECUTION DETAILS:**
        - Agent: {failed_execution.agent_name}
        - Goal: {failed_execution.goal}
        - Inputs: {json.dumps(failed_execution.inputs, indent=2)}
        - Failure Message: {failed_execution.result.message}
        - Error Details: {error_details}
        - Execution Duration: {failure_outputs.get('execution_duration_seconds', 'unknown')} seconds
        - Failure Type Hint: {failure_type_hint}
        
        **RECENT EXECUTION HISTORY:**
        {history_summary}
        
        **WORKFLOW CONTEXT:**
        - Current Hop: {workflow_context.current_hop}/{workflow_context.max_hops}
        - Total Executions: {len(workflow_context.execution_history)}
        - Accumulated Outputs: {len(workflow_context.accumulated_outputs)} keys
        
        **WORKSPACE CONTEXT:**
        {workspace_context}
        
        **FAILURE TYPE CATEGORIES TO ANALYZE:**
        
        1. **INCORRECT_ROUTING** - The request was sent to the wrong agent
           - Signs: Agent capabilities don't match the goal
           - Signs: Agent confusion or "I don't handle this" responses
           - Signs: Goal requires different foundational capabilities
        
        2. **INSUFFICIENT_CONTEXT** - Routing was correct, but provided context was incomplete
           - Signs: Agent asks for missing information
           - Signs: "Cannot proceed without X" type errors
           - Signs: Agent has capability but lacks necessary inputs
        
        3. **TARGET_AGENT_MISROUTING** - Agent's internal LLM router chose wrong capability
           - Signs: Agent executed but chose wrong internal operation
           - Signs: Right agent, wrong sub-capability invoked
           - Signs: Agent seemed confused about which internal tool to use
        
        4. **CAPABILITY_EXECUTION_FAILURE** - Right agent, right capability, but execution failed
           - Signs: Technical errors (file not found, syntax errors, etc.)
           - Signs: Environment issues (missing dependencies, permissions)
           - Signs: Code execution failures or runtime errors
        
        **SELF-DEBUGGING ANALYSIS PROCESS:**
        
        1. **FAILURE CLASSIFICATION:** Which failure type best matches the evidence?
        2. **ROOT CAUSE ANALYSIS:** What specifically went wrong and why?
        3. **CONTEXT EVALUATION:** Was the routing decision reasonable given available information?
        4. **PATTERN RECOGNITION:** Is this failure part of a larger pattern in the workflow?
        5. **REMEDIATION STRATEGY:** What specific actions will address the root cause?
        
        **REMEDIATION OPTIONS:**
        
        - **Route to Different Agent:** If wrong agent was selected
        - **Enhance Context:** If inputs were insufficient, gather missing information
        - **Retry with Better Instructions:** If agent misrouted internally
        - **Technical Problem Solving:** If execution failed due to technical issues
        - **User Consultation:** If problem is unclear or requires human judgment
        
        **CRITICAL INSTRUCTIONS:**
        
        - Be HONEST about routing mistakes - admit when the router made a poor decision
        - Be SPECIFIC about what information is missing or what went wrong
        - FOCUS on actionable remediation - what concrete steps will fix this?
        - Consider the WORKFLOW CONTEXT - is this failure blocking overall progress?
        - If you cannot determine a clear remediation path, recommend USER CONSULTATION
        
        **RESPONSE FORMAT:**
        Your response must be a single JSON object with these exact keys:
        {{
            "failure_type": "incorrect_routing|insufficient_context|target_agent_misrouting|capability_execution_failure",
            "root_cause": "Detailed explanation of what specifically went wrong and why",
            "failure_description": "Clear summary of the failure for logging/reporting",
            "remediation_plan": "Specific, actionable steps to address the root cause",
            "confidence": 0.85,
            "requires_user_consultation": false,
            "consultation_prompt": "If requires_user_consultation is true, what to ask the user"
        }}
        
        **REMEMBER:** Your analysis should be thorough, honest, and focused on getting the workflow back on track.
        Analyze the failure and provide a clear path forward.
        """
    
    def _parse_failure_analysis(self, response_data: Dict[str, Any], 
                              failed_execution: AgentExecution) -> FailureAnalysis:
        """Parse LLM response into FailureAnalysis object."""
        
        failure_type_str = response_data.get("failure_type", "capability_execution_failure")
        
        # Map string to FailureType enum
        failure_type = None
        for ft in FailureType:
            if ft.value == failure_type_str:
                failure_type = ft
                break
        
        if not failure_type:
            logger.warning(f"Unknown failure type: {failure_type_str}, defaulting to capability_execution_failure")
            failure_type = FailureType.CAPABILITY_EXECUTION_FAILURE
        
        return FailureAnalysis(
            failure_type=failure_type,
            failed_agent=failed_execution.agent_name,
            failure_description=response_data.get("failure_description", f"{failed_execution.agent_name} execution failed"),
            root_cause=response_data.get("root_cause", "Unknown root cause"),
            remediation_plan=response_data.get("remediation_plan", "Retry with DebuggingAgent"),
            confidence=float(response_data.get("confidence", 0.5)),
            requires_user_consultation=bool(response_data.get("requires_user_consultation", False)),
            consultation_prompt=response_data.get("consultation_prompt", "")
        )
    
    def _create_fallback_failure_analysis(self, failed_execution: AgentExecution,
                                        workflow_context: WorkflowContext) -> FailureAnalysis:
        """Create basic failure analysis when LLM analysis is not available."""
        
        failure_outputs = failed_execution.result.outputs or {}
        failure_type_hint = failure_outputs.get("failure_type", "unknown")
        
        # Map hint to failure type
        if failure_type_hint == "agent_not_found":
            failure_type = FailureType.AGENT_NOT_FOUND
        elif failure_type_hint in ["execution_exception", "instantiation_error", "unexpected_error"]:
            failure_type = FailureType.EXECUTION_EXCEPTION
        else:
            failure_type = FailureType.CAPABILITY_EXECUTION_FAILURE
        
        return FailureAnalysis(
            failure_type=failure_type,
            failed_agent=failed_execution.agent_name,
            failure_description=f"Fallback analysis: {failed_execution.agent_name} failed",
            root_cause=failed_execution.result.message,
            remediation_plan="Route to DebuggingAgent for systematic analysis",
            confidence=0.3,
            requires_user_consultation=workflow_context.current_hop >= workflow_context.max_hops - 2  # Near max hops
        )
    
    def _apply_remediation_plan(self, failure_analysis: FailureAnalysis,
                              workflow_context: WorkflowContext,
                              global_context: GlobalContext) -> NextAgentDecision:
        """Apply the remediation plan to determine the next course of action."""
        
        logger.info(f"Applying remediation plan for {failure_analysis.failure_type.value}")
        
        if failure_analysis.failure_type == FailureType.INCORRECT_ROUTING:
            return self._remediate_incorrect_routing(failure_analysis, workflow_context, global_context)
        
        elif failure_analysis.failure_type == FailureType.INSUFFICIENT_CONTEXT:
            return self._remediate_insufficient_context(failure_analysis, workflow_context, global_context)
        
        elif failure_analysis.failure_type == FailureType.TARGET_AGENT_MISROUTING:
            return self._remediate_target_agent_misrouting(failure_analysis, workflow_context, global_context)
        
        elif failure_analysis.failure_type == FailureType.CAPABILITY_EXECUTION_FAILURE:
            return self._remediate_capability_failure(failure_analysis, workflow_context, global_context)
        
        else:
            # Default remediation
            return NextAgentDecision(
                agent_name="DebuggingAgent",
                confidence=0.7,
                reasoning=f"Fallback remediation for {failure_analysis.failure_type.value}",
                recommended_inputs={
                    "problem_description": failure_analysis.failure_description,
                    "root_cause": failure_analysis.root_cause,
                    "failed_agent": failure_analysis.failed_agent,
                    "debugging_mode": "full_debugging"
                },
                goal_for_agent=f"Debug and resolve: {failure_analysis.failure_description}",
                failure_analysis=failure_analysis
            )
    
    def _remediate_incorrect_routing(self, analysis: FailureAnalysis,
                                   workflow_context: WorkflowContext,
                                   global_context: GlobalContext) -> NextAgentDecision:
        """Remediate incorrect routing decisions."""
        
        # Route to AnalystAgent to reassess the approach
        return NextAgentDecision(
            agent_name="AnalystAgent",
            confidence=0.8,
            reasoning=f"Incorrect routing detected for {analysis.failed_agent}. Reassessing approach.",
            recommended_inputs={
                "analysis_type": "problem_analysis",
                "failed_routing_context": {
                    "failed_agent": analysis.failed_agent,
                    "original_goal": workflow_context.user_goal,
                    "root_cause": analysis.root_cause
                }
            },
            goal_for_agent=f"Reassess approach after routing failure: {analysis.failure_description}",
            failure_analysis=analysis
        )
    
    def _remediate_insufficient_context(self, analysis: FailureAnalysis,
                                      workflow_context: WorkflowContext,
                                      global_context: GlobalContext) -> NextAgentDecision:
        """Remediate insufficient context issues."""
        
        # Route to AnalystAgent to gather missing context
        return NextAgentDecision(
            agent_name="AnalystAgent",
            confidence=0.8,
            reasoning=f"Insufficient context for {analysis.failed_agent}. Gathering missing information.",
            recommended_inputs={
                "analysis_type": "comprehensive_analysis",
                "context_gathering_focus": analysis.root_cause,
                "failed_agent_needs": analysis.failed_agent
            },
            goal_for_agent=f"Gather missing context: {analysis.failure_description}",
            failure_analysis=analysis
        )
    
    def _remediate_target_agent_misrouting(self, analysis: FailureAnalysis,
                                         workflow_context: WorkflowContext,
                                         global_context: GlobalContext) -> NextAgentDecision:
        """Remediate target agent internal misrouting."""
        
        # Get the last execution to extract the goal and inputs
        last_execution = workflow_context.execution_history[-1] if workflow_context.execution_history else None
        
        if last_execution:
            # Retry the same agent with more specific instructions
            enhanced_inputs = last_execution.inputs.copy()
            enhanced_inputs.update({
                "retry_context": {
                    "previous_failure": analysis.failure_description,
                    "routing_guidance": analysis.remediation_plan,
                    "force_specific_operation": True
                }
            })
            
            return NextAgentDecision(
                agent_name=analysis.failed_agent,
                confidence=0.7,
                reasoning=f"Retrying {analysis.failed_agent} with enhanced routing guidance",
                recommended_inputs=enhanced_inputs,
                goal_for_agent=f"RETRY with specific guidance: {last_execution.goal}",
                failure_analysis=analysis
            )
        else:
            # Fallback to debugging
            return self._remediate_capability_failure(analysis, workflow_context, global_context)
    
    def _remediate_capability_failure(self, analysis: FailureAnalysis,
                                    workflow_context: WorkflowContext,
                                    global_context: GlobalContext) -> NextAgentDecision:
        """Remediate capability execution failures."""
        
        # Route to DebuggingAgent for systematic troubleshooting
        return NextAgentDecision(
            agent_name="DebuggingAgent",
            confidence=0.9,
            reasoning=f"Capability execution failure in {analysis.failed_agent}. Systematic debugging required.",
            recommended_inputs={
                "problem_description": analysis.failure_description,
                "root_cause": analysis.root_cause,
                "failed_agent": analysis.failed_agent,
                "debugging_mode": "fix_validation",
                "workspace_path": str(global_context.workspace.repo_path) if global_context.workspace else "/tmp"
            },
            goal_for_agent=f"Debug and fix: {analysis.failure_description}",
            failure_analysis=analysis
        )


def execute_multi_agent_workflow(user_goal: str, initial_inputs: Dict[str, Any],
                                global_context: GlobalContext, llm_client: Any = None,
                                max_hops: int = 10, ui_interface: Any = None) -> WorkflowContext:
    """
    Convenience function to execute a complete multi-agent workflow.
    
    Args:
        user_goal: The user's high-level goal
        initial_inputs: Initial inputs provided by the user  
        global_context: Global execution context
        llm_client: LLM client for intelligent routing
        max_hops: Maximum number of agent invocations
        ui_interface: UI interface for displaying agent I/O to users
        
    Returns:
        WorkflowContext with complete execution results
    """
    router = InterAgentRouter(llm_client=llm_client, ui_interface=ui_interface)
    return router.execute_workflow(user_goal, initial_inputs, global_context, max_hops)