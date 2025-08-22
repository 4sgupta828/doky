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
class NextAgentDecision:
    """Decision about which agent to invoke next."""
    agent_name: str
    confidence: float
    reasoning: str
    recommended_inputs: Dict[str, Any]
    goal_for_agent: str
    is_completion: bool = False
    completion_summary: str = ""


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
                
                # Determine next agent
                next_decision = self._determine_next_agent(workflow_context, global_context)
                
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
        """Execute a single agent and return the execution record."""
        
        try:
            # Display agent input to user if UI is available
            if self.ui_interface and hasattr(self.ui_interface, 'display_agent_input'):
                self.ui_interface.display_agent_input(agent_name, goal, inputs)
            
            # Get agent class from registry
            agent_class = self.agent_registry.get(agent_name)
            if not agent_class:
                logger.error(f"Agent not found in registry: {agent_name}")
                return None
            
            # Instantiate agent with LLM client
            if hasattr(agent_class, '__init__'):
                try:
                    # Try to pass llm_client if agent accepts it
                    agent_instance = agent_class(llm_client=self.llm_client)
                except TypeError:
                    # Fallback to default constructor
                    agent_instance = agent_class()
            else:
                agent_instance = agent_class()
            
            # Execute agent
            result = agent_instance.execute(goal, inputs, global_context)
            
            # Display agent output to user if UI is available
            if self.ui_interface and hasattr(self.ui_interface, 'display_agent_output'):
                self.ui_interface.display_agent_output(agent_name, result.success, result.message, result.outputs)
            
            # Create execution record
            execution = AgentExecution(
                agent_name=agent_name,
                goal=goal,
                inputs=inputs,
                result=result,
                execution_order=len(workflow_context.execution_history) + 1,
                reasoning=f"Executed {agent_name} for: {goal}",
                confidence=1.0  # Actual execution always has confidence 1.0
            )
            
            logger.info(f"Agent {agent_name} executed: {'SUCCESS' if result.success else 'FAILED'}")
            if not result.success:
                logger.warning(f"Agent failure: {result.message}")
            
            return execution
            
        except Exception as e:
            logger.error(f"Failed to execute agent {agent_name}: {e}", exc_info=True)
            return None
    
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
            return self._parse_next_agent_decision(response_data, workflow_context)
            
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
        
        **AGENT SELECTION CRITERIA:**
        - **AnalystAgent**: Use for initial analysis, final validation, or when understanding is needed
        - **StrategistAgent**: Use for complex planning, task decomposition, or orchestration
        - **CreatorAgent**: Use when something new needs to be created (code, tests, docs, specs)  
        - **SurgeonAgent**: Use for precise modifications, fixes, or surgical operations
        - **ExecutorAgent**: Use for running tests, validation, builds, or system operations
        - **DebuggingAgent**: Use for systematic debugging and troubleshooting
        
        **RECOMMENDED_INPUTS GUIDANCE:**
        When routing to specific agents, provide structured inputs in recommended_inputs:
        
        - **ExecutorAgent**: For shell operations, provide:
          {{"commands": ["command1", "command2"], "working_directory": "/path", "purpose": "description"}}
          Example Git setup: {{"commands": ["git init", "git config user.name 'Agent User'", "git config user.email 'agent@example.com'"], "working_directory": "{str(global_context.workspace_path)}", "purpose": "Git repository initialization"}}
        
        - **CreatorAgent**: For content creation, provide structured inputs based on creation type:
          
          For **test creation**:
          {{"creation_type": "tests", "test_type": "unit|integration|cli|api|performance", "test_framework": "pytest|unittest|jest", "test_quality": "fast|decent|production", "target_files": ["file1.py"], "requirements": ["test requirement 1", "test requirement 2"]}}
          
          For **code creation**:
          {{"creation_type": "code", "language": "python|javascript|java", "code_type": "function|class|module|library", "file_path": "path/to/file", "requirements": ["code requirement 1", "code requirement 2"]}}
          
          For **documentation creation**:
          {{"creation_type": "documentation", "doc_type": "readme|api_doc|user_guide|technical_spec", "format": "markdown|html|rst", "requirements": ["doc requirement 1", "doc requirement 2"]}}
          
          For **project creation**:
          {{"creation_type": "project", "project_type": "library|cli_app|web_app|api", "language": "python|javascript|java", "structure": ["component1", "component2"], "requirements": ["project requirement 1", "project requirement 2"]}}
        
        - **SurgeonAgent**: For modifications, provide:
          {{"target_files": ["file1.py", "file2.py"], "operation": "fix|refactor|update", "specific_changes": ["change description"]}}
        
        - **Other agents**: Provide relevant context and parameters as appropriate
        
        **COMPLETION DETECTION:**
        If you believe the user's goal has been substantially completed based on the execution history
        and accumulated outputs, set "is_completion" to true and route to AnalystAgent for final validation.
        
        **CRITICAL INSTRUCTIONS:**
        - Analyze what has been done vs. what remains to achieve the user's goal
        - Choose the agent that makes the most direct progress toward completion
        - Provide specific, actionable goals for the next agent
        - If multiple agents could help, choose the one that advances the workflow most significantly
        
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
          {{"commands": ["command1", "command2"], "working_directory": "/path", "purpose": "description"}}
        - **CreatorAgent**: For content creation, provide structured inputs based on creation type:
          
          For **test creation**:
          {{"creation_type": "tests", "test_type": "unit|integration|cli|api|performance", "test_framework": "pytest|unittest|jest", "test_quality": "fast|decent|production", "target_files": ["file1.py"], "requirements": ["test requirement 1", "test requirement 2"]}}
          
          For **code creation**:
          {{"creation_type": "code", "language": "python|javascript|java", "code_type": "function|class|module|library", "file_path": "path/to/file", "requirements": ["code requirement 1", "code requirement 2"]}}
          
          For **documentation creation**:
          {{"creation_type": "documentation", "doc_type": "readme|api_doc|user_guide|technical_spec", "format": "markdown|html|rst", "requirements": ["doc requirement 1", "doc requirement 2"]}}
          
          For **project creation**:
          {{"creation_type": "project", "project_type": "library|cli_app|web_app|api", "language": "python|javascript|java", "structure": ["component1", "component2"], "requirements": ["project requirement 1", "project requirement 2"]}}
        - **SurgeonAgent**: For modifications, provide:
          {{"target_files": ["file1.py"], "operation": "fix|refactor|update", "specific_changes": ["change description"]}}
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