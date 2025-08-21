# fagents/routing.py
"""
LLM-based routing system for foundation agents.

This module provides intelligent routing capabilities that replace hardcoded 
rule-based routing with LLM-based decision making to ensure directional 
progress toward user goals.
"""

import json
import logging
from typing import Dict, Any, List, Optional
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class RoutingDecision(Enum):
    """Possible routing decisions for foundation agents."""
    # AnalystAgent routing
    CODE_ANALYSIS = "code_analysis"
    ENVIRONMENT_ANALYSIS = "environment_analysis"
    PROBLEM_ANALYSIS = "problem_analysis"
    QUALITY_ANALYSIS = "quality_analysis"
    COMPREHENSIVE_ANALYSIS = "comprehensive_analysis"
    
    # CreatorAgent routing
    CODE_CREATION = "code"
    TEST_CREATION = "tests"
    DOCUMENTATION_CREATION = "documentation"
    SPECIFICATION_CREATION = "specification"
    MANIFEST_CREATION = "manifest"
    FULL_PROJECT_CREATION = "full_project"
    
    # StrategistAgent routing
    TASK_PLANNING = "task_planning"
    WORKFLOW_ORCHESTRATION = "workflow_orchestration"
    PLAN_REFINEMENT = "plan_refinement"
    RESOURCE_OPTIMIZATION = "resource_optimization"
    PROGRESS_MONITORING = "progress_monitoring"
    
    # ExecutorAgent routing
    TEST_EXECUTION = "test_execution"
    CODE_VALIDATION = "code_validation"
    SHELL_EXECUTION = "shell_execution"
    FILE_OPERATIONS = "file_operations"
    ENVIRONMENT_SETUP = "environment_setup"
    
    # SurgeonAgent routing
    SCRIPT_EXECUTION = "script_execution"
    CODE_MODIFICATION = "code_modification"
    REQUIREMENTS_MANAGEMENT = "requirements_management"
    CONFIGURATION_MANAGEMENT = "configuration_management"
    DEPENDENCY_UPDATE = "dependency_update"
    PRECISE_REPAIR = "precise_repair"


@dataclass
class RoutingContext:
    """Context information for routing decisions."""
    agent_type: str
    goal: str
    inputs: Dict[str, Any]
    workspace_files: List[str]
    available_capabilities: List[str]
    previous_actions: List[Dict[str, Any]]
    user_intent_indicators: List[str]


@dataclass
class RoutingResult:
    """Result of routing decision with confidence and reasoning."""
    decision: RoutingDecision
    confidence: float
    reasoning: str
    recommended_inputs: Dict[str, Any]


class LLMRouter:
    """
    LLM-based intelligent routing system for foundation agents.
    
    This router uses structured prompts and clear instructions to make
    routing decisions that ensure directional progress toward user goals.
    """
    
    def __init__(self, llm_client: Any = None):
        self.llm_client = llm_client
    
    def route_request(self, context: RoutingContext) -> RoutingResult:
        """
        Route a request to the appropriate handler using LLM-based decision making.
        
        Args:
            context: Complete context for routing decision
            
        Returns:
            RoutingResult with decision, confidence, and reasoning
        """
        if not self.llm_client:
            raise RuntimeError(
                "LLM client is required for routing decisions. "
                "No fallback routing available - please ensure OpenAI LLM Tool is properly configured."
            )
        
        try:
            # Build routing prompt based on agent type
            prompt = self._build_routing_prompt(context)
            
            # Get LLM decision
            response_str = self.llm_client.invoke(prompt)
            
            # Parse response
            response_data = json.loads(response_str)
            
            # Validate and create result
            return self._create_routing_result(response_data, context)
            
        except Exception as e:
            logger.error(f"LLM routing failed: {e}")
            raise RuntimeError(f"LLM routing failed: {e}. No fallback available - please check LLM configuration.")
    
    def _build_routing_prompt(self, context: RoutingContext) -> str:
        """Build agent-specific routing prompt."""
        
        if context.agent_type == "AnalystAgent":
            return self._build_analyst_routing_prompt(context)
        elif context.agent_type == "CreatorAgent":
            return self._build_creator_routing_prompt(context)
        elif context.agent_type == "StrategistAgent":
            return self._build_strategist_routing_prompt(context)
        elif context.agent_type == "ExecutorAgent":
            return self._build_executor_routing_prompt(context)
        elif context.agent_type == "SurgeonAgent":
            return self._build_surgeon_routing_prompt(context)
        else:
            return self._build_generic_routing_prompt(context)
    
    def _build_analyst_routing_prompt(self, context: RoutingContext) -> str:
        """Build routing prompt for AnalystAgent."""
        workspace_context = self._format_workspace_context(context.workspace_files)
        inputs_summary = self._format_inputs_summary(context.inputs)
        
        return f"""
        You are the routing intelligence for the AnalystAgent, a foundational agent specialized in deep comprehension and diagnosis.
        
        Your task is to determine which specific analysis operation will make the most directional progress toward the user's goal.
        
        **User Goal:** "{context.goal}"
        
        **Available Inputs:**
        {inputs_summary}
        
        **Workspace Context:**
        {workspace_context}
        
        **Available Analysis Operations:**
        - **code_analysis**: Analyze code syntax, imports, structure, and validate Python code
        - **environment_analysis**: Analyze development environment, dependencies, system state
        - **problem_analysis**: Diagnose problems, errors, failures, and root cause analysis
        - **quality_analysis**: Assess code quality, security vulnerabilities, and maintainability
        - **comprehensive_analysis**: Run multiple relevant analyses for complete assessment
        
        **Decision Criteria:**
        1. **Directional Progress**: Which operation moves closest to solving the user's actual need?
        2. **Input Alignment**: Which operation best utilizes the available inputs?
        3. **Goal Clarity**: Which operation provides the clearest path to actionable results?
        4. **User Value**: Which operation delivers the most immediate value to the user?
        
        **Critical Instructions:**
        - Choose the operation that makes the most DIRECT progress toward the goal
        - If multiple operations could help, choose the most specific and actionable one
        - If the goal involves error diagnosis or debugging, favor problem_analysis
        - If the goal involves code validation or syntax checking, favor code_analysis
        - If the goal involves security or quality concerns, favor quality_analysis
        - If the goal is broad or unclear, favor comprehensive_analysis
        
        **Response Format:**
        Your response must be a single JSON object with these exact keys:
        {{
            "decision": "code_analysis|environment_analysis|problem_analysis|quality_analysis|comprehensive_analysis",
            "confidence": 0.8,
            "reasoning": "Clear explanation of why this operation makes the most directional progress",
            "recommended_inputs": {{"key": "value"}}
        }}
        
        Analyze the goal and context, then provide your routing decision.
        """
    
    def _build_creator_routing_prompt(self, context: RoutingContext) -> str:
        """Build routing prompt for CreatorAgent."""
        workspace_context = self._format_workspace_context(context.workspace_files)
        inputs_summary = self._format_inputs_summary(context.inputs)
        
        return f"""
        You are the routing intelligence for the CreatorAgent, a foundational agent specialized in generating all types of software development content.
        
        Your task is to determine which specific creation operation will make the most directional progress toward the user's goal.
        
        **User Goal:** "{context.goal}"
        
        **Available Inputs:**
        {inputs_summary}
        
        **Workspace Context:**
        {workspace_context}
        
        **Available Creation Operations:**
        - **code**: Generate code in multiple programming languages
        - **tests**: Generate unit tests, integration tests, CLI tests, etc.
        - **documentation**: Generate README, API docs, user guides, technical documentation
        - **specification**: Create technical specifications from requirements
        - **manifest**: Plan project file structures and generate manifests
        - **full_project**: Create complete project scaffolds with all components
        
        **Decision Criteria:**
        1. **Creation Intent**: What type of content does the user actually want created?
        2. **Scope Analysis**: Is this a single component or a full project creation task?
        3. **Quality Level**: Should this be FAST (simple/prototype), DECENT (standard), or PRODUCTION (enterprise)?
        4. **Input Utilization**: Which operation best uses the provided inputs?
        5. **Development Phase**: What stage of development is this request addressing?
        
        **Critical Instructions:**
        - Choose the operation that directly creates what the user is asking for
        - **Quality Level Selection:**
          * FAST: Simple requests, prototypes, single functions, basic scripts
          * DECENT: Standard development tasks, multi-component features
          * PRODUCTION: Enterprise features, security-critical code, comprehensive systems
        - If the goal mentions "test" or "testing", favor tests creation
        - If the goal mentions "documentation", "readme", or "docs", favor documentation
        - If the goal mentions "spec", "requirements", or "technical specification", favor specification
        - If the goal mentions "project structure" or "scaffold", favor manifest or full_project
        - If the goal is about implementing functionality, favor code creation
        - If multiple creation types are implied, favor full_project
        
        **Response Format:**
        Your response must be a single JSON object with these exact keys:
        {{
            "decision": "code|tests|documentation|specification|manifest|full_project",
            "confidence": 0.9,
            "reasoning": "Clear explanation of what content type the user wants created",
            "recommended_inputs": {{
                "quality": "fast|decent|production",
                "other_key": "value"
            }}
        }}
        
        Analyze the goal and determine what the user wants to create.
        """
    
    def _build_strategist_routing_prompt(self, context: RoutingContext) -> str:
        """Build routing prompt for StrategistAgent."""
        workspace_context = self._format_workspace_context(context.workspace_files)
        inputs_summary = self._format_inputs_summary(context.inputs)
        
        return f"""
        You are the routing intelligence for the StrategistAgent, a foundational agent specialized in task decomposition and intelligent orchestration.
        
        Your task is to determine which specific strategy operation will make the most directional progress toward the user's goal.
        
        **User Goal:** "{context.goal}"
        
        **Available Inputs:**
        {inputs_summary}
        
        **Workspace Context:**
        {workspace_context}
        
        **Available Strategy Operations:**
        - **task_planning**: Decompose goals into executable task graphs and create plans
        - **workflow_orchestration**: Execute complete workflows with multiple agents
        - **plan_refinement**: Refine existing plans based on feedback or new requirements
        - **resource_optimization**: Optimize workflows for parallel execution and efficiency
        - **progress_monitoring**: Monitor and report on workflow execution progress
        
        **Decision Criteria:**
        1. **Planning Stage**: Is this about creating a plan or executing one?
        2. **Complexity Level**: Does this require decomposition or orchestration?
        3. **Execution State**: Are we planning, executing, or monitoring?
        4. **Resource Needs**: Does this require optimization or coordination?
        
        **Critical Instructions:**
        - If the goal involves breaking down or planning a complex task, choose task_planning
        - If there's already a task_graph or plan that needs execution, choose workflow_orchestration
        - If there's an existing plan that needs modification, choose plan_refinement
        - If the goal involves improving efficiency or parallel execution, choose resource_optimization
        - If the goal involves checking status or tracking progress, choose progress_monitoring
        - For new complex goals that need decomposition, always start with task_planning
        
        **Response Format:**
        Your response must be a single JSON object with these exact keys:
        {{
            "decision": "task_planning|workflow_orchestration|plan_refinement|resource_optimization|progress_monitoring",
            "confidence": 0.85,
            "reasoning": "Clear explanation of what strategic operation is needed",
            "recommended_inputs": {{"key": "value"}}
        }}
        
        Analyze the strategic needs and choose the appropriate operation.
        """
    
    def _build_executor_routing_prompt(self, context: RoutingContext) -> str:
        """Build routing prompt for ExecutorAgent."""
        workspace_context = self._format_workspace_context(context.workspace_files)
        inputs_summary = self._format_inputs_summary(context.inputs)
        
        return f"""
        You are the routing intelligence for the ExecutorAgent, a foundational agent specialized in execution operations.
        
        Your task is to determine which specific execution operation will make the most directional progress toward the user's goal.
        
        **User Goal:** "{context.goal}"
        
        **Available Inputs:**
        {inputs_summary}
        
        **Workspace Context:**
        {workspace_context}
        
        **Available Execution Operations:**
        - **test_execution**: Run test suites, analyze results, execute validation tests
        - **code_validation**: Validate code syntax, imports, and execution correctness
        - **shell_execution**: Execute shell commands, build operations, deployment tasks
        - **file_operations**: Read, write, discover, and manage files and directories
        - **environment_setup**: Create environments, install dependencies, setup development
        
        **Decision Criteria:**
        1. **Action Type**: What kind of execution is the user requesting?
        2. **Input Analysis**: What do the inputs suggest about the operation type?
        3. **Goal Verbs**: What verbs indicate the desired action (run, test, validate, build, etc.)?
        4. **System Impact**: What level of system interaction is required?
        
        **Critical Instructions:**
        - If goal mentions "test", "run tests", or "execute tests", choose test_execution
        - If goal mentions "validate", "check syntax", or "verify code", choose code_validation
        - If goal mentions "command", "build", "deploy", "execute", choose shell_execution
        - If goal mentions "file", "read", "write", "copy", "move", choose file_operations
        - If goal mentions "environment", "install", "setup", "dependencies", choose environment_setup
        - Look at inputs: commands→shell_execution, code_files→code_validation, test_target→test_execution
        
        **Response Format:**
        Your response must be a single JSON object with these exact keys:
        {{
            "decision": "test_execution|code_validation|shell_execution|file_operations|environment_setup",
            "confidence": 0.9,
            "reasoning": "Clear explanation of what execution operation is needed",
            "recommended_inputs": {{"key": "value"}}
        }}
        
        Analyze the execution requirements and choose the appropriate operation.
        """
    
    def _build_surgeon_routing_prompt(self, context: RoutingContext) -> str:
        """Build routing prompt for SurgeonAgent."""
        workspace_context = self._format_workspace_context(context.workspace_files)
        inputs_summary = self._format_inputs_summary(context.inputs)
        
        return f"""
        You are the routing intelligence for the SurgeonAgent, a foundational agent specialized in precise modifications and surgical operations.
        
        Your task is to determine which specific surgical operation will make the most directional progress toward the user's goal.
        
        **User Goal:** "{context.goal}"
        
        **Available Inputs:**
        {inputs_summary}
        
        **Workspace Context:**
        {workspace_context}
        
        **Available Surgical Operations:**
        - **script_execution**: Execute structured modification scripts with instructions
        - **code_modification**: Make precise code changes, fixes, and targeted repairs
        - **requirements_management**: Analyze dependencies, manage requirements files
        - **configuration_management**: Create, update, and manage configuration files
        - **dependency_update**: Update project dependencies and package versions
        - **precise_repair**: Multi-step surgical repairs with backup and validation
        
        **Decision Criteria:**
        1. **Modification Type**: What kind of precise change is needed?
        2. **Scope**: Is this a single change or multiple coordinated changes?
        3. **Target**: What is being modified (code, config, dependencies, etc.)?
        4. **Complexity**: How many steps or validations are required?
        
        **Critical Instructions:**
        - If goal mentions "script" or there's an instruction_script input, choose script_execution
        - If goal mentions "requirements", "dependencies", or "packages", choose requirements_management
        - If goal mentions "config", "configuration", or config file formats, choose configuration_management
        - If goal mentions "update dependencies" or version changes, choose dependency_update
        - If goal mentions "repair", "fix multiple", or complex repairs, choose precise_repair
        - For simple code changes or modifications, choose code_modification
        - If multiple operations are needed, choose precise_repair to coordinate them
        
        **Response Format:**
        Your response must be a single JSON object with these exact keys:
        {{
            "decision": "script_execution|code_modification|requirements_management|configuration_management|dependency_update|precise_repair",
            "confidence": 0.9,
            "reasoning": "Clear explanation of what surgical operation is needed",
            "recommended_inputs": {{"key": "value"}}
        }}
        
        Analyze the surgical requirements and choose the appropriate operation.
        """
    
    def _build_generic_routing_prompt(self, context: RoutingContext) -> str:
        """Build generic routing prompt for unknown agent types."""
        return f"""
        You are a routing intelligence system for the {context.agent_type} agent.
        
        **Goal:** "{context.goal}"
        **Inputs:** {json.dumps(context.inputs, indent=2)}
        **Capabilities:** {context.available_capabilities}
        
        Analyze the goal and inputs to determine the most appropriate action.
        
        Response format:
        {{
            "decision": "appropriate_action",
            "confidence": 0.7,
            "reasoning": "explanation of decision",
            "recommended_inputs": {{"key": "value"}}
        }}
        """
    
    def _format_workspace_context(self, workspace_files: List[str]) -> str:
        """Format workspace files for prompt context."""
        if not workspace_files:
            return "No workspace files provided"
        
        # Limit to first 20 files to avoid token limits
        files_subset = workspace_files[:20]
        files_text = "\n".join(f"  - {file}" for file in files_subset)
        
        if len(workspace_files) > 20:
            files_text += f"\n  ... and {len(workspace_files) - 20} more files"
        
        return f"Workspace files ({len(workspace_files)} total):\n{files_text}"
    
    def _format_inputs_summary(self, inputs: Dict[str, Any]) -> str:
        """Format inputs for prompt context."""
        if not inputs:
            return "No inputs provided"
        
        summary_lines = []
        for key, value in inputs.items():
            if isinstance(value, (dict, list)) and len(str(value)) > 100:
                summary_lines.append(f"  - {key}: {type(value).__name__} (length: {len(value)})")
            else:
                summary_lines.append(f"  - {key}: {value}")
        
        return "\n".join(summary_lines)
    
    def _create_routing_result(self, response_data: Dict[str, Any], context: RoutingContext) -> RoutingResult:
        """Create RoutingResult from LLM response."""
        decision_str = response_data.get("decision", "")
        
        # Find matching RoutingDecision enum value
        decision = None
        for route_decision in RoutingDecision:
            if route_decision.value == decision_str:
                decision = route_decision
                break
        
        if not decision:
            # Fail fast on invalid routing decision
            logger.error(f"Invalid routing decision: {decision_str}. Cannot continue without valid LLM routing.")
            raise ValueError(f"Invalid routing decision from LLM: {decision_str}")
        
        return RoutingResult(
            decision=decision,
            confidence=float(response_data.get("confidence", 0.5)),
            reasoning=response_data.get("reasoning", "LLM routing decision"),
            recommended_inputs=response_data.get("recommended_inputs", {})
        )
    


def create_routing_context(agent_type: str, goal: str, inputs: Dict[str, Any], 
                          workspace_files: List[str] = None, 
                          available_capabilities: List[str] = None,
                          previous_actions: List[Dict[str, Any]] = None) -> RoutingContext:
    """Create a RoutingContext for routing decisions."""
    return RoutingContext(
        agent_type=agent_type,
        goal=goal,
        inputs=inputs,
        workspace_files=workspace_files or [],
        available_capabilities=available_capabilities or [],
        previous_actions=previous_actions or [],
        user_intent_indicators=[]
    )


def route_with_llm(llm_client: Any, agent_type: str, goal: str, inputs: Dict[str, Any],
                  workspace_files: List[str] = None) -> RoutingResult:
    """Convenience function for LLM-based routing."""
    router = LLMRouter(llm_client)
    context = create_routing_context(agent_type, goal, inputs, workspace_files)
    return router.route_request(context)