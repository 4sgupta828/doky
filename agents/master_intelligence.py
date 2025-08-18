# agents/master_intelligence.py
import json
import logging
from typing import Dict, Any, List, Literal, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

# Foundational dependencies
from .base import BaseAgent
from core.context import GlobalContext
from core.models import AgentResponse, AgentResult, TaskNode

# Get a logger instance for this module
logger = logging.getLogger(__name__)


# --- Data Models ---

class IntentType(Enum):
    DISCOVERY = "DISCOVERY"          # "What's wrong?", "Find the bug", "Review code"
    CREATION = "CREATION"            # "Add feature X", "Generate tests", "Create docs"
    MODIFICATION = "MODIFICATION"    # "Fix bug in line 45", "Update function", "Refactor"
    TROUBLESHOOTING = "TROUBLESHOOTING"  # Error logs, "App won't start", "Tests failing"
    MAINTENANCE = "MAINTENANCE"      # "Update dependencies", "Setup environment"
    VAGUE = "VAGUE"                 # "Something is broken", "Make it better", unclear requests


class Specificity(Enum):
    HIGH = "HIGH"      # Clear, specific requests with enough detail
    MEDIUM = "MEDIUM"  # Some details but needs clarification
    LOW = "LOW"        # Vague, requires significant clarification


class Urgency(Enum):
    HIGH = "HIGH"      # Production issues, blocking bugs
    MEDIUM = "MEDIUM"  # Important but not critical
    LOW = "LOW"        # Nice to have, optimization


class Scope(Enum):
    PROJECT = "PROJECT"    # Affects entire project
    FILE = "FILE"          # Single file changes
    FUNCTION = "FUNCTION"  # Function-level changes
    LINE = "LINE"          # Specific lines


class Domain(Enum):
    BACKEND = "BACKEND"
    FRONTEND = "FRONTEND" 
    TESTING = "TESTING"
    DEVOPS = "DEVOPS"
    QUALITY = "QUALITY"
    GENERAL = "GENERAL"


class UserExperience(Enum):
    BEGINNER = "BEGINNER"
    INTERMEDIATE = "INTERMEDIATE"
    EXPERT = "EXPERT"


@dataclass
class UserIntent:
    """Comprehensive analysis of user's intent"""
    intent_type: IntentType
    specificity: Specificity
    urgency: Urgency
    scope: Scope
    domain: Domain
    user_experience_level: UserExperience
    requires_clarification: bool
    extracted_entities: Dict[str, Any]
    confidence_score: float
    original_input: str
    processing_timestamp: datetime


class ApproachType(Enum):
    FAST = "FAST"              # Quick, minimal steps for urgent issues
    THOROUGH = "THOROUGH"      # Comprehensive, detailed analysis
    EXPERIMENTAL = "EXPERIMENTAL"  # Try new approaches, learning mode
    CONSERVATIVE = "CONSERVATIVE"  # Safe, proven approaches only


class WorkflowType(Enum):
    LINEAR = "LINEAR"          # Simple A → B → C sequence
    BRANCHING = "BRANCHING"    # Decision points with multiple paths
    ITERATIVE = "ITERATIVE"    # Repeat until success
    PARALLEL = "PARALLEL"      # Multiple agents working simultaneously
    ADAPTIVE = "ADAPTIVE"      # Dynamic workflow that changes based on results


@dataclass
class AgentStep:
    """Individual step in workflow execution"""
    agent_name: str
    goal: str
    inputs: Dict[str, Any]
    expected_outputs: List[str]
    dependencies: List[str]
    optional: bool = False
    timeout: Optional[int] = None
    confidence_threshold: float = 0.7


@dataclass
class SuccessCriterion:
    """Defines what constitutes success"""
    criterion: str
    validation_method: str
    required: bool = True


@dataclass
class LearningGoal:
    """What to learn from this workflow"""
    goal: str
    success_metric: str
    pattern_to_capture: str


@dataclass
class StrategicPlan:
    """Complete plan for achieving user's goal"""
    approach: ApproachType
    workflow_type: WorkflowType
    agent_sequence: List[AgentStep]
    parallel_opportunities: List[List[str]]
    success_criteria: List[SuccessCriterion]
    learning_objectives: List[LearningGoal]
    estimated_duration: Optional[timedelta] = None
    resource_requirements: Dict[str, Any] = None
    fallback_strategies: List[str] = None


# --- LLM Integration ---

class LLMClient:
    """Placeholder for LLM integration"""
    def invoke(self, prompt: str) -> str:
        raise NotImplementedError("LLMClient.invoke must be implemented by a concrete class.")


# --- Agent Implementation ---

class MasterIntelligenceAgent(BaseAgent):
    """
    The supreme decision-maker that understands user intent and creates strategic plans.
    
    This agent is the brain of the system - it analyzes ANY user input and determines
    the optimal approach using LLM-driven intelligence.
    """

    def __init__(self, llm_client: Any = None, agent_registry: Dict[str, Any] = None):
        super().__init__(
            name="MasterIntelligenceAgent",
            description="Supreme decision-maker that analyzes user intent and creates strategic plans for any software development scenario."
        )
        self.llm_client = llm_client or LLMClient()
        self.agent_registry = agent_registry or {}
        
        # Learning storage - in production this would be persistent
        self.learned_patterns = {
            "user_intent_patterns": {},
            "approach_effectiveness": {},
            "failure_predictions": {},
            "success_optimizations": {}
        }

    def required_inputs(self) -> List[str]:
        """Required inputs for MasterIntelligenceAgent execution."""
        return ["user_input"]

    def optional_inputs(self) -> List[str]:
        """Optional inputs for MasterIntelligenceAgent execution."""
        return ["context_history", "user_preferences", "time_constraints", "resource_constraints"]

    def execute_v2(self, goal: str, inputs: Dict[str, Any], global_context: GlobalContext) -> AgentResult:
        """
        NEW INTERFACE: Analyze user intent and create strategic plan.
        """
        logger.info(f"MasterIntelligenceAgent executing: '{goal}'")
        
        # Validate inputs with graceful handling
        try:
            self.validate_inputs(inputs)
        except Exception as validation_error:
            return self.create_result(
                success=False,
                message=str(validation_error),
                error_details={"validation_error": str(validation_error)}
            )
        
        user_input = inputs["user_input"]
        context_history = inputs.get("context_history", {})
        
        try:
            self.report_progress("Analyzing user intent", f"Processing: '{user_input[:60]}...'")
            
            # Phase 1: Analyze user intent
            user_intent = self.analyze_user_input(user_input, global_context)
            
            # Phase 2: Build strategic plan
            strategic_plan = self.build_strategic_plan(user_intent, global_context, context_history)
            
            # Phase 3: Apply learned optimizations
            optimized_plan = self.apply_learned_optimizations(strategic_plan, user_intent)
            
            return self.create_result(
                success=True,
                message=f"Strategic plan created for {user_intent.intent_type.value} request",
                outputs={
                    "user_intent": self._serialize_intent(user_intent),
                    "strategic_plan": self._serialize_plan(optimized_plan),
                    "confidence_score": user_intent.confidence_score
                }
            )
            
        except Exception as e:
            error_msg = f"MasterIntelligenceAgent failed: {e}"
            logger.error(error_msg, exc_info=True)
            return self.create_result(
                success=False,
                message=error_msg,
                error_details={"exception": str(e)}
            )

    def analyze_user_input(self, user_input: str, context: GlobalContext) -> UserIntent:
        """
        Analyze ANY user input and extract comprehensive intent information.
        
        Handles:
        - Explicit requests: "Add login feature", "Fix bug in line 45"
        - Vague requests: "Something is broken", "Make it better"
        - Implicit requests: [Error log paste], "Getting this error..."
        - Context-dependent: "Fix this", "Add tests"
        """
        self.report_thinking("Analyzing user input with LLM to understand intent, scope, and requirements")
        
        try:
            # Build comprehensive analysis prompt
            analysis_prompt = f"""
            You are an expert software development analyst. Analyze the user's request below and extract comprehensive intent information.

            USER INPUT: "{user_input}"
            
            PROJECT CONTEXT:
            - Workspace: {context.workspace_path}
            - Recent artifacts: {list(context.list_artifacts())[:5]}
            - Available agents: {list(self.agent_registry.keys())[:10]}
            
            Analyze this request and return a JSON object with this EXACT structure:
            {{
                "intent_type": "DISCOVERY|CREATION|MODIFICATION|TROUBLESHOOTING|MAINTENANCE|VAGUE",
                "specificity": "HIGH|MEDIUM|LOW",
                "urgency": "HIGH|MEDIUM|LOW", 
                "scope": "PROJECT|FILE|FUNCTION|LINE",
                "domain": "BACKEND|FRONTEND|TESTING|DEVOPS|QUALITY|GENERAL",
                "user_experience_level": "BEGINNER|INTERMEDIATE|EXPERT",
                "requires_clarification": true/false,
                "extracted_entities": {{
                    "mentioned_files": ["file1.py", "file2.js"],
                    "mentioned_functions": ["function_name"],
                    "error_types": ["TypeError", "ImportError"],
                    "technologies": ["pytest", "flask", "react"],
                    "keywords": ["login", "database", "api"]
                }},
                "confidence_score": 0.85,
                "reasoning": "Brief explanation of the analysis",
                "suggested_clarifications": ["What specific functionality?", "Which file?"]
            }}
            
            ANALYSIS GUIDELINES:
            - DISCOVERY: Understanding, investigating, reviewing, analyzing existing code
            - CREATION: Building new features, generating new code, creating from scratch
            - MODIFICATION: Fixing, updating, changing existing code
            - TROUBLESHOOTING: Error logs, broken functionality, debugging
            - MAINTENANCE: Dependencies, environment, setup, configuration
            - VAGUE: Unclear requests needing clarification
            
            Focus on what the user actually wants to achieve.
            """
            
            # Get LLM analysis
            response = self.llm_client.invoke(analysis_prompt)
            analysis_data = json.loads(response)
            
            # Create UserIntent object
            user_intent = UserIntent(
                intent_type=IntentType(analysis_data["intent_type"]),
                specificity=Specificity(analysis_data["specificity"]),
                urgency=Urgency(analysis_data["urgency"]),
                scope=Scope(analysis_data["scope"]),
                domain=Domain(analysis_data["domain"]),
                user_experience_level=UserExperience(analysis_data["user_experience_level"]),
                requires_clarification=analysis_data["requires_clarification"],
                extracted_entities=analysis_data["extracted_entities"],
                confidence_score=analysis_data["confidence_score"],
                original_input=user_input,
                processing_timestamp=datetime.now()
            )
            
            logger.info(f"User intent analyzed: {user_intent.intent_type.value} ({user_intent.confidence_score:.2f} confidence)")
            self.report_progress("Intent analysis complete", 
                               f"{user_intent.intent_type.value} request with {user_intent.specificity.value} specificity")
            
            return user_intent
            
        except Exception as e:
            logger.error(f"Intent analysis failed: {e}")
            # Fallback to basic intent
            return UserIntent(
                intent_type=IntentType.VAGUE,
                specificity=Specificity.LOW,
                urgency=Urgency.MEDIUM,
                scope=Scope.PROJECT,
                domain=Domain.GENERAL,
                user_experience_level=UserExperience.INTERMEDIATE,
                requires_clarification=True,
                extracted_entities={},
                confidence_score=0.1,
                original_input=user_input,
                processing_timestamp=datetime.now()
            )

    def build_strategic_plan(self, intent: UserIntent, context: GlobalContext, 
                           context_history: Dict[str, Any]) -> StrategicPlan:
        """
        Create a comprehensive strategic plan based on user intent and context.
        """
        self.report_thinking("Building strategic plan with LLM based on intent analysis and available agents")
        
        try:
            # Gather context for planning
            planning_context = self._build_planning_context(intent, context, context_history)
            
            # Build planning prompt
            planning_prompt = f"""
            You are a software development workflow architect. Create a strategic plan to achieve the user's goal.

            USER INTENT ANALYSIS:
            {json.dumps(self._serialize_intent(intent), indent=2)}
            
            PLANNING CONTEXT:
            {json.dumps(planning_context, indent=2)}
            
            AVAILABLE AGENTS:
            {json.dumps({name: f"Purpose: {agent.description}" for name, agent in self.agent_registry.items()}, indent=2)}
            
            Create a strategic plan as a JSON object with this EXACT structure:
            {{
                "approach": "FAST|THOROUGH|EXPERIMENTAL|CONSERVATIVE",
                "workflow_type": "LINEAR|BRANCHING|ITERATIVE|PARALLEL|ADAPTIVE",
                "agent_sequence": [
                    {{
                        "agent_name": "AgentName",
                        "goal": "Specific goal for this agent",
                        "inputs": {{"key": "value"}},
                        "expected_outputs": ["output1", "output2"],
                        "dependencies": ["previous_agent_output"],
                        "optional": false,
                        "confidence_threshold": 0.8
                    }}
                ],
                "parallel_opportunities": [["Agent1", "Agent2"], ["Agent3", "Agent4"]],
                "success_criteria": [
                    {{
                        "criterion": "All tests pass",
                        "validation_method": "run_tests",
                        "required": true
                    }}
                ],
                "learning_objectives": [
                    {{
                        "goal": "Learn user preference patterns",
                        "success_metric": "user_satisfaction > 0.8",
                        "pattern_to_capture": "user prefers fast over thorough"
                    }}
                ],
                "estimated_duration_minutes": 15,
                "resource_requirements": {{"llm_calls": 3, "file_operations": 5}},
                "fallback_strategies": ["Manual investigation", "Request clarification"]
            }}
            
            PLANNING GUIDELINES:
            - FAST: Minimal steps, quick results for urgent issues
            - THOROUGH: Comprehensive analysis, detailed validation
            - LINEAR: Simple A→B→C for clear requests
            - PARALLEL: Independent tasks that can run simultaneously
            - Choose agents that actually exist in the registry
            - Be specific about inputs and expected outputs
            - Consider user experience level in complexity choices
            """
            
            # Get LLM planning
            response = self.llm_client.invoke(planning_prompt)
            plan_data = json.loads(response)
            
            # Create StrategicPlan object
            strategic_plan = StrategicPlan(
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
                ],
                estimated_duration=timedelta(minutes=plan_data.get("estimated_duration_minutes", 10)),
                resource_requirements=plan_data.get("resource_requirements", {}),
                fallback_strategies=plan_data.get("fallback_strategies", [])
            )
            
            logger.info(f"Strategic plan created: {strategic_plan.approach.value} {strategic_plan.workflow_type.value} with {len(strategic_plan.agent_sequence)} steps")
            self.report_progress("Strategic plan complete", 
                               f"{strategic_plan.approach.value} approach with {len(strategic_plan.agent_sequence)} agent steps")
            
            return strategic_plan
            
        except Exception as e:
            logger.error(f"Strategic planning failed: {e}")
            # Fallback to basic plan
            return self._create_fallback_plan(intent)

    def apply_learned_optimizations(self, plan: StrategicPlan, intent: UserIntent) -> StrategicPlan:
        """
        Apply learned patterns and optimizations to improve the plan.
        """
        # In production, this would query a persistent learning database
        # For now, apply simple heuristic optimizations
        
        # Example optimization: Fast-track for simple, high-confidence requests
        if (intent.specificity == Specificity.HIGH and 
            intent.confidence_score > 0.8 and 
            intent.intent_type in [IntentType.MODIFICATION, IntentType.TROUBLESHOOTING]):
            
            if plan.approach == ApproachType.THOROUGH:
                plan.approach = ApproachType.FAST
                logger.info("Applied optimization: Switched from THOROUGH to FAST for high-confidence request")
        
        return plan

    def learn_from_outcome(self, plan: StrategicPlan, intent: UserIntent, outcome: Dict[str, Any]):
        """
        Learn from workflow outcomes to improve future decisions.
        """
        # Store pattern for future use
        pattern_key = f"{intent.intent_type.value}_{intent.specificity.value}_{plan.approach.value}"
        
        if pattern_key not in self.learned_patterns["approach_effectiveness"]:
            self.learned_patterns["approach_effectiveness"][pattern_key] = []
        
        self.learned_patterns["approach_effectiveness"][pattern_key].append({
            "success": outcome.get("success", False),
            "user_satisfaction": outcome.get("user_satisfaction", 0.5),
            "execution_time": outcome.get("execution_time", 0),
            "timestamp": datetime.now().isoformat()
        })
        
        logger.info(f"Learned from outcome: {pattern_key} → {outcome.get('success', False)}")

    def _build_planning_context(self, intent: UserIntent, context: GlobalContext, 
                              history: Dict[str, Any]) -> Dict[str, Any]:
        """Build comprehensive context for strategic planning."""
        return {
            "project_info": {
                "workspace_path": context.workspace_path,
                "available_artifacts": list(context.list_artifacts()),
                "recent_activities": history.get("recent_activities", [])
            },
            "user_context": {
                "experience_level": intent.user_experience_level.value,
                "domain_focus": intent.domain.value,
                "urgency_level": intent.urgency.value
            },
            "system_capabilities": {
                "available_agents": list(self.agent_registry.keys()),
                "learned_patterns_count": len(self.learned_patterns["approach_effectiveness"])
            }
        }

    def _create_fallback_plan(self, intent: UserIntent) -> StrategicPlan:
        """Create a basic fallback plan when LLM planning fails."""
        return StrategicPlan(
            approach=ApproachType.CONSERVATIVE,
            workflow_type=WorkflowType.LINEAR,
            agent_sequence=[
                AgentStep(
                    agent_name="ClarifierAgent",
                    goal="Clarify user requirements",
                    inputs={"user_input": intent.original_input},
                    expected_outputs=["clarified_requirements"],
                    dependencies=[]
                )
            ],
            parallel_opportunities=[],
            success_criteria=[
                SuccessCriterion(
                    criterion="User requirements clarified",
                    validation_method="user_confirmation",
                    required=True
                )
            ],
            learning_objectives=[],
            estimated_duration=timedelta(minutes=5),
            fallback_strategies=["Manual assistance required"]
        )

    def _serialize_intent(self, intent: UserIntent) -> Dict[str, Any]:
        """Serialize UserIntent for JSON output."""
        return {
            "intent_type": intent.intent_type.value,
            "specificity": intent.specificity.value,
            "urgency": intent.urgency.value,
            "scope": intent.scope.value,
            "domain": intent.domain.value,
            "user_experience_level": intent.user_experience_level.value,
            "requires_clarification": intent.requires_clarification,
            "extracted_entities": intent.extracted_entities,
            "confidence_score": intent.confidence_score,
            "original_input": intent.original_input,
            "processing_timestamp": intent.processing_timestamp.isoformat()
        }

    def _serialize_plan(self, plan: StrategicPlan) -> Dict[str, Any]:
        """Serialize StrategicPlan for JSON output."""
        return {
            "approach": plan.approach.value,
            "workflow_type": plan.workflow_type.value,
            "agent_sequence": [
                {
                    "agent_name": step.agent_name,
                    "goal": step.goal,
                    "inputs": step.inputs,
                    "expected_outputs": step.expected_outputs,
                    "dependencies": step.dependencies,
                    "optional": step.optional,
                    "confidence_threshold": step.confidence_threshold
                } for step in plan.agent_sequence
            ],
            "parallel_opportunities": plan.parallel_opportunities,
            "success_criteria": [
                {
                    "criterion": sc.criterion,
                    "validation_method": sc.validation_method,
                    "required": sc.required
                } for sc in plan.success_criteria
            ],
            "learning_objectives": [
                {
                    "goal": lg.goal,
                    "success_metric": lg.success_metric,
                    "pattern_to_capture": lg.pattern_to_capture
                } for lg in plan.learning_objectives
            ],
            "estimated_duration_minutes": plan.estimated_duration.total_seconds() / 60 if plan.estimated_duration else None,
            "resource_requirements": plan.resource_requirements,
            "fallback_strategies": plan.fallback_strategies
        }


# Legacy execute method for compatibility
    def execute(self, goal: str, context: GlobalContext, current_task: TaskNode) -> AgentResponse:
        """Legacy execute method for backward compatibility."""
        result = self.execute_v2(
            goal=goal,
            inputs={"user_input": goal},
            global_context=context
        )
        
        return AgentResponse(
            success=result.success,
            message=result.message,
            artifacts_generated=[]
        )