# core/meta_cognition.py
"""
Meta-Cognitive Oversight Systems for Agent Mesh Architecture

This module provides intelligent oversight, completion tracking, loop detection,
and intervention mechanisms for the foundational agent mesh.
"""

import time
import uuid
import logging
from typing import Dict, Any, List, Optional, Set
from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path

from core.models import AgentResult
from core.context import GlobalContext

logger = logging.getLogger(__name__)


class InterventionLevel(Enum):
    """Levels of intervention for stuck workflows"""
    GENTLE_NUDGE = "gentle_nudge"
    WORKFLOW_REDIRECT = "workflow_redirect" 
    FORCED_COORDINATION = "forced_coordination"
    EMERGENCY_COMPLETION = "emergency_completion"


class LoopType(Enum):
    """Types of detected loops in agent collaboration"""
    PING_PONG = "ping_pong_loop"
    RECURSIVE_SELF = "recursive_self_loop"
    CIRCULAR_TASK = "circular_task_pattern"
    EXCESSIVE_ITERATIONS = "excessive_iterations"


@dataclass
class WorkflowState:
    """Complete state tracking for a workflow"""
    workflow_id: str
    user_request: str
    start_time: float
    last_progress_time: float
    current_agent: str
    status: str = "active"
    
    # Progress tracking
    progress_milestones: List[Dict[str, Any]] = field(default_factory=list)
    completion_percentage: float = 0.0
    expected_deliverables: List[str] = field(default_factory=list)
    delivered_artifacts: List[str] = field(default_factory=list)
    
    # Agent collaboration tracking
    agent_handoff_chain: List[str] = field(default_factory=list)
    collaboration_graph: List[Dict[str, Any]] = field(default_factory=list)
    
    # Problem detection
    stuck_warnings: List[Dict[str, Any]] = field(default_factory=list)
    loop_warnings: List[Dict[str, Any]] = field(default_factory=list)
    intervention_history: List[Dict[str, Any]] = field(default_factory=list)


class CompletionTrackingSystem:
    """Ensures every request reaches proper completion"""
    
    def __init__(self):
        self.completion_criteria = {}
        self.validation_rules = {
            "analyst_validation_required": True,
            "deliverables_must_match_request": True,
            "no_critical_errors": True,
            "progress_must_be_meaningful": True
        }
    
    def define_completion_criteria(self, workflow_id: str, user_request: str, initial_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Define what constitutes completion for this specific request"""
        
        criteria = {
            "workflow_id": workflow_id,
            "primary_objectives": self._extract_primary_objectives(user_request),
            "required_deliverables": self._identify_required_deliverables(user_request, initial_analysis),
            "validation_requirements": self._determine_validation_needs(user_request),
            "success_conditions": self._define_success_conditions(user_request, initial_analysis),
            "analyst_final_check": True  # ALWAYS required
        }
        
        self.completion_criteria[workflow_id] = criteria
        logger.info(f"Completion criteria defined for workflow {workflow_id}: {len(criteria['primary_objectives'])} objectives")
        
        return criteria
    
    def validate_completion(self, workflow_id: str, final_result: AgentResult, workflow_state: WorkflowState) -> Dict[str, Any]:
        """Comprehensive completion validation"""
        
        criteria = self.completion_criteria.get(workflow_id, {})
        if not criteria:
            logger.warning(f"No completion criteria found for workflow {workflow_id}")
            return {"is_complete": False, "error": "No completion criteria defined"}
        
        validation_results = {
            "is_complete": False,
            "missing_requirements": [],
            "analyst_validated": False,
            "confidence_score": 0.0,
            "completion_details": {}
        }
        
        try:
            # Check primary objectives
            objectives_met = 0
            for objective in criteria.get("primary_objectives", []):
                if self._objective_satisfied(objective, final_result, workflow_state):
                    objectives_met += 1
                else:
                    validation_results["missing_requirements"].append(f"Objective not met: {objective}")
            
            # Check required deliverables
            deliverables_met = 0
            for deliverable in criteria.get("required_deliverables", []):
                if self._deliverable_present(deliverable, final_result, workflow_state):
                    deliverables_met += 1
                else:
                    validation_results["missing_requirements"].append(f"Missing deliverable: {deliverable}")
            
            # CRITICAL: Ensure Analyst performed final validation
            validation_results["analyst_validated"] = self._analyst_validated_result(final_result, workflow_state)
            if not validation_results["analyst_validated"]:
                validation_results["missing_requirements"].append("Analyst final validation missing")
            
            # Calculate completion score
            total_objectives = len(criteria.get("primary_objectives", []))
            total_deliverables = len(criteria.get("required_deliverables", []))
            
            if total_objectives + total_deliverables > 0:
                validation_results["confidence_score"] = (objectives_met + deliverables_met) / (total_objectives + total_deliverables)
            else:
                validation_results["confidence_score"] = 1.0 if validation_results["analyst_validated"] else 0.0
            
            # Overall completion assessment
            validation_results["is_complete"] = (
                len(validation_results["missing_requirements"]) == 0 and 
                validation_results["analyst_validated"] and
                validation_results["confidence_score"] >= 0.8
            )
            
            validation_results["completion_details"] = {
                "objectives_met": f"{objectives_met}/{total_objectives}",
                "deliverables_met": f"{deliverables_met}/{total_deliverables}",
                "workflow_duration": time.time() - workflow_state.start_time,
                "agent_handoffs": len(workflow_state.agent_handoff_chain),
                "milestones_completed": len(workflow_state.progress_milestones)
            }
            
            logger.info(f"Completion validation for workflow {workflow_id}: {'COMPLETE' if validation_results['is_complete'] else 'INCOMPLETE'}")
            
        except Exception as e:
            logger.error(f"Error validating completion for workflow {workflow_id}: {e}")
            validation_results["missing_requirements"].append(f"Validation error: {e}")
        
        return validation_results
    
    def _extract_primary_objectives(self, user_request: str) -> List[str]:
        """Extract primary objectives from user request"""
        objectives = []
        
        # Keywords that indicate specific objectives
        objective_keywords = {
            "create": "creation",
            "generate": "generation", 
            "build": "construction",
            "implement": "implementation",
            "add": "addition",
            "fix": "repair",
            "debug": "debugging",
            "test": "testing",
            "refactor": "refactoring",
            "optimize": "optimization",
            "analyze": "analysis",
            "review": "review"
        }
        
        request_lower = user_request.lower()
        for keyword, objective_type in objective_keywords.items():
            if keyword in request_lower:
                objectives.append(f"{objective_type}_completed")
        
        # Default objective if none detected
        if not objectives:
            objectives.append("user_request_addressed")
        
        return objectives
    
    def _identify_required_deliverables(self, user_request: str, initial_analysis: Dict[str, Any]) -> List[str]:
        """Identify required deliverables based on request and analysis"""
        deliverables = []
        
        # Check for explicit deliverable mentions in request
        deliverable_keywords = {
            "code": "generated_code",
            "test": "test_files",
            "documentation": "documentation", 
            "spec": "specification",
            "report": "analysis_report",
            "fix": "code_changes",
            "config": "configuration_files"
        }
        
        request_lower = user_request.lower()
        for keyword, deliverable in deliverable_keywords.items():
            if keyword in request_lower:
                deliverables.append(deliverable)
        
        # Add deliverables from initial analysis if available
        if initial_analysis:
            analysis_deliverables = initial_analysis.get("expected_deliverables", [])
            deliverables.extend(analysis_deliverables)
        
        return list(set(deliverables))  # Remove duplicates
    
    def _determine_validation_needs(self, user_request: str) -> List[str]:
        """Determine what validation is needed"""
        validations = ["analyst_final_validation"]  # Always required
        
        if any(word in user_request.lower() for word in ["test", "testing"]):
            validations.append("test_execution")
        
        if any(word in user_request.lower() for word in ["code", "implement", "create"]):
            validations.append("code_validation")
        
        return validations
    
    def _define_success_conditions(self, user_request: str, initial_analysis: Dict[str, Any]) -> List[str]:
        """Define success conditions"""
        conditions = [
            "no_critical_errors",
            "meaningful_progress_made", 
            "analyst_approval"
        ]
        
        # Add specific conditions based on request type
        if "test" in user_request.lower():
            conditions.append("tests_pass")
        
        if "fix" in user_request.lower() or "debug" in user_request.lower():
            conditions.append("problem_resolved")
        
        return conditions
    
    def _objective_satisfied(self, objective: str, result: AgentResult, workflow_state: WorkflowState) -> bool:
        """Check if a specific objective is satisfied"""
        if not result.success:
            return False
        
        # Check based on objective type
        if "creation" in objective or "generation" in objective:
            return any("generated" in key for key in result.outputs.keys()) if result.outputs else False
        
        if "testing" in objective:
            return any("test" in key.lower() for key in result.outputs.keys()) if result.outputs else False
        
        if "analysis" in objective:
            return any("analysis" in key.lower() for key in result.outputs.keys()) if result.outputs else False
        
        # Default: check if we have meaningful outputs
        return bool(result.outputs and len(result.outputs) > 0)
    
    def _deliverable_present(self, deliverable: str, result: AgentResult, workflow_state: WorkflowState) -> bool:
        """Check if a specific deliverable is present"""
        if not result.outputs:
            return False
        
        # Check in result outputs
        deliverable_lower = deliverable.lower()
        for key in result.outputs.keys():
            if deliverable_lower in key.lower():
                return True
        
        # Check in workflow state delivered artifacts
        return deliverable in workflow_state.delivered_artifacts
    
    def _analyst_validated_result(self, result: AgentResult, workflow_state: WorkflowState) -> bool:
        """Verify that Analyst performed final validation"""
        # Check if last agent in chain was Analyst
        if workflow_state.agent_handoff_chain and workflow_state.agent_handoff_chain[-1] == "AnalystAgent":
            return True
        
        # Check if result contains analyst validation markers
        if result.outputs:
            analyst_markers = ["analyst_validation", "final_validation", "completion_validation"]
            for marker in analyst_markers:
                if marker in result.outputs:
                    return True
        
        return False


class LoopDetectionSystem:
    """Advanced loop detection and prevention system"""
    
    def __init__(self, max_iterations=15, max_same_agent_consecutive=4):
        self.max_iterations = max_iterations
        self.max_same_agent_consecutive = max_same_agent_consecutive
        self.workflow_graphs = {}
        self.loop_patterns = {}
    
    def track_agent_interaction(self, workflow_id: str, from_agent: str, to_agent: str, task: str) -> bool:
        """Track agent interaction and detect loops. Returns False if dangerous loop detected."""
        
        if workflow_id not in self.workflow_graphs:
            self.workflow_graphs[workflow_id] = {
                "interactions": [],
                "agent_sequence": [],
                "task_history": [],
                "loop_warnings": []
            }
        
        graph = self.workflow_graphs[workflow_id]
        current_time = time.time()
        
        # Record interaction
        interaction = {
            "from": from_agent,
            "to": to_agent,
            "task": task,
            "timestamp": current_time,
            "step": len(graph["interactions"]) + 1
        }
        
        graph["interactions"].append(interaction)
        graph["agent_sequence"].append(to_agent)
        graph["task_history"].append(task)
        
        # Detect loops
        loop_analysis = self._detect_loops(workflow_id)
        
        if loop_analysis["is_dangerous_loop"]:
            self._record_loop_warning(workflow_id, loop_analysis)
            logger.warning(f"Dangerous loop detected in workflow {workflow_id}: {loop_analysis['loop_type']}")
            return False
        
        return True
    
    def _detect_loops(self, workflow_id: str) -> Dict[str, Any]:
        """Comprehensive loop detection"""
        
        graph = self.workflow_graphs[workflow_id]
        agent_sequence = graph["agent_sequence"]
        task_history = graph["task_history"]
        
        loop_analysis = {
            "is_dangerous_loop": False,
            "loop_type": None,
            "detection_reasons": [],
            "suggested_intervention": None,
            "confidence": 0.0
        }
        
        # 1. Check for excessive iterations
        if len(agent_sequence) > self.max_iterations:
            loop_analysis.update({
                "is_dangerous_loop": True,
                "loop_type": LoopType.EXCESSIVE_ITERATIONS,
                "confidence": 1.0
            })
            loop_analysis["detection_reasons"].append(f"Exceeded {self.max_iterations} agent interactions")
        
        # 2. Check for ping-pong patterns (A→B→A→B)
        if len(agent_sequence) >= 4:
            recent_agents = agent_sequence[-4:]
            if recent_agents[0] == recent_agents[2] and recent_agents[1] == recent_agents[3]:
                loop_analysis.update({
                    "is_dangerous_loop": True,
                    "loop_type": LoopType.PING_PONG,
                    "confidence": 0.9
                })
                loop_analysis["detection_reasons"].append(f"Ping-pong between {recent_agents[0]} and {recent_agents[1]}")
        
        # 3. Check for consecutive same agent calls
        consecutive_same = self._count_consecutive_same_agent(agent_sequence)
        if consecutive_same > self.max_same_agent_consecutive:
            loop_analysis.update({
                "is_dangerous_loop": True,
                "loop_type": LoopType.RECURSIVE_SELF,
                "confidence": 0.8
            })
            loop_analysis["detection_reasons"].append(f"Agent called itself {consecutive_same} times consecutively")
        
        # 4. Check for circular task patterns
        if self._detect_circular_task_pattern(task_history):
            loop_analysis.update({
                "is_dangerous_loop": True,
                "loop_type": LoopType.CIRCULAR_TASK,
                "confidence": 0.7
            })
            loop_analysis["detection_reasons"].append("Circular task pattern detected")
        
        # 5. Suggest intervention
        if loop_analysis["is_dangerous_loop"]:
            loop_analysis["suggested_intervention"] = self._suggest_intervention(loop_analysis["loop_type"])
        
        return loop_analysis
    
    def _count_consecutive_same_agent(self, agent_sequence: List[str]) -> int:
        """Count consecutive calls to the same agent"""
        if not agent_sequence:
            return 0
        
        current_agent = agent_sequence[-1]
        count = 1
        
        for i in range(len(agent_sequence) - 2, -1, -1):
            if agent_sequence[i] == current_agent:
                count += 1
            else:
                break
        
        return count
    
    def _detect_circular_task_pattern(self, task_history: List[str]) -> bool:
        """Detect if tasks are repeating in a circular pattern"""
        if len(task_history) < 6:
            return False
        
        # Check for repeating patterns of length 2-3
        for pattern_length in [2, 3]:
            if len(task_history) >= pattern_length * 2:
                recent_tasks = task_history[-pattern_length * 2:]
                first_half = recent_tasks[:pattern_length]
                second_half = recent_tasks[pattern_length:]
                
                if first_half == second_half:
                    return True
        
        return False
    
    def _suggest_intervention(self, loop_type: LoopType) -> str:
        """Suggest intervention strategy based on loop type"""
        interventions = {
            LoopType.EXCESSIVE_ITERATIONS: "force_analyst_validation",
            LoopType.PING_PONG: "introduce_strategist_coordination",
            LoopType.RECURSIVE_SELF: "prevent_self_delegation", 
            LoopType.CIRCULAR_TASK: "escalate_to_strategist"
        }
        
        return interventions.get(loop_type, "escalate_to_analyst")
    
    def _record_loop_warning(self, workflow_id: str, loop_analysis: Dict[str, Any]):
        """Record loop warning in workflow graph"""
        graph = self.workflow_graphs[workflow_id]
        graph["loop_warnings"].append({
            "timestamp": time.time(),
            "loop_analysis": loop_analysis
        })
    
    def get_workflow_collaboration_summary(self, workflow_id: str) -> Dict[str, Any]:
        """Get summary of agent collaboration for a workflow"""
        graph = self.workflow_graphs.get(workflow_id, {})
        
        if not graph:
            return {"error": "Workflow not found"}
        
        agent_counts = {}
        for agent in graph["agent_sequence"]:
            agent_counts[agent] = agent_counts.get(agent, 0) + 1
        
        return {
            "total_interactions": len(graph["interactions"]),
            "unique_agents_involved": len(set(graph["agent_sequence"])),
            "agent_interaction_counts": agent_counts,
            "loop_warnings": len(graph["loop_warnings"]),
            "most_active_agent": max(agent_counts.items(), key=lambda x: x[1])[0] if agent_counts else None,
            "collaboration_pattern": graph["agent_sequence"][-5:] if len(graph["agent_sequence"]) >= 5 else graph["agent_sequence"]
        }


class ProgressMonitoringSystem:
    """Monitors agent collaboration progress and maintains accountability"""
    
    def __init__(self, progress_timeout=600):  # 10 minutes default
        self.progress_timeout = progress_timeout
        self.workflow_states = {}
        self.stuck_detection_thresholds = {
            "no_progress_timeout": 180,  # 3 minutes without progress
            "same_task_repeated": 3,     # Same task attempted 3 times
            "agent_not_responding": 120,  # Agent silent for 2 minutes
            "low_completion_rate": 0.1   # Less than 10% progress per hour
        }
    
    def initialize_workflow(self, workflow_id: str, user_request: str, initial_analysis: Dict[str, Any] = None) -> WorkflowState:
        """Initialize comprehensive progress tracking for a workflow"""
        
        current_time = time.time()
        workflow_state = WorkflowState(
            workflow_id=workflow_id,
            user_request=user_request,
            start_time=current_time,
            last_progress_time=current_time,
            current_agent="AnalystAgent",
            agent_handoff_chain=["AnalystAgent"]
        )
        
        # Set expected deliverables from initial analysis
        if initial_analysis:
            workflow_state.expected_deliverables = initial_analysis.get("expected_deliverables", [])
        
        self.workflow_states[workflow_id] = workflow_state
        logger.info(f"Initialized progress tracking for workflow {workflow_id}")
        
        return workflow_state
    
    def update_progress(self, workflow_id: str, agent_name: str, task: str, result: AgentResult) -> Dict[str, Any]:
        """Update progress tracking with agent results"""
        
        workflow_state = self.workflow_states.get(workflow_id)
        if not workflow_state:
            logger.warning(f"No workflow state found for {workflow_id}")
            return {"error": "Workflow not found"}
        
        current_time = time.time()
        
        # Update progress milestone
        milestone = {
            "timestamp": current_time,
            "agent": agent_name,
            "task": task,
            "success": result.success,
            "outputs": list(result.outputs.keys()) if result.outputs else [],
            "duration": current_time - workflow_state.last_progress_time,
            "message": result.message
        }
        
        workflow_state.progress_milestones.append(milestone)
        workflow_state.last_progress_time = current_time
        workflow_state.current_agent = agent_name
        
        # Update agent handoff chain
        if not workflow_state.agent_handoff_chain or workflow_state.agent_handoff_chain[-1] != agent_name:
            workflow_state.agent_handoff_chain.append(agent_name)
        
        # Update delivered artifacts
        if result.outputs:
            for artifact_key, artifact_value in result.outputs.items():
                if artifact_key not in workflow_state.delivered_artifacts and artifact_value:
                    workflow_state.delivered_artifacts.append(artifact_key)
        
        # Calculate completion percentage
        workflow_state.completion_percentage = self._calculate_completion_percentage(workflow_state)
        
        # Check for stuck conditions
        stuck_conditions = self._check_for_stuck_conditions(workflow_state)
        
        logger.info(f"Progress updated for workflow {workflow_id}: {workflow_state.completion_percentage:.1f}% complete")
        
        return {
            "workflow_id": workflow_id,
            "completion_percentage": workflow_state.completion_percentage,
            "stuck_conditions": stuck_conditions,
            "current_agent": agent_name,
            "milestone_recorded": True
        }
    
    def _calculate_completion_percentage(self, workflow_state: WorkflowState) -> float:
        """Calculate workflow completion percentage"""
        
        # Method 1: Based on expected deliverables
        if workflow_state.expected_deliverables:
            delivered_count = len(workflow_state.delivered_artifacts)
            expected_count = len(workflow_state.expected_deliverables)
            deliverable_progress = (delivered_count / expected_count) * 100
        else:
            deliverable_progress = 0
        
        # Method 2: Based on milestone progress (each meaningful milestone = progress)
        meaningful_milestones = len([m for m in workflow_state.progress_milestones if m["success"]])
        milestone_progress = min(100, meaningful_milestones * 20)  # Each milestone = 20% max
        
        # Method 3: Time-based progress estimation (diminishing returns over time)
        elapsed_time = time.time() - workflow_state.start_time
        time_progress = min(90, (elapsed_time / 300) * 50)  # 5 minutes = 50% time progress, max 90%
        
        # Weighted combination
        if workflow_state.expected_deliverables:
            completion = (deliverable_progress * 0.6) + (milestone_progress * 0.3) + (time_progress * 0.1)
        else:
            completion = (milestone_progress * 0.7) + (time_progress * 0.3)
        
        return min(100.0, max(0.0, completion))
    
    def _check_for_stuck_conditions(self, workflow_state: WorkflowState) -> List[Dict[str, Any]]:
        """Detect if workflow is stuck and needs intervention"""
        
        current_time = time.time()
        stuck_conditions = []
        
        # 1. No progress timeout
        time_since_progress = current_time - workflow_state.last_progress_time
        if time_since_progress > self.stuck_detection_thresholds["no_progress_timeout"]:
            stuck_conditions.append({
                "type": "no_progress_timeout",
                "details": f"No progress for {time_since_progress:.1f} seconds",
                "severity": "high",
                "suggested_action": "force_progress_check"
            })
        
        # 2. Same task repeated multiple times
        if len(workflow_state.progress_milestones) >= 3:
            recent_tasks = [m["task"] for m in workflow_state.progress_milestones[-3:]]
            if len(set(recent_tasks)) == 1:
                stuck_conditions.append({
                    "type": "task_repetition", 
                    "details": f"Task '{recent_tasks[0]}' repeated {len(recent_tasks)} times",
                    "severity": "medium",
                    "suggested_action": "change_approach"
                })
        
        # 3. Excessive agent handoffs
        if len(workflow_state.agent_handoff_chain) > 10:
            stuck_conditions.append({
                "type": "excessive_handoffs",
                "details": f"{len(workflow_state.agent_handoff_chain)} agent handoffs",
                "severity": "medium",
                "suggested_action": "force_strategist_coordination"
            })
        
        # 4. Low completion rate over time
        elapsed_time = current_time - workflow_state.start_time
        if elapsed_time > 600:  # More than 10 minutes
            expected_progress = (elapsed_time / 3600) * 100  # Expected 100% per hour
            if workflow_state.completion_percentage < expected_progress * self.stuck_detection_thresholds["low_completion_rate"]:
                stuck_conditions.append({
                    "type": "low_completion_rate",
                    "details": f"Only {workflow_state.completion_percentage:.1f}% progress in {elapsed_time/60:.1f} minutes",
                    "severity": "medium",
                    "suggested_action": "escalate_to_analyst"
                })
        
        # 5. Consecutive failed milestones
        if len(workflow_state.progress_milestones) >= 3:
            recent_failures = [m for m in workflow_state.progress_milestones[-3:] if not m["success"]]
            if len(recent_failures) >= 2:
                stuck_conditions.append({
                    "type": "consecutive_failures",
                    "details": f"{len(recent_failures)} consecutive failed attempts",
                    "severity": "high",
                    "suggested_action": "intervention_required"
                })
        
        # Record stuck warnings in workflow state
        for condition in stuck_conditions:
            workflow_state.stuck_warnings.append({
                "timestamp": current_time,
                "condition": condition
            })
        
        return stuck_conditions
    
    def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get comprehensive workflow status report"""
        
        workflow_state = self.workflow_states.get(workflow_id)
        if not workflow_state:
            return {"error": "Workflow not found"}
        
        current_time = time.time()
        
        return {
            "workflow_id": workflow_id,
            "status": workflow_state.status,
            "user_request": workflow_state.user_request,
            "start_time": workflow_state.start_time,
            "total_duration_seconds": current_time - workflow_state.start_time,
            "completion_percentage": workflow_state.completion_percentage,
            "current_agent": workflow_state.current_agent,
            "agent_handoff_count": len(workflow_state.agent_handoff_chain),
            "milestones_completed": len(workflow_state.progress_milestones),
            "successful_milestones": len([m for m in workflow_state.progress_milestones if m["success"]]),
            "deliverables_status": {
                "expected": len(workflow_state.expected_deliverables),
                "delivered": len(workflow_state.delivered_artifacts),
                "missing": list(set(workflow_state.expected_deliverables) - set(workflow_state.delivered_artifacts))
            },
            "recent_activity": workflow_state.progress_milestones[-3:] if len(workflow_state.progress_milestones) >= 3 else workflow_state.progress_milestones,
            "stuck_warnings": len(workflow_state.stuck_warnings),
            "loop_warnings": len(workflow_state.loop_warnings),
            "last_activity_seconds_ago": current_time - workflow_state.last_progress_time,
            "is_healthy": self._assess_workflow_health(workflow_state),
            "collaboration_pattern": workflow_state.agent_handoff_chain
        }
    
    def _assess_workflow_health(self, workflow_state: WorkflowState) -> bool:
        """Assess overall workflow health"""
        
        current_time = time.time()
        health_indicators = []
        
        # Check progress rate
        elapsed_time = current_time - workflow_state.start_time
        if elapsed_time > 0:
            progress_rate = workflow_state.completion_percentage / (elapsed_time / 60)  # Progress per minute
            health_indicators.append(progress_rate > 1.0)  # At least 1% per minute
        
        # Check recent activity
        time_since_progress = current_time - workflow_state.last_progress_time
        health_indicators.append(time_since_progress < self.stuck_detection_thresholds["no_progress_timeout"])
        
        # Check success rate of recent milestones
        if workflow_state.progress_milestones:
            recent_milestones = workflow_state.progress_milestones[-5:]  # Last 5 milestones
            success_rate = len([m for m in recent_milestones if m["success"]]) / len(recent_milestones)
            health_indicators.append(success_rate >= 0.6)  # At least 60% success rate
        
        # Check for excessive warnings
        health_indicators.append(len(workflow_state.stuck_warnings) < 3)
        health_indicators.append(len(workflow_state.loop_warnings) == 0)
        
        # Overall health: majority of indicators must be positive
        return sum(health_indicators) >= len(health_indicators) / 2


class FailSafeInterventionSystem:
    """Intelligent fail-safe mechanisms to rescue stuck workflows"""
    
    def __init__(self):
        self.intervention_history = {}
        self.intervention_strategies = {
            "force_analyst_validation": self._force_analyst_return,
            "introduce_strategist_coordination": self._escalate_to_strategist,
            "prevent_self_delegation": self._prevent_self_delegation,
            "escalate_to_strategist": self._escalate_to_strategist,
            "escalate_to_analyst": self._force_analyst_return,
            "change_approach": self._suggest_approach_change,
            "intervention_required": self._emergency_intervention
        }
    
    async def assess_and_intervene(
        self, 
        workflow_id: str,
        workflow_state: WorkflowState,
        stuck_conditions: List[Dict[str, Any]],
        agents: Dict[str, Any]
    ) -> Optional[AgentResult]:
        """Assess situation and apply appropriate intervention"""
        
        if not stuck_conditions:
            return None
        
        # Determine intervention level
        intervention_level = self._determine_intervention_level(stuck_conditions)
        
        # Select intervention strategy
        primary_condition = max(stuck_conditions, key=lambda x: self._get_severity_score(x["severity"]))
        strategy_name = primary_condition.get("suggested_action", "escalate_to_analyst")
        
        logger.warning(f"Applying intervention for workflow {workflow_id}: {strategy_name} (level: {intervention_level})")
        
        # Apply intervention
        intervention_result = None
        if strategy_name in self.intervention_strategies:
            strategy_function = self.intervention_strategies[strategy_name]
            intervention_result = await strategy_function(workflow_id, workflow_state, stuck_conditions, agents)
        
        # Record intervention
        self._record_intervention(workflow_id, strategy_name, intervention_level, intervention_result)
        
        return intervention_result
    
    def _determine_intervention_level(self, stuck_conditions: List[Dict[str, Any]]) -> InterventionLevel:
        """Determine appropriate intervention level"""
        
        severity_scores = [self._get_severity_score(condition["severity"]) for condition in stuck_conditions]
        max_severity = max(severity_scores)
        condition_count = len(stuck_conditions)
        
        if max_severity >= 3 or condition_count >= 3:
            return InterventionLevel.EMERGENCY_COMPLETION
        elif max_severity >= 2 or condition_count >= 2:
            return InterventionLevel.FORCED_COORDINATION
        elif max_severity >= 1:
            return InterventionLevel.WORKFLOW_REDIRECT
        else:
            return InterventionLevel.GENTLE_NUDGE
    
    def _get_severity_score(self, severity: str) -> int:
        """Convert severity to numeric score"""
        severity_map = {"low": 1, "medium": 2, "high": 3, "critical": 4}
        return severity_map.get(severity, 1)
    
    async def _force_analyst_return(
        self, 
        workflow_id: str,
        workflow_state: WorkflowState,
        stuck_conditions: List[Dict[str, Any]],
        agents: Dict[str, Any]
    ) -> AgentResult:
        """Force return to Analyst for validation"""
        
        analyst = agents.get("AnalystAgent")
        if not analyst:
            return AgentResult(success=False, message="Analyst not available for intervention")
        
        intervention_context = {
            "workflow_id": workflow_id,
            "intervention_reason": "forced_validation",
            "stuck_conditions": stuck_conditions,
            "current_progress": {
                "completion_percentage": workflow_state.completion_percentage,
                "milestones": len(workflow_state.progress_milestones),
                "delivered_artifacts": workflow_state.delivered_artifacts
            },
            "force_completion_check": True
        }
        
        result = await analyst.execute(
            goal="intervention_forced_validation",
            inputs=intervention_context,
            global_context=analyst.global_context
        )
        
        return result
    
    async def _escalate_to_strategist(
        self,
        workflow_id: str,
        workflow_state: WorkflowState,
        stuck_conditions: List[Dict[str, Any]],
        agents: Dict[str, Any]
    ) -> AgentResult:
        """Escalate to Strategist for coordination"""
        
        strategist = agents.get("StrategistAgent")
        if not strategist:
            return AgentResult(success=False, message="Strategist not available for intervention")
        
        escalation_context = {
            "workflow_id": workflow_id,
            "intervention_reason": "workflow_stuck_escalation",
            "original_request": workflow_state.user_request,
            "stuck_conditions": stuck_conditions,
            "current_state": {
                "agent_chain": workflow_state.agent_handoff_chain,
                "progress": workflow_state.completion_percentage,
                "milestones": workflow_state.progress_milestones[-3:] if len(workflow_state.progress_milestones) >= 3 else workflow_state.progress_milestones
            },
            "require_replanning": True
        }
        
        result = await strategist.execute(
            goal="intervention_workflow_rescue",
            inputs=escalation_context,
            global_context=strategist.global_context
        )
        
        return result
    
    async def _prevent_self_delegation(
        self,
        workflow_id: str,
        workflow_state: WorkflowState,
        stuck_conditions: List[Dict[str, Any]],
        agents: Dict[str, Any]
    ) -> AgentResult:
        """Prevent agent from delegating to itself"""
        
        # This is handled by modifying the agent's collaboration behavior
        # For now, escalate to Analyst
        return await self._force_analyst_return(workflow_id, workflow_state, stuck_conditions, agents)
    
    async def _suggest_approach_change(
        self,
        workflow_id: str,
        workflow_state: WorkflowState,
        stuck_conditions: List[Dict[str, Any]],
        agents: Dict[str, Any]
    ) -> AgentResult:
        """Suggest different approach via Strategist"""
        
        return await self._escalate_to_strategist(workflow_id, workflow_state, stuck_conditions, agents)
    
    async def _emergency_intervention(
        self,
        workflow_id: str,
        workflow_state: WorkflowState,
        stuck_conditions: List[Dict[str, Any]],
        agents: Dict[str, Any]
    ) -> AgentResult:
        """Emergency intervention with partial completion"""
        
        analyst = agents.get("AnalystAgent")
        if not analyst:
            return AgentResult(
                success=False, 
                message="Emergency intervention failed: Analyst not available",
                outputs={"emergency_completion": True}
            )
        
        emergency_context = {
            "workflow_id": workflow_id,
            "emergency_completion": True,
            "intervention_reason": "emergency_stuck_rescue",
            "available_progress": {
                "completion_percentage": workflow_state.completion_percentage,
                "delivered_artifacts": workflow_state.delivered_artifacts,
                "successful_milestones": [m for m in workflow_state.progress_milestones if m["success"]]
            },
            "stuck_conditions": stuck_conditions,
            "accept_partial_results": True
        }
        
        result = await analyst.execute(
            goal="emergency_completion_intervention",
            inputs=emergency_context,
            global_context=analyst.global_context
        )
        
        return result
    
    def _record_intervention(
        self, 
        workflow_id: str, 
        strategy: str, 
        level: InterventionLevel, 
        result: Optional[AgentResult]
    ):
        """Record intervention for learning and monitoring"""
        
        if workflow_id not in self.intervention_history:
            self.intervention_history[workflow_id] = []
        
        intervention_record = {
            "timestamp": time.time(),
            "strategy": strategy,
            "level": level.value,
            "success": result.success if result else False,
            "message": result.message if result else "No result"
        }
        
        self.intervention_history[workflow_id].append(intervention_record)
        logger.info(f"Intervention recorded for workflow {workflow_id}: {strategy} ({'SUCCESS' if intervention_record['success'] else 'FAILED'})")