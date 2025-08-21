# core/agent_collaboration.py
"""
Enhanced Agent Collaboration Framework for Meta-Cognitive Agent Mesh

This module provides the foundational collaboration capabilities that enable
agents to communicate and coordinate with each other intelligently.
"""

import asyncio
import time
import logging
from typing import Dict, Any, List, Optional, Set
from abc import ABC, abstractmethod

from core.models import AgentResult
from core.context import GlobalContext
from core.meta_cognition import (
    LoopDetectionSystem, ProgressMonitoringSystem, 
    CompletionTrackingSystem, FailSafeInterventionSystem,
    WorkflowState
)

logger = logging.getLogger(__name__)


class AgentCapability:
    """Represents a specific capability of an agent"""
    
    def __init__(self, name: str, description: str, confidence: float = 1.0, keywords: List[str] = None):
        self.name = name
        self.description = description
        self.confidence = confidence
        self.keywords = keywords or []
        self.usage_count = 0
        self.success_rate = 1.0


class CollaborationMixin:
    """
    Mixin class that provides meta-cognitive collaboration capabilities to any agent.
    
    This mixin enables:
    - Intelligent peer discovery and routing
    - Loop-aware delegation
    - Progress tracking integration
    - Meta-cognitive oversight compliance
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.peer_agents = {}
        self.capabilities_index = {}
        self.collaboration_history = []
        self.meta_cognition_enabled = False
        
        # Meta-cognitive systems (injected during setup)
        self.loop_detector = None
        self.progress_monitor = None
        self.completion_tracker = None
        self.intervention_system = None
    
    def setup_meta_cognition(
        self,
        loop_detector: LoopDetectionSystem,
        progress_monitor: ProgressMonitoringSystem,
        completion_tracker: CompletionTrackingSystem,
        intervention_system: FailSafeInterventionSystem,
        global_context: GlobalContext
    ):
        """Setup meta-cognitive oversight systems"""
        self.loop_detector = loop_detector
        self.progress_monitor = progress_monitor
        self.completion_tracker = completion_tracker
        self.intervention_system = intervention_system
        self.global_context = global_context
        self.meta_cognition_enabled = True
        
        logger.debug(f"Meta-cognition enabled for {self.__class__.__name__}")
    
    def register_peer_network(self, agents: Dict[str, Any]):
        """Register network of peer agents for collaboration"""
        # Filter out self from peer network
        self.peer_agents = {name: agent for name, agent in agents.items() 
                           if name != self.__class__.__name__ and agent != self}
        
        # Build capability index for intelligent routing
        self._build_capability_index()
        
        logger.info(f"{self.__class__.__name__} registered with {len(self.peer_agents)} peer agents")
    
    def _build_capability_index(self):
        """Build searchable index of agent capabilities"""
        self.capabilities_index = {}
        
        for agent_name, agent in self.peer_agents.items():
            try:
                if hasattr(agent, 'get_capabilities'):
                    capabilities_info = agent.get_capabilities()
                    
                    # Index by primary functions
                    primary_functions = capabilities_info.get("primary_functions", [])
                    for function in primary_functions:
                        if function not in self.capabilities_index:
                            self.capabilities_index[function] = []
                        
                        capability = AgentCapability(
                            name=function,
                            description=capabilities_info.get("description", ""),
                            keywords=function.lower().split()
                        )
                        self.capabilities_index[function].append((agent_name, capability))
                    
                    # Index by agent type keywords
                    description = capabilities_info.get("description", "").lower()
                    agent_keywords = description.split()
                    
                    for keyword in agent_keywords:
                        if len(keyword) > 3:  # Skip short words
                            if keyword not in self.capabilities_index:
                                self.capabilities_index[keyword] = []
                            
                            capability = AgentCapability(
                                name=f"{agent_name}_general",
                                description=capabilities_info.get("description", ""),
                                keywords=[keyword]
                            )
                            self.capabilities_index[keyword].append((agent_name, capability))
                
            except Exception as e:
                logger.warning(f"Failed to index capabilities for {agent_name}: {e}")
        
        logger.debug(f"Capability index built with {len(self.capabilities_index)} categories")
    
    async def smart_delegate(
        self, 
        task_description: str, 
        inputs: Dict[str, Any], 
        workflow_id: str = None,
        exclude_agents: Set[str] = None
    ) -> AgentResult:
        """
        Intelligently delegate task to most suitable peer agent with meta-cognitive oversight.
        """
        
        # Find best agent for task
        best_agent_name = self._find_best_agent_for_task(task_description, exclude_agents)
        
        if not best_agent_name:
            return AgentResult(
                success=False,
                message=f"No suitable agent found for task: {task_description}",
                outputs={"delegation_failed": True}
            )
        
        # Check for loops if meta-cognition is enabled
        if workflow_id and self.meta_cognition_enabled:
            can_proceed = self.loop_detector.track_agent_interaction(
                workflow_id, 
                self.__class__.__name__, 
                best_agent_name, 
                task_description
            )
            
            if not can_proceed:
                logger.warning(f"Loop detected: {self.__class__.__name__} → {best_agent_name}")
                return AgentResult(
                    success=False,
                    message=f"Collaboration blocked due to loop detection: {self.__class__.__name__} → {best_agent_name}",
                    outputs={"loop_prevented": True, "blocked_agent": best_agent_name}
                )
        
        # Proceed with delegation
        peer_agent = self.peer_agents[best_agent_name]
        
        # Add collaboration context to inputs
        enhanced_inputs = {
            **inputs,
            "workflow_id": workflow_id,
            "delegated_from": self.__class__.__name__,
            "delegation_reason": f"Best suited for: {task_description}",
            "meta_cognition_enabled": self.meta_cognition_enabled
        }
        
        try:
            # Execute with peer agent
            if hasattr(peer_agent, 'execute_with_oversight') and self.meta_cognition_enabled:
                result = await peer_agent.execute_with_oversight(
                    task_description, enhanced_inputs, self.global_context
                )
            else:
                result = await peer_agent.execute(
                    task_description, enhanced_inputs, self.global_context
                )
            
            # Update progress monitoring if enabled
            if workflow_id and self.meta_cognition_enabled:
                self.progress_monitor.update_progress(workflow_id, best_agent_name, task_description, result)
            
            # Track collaboration
            self._record_collaboration(best_agent_name, task_description, result)
            
            logger.info(f"Delegation successful: {self.__class__.__name__} → {best_agent_name} ({'SUCCESS' if result.success else 'FAILED'})")
            
            return result
            
        except Exception as e:
            error_result = AgentResult(
                success=False,
                message=f"Delegation failed: {e}",
                outputs={"delegation_error": str(e)}
            )
            
            # Still track the failed collaboration
            self._record_collaboration(best_agent_name, task_description, error_result)
            
            return error_result
    
    def _find_best_agent_for_task(self, task_description: str, exclude_agents: Set[str] = None) -> Optional[str]:
        """Find the best agent for a specific task using intelligent matching"""
        
        if exclude_agents is None:
            exclude_agents = set()
        
        task_words = set(task_description.lower().split())
        agent_scores = {}
        
        # Score agents based on capability relevance
        for capability_key, agent_capabilities in self.capabilities_index.items():
            capability_words = set(capability_key.lower().split())
            
            # Calculate word overlap between task and capability
            overlap = len(task_words.intersection(capability_words))
            
            if overlap > 0:
                for agent_name, capability in agent_capabilities:
                    if agent_name in exclude_agents:
                        continue
                    
                    # Base score from word overlap
                    base_score = overlap
                    
                    # Boost score based on capability confidence and success rate
                    confidence_boost = capability.confidence * capability.success_rate
                    
                    # Penalty for overused agents (encourage distribution)
                    usage_penalty = capability.usage_count * 0.1
                    
                    final_score = base_score + confidence_boost - usage_penalty
                    
                    if agent_name not in agent_scores:
                        agent_scores[agent_name] = 0
                    
                    agent_scores[agent_name] += final_score
        
        # Return agent with highest score
        if agent_scores:
            best_agent = max(agent_scores.items(), key=lambda x: x[1])
            logger.debug(f"Best agent for '{task_description}': {best_agent[0]} (score: {best_agent[1]:.2f})")
            return best_agent[0]
        
        # Fallback: return first available agent not in exclude list
        available_agents = [name for name in self.peer_agents.keys() if name not in exclude_agents]
        return available_agents[0] if available_agents else None
    
    async def multi_agent_consultation(
        self, 
        question: str, 
        consulting_agents: List[str] = None,
        workflow_id: str = None
    ) -> Dict[str, AgentResult]:
        """
        Consult multiple agents for different perspectives on a question.
        """
        
        if not consulting_agents:
            # Consult up to 3 most relevant agents
            consulting_agents = list(self.peer_agents.keys())[:3]
        
        consultation_results = {}
        consultation_tasks = []
        
        # Create consultation tasks
        for agent_name in consulting_agents:
            if agent_name in self.peer_agents:
                consultation_inputs = {
                    "consultation_question": question,
                    "consultation_type": "peer_consultation",
                    "workflow_id": workflow_id,
                    "requesting_agent": self.__class__.__name__
                }
                
                task = self.smart_delegate(
                    f"provide_consultation: {question}",
                    consultation_inputs,
                    workflow_id
                )
                consultation_tasks.append((agent_name, task))
        
        # Gather results
        for agent_name, task in consultation_tasks:
            try:
                result = await task
                consultation_results[agent_name] = result
            except Exception as e:
                consultation_results[agent_name] = AgentResult(
                    success=False,
                    message=f"Consultation failed: {e}",
                    outputs={"consultation_error": str(e)}
                )
        
        logger.info(f"Multi-agent consultation completed: {len(consultation_results)} responses")
        return consultation_results
    
    async def coordinated_workflow(
        self, 
        workflow_steps: List[Dict[str, Any]], 
        workflow_id: str = None
    ) -> List[AgentResult]:
        """
        Execute a coordinated workflow across multiple agents.
        
        workflow_steps format:
        [
            {"agent": "agent_name", "task": "task_description", "inputs": {...}},
            {"agent": "auto", "task": "task_description", "inputs": {...}}  # auto-select agent
        ]
        """
        
        workflow_results = []
        accumulated_context = {}
        
        for step_index, step in enumerate(workflow_steps):
            step_agent = step.get("agent")
            step_task = step.get("task")
            step_inputs = {**step.get("inputs", {}), **accumulated_context}
            
            try:
                if step_agent == "auto":
                    # Auto-select best agent
                    result = await self.smart_delegate(step_task, step_inputs, workflow_id)
                else:
                    # Specific agent requested
                    if step_agent in self.peer_agents:
                        peer_agent = self.peer_agents[step_agent]
                        
                        enhanced_inputs = {
                            **step_inputs,
                            "workflow_id": workflow_id,
                            "workflow_step": step_index + 1,
                            "total_steps": len(workflow_steps)
                        }
                        
                        result = await peer_agent.execute(
                            step_task, enhanced_inputs, self.global_context
                        )
                    else:
                        result = AgentResult(
                            success=False,
                            message=f"Requested agent not available: {step_agent}",
                            outputs={"missing_agent": step_agent}
                        )
                
                workflow_results.append(result)
                
                # Accumulate context from successful steps
                if result.success and result.outputs:
                    accumulated_context.update(result.outputs)
                
                # Stop workflow on critical failures
                if not result.success and step.get("critical", False):
                    logger.error(f"Critical workflow step failed: {step_task}")
                    break
                    
            except Exception as e:
                error_result = AgentResult(
                    success=False,
                    message=f"Workflow step failed: {e}",
                    outputs={"step_error": str(e)}
                )
                workflow_results.append(error_result)
                
                if step.get("critical", False):
                    break
        
        logger.info(f"Coordinated workflow completed: {len(workflow_results)} steps executed")
        return workflow_results
    
    def _record_collaboration(self, agent_name: str, task: str, result: AgentResult):
        """Record collaboration for learning and optimization"""
        
        collaboration_record = {
            "timestamp": time.time(),
            "peer_agent": agent_name,
            "task": task,
            "success": result.success,
            "message": result.message[:100] if result.message else "",  # Truncate for storage
            "outputs_count": len(result.outputs) if result.outputs else 0
        }
        
        self.collaboration_history.append(collaboration_record)
        
        # Update capability success rates
        for capability_key, agent_capabilities in self.capabilities_index.items():
            for stored_agent_name, capability in agent_capabilities:
                if stored_agent_name == agent_name:
                    capability.usage_count += 1
                    
                    # Update success rate with exponential moving average
                    alpha = 0.1  # Learning rate
                    new_success = 1.0 if result.success else 0.0
                    capability.success_rate = (1 - alpha) * capability.success_rate + alpha * new_success
        
        # Keep history manageable
        if len(self.collaboration_history) > 100:
            self.collaboration_history = self.collaboration_history[-50:]
    
    def get_collaboration_stats(self) -> Dict[str, Any]:
        """Get collaboration statistics and performance metrics"""
        
        if not self.collaboration_history:
            return {"total_collaborations": 0}
        
        total_collaborations = len(self.collaboration_history)
        successful_collaborations = len([c for c in self.collaboration_history if c["success"]])
        
        # Agent collaboration frequency
        agent_frequency = {}
        for collab in self.collaboration_history:
            agent = collab["peer_agent"]
            agent_frequency[agent] = agent_frequency.get(agent, 0) + 1
        
        # Recent collaboration trend (last 10)
        recent_collaborations = self.collaboration_history[-10:]
        recent_success_rate = len([c for c in recent_collaborations if c["success"]]) / len(recent_collaborations)
        
        return {
            "total_collaborations": total_collaborations,
            "overall_success_rate": successful_collaborations / total_collaborations,
            "recent_success_rate": recent_success_rate,
            "most_collaborated_agents": sorted(agent_frequency.items(), key=lambda x: x[1], reverse=True)[:3],
            "unique_peer_agents": len(set(c["peer_agent"] for c in self.collaboration_history)),
            "average_collaborations_per_agent": total_collaborations / len(self.peer_agents) if self.peer_agents else 0
        }


class MetaCognitiveFoundationalAgent(CollaborationMixin):
    """
    Enhanced base class for foundational agents with meta-cognitive collaboration capabilities.
    
    This class provides:
    - Meta-cognitive oversight integration
    - Advanced collaboration capabilities  
    - Loop detection and intervention handling
    - Progress tracking and reporting
    """
    
    def __init__(self, name: str, description: str):
        super().__init__()
        self.name = name
        self.description = description
        self.execution_context = {}
    
    @abstractmethod
    async def execute(self, goal: str, inputs: Dict[str, Any], global_context: GlobalContext) -> AgentResult:
        """Execute the agent's core functionality"""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> Dict[str, Any]:
        """Return the agent's capabilities"""
        pass
    
    async def execute_with_oversight(
        self, 
        goal: str, 
        inputs: Dict[str, Any], 
        global_context: GlobalContext
    ) -> AgentResult:
        """
        Execute with full meta-cognitive oversight enabled.
        """
        
        workflow_id = inputs.get("workflow_id")
        
        if workflow_id and self.meta_cognition_enabled:
            # Check for intervention needs before execution
            workflow_state = self.progress_monitor.workflow_states.get(workflow_id)
            
            if workflow_state:
                stuck_conditions = self.progress_monitor._check_for_stuck_conditions(workflow_state)
                
                if stuck_conditions:
                    # Apply intervention if needed
                    intervention_result = await self.intervention_system.assess_and_intervene(
                        workflow_id, workflow_state, stuck_conditions, self.peer_agents
                    )
                    
                    if intervention_result:
                        logger.info(f"Intervention applied for {self.name}: {intervention_result.message}")
                        return intervention_result
        
        # Execute normally
        try:
            result = await self.execute(goal, inputs, global_context)
            
            # Update progress if workflow tracking is enabled
            if workflow_id and self.meta_cognition_enabled:
                self.progress_monitor.update_progress(workflow_id, self.name, goal, result)
            
            return result
            
        except Exception as e:
            error_result = AgentResult(
                success=False,
                message=f"Agent execution failed: {e}",
                outputs={"execution_error": str(e)}
            )
            
            # Still update progress to track the failure
            if workflow_id and self.meta_cognition_enabled:
                self.progress_monitor.update_progress(workflow_id, self.name, goal, error_result)
            
            return error_result
    
    def report_progress(self, phase: str, message: str):
        """Report progress for user visibility"""
        logger.info(f"[{self.name}] {phase}: {message}")
    
    def report_error(self, message: str, exception: Exception = None):
        """Report errors with proper logging"""
        logger.error(f"[{self.name}] ERROR: {message}", exc_info=exception)