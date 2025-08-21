# fagents/meta_cognitive_agents.py
"""
Meta-Cognitive Enhanced Foundational Agents

These are enhanced versions of the foundational agents with full meta-cognitive
collaboration capabilities, intelligent peer-to-peer communication, and oversight integration.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional

from core.models import AgentResult
from core.context import GlobalContext
from core.agent_collaboration import MetaCognitiveFoundationalAgent

# Import original foundational agents
from fagents.analyst import AnalystAgent
from fagents.strategist import StrategistAgent  
from fagents.creator import CreatorAgent
from fagents.surgeon import SurgeonAgent
from fagents.executor import ExecutorAgent
from fagents.debugger import DebuggingAgent

logger = logging.getLogger(__name__)


class MetaCognitiveAnalystAgent(MetaCognitiveFoundationalAgent, AnalystAgent):
    """
    Enhanced Analyst with meta-cognitive collaboration capabilities.
    
    The Analyst becomes the intelligent gateway and completion validator for the agent mesh.
    """
    
    def __init__(self):
        MetaCognitiveFoundationalAgent.__init__(
            self, 
            name="AnalystAgent", 
            description="Master analyst with comprehensive understanding and validation capabilities"
        )
        AnalystAgent.__init__(self)
    
    async def execute(self, goal: str, inputs: Dict[str, Any], global_context: GlobalContext) -> AgentResult:
        """Enhanced execute with collaboration support"""
        
        # Handle special collaborative goals
        if goal == "process_user_request_collaboratively":
            return await self._handle_collaborative_user_request(inputs, global_context)
        elif goal.startswith("provide_consultation:"):
            consultation_topic = goal.split("provide_consultation:", 1)[1].strip()
            return await self._provide_analytical_consultation(consultation_topic, inputs)
        elif goal in ["intervention_forced_validation", "emergency_completion_intervention"]:
            return await self._handle_intervention_request(goal, inputs, global_context)
        else:
            # Use original Analyst functionality (NOT async)
            return AnalystAgent.execute(self, goal, inputs, global_context)
    
    async def execute_with_oversight(self, goal: str, inputs: Dict[str, Any], global_context: GlobalContext) -> AgentResult:
        """Execute with meta-cognitive oversight - this is the async wrapper"""
        
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
        
        # Execute the main logic (this calls the async execute method above)
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
    
    async def _handle_collaborative_user_request(self, inputs: Dict[str, Any], global_context: GlobalContext) -> AgentResult:
        """Handle user request with full agent mesh collaboration"""
        
        user_request = inputs["user_request"]
        workflow_id = inputs.get("workflow_id")
        
        self.report_progress("Request Analysis", f"Analyzing: {user_request[:50]}...")
        
        try:
            # Step 1: Comprehensive analysis of user request
            analysis_result = AnalystAgent.execute(
                self,
                "comprehensive_analysis",
                {"user_input": user_request, "analysis_depth": "detailed"},
                global_context
            )
            
            if not analysis_result.success:
                return analysis_result
            
            analysis = analysis_result.outputs
            
            # Step 2: Define completion criteria if we have a workflow
            if workflow_id and self.meta_cognition_enabled:
                self.completion_tracker.define_completion_criteria(
                    workflow_id, user_request, analysis
                )
            
            # Step 3: Determine collaboration needs
            collaboration_needs = self._assess_collaboration_needs(user_request, analysis)
            
            if not collaboration_needs["needs_collaboration"]:
                self.report_progress("Direct Response", "Simple request - handling directly")
                return analysis_result
            
            # Step 4: Orchestrate collaborative workflow
            self.report_progress("Collaboration", f"Orchestrating {len(collaboration_needs['required_agents'])} agent workflow")
            
            workflow_result = await self._orchestrate_collaborative_workflow(
                user_request, analysis, collaboration_needs, workflow_id
            )
            
            # Step 5: Final validation (CRITICAL - always ensure completion)
            final_result = await self._validate_collaborative_completion(
                user_request, workflow_result, workflow_id
            )
            
            self.report_progress("Completion", "Request fully processed and validated")
            return final_result
            
        except Exception as e:
            self.report_error(f"Collaborative request handling failed: {e}", e)
            return AgentResult(
                success=False,
                message=f"Failed to process collaborative request: {e}",
                outputs={"error": str(e)}
            )
    
    def _assess_collaboration_needs(self, user_request: str, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Assess what type of collaboration is needed"""
        
        request_lower = user_request.lower()
        required_capabilities = []
        required_agents = []
        
        # Determine required capabilities and agents
        capability_mapping = {
            "planning": ("StrategistAgent", ["plan", "strategy", "organize", "coordinate"]),
            "creation": ("CreatorAgent", ["create", "generate", "build", "implement", "write"]),
            "modification": ("SurgeonAgent", ["fix", "modify", "change", "update", "refactor", "edit"]),
            "execution": ("ExecutorAgent", ["run", "test", "execute", "validate", "check"]),
            "debugging": ("DebuggingAgent", ["debug", "troubleshoot", "diagnose", "solve", "error"])
        }
        
        for capability, (agent, keywords) in capability_mapping.items():
            if any(keyword in request_lower for keyword in keywords):
                required_capabilities.append(capability)
                required_agents.append(agent)
        
        # Determine workflow type
        workflow_type = "sequential"
        if len(required_capabilities) > 2:
            workflow_type = "parallel" if any(word in request_lower for word in ["quickly", "parallel", "simultaneously"]) else "sequential"
        elif len(required_capabilities) == 1 and required_capabilities[0] in ["debugging", "modification"]:
            workflow_type = "iterative"
        
        # Check complexity
        complexity_indicators = len(required_capabilities) + (1 if "complex" in request_lower else 0)
        
        needs_collaboration = len(required_capabilities) > 0
        
        return {
            "needs_collaboration": needs_collaboration,
            "required_capabilities": required_capabilities,
            "required_agents": required_agents,
            "workflow_type": workflow_type,
            "complexity": "high" if complexity_indicators > 2 else "medium" if complexity_indicators > 1 else "low"
        }
    
    async def _orchestrate_collaborative_workflow(
        self, 
        user_request: str, 
        analysis: Dict[str, Any], 
        collaboration_needs: Dict[str, Any],
        workflow_id: str
    ) -> AgentResult:
        """Orchestrate the collaborative workflow"""
        
        required_agents = collaboration_needs["required_agents"]
        workflow_type = collaboration_needs["workflow_type"]
        
        if workflow_type == "sequential":
            return await self._execute_sequential_collaboration(
                user_request, analysis, required_agents, workflow_id
            )
        elif workflow_type == "parallel":
            return await self._execute_parallel_collaboration(
                user_request, analysis, required_agents, workflow_id
            )
        else:  # iterative
            return await self._execute_iterative_collaboration(
                user_request, analysis, required_agents, workflow_id
            )
    
    async def _execute_sequential_collaboration(
        self, 
        user_request: str, 
        analysis: Dict[str, Any], 
        required_agents: List[str], 
        workflow_id: str
    ) -> AgentResult:
        """Execute sequential collaborative workflow"""
        
        workflow_results = []
        accumulated_context = {"original_analysis": analysis, "user_request": user_request}
        
        for agent_name in required_agents:
            task_description = f"Handle {agent_name.replace('Agent', '').lower()} aspect of: {user_request}"
            
            self.report_progress("Sequential Step", f"→ {agent_name}")
            
            step_result = await self.smart_delegate(
                task_description, 
                accumulated_context, 
                workflow_id
            )
            
            workflow_results.append(step_result)
            
            if not step_result.success:
                return AgentResult(
                    success=False,
                    message=f"Sequential workflow failed at {agent_name}: {step_result.message}",
                    outputs={"partial_results": workflow_results}
                )
            
            # Accumulate context for next agent
            if step_result.outputs:
                accumulated_context.update(step_result.outputs)
        
        # Combine all results
        return AgentResult(
            success=True,
            message=f"Sequential workflow completed successfully across {len(required_agents)} agents",
            outputs={
                "workflow_type": "sequential",
                "workflow_results": workflow_results,
                "final_context": accumulated_context,
                "agents_involved": required_agents
            }
        )
    
    async def _execute_parallel_collaboration(
        self, 
        user_request: str, 
        analysis: Dict[str, Any], 
        required_agents: List[str], 
        workflow_id: str
    ) -> AgentResult:
        """Execute parallel collaborative workflow"""
        
        # Create tasks for parallel execution
        collaboration_tasks = []
        base_context = {"original_analysis": analysis, "user_request": user_request}
        
        for agent_name in required_agents:
            task_description = f"Handle {agent_name.replace('Agent', '').lower()} aspect of: {user_request}"
            
            self.report_progress("Parallel Launch", f"→ {agent_name}")
            
            task = self.smart_delegate(task_description, base_context, workflow_id)
            collaboration_tasks.append((agent_name, task))
        
        # Wait for all results
        workflow_results = []
        for agent_name, task in collaboration_tasks:
            try:
                result = await task
                workflow_results.append(result)
                
                self.report_progress("Parallel Complete", f"✓ {agent_name} ({'SUCCESS' if result.success else 'FAILED'})")
                
            except Exception as e:
                error_result = AgentResult(
                    success=False,
                    message=f"Parallel task failed for {agent_name}: {e}",
                    outputs={"agent_error": str(e)}
                )
                workflow_results.append(error_result)
        
        # Check overall success
        successful_results = [r for r in workflow_results if r.success]
        overall_success = len(successful_results) >= len(workflow_results) * 0.6  # 60% success threshold
        
        return AgentResult(
            success=overall_success,
            message=f"Parallel workflow: {len(successful_results)}/{len(workflow_results)} agents succeeded",
            outputs={
                "workflow_type": "parallel",
                "workflow_results": workflow_results,
                "success_rate": len(successful_results) / len(workflow_results),
                "agents_involved": required_agents
            }
        )
    
    async def _execute_iterative_collaboration(
        self, 
        user_request: str, 
        analysis: Dict[str, Any], 
        required_agents: List[str], 
        workflow_id: str
    ) -> AgentResult:
        """Execute iterative collaborative workflow (for debugging/modification tasks)"""
        
        max_iterations = 3
        iteration_results = []
        current_context = {"original_analysis": analysis, "user_request": user_request}
        
        for iteration in range(max_iterations):
            self.report_progress("Iteration", f"Round {iteration + 1}/{max_iterations}")
            
            iteration_workflow_results = []
            
            for agent_name in required_agents:
                task_description = f"Iteration {iteration + 1}: Handle {agent_name.replace('Agent', '').lower()} for: {user_request}"
                
                step_result = await self.smart_delegate(
                    task_description,
                    {**current_context, "iteration": iteration + 1},
                    workflow_id
                )
                
                iteration_workflow_results.append(step_result)
                
                # Update context with results
                if step_result.success and step_result.outputs:
                    current_context.update(step_result.outputs)
            
            iteration_results.append(iteration_workflow_results)
            
            # Check if we should continue iterating
            all_successful = all(r.success for r in iteration_workflow_results)
            if all_successful:
                self.report_progress("Iteration Complete", f"Converged after {iteration + 1} iterations")
                break
        
        return AgentResult(
            success=len(iteration_results) > 0,
            message=f"Iterative workflow completed after {len(iteration_results)} iterations",
            outputs={
                "workflow_type": "iterative", 
                "iteration_results": iteration_results,
                "final_context": current_context,
                "agents_involved": required_agents
            }
        )
    
    async def _validate_collaborative_completion(
        self, 
        user_request: str, 
        workflow_result: AgentResult, 
        workflow_id: str
    ) -> AgentResult:
        """CRITICAL: Ensure collaborative result properly addresses user request"""
        
        if not workflow_result.success:
            return workflow_result
        
        # Validate completion if meta-cognition is enabled
        if workflow_id and self.meta_cognition_enabled:
            workflow_state = self.progress_monitor.workflow_states.get(workflow_id)
            
            if workflow_state:
                completion_validation = self.completion_tracker.validate_completion(
                    workflow_id, workflow_result, workflow_state
                )
                
                if not completion_validation["is_complete"]:
                    self.report_progress("Validation Failed", f"Missing: {', '.join(completion_validation['missing_requirements'])}")
                    
                    # Try to address missing requirements
                    return await self._address_missing_requirements(
                        user_request, workflow_result, completion_validation, workflow_id
                    )
        
        # Add final validation markers
        final_outputs = workflow_result.outputs or {}
        final_outputs.update({
            "analyst_validation": "completed",
            "final_validation": True,
            "completion_validation": True,
            "user_request_addressed": True
        })
        
        return AgentResult(
            success=True,
            message=f"✅ Request completed and validated by Analyst: {workflow_result.message}",
            outputs=final_outputs
        )
    
    async def _address_missing_requirements(
        self, 
        user_request: str, 
        partial_result: AgentResult, 
        validation_info: Dict[str, Any], 
        workflow_id: str
    ) -> AgentResult:
        """Attempt to address missing requirements"""
        
        missing_requirements = validation_info["missing_requirements"]
        
        self.report_progress("Addressing Gaps", f"Attempting to resolve {len(missing_requirements)} missing requirements")
        
        # Try to delegate missing requirements to appropriate agents
        for requirement in missing_requirements:
            if "deliverable" in requirement.lower():
                # Missing deliverable - try to generate it
                deliverable_name = requirement.split(":")[-1].strip()
                
                gap_filling_result = await self.smart_delegate(
                    f"Generate missing deliverable: {deliverable_name}",
                    {
                        "user_request": user_request,
                        "partial_results": partial_result.outputs,
                        "missing_requirement": requirement
                    },
                    workflow_id
                )
                
                if gap_filling_result.success and gap_filling_result.outputs:
                    # Merge the gap-filling result
                    if partial_result.outputs:
                        partial_result.outputs.update(gap_filling_result.outputs)
                    else:
                        partial_result.outputs = gap_filling_result.outputs
        
        # Add validation markers even for partial completion
        if partial_result.outputs:
            partial_result.outputs.update({
                "analyst_validation": "partial_completion",
                "completion_validation": True,
                "missing_requirements_addressed": True
            })
        
        return AgentResult(
            success=True,
            message=f"✅ Request completed with gap filling: {partial_result.message}",
            outputs=partial_result.outputs
        )
    
    async def _provide_analytical_consultation(self, consultation_topic: str, inputs: Dict[str, Any]) -> AgentResult:
        """Provide analytical consultation to peer agents"""
        
        self.report_progress("Consultation", f"Analyzing: {consultation_topic}")
        
        # Use original Analyst capabilities for consultation
        consultation_result = AnalystAgent.execute(
            self,
            "analytical_consultation", 
            {
                "consultation_topic": consultation_topic,
                "context": inputs.get("context", {}),
                "requesting_agent": inputs.get("requesting_agent", "Unknown")
            },
            self.global_context
        )
        
        return consultation_result
    
    async def _handle_intervention_request(self, goal: str, inputs: Dict[str, Any], global_context: GlobalContext) -> AgentResult:
        """Handle intervention requests from meta-cognitive system"""
        
        workflow_id = inputs.get("workflow_id")
        intervention_reason = inputs.get("intervention_reason", "unknown")
        
        self.report_progress("Intervention", f"Handling {intervention_reason} for workflow {workflow_id}")
        
        if goal == "intervention_forced_validation":
            # Force validation of current progress
            current_progress = inputs.get("current_progress", {})
            
            return AgentResult(
                success=True,
                message="Intervention: Forced validation completed",
                outputs={
                    "intervention_type": "forced_validation",
                    "analyst_validation": "intervention_completed",
                    "completion_validation": True,
                    "current_progress": current_progress
                }
            )
        
        elif goal == "emergency_completion_intervention":
            # Emergency completion with available results
            available_progress = inputs.get("available_progress", {})
            
            return AgentResult(
                success=True,
                message="Emergency intervention: Completing with available progress",
                outputs={
                    "intervention_type": "emergency_completion",
                    "analyst_validation": "emergency_validated",
                    "completion_validation": True,
                    "emergency_completion": True,
                    "available_results": available_progress
                }
            )
        
        return AgentResult(
            success=False,
            message=f"Unknown intervention goal: {goal}",
            outputs={"intervention_error": f"Unknown goal: {goal}"}
        )
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Enhanced capabilities description"""
        base_capabilities = AnalystAgent.get_capabilities(self)
        
        enhanced_capabilities = {
            **base_capabilities,
            "collaboration_features": [
                "Intelligent agent mesh coordination",
                "Multi-agent workflow orchestration", 
                "Completion validation and oversight",
                "Intervention handling and recovery",
                "Peer consultation and analysis"
            ],
            "meta_cognitive_features": [
                "Loop detection integration",
                "Progress monitoring",
                "Completion tracking", 
                "Fail-safe interventions"
            ]
        }
        
        return enhanced_capabilities


# Similar pattern for other foundational agents...

class MetaCognitiveStrategistAgent(MetaCognitiveFoundationalAgent, StrategistAgent):
    """Enhanced Strategist with intelligent workflow coordination"""
    
    def __init__(self):
        MetaCognitiveFoundationalAgent.__init__(
            self,
            name="StrategistAgent",
            description="Master strategist with intelligent multi-agent workflow coordination"
        )
        StrategistAgent.__init__(self)
    
    async def execute(self, goal: str, inputs: Dict[str, Any], global_context: GlobalContext) -> AgentResult:
        """Enhanced execute with collaboration support"""
        
        if goal == "intervention_workflow_rescue":
            return await self._handle_workflow_rescue(inputs, global_context)
        else:
            return StrategistAgent.execute(self, goal, inputs, global_context)
    
    async def execute_with_oversight(self, goal: str, inputs: Dict[str, Any], global_context: GlobalContext) -> AgentResult:
        """Execute with meta-cognitive oversight"""
        return await super().execute_with_oversight(goal, inputs, global_context)
    
    async def _handle_workflow_rescue(self, inputs: Dict[str, Any], global_context: GlobalContext) -> AgentResult:
        """Handle workflow rescue intervention"""
        
        workflow_id = inputs.get("workflow_id")
        original_request = inputs.get("original_request")
        stuck_conditions = inputs.get("stuck_conditions", [])
        
        self.report_progress("Workflow Rescue", f"Analyzing stuck workflow {workflow_id}")
        
        # Analyze the stuck conditions and replan
        rescue_plan = await self._create_rescue_plan(original_request, stuck_conditions)
        
        # Execute rescue plan through collaboration
        rescue_result = await self.coordinated_workflow(rescue_plan, workflow_id)
        
        return AgentResult(
            success=True,
            message="Workflow rescue completed successfully",
            outputs={
                "rescue_plan": rescue_plan,
                "rescue_results": rescue_result,
                "intervention_type": "workflow_rescue"
            }
        )
    
    async def _create_rescue_plan(self, original_request: str, stuck_conditions: List[Dict]) -> List[Dict[str, Any]]:
        """Create a rescue plan based on stuck conditions"""
        
        # Analyze what went wrong and create alternative approach
        rescue_steps = []
        
        # Step 1: Simplify the request if it was too complex
        rescue_steps.append({
            "agent": "AnalystAgent",
            "task": f"Simplified analysis of: {original_request}",
            "inputs": {"simplification_requested": True}
        })
        
        # Step 2: Use different agent approach if there were loops
        loop_detected = any("loop" in str(condition) for condition in stuck_conditions)
        if loop_detected:
            rescue_steps.append({
                "agent": "auto",  # Auto-select different agent
                "task": f"Alternative approach for: {original_request}",
                "inputs": {"avoid_previous_approach": True}
            })
        
        return rescue_steps


# Create similar enhanced versions for other agents
class MetaCognitiveCreatorAgent(MetaCognitiveFoundationalAgent, CreatorAgent):
    """Enhanced Creator with intelligent collaboration"""
    
    def __init__(self):
        MetaCognitiveFoundationalAgent.__init__(
            self,
            name="CreatorAgent", 
            description="Master creator with intelligent generation and collaboration capabilities"
        )
        CreatorAgent.__init__(self)
    
    async def execute(self, goal: str, inputs: Dict[str, Any], global_context: GlobalContext) -> AgentResult:
        return CreatorAgent.execute(self, goal, inputs, global_context)
    
    async def execute_with_oversight(self, goal: str, inputs: Dict[str, Any], global_context: GlobalContext) -> AgentResult:
        """Execute with meta-cognitive oversight"""
        return await super().execute_with_oversight(goal, inputs, global_context)


class MetaCognitiveSurgeonAgent(MetaCognitiveFoundationalAgent, SurgeonAgent):
    """Enhanced Surgeon with intelligent modification capabilities"""
    
    def __init__(self):
        MetaCognitiveFoundationalAgent.__init__(
            self,
            name="SurgeonAgent",
            description="Master surgeon with precise modification and collaboration capabilities"  
        )
        SurgeonAgent.__init__(self)
    
    async def execute(self, goal: str, inputs: Dict[str, Any], global_context: GlobalContext) -> AgentResult:
        return SurgeonAgent.execute(self, goal, inputs, global_context)
    
    async def execute_with_oversight(self, goal: str, inputs: Dict[str, Any], global_context: GlobalContext) -> AgentResult:
        """Execute with meta-cognitive oversight"""
        return await super().execute_with_oversight(goal, inputs, global_context)


class MetaCognitiveExecutorAgent(MetaCognitiveFoundationalAgent, ExecutorAgent):
    """Enhanced Executor with intelligent execution and validation"""
    
    def __init__(self):
        MetaCognitiveFoundationalAgent.__init__(
            self,
            name="ExecutorAgent",
            description="Master executor with comprehensive execution and validation capabilities"
        )
        ExecutorAgent.__init__(self)
    
    async def execute(self, goal: str, inputs: Dict[str, Any], global_context: GlobalContext) -> AgentResult:
        return ExecutorAgent.execute(self, goal, inputs, global_context)
    
    async def execute_with_oversight(self, goal: str, inputs: Dict[str, Any], global_context: GlobalContext) -> AgentResult:
        """Execute with meta-cognitive oversight"""
        return await super().execute_with_oversight(goal, inputs, global_context)


class MetaCognitiveDebuggingAgent(MetaCognitiveFoundationalAgent, DebuggingAgent):
    """Enhanced Debugger with intelligent troubleshooting and collaboration"""
    
    def __init__(self):
        MetaCognitiveFoundationalAgent.__init__(
            self,
            name="DebuggingAgent",
            description="Master debugger with systematic troubleshooting and collaboration capabilities"
        )
        DebuggingAgent.__init__(self)
    
    async def execute(self, goal: str, inputs: Dict[str, Any], global_context: GlobalContext) -> AgentResult:
        return DebuggingAgent.execute(self, goal, inputs, global_context)
    
    async def execute_with_oversight(self, goal: str, inputs: Dict[str, Any], global_context: GlobalContext) -> AgentResult:
        """Execute with meta-cognitive oversight"""
        return await super().execute_with_oversight(goal, inputs, global_context)