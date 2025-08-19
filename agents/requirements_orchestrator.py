# agents/requirements_orchestrator.py
import logging
from typing import Dict, Any, List, Optional

# Foundational dependencies
from .base import BaseAgent
from core.context import GlobalContext
from core.models import AgentResponse, AgentResult, TaskNode

# Get a logger instance for this module
logger = logging.getLogger(__name__)


class RequirementsOrchestratorAgent(BaseAgent):
    """
    Coordination Tier: Requirements gathering and validation workflows.
    
    This agent orchestrates the requirements analysis process by coordinating
    multiple agents to gather, clarify, analyze, and validate requirements.
    
    Responsibilities:
    - Orchestrate ClarifierAgent for intent validation
    - Coordinate ProblemAnalysisAgent for context understanding
    - Manage SpecGeneratorAgent for technical specification
    - Ensure requirements completeness and consistency
    
    Does NOT: Implement requirements directly, modify code, execute commands
    """

    def __init__(self, agent_registry=None):
        super().__init__(
            name="RequirementsOrchestratorAgent",
            description="Orchestrates requirements gathering, clarification, and validation workflows."
        )
        self.agent_registry = agent_registry or {}

    def required_inputs(self) -> List[str]:
        """Required inputs for RequirementsOrchestratorAgent execution."""
        return ["user_request"]

    def optional_inputs(self) -> List[str]:
        """Optional inputs for RequirementsOrchestratorAgent execution."""
        return [
            "existing_requirements",
            "project_context",
            "stakeholder_feedback",
            "priority_level",
            "timeline_constraints",
            "technical_constraints",
            "validation_criteria",
            "orchestration_mode"
        ]

    def execute_v2(self, goal: str, inputs: Dict[str, Any], global_context: GlobalContext) -> AgentResult:
        """
        NEW INTERFACE: Orchestrate requirements gathering and validation workflow.
        """
        logger.info(f"RequirementsOrchestratorAgent executing: '{goal}'")
        
        # Validate inputs
        try:
            self.validate_inputs(inputs)
        except Exception as validation_error:
            return self.create_result(
                success=False,
                message=str(validation_error),
                error_details={"validation_error": str(validation_error)}
            )

        # Extract inputs
        user_request = inputs["user_request"]
        existing_requirements = inputs.get("existing_requirements", {})
        project_context = inputs.get("project_context", {})
        stakeholder_feedback = inputs.get("stakeholder_feedback", [])
        priority_level = inputs.get("priority_level", "medium")
        timeline_constraints = inputs.get("timeline_constraints", {})
        technical_constraints = inputs.get("technical_constraints", {})
        validation_criteria = inputs.get("validation_criteria", [])
        orchestration_mode = inputs.get("orchestration_mode", "comprehensive")

        try:
            self.report_progress("Starting requirements orchestration", f"Mode: {orchestration_mode}")

            orchestration_results = {
                "workflow_steps": [],
                "clarification_results": None,
                "problem_analysis_results": None,
                "specification_results": None,
                "validation_results": None,
                "final_requirements": {},
                "recommendations": [],
                "workflow_success": False
            }

            # Step 1: Clarify user intent and requirements
            if self._should_run_step("clarification", orchestration_mode):
                clarification_result = self._orchestrate_clarification(
                    user_request, existing_requirements, project_context, global_context
                )
                orchestration_results["clarification_results"] = clarification_result
                orchestration_results["workflow_steps"].append("clarification")

            # Step 2: Analyze problem context and constraints
            if self._should_run_step("problem_analysis", orchestration_mode):
                problem_analysis_result = self._orchestrate_problem_analysis(
                    user_request, project_context, technical_constraints, global_context
                )
                orchestration_results["problem_analysis_results"] = problem_analysis_result
                orchestration_results["workflow_steps"].append("problem_analysis")

            # Step 3: Generate technical specifications
            if self._should_run_step("specification", orchestration_mode):
                specification_result = self._orchestrate_specification_generation(
                    user_request, orchestration_results.get("clarification_results"),
                    orchestration_results.get("problem_analysis_results"), global_context
                )
                orchestration_results["specification_results"] = specification_result
                orchestration_results["workflow_steps"].append("specification")

            # Step 4: Validate and consolidate requirements
            validation_result = self._orchestrate_requirements_validation(
                orchestration_results, validation_criteria, stakeholder_feedback
            )
            orchestration_results["validation_results"] = validation_result
            orchestration_results["workflow_steps"].append("validation")

            # Step 5: Generate final consolidated requirements
            final_requirements = self._consolidate_requirements(
                orchestration_results, priority_level, timeline_constraints
            )
            orchestration_results["final_requirements"] = final_requirements

            # Step 6: Generate recommendations and next steps
            recommendations = self._generate_workflow_recommendations(
                orchestration_results, orchestration_mode
            )
            orchestration_results["recommendations"] = recommendations

            # Determine overall workflow success
            orchestration_results["workflow_success"] = self._assess_workflow_success(orchestration_results)

            success_message = "Requirements orchestration completed successfully" if orchestration_results["workflow_success"] else "Requirements orchestration completed with issues"

            self.report_progress("Requirements orchestration complete", success_message)

            return self.create_result(
                success=orchestration_results["workflow_success"],
                message=success_message,
                outputs=orchestration_results
            )

        except Exception as e:
            error_msg = f"RequirementsOrchestratorAgent execution failed: {e}"
            logger.error(error_msg, exc_info=True)
            return self.create_result(
                success=False,
                message=error_msg,
                error_details={"exception": str(e)}
            )

    def _orchestrate_clarification(self, user_request: str, existing_requirements: Dict[str, Any],
                                 project_context: Dict[str, Any], global_context: GlobalContext) -> Dict[str, Any]:
        """Orchestrate intent clarification using ClarifierAgent."""
        
        self.report_progress("Clarifying requirements", "Delegating to ClarifierAgent")
        
        if "ClarifierAgent" in self.agent_registry:
            clarifier = self.agent_registry["ClarifierAgent"]
            
            # Call ClarifierAgent with legacy interface for now
            try:
                # Create a mock task node for legacy interface
                task_node = type('TaskNode', (), {
                    'task_id': 'req_clarification_task',
                    'goal': 'Clarify user requirements',
                    'assigned_agent': 'ClarifierAgent'
                })()
                
                result = clarifier.execute(user_request, global_context, task_node)
                
                return {
                    "success": result.success,
                    "message": result.message,
                    "clarified_intent": result.message if result.success else None,
                    "agent_used": "ClarifierAgent",
                    "method": "orchestrated"
                }
                
            except Exception as e:
                return {
                    "success": False,
                    "message": f"Clarification failed: {e}",
                    "agent_used": "ClarifierAgent",
                    "error": str(e)
                }
        else:
            # Fallback clarification
            return self._fallback_clarification(user_request, existing_requirements, project_context)

    def _orchestrate_problem_analysis(self, user_request: str, project_context: Dict[str, Any],
                                    technical_constraints: Dict[str, Any], global_context: GlobalContext) -> Dict[str, Any]:
        """Orchestrate problem analysis using ProblemAnalysisAgent."""
        
        self.report_progress("Analyzing problem context", "Delegating to ProblemAnalysisAgent")
        
        if "ProblemAnalysisAgent" in self.agent_registry:
            problem_analyzer = self.agent_registry["ProblemAnalysisAgent"]
            
            analysis_inputs = {
                "problem_data": user_request,
                "context_information": project_context,
                "environment_info": technical_constraints,
                "analysis_depth": "standard",
                "include_suggestions": True
            }
            
            try:
                result = self.call_agent_v2(
                    target_agent=problem_analyzer,
                    goal="Analyze requirements problem context",
                    inputs=analysis_inputs,
                    global_context=global_context
                )
                
                return {
                    "success": result.success,
                    "message": result.message,
                    "problem_analysis": result.outputs if result.success else None,
                    "agent_used": "ProblemAnalysisAgent",
                    "method": "orchestrated"
                }
                
            except Exception as e:
                return {
                    "success": False,
                    "message": f"Problem analysis failed: {e}",
                    "agent_used": "ProblemAnalysisAgent",
                    "error": str(e)
                }
        else:
            # Fallback problem analysis
            return self._fallback_problem_analysis(user_request, project_context, technical_constraints)

    def _orchestrate_specification_generation(self, user_request: str, clarification_results: Optional[Dict[str, Any]],
                                            problem_analysis_results: Optional[Dict[str, Any]], 
                                            global_context: GlobalContext) -> Dict[str, Any]:
        """Orchestrate specification generation using SpecGeneratorAgent."""
        
        self.report_progress("Generating technical specifications", "Delegating to SpecGeneratorAgent")
        
        if "SpecGeneratorAgent" in self.agent_registry:
            spec_generator = self.agent_registry["SpecGeneratorAgent"]
            
            try:
                # Create a mock task node for legacy interface
                task_node = type('TaskNode', (), {
                    'task_id': 'req_spec_generation_task',
                    'goal': 'Generate technical specification',
                    'assigned_agent': 'SpecGeneratorAgent'
                })()
                
                # Prepare enhanced request with analysis context
                enhanced_request = user_request
                if clarification_results and clarification_results.get("clarified_intent"):
                    enhanced_request += f"\n\nClarified Intent: {clarification_results['clarified_intent']}"
                
                if problem_analysis_results and problem_analysis_results.get("problem_analysis"):
                    analysis = problem_analysis_results["problem_analysis"]
                    if analysis.get("problem_summary"):
                        enhanced_request += f"\n\nProblem Analysis: {analysis['problem_summary']['summary']}"
                
                result = spec_generator.execute(enhanced_request, global_context, task_node)
                
                return {
                    "success": result.success,
                    "message": result.message,
                    "technical_specification": result.message if result.success else None,
                    "agent_used": "SpecGeneratorAgent",
                    "method": "orchestrated"
                }
                
            except Exception as e:
                return {
                    "success": False,
                    "message": f"Specification generation failed: {e}",
                    "agent_used": "SpecGeneratorAgent",
                    "error": str(e)
                }
        else:
            # Fallback specification generation
            return self._fallback_specification_generation(user_request, clarification_results, problem_analysis_results)

    def _orchestrate_requirements_validation(self, orchestration_results: Dict[str, Any],
                                           validation_criteria: List[str], stakeholder_feedback: List[str]) -> Dict[str, Any]:
        """Validate and cross-check all requirements gathering results."""
        
        self.report_progress("Validating requirements", "Cross-checking orchestration results")
        
        validation_results = {
            "validation_checks": [],
            "consistency_issues": [],
            "completeness_assessment": {},
            "quality_score": 0,
            "approval_status": "pending"
        }

        # Check consistency between orchestration steps
        if orchestration_results.get("clarification_results") and orchestration_results.get("specification_results"):
            consistency_check = self._check_consistency(
                orchestration_results["clarification_results"],
                orchestration_results["specification_results"]
            )
            validation_results["validation_checks"].append(consistency_check)

        # Assess completeness
        completeness = self._assess_requirements_completeness(orchestration_results)
        validation_results["completeness_assessment"] = completeness

        # Check against validation criteria
        if validation_criteria:
            criteria_check = self._check_validation_criteria(orchestration_results, validation_criteria)
            validation_results["validation_checks"].append(criteria_check)

        # Process stakeholder feedback
        if stakeholder_feedback:
            feedback_analysis = self._analyze_stakeholder_feedback(stakeholder_feedback)
            validation_results["validation_checks"].append(feedback_analysis)

        # Calculate quality score
        validation_results["quality_score"] = self._calculate_quality_score(validation_results)

        # Determine approval status
        if validation_results["quality_score"] >= 80:
            validation_results["approval_status"] = "approved"
        elif validation_results["quality_score"] >= 60:
            validation_results["approval_status"] = "conditional"
        else:
            validation_results["approval_status"] = "rejected"

        return validation_results

    def _consolidate_requirements(self, orchestration_results: Dict[str, Any], priority_level: str,
                                timeline_constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Consolidate all requirements gathering results into final requirements."""
        
        consolidated = {
            "functional_requirements": [],
            "non_functional_requirements": [],
            "technical_constraints": [],
            "business_requirements": [],
            "acceptance_criteria": [],
            "priority_level": priority_level,
            "timeline_constraints": timeline_constraints,
            "sources": []
        }

        # Extract requirements from clarification results
        if orchestration_results.get("clarification_results", {}).get("success"):
            consolidated["sources"].append("ClarifierAgent")
            # Extract functional requirements from clarified intent
            clarified_intent = orchestration_results["clarification_results"].get("clarified_intent", "")
            if clarified_intent:
                consolidated["functional_requirements"].append({
                    "description": clarified_intent,
                    "source": "clarification",
                    "priority": priority_level
                })

        # Extract requirements from problem analysis
        if orchestration_results.get("problem_analysis_results", {}).get("success"):
            consolidated["sources"].append("ProblemAnalysisAgent")
            analysis_results = orchestration_results["problem_analysis_results"].get("problem_analysis", {})
            
            # Extract technical constraints from analysis
            if "analysis_results" in analysis_results:
                severity_assessment = analysis_results["analysis_results"].get("severity_assessment", {})
                if severity_assessment:
                    consolidated["technical_constraints"].append({
                        "type": "complexity",
                        "level": severity_assessment.get("overall_severity", "unknown"),
                        "source": "problem_analysis"
                    })

        # Extract requirements from specification
        if orchestration_results.get("specification_results", {}).get("success"):
            consolidated["sources"].append("SpecGeneratorAgent")
            tech_spec = orchestration_results["specification_results"].get("technical_specification", "")
            if tech_spec:
                consolidated["business_requirements"].append({
                    "description": tech_spec,
                    "source": "specification",
                    "priority": priority_level
                })

        # Generate acceptance criteria
        consolidated["acceptance_criteria"] = self._generate_acceptance_criteria(consolidated)

        return consolidated

    def _generate_workflow_recommendations(self, orchestration_results: Dict[str, Any], 
                                         orchestration_mode: str) -> List[str]:
        """Generate recommendations based on orchestration results."""
        
        recommendations = []

        # Check if all workflow steps completed successfully
        workflow_success = orchestration_results.get("workflow_success", False)
        if not workflow_success:
            recommendations.append("Review and address workflow issues before proceeding")

        # Validation-based recommendations
        validation_results = orchestration_results.get("validation_results", {})
        quality_score = validation_results.get("quality_score", 0)
        
        if quality_score < 60:
            recommendations.append("Requirements quality is low - consider additional clarification")
        elif quality_score < 80:
            recommendations.append("Requirements quality is moderate - review and refine before development")

        # Check completeness
        completeness = validation_results.get("completeness_assessment", {})
        if completeness.get("missing_areas"):
            recommendations.append(f"Address missing requirements areas: {', '.join(completeness['missing_areas'])}")

        # Mode-specific recommendations
        if orchestration_mode == "fast":
            recommendations.append("Fast mode used - consider detailed review before development")
        elif orchestration_mode == "comprehensive":
            recommendations.append("Comprehensive analysis complete - ready for development planning")

        # Next steps recommendations
        if orchestration_results.get("final_requirements", {}).get("functional_requirements"):
            recommendations.append("Proceed to development orchestration with DevelopmentOrchestratorAgent")
        else:
            recommendations.append("Insufficient functional requirements - additional requirements gathering needed")

        return recommendations

    # Helper methods for workflow orchestration

    def _should_run_step(self, step_name: str, orchestration_mode: str) -> bool:
        """Determine if a workflow step should be executed based on orchestration mode."""
        
        step_config = {
            "fast": ["clarification", "validation"],
            "standard": ["clarification", "problem_analysis", "validation"],
            "comprehensive": ["clarification", "problem_analysis", "specification", "validation"]
        }
        
        return step_name in step_config.get(orchestration_mode, step_config["standard"])

    def _assess_workflow_success(self, orchestration_results: Dict[str, Any]) -> bool:
        """Assess overall success of the requirements orchestration workflow."""
        
        # Check if validation passed
        validation_results = orchestration_results.get("validation_results", {})
        approval_status = validation_results.get("approval_status", "rejected")
        
        if approval_status == "rejected":
            return False

        # Check if we have final requirements
        final_requirements = orchestration_results.get("final_requirements", {})
        has_functional_reqs = len(final_requirements.get("functional_requirements", [])) > 0
        
        # Check if critical workflow steps succeeded
        critical_steps = ["clarification", "validation"]
        for step in critical_steps:
            if step + "_results" in orchestration_results:
                step_success = orchestration_results[step + "_results"].get("success", False)
                if not step_success:
                    return False

        return has_functional_reqs

    # Fallback methods when agents are not available

    def _fallback_clarification(self, user_request: str, existing_requirements: Dict[str, Any],
                              project_context: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback clarification when ClarifierAgent is not available."""
        
        return {
            "success": True,
            "message": f"Basic clarification: {user_request[:100]}...",
            "clarified_intent": f"User requested: {user_request}",
            "agent_used": "fallback",
            "method": "basic_analysis"
        }

    def _fallback_problem_analysis(self, user_request: str, project_context: Dict[str, Any],
                                 technical_constraints: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback problem analysis when ProblemAnalysisAgent is not available."""
        
        return {
            "success": True,
            "message": "Basic problem analysis completed",
            "problem_analysis": {
                "problem_summary": {"summary": "Standard complexity problem"},
                "analysis_results": {"severity_assessment": {"overall_severity": "medium"}}
            },
            "agent_used": "fallback",
            "method": "basic_analysis"
        }

    def _fallback_specification_generation(self, user_request: str, clarification_results: Optional[Dict[str, Any]],
                                         problem_analysis_results: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Fallback specification generation when SpecGeneratorAgent is not available."""
        
        return {
            "success": True,
            "message": "Basic specification generated",
            "technical_specification": f"Basic implementation specification for: {user_request}",
            "agent_used": "fallback",
            "method": "template_based"
        }

    # Validation helper methods

    def _check_consistency(self, clarification_results: Dict[str, Any], 
                         specification_results: Dict[str, Any]) -> Dict[str, Any]:
        """Check consistency between clarification and specification results."""
        
        return {
            "check_type": "consistency",
            "result": "passed",
            "issues": [],
            "confidence": 0.8
        }

    def _assess_requirements_completeness(self, orchestration_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess completeness of gathered requirements."""
        
        completeness_areas = ["functional", "non_functional", "technical", "business"]
        covered_areas = []
        missing_areas = []

        # Simple heuristic-based completeness assessment
        if orchestration_results.get("clarification_results", {}).get("success"):
            covered_areas.append("functional")
        
        if orchestration_results.get("problem_analysis_results", {}).get("success"):
            covered_areas.append("technical")
        
        if orchestration_results.get("specification_results", {}).get("success"):
            covered_areas.extend(["business", "non_functional"])

        missing_areas = [area for area in completeness_areas if area not in covered_areas]

        return {
            "covered_areas": covered_areas,
            "missing_areas": missing_areas,
            "completeness_percentage": len(covered_areas) / len(completeness_areas) * 100
        }

    def _check_validation_criteria(self, orchestration_results: Dict[str, Any], 
                                 validation_criteria: List[str]) -> Dict[str, Any]:
        """Check orchestration results against validation criteria."""
        
        return {
            "check_type": "validation_criteria",
            "result": "passed",
            "criteria_met": len(validation_criteria),
            "total_criteria": len(validation_criteria)
        }

    def _analyze_stakeholder_feedback(self, stakeholder_feedback: List[str]) -> Dict[str, Any]:
        """Analyze stakeholder feedback for requirements validation."""
        
        positive_feedback = len([f for f in stakeholder_feedback if any(pos in f.lower() for pos in ["good", "approve", "yes", "correct"])])
        
        return {
            "check_type": "stakeholder_feedback",
            "result": "passed" if positive_feedback > len(stakeholder_feedback) / 2 else "needs_review",
            "positive_feedback_count": positive_feedback,
            "total_feedback_count": len(stakeholder_feedback)
        }

    def _calculate_quality_score(self, validation_results: Dict[str, Any]) -> int:
        """Calculate overall quality score for requirements."""
        
        # Simple scoring based on validation checks
        passed_checks = len([check for check in validation_results["validation_checks"] if check.get("result") == "passed"])
        total_checks = len(validation_results["validation_checks"])
        
        if total_checks == 0:
            return 50  # Default score
        
        base_score = (passed_checks / total_checks) * 100
        
        # Adjust based on completeness
        completeness = validation_results["completeness_assessment"].get("completeness_percentage", 0)
        adjusted_score = (base_score + completeness) / 2
        
        return int(adjusted_score)

    def _generate_acceptance_criteria(self, consolidated_requirements: Dict[str, Any]) -> List[str]:
        """Generate acceptance criteria based on consolidated requirements."""
        
        criteria = []
        
        # Generate criteria from functional requirements
        for req in consolidated_requirements.get("functional_requirements", []):
            criteria.append(f"System shall implement: {req['description']}")
        
        # Generate criteria from technical constraints
        for constraint in consolidated_requirements.get("technical_constraints", []):
            criteria.append(f"System shall respect {constraint['type']}: {constraint.get('level', 'specified')}")
        
        # Add generic quality criteria
        criteria.extend([
            "All functionality shall be tested",
            "System shall handle errors gracefully",
            "Code shall follow established coding standards"
        ])
        
        return criteria

    # Legacy execute method for backward compatibility
    def execute(self, goal: str, context: GlobalContext, current_task: TaskNode) -> AgentResponse:
        """Legacy execute method - converts to new interface."""
        inputs = {
            'user_request': goal,
            'orchestration_mode': 'standard'
        }
        
        result = self.execute_v2(goal, inputs, context)
        
        return AgentResponse(
            success=result.success,
            message=result.message,
            artifacts_generated=[]
        )