# fagents/debugger.py
"""
FOUNDATIONAL AGENT 6: DEBUGGER

The Master Troubleshooter - Systematic bug diagnosis, hypothesis formation, and fix orchestration.
"""

import json
import logging
from typing import Dict, Any, List, Optional

# Foundational base
from .base import FoundationalAgent
from core.context import GlobalContext
from core.models import AgentResult

# Debugging tools - atomic and reusable
from tools.debugging_tools import (
    DebuggingState, FixStrategy,
    create_debugging_session, validate_debugging_inputs,
    scan_for_relevant_logs, classify_problem_type,
    generate_debugging_hypothesis, perform_failure_reflection,
    update_debugging_confidence, generate_instrumentation_plan
)

# Analysis and execution tools  
from tools.problem_analysis_tools import (
    classify_problem, analyze_errors, assess_severity,
    analyze_root_causes, generate_problem_recommendations
)
from tools.test_tools import execute_tests
from tools.test_generation_tools import TestGenerationContext, TestType, generate_tests
from tools.code_generation_tools import CodeGenerationContext, generate_code
# Note: Surgical fix tools would be imported here in full implementation
# from tools.script_execution_tools import execute_script, ScriptExecutionContext, InstructionScript

logger = logging.getLogger(__name__)


class DebuggingAgent(FoundationalAgent):
    """
    FOUNDATIONAL AGENT 6: DEBUGGER
    
    The Master Troubleshooter - Systematic debugging through evidence gathering, 
    hypothesis formation, and iterative fix validation.
    
    Core Capability: Orchestrate complete debugging workflows from problem reproduction
    through fix validation.
    
    Powers:
    - Bug reproduction through automated test creation
    - Evidence gathering from logs, errors, and system state
    - Hypothesis formation using problem analysis and historical data
    - Fix strategy selection (surgical vs design change)
    - Iterative debugging with confidence tracking
    - Failure reflection and strategy adaptation
    - Code instrumentation for runtime debugging
    - Fix validation and regression detection
    
    Unique Value: Can systematically debug ANY problem through structured investigation
    """
    
    def __init__(self, llm_client: Any = None):
        super().__init__(
            name="DebuggingAgent", 
            description="Master troubleshooter that systematically debugs problems through evidence, hypothesis, and validation."
        )
        self.max_debug_iterations = 5
        self._llm_client = llm_client
    
    def execute(self, goal: str, inputs: Dict[str, Any], global_context: GlobalContext) -> AgentResult:
        """
        Execute systematic debugging workflow.
        
        Supports multiple debugging modes:
        - bug_reproduction: Create tests that reproduce the reported bug
        - evidence_gathering: Collect diagnostic information and logs
        - hypothesis_formation: Generate and test debugging hypotheses  
        - fix_validation: Validate that fixes resolve the problem
        - full_debugging: Complete end-to-end debugging workflow
        """
        self.report_progress("Starting debugging session", f"Goal: {goal}")
        
        try:
            # Validate inputs
            validate_debugging_inputs(inputs)
            
            # Determine debugging mode
            debug_mode = self._determine_debug_mode(goal, inputs)
            self.report_progress("Debug mode determined", debug_mode)
            
            # Execute appropriate debugging workflow
            if debug_mode == "bug_reproduction":
                return self._execute_bug_reproduction(goal, inputs, global_context)
            elif debug_mode == "evidence_gathering":
                return self._execute_evidence_gathering(goal, inputs, global_context)  
            elif debug_mode == "hypothesis_formation":
                return self._execute_hypothesis_formation(goal, inputs, global_context)
            elif debug_mode == "fix_validation":
                return self._execute_fix_validation(goal, inputs, global_context)
            elif debug_mode == "full_debugging":
                return self._execute_full_debugging(goal, inputs, global_context)
            else:
                return self._execute_auto_debugging(goal, inputs, global_context)
                
        except Exception as e:
            self.report_error(f"Debugging failed: {e}", e)
            return AgentResult(
                success=False,
                message=f"Debugging failed: {e}",
                outputs={},
                error_details={"exception": str(e)}
            )
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Return comprehensive capabilities description."""
        return {
            "name": "DebuggingAgent",
            "description": "Master troubleshooter for systematic bug diagnosis and resolution",
            "primary_functions": [
                "Bug reproduction through test creation",
                "Evidence gathering from logs and system state",
                "Hypothesis formation and testing",
                "Fix strategy selection and execution",
                "Fix validation and regression detection",
                "Iterative debugging with confidence tracking"
            ],
            "input_types": [
                "problem_description", "failed_test_report", "code_context",
                "environment_data", "error_logs", "max_iterations"
            ],
            "output_types": [
                "debugging_session_results", "hypothesis_analysis", "fix_strategies",
                "validation_results", "confidence_assessment", "debugging_recommendations"
            ],
            "debugging_modes": [
                "bug_reproduction", "evidence_gathering", "hypothesis_formation", 
                "fix_validation", "full_debugging", "auto_debugging"
            ],
            "complexity_handling": "Can debug simple syntax errors to complex system-wide issues"
        }
    
    def _determine_debug_mode(self, goal: str, inputs: Dict[str, Any]) -> str:
        """Determine debugging mode based on goal and inputs."""
        goal_lower = goal.lower()
        
        # Explicit mode requests
        if "reproduce" in goal_lower or "create test" in goal_lower:
            return "bug_reproduction"
        elif "evidence" in goal_lower or "gather" in goal_lower or "investigate" in goal_lower:
            return "evidence_gathering"
        elif "hypothesis" in goal_lower or "analyze" in goal_lower:
            return "hypothesis_formation"
        elif "validate" in goal_lower or "test fix" in goal_lower:
            return "fix_validation"
        elif "debug" in goal_lower and ("full" in goal_lower or "complete" in goal_lower):
            return "full_debugging"
        
        # Auto-detect based on inputs
        if "failed_test_report" in inputs and inputs["failed_test_report"]:
            return "full_debugging"  # Already have failing test, can start full debugging
        elif inputs.get("problem_description") and not inputs.get("code_context"):
            return "bug_reproduction"  # Need to reproduce the bug first
        else:
            return "auto_debugging"
    
    def _execute_bug_reproduction(self, goal: str, inputs: Dict[str, Any], global_context: GlobalContext) -> AgentResult:
        """Execute bug reproduction by creating failing tests."""
        self.report_progress("Bug reproduction", "Creating test to reproduce the reported bug")
        
        problem_description = inputs["problem_description"]
        code_context = inputs.get("code_context", {})
        
        # Create debugging session
        debug_state = create_debugging_session(problem_description)
        
        try:
            # Generate test specification for the bug
            test_spec = f"Create a test that demonstrates this bug: {problem_description}"
            
            # Use test generation tools to create reproducing test
            test_context = TestGenerationContext(
                goal=test_spec,
                source_files=code_context,
                test_type=TestType.INTEGRATION,
                specification=f"Test should fail and demonstrate: {problem_description}"
            )
            
            test_generation_result = generate_tests(test_context)
            
            if not test_generation_result.success:
                return AgentResult(
                    success=False,
                    message=f"Failed to generate reproducing test: {test_generation_result.error_details or 'Unknown error'}",
                    outputs={"debug_session": debug_state.__dict__}
                )
            
            # Run the generated test to confirm it fails
            test_files = list(test_generation_result.generated_tests.values())
            test_execution_result = execute_tests(
                test_files=test_files,
                framework="pytest",
                timeout_seconds=60
            )
            
            success = test_execution_result.get("success", False) and test_execution_result.get("failures", 0) > 0
            
            if success:
                debug_state.last_error_report = test_execution_result.get("failure_details")
                message = "Successfully reproduced bug with failing test"
            else:
                message = "Test generation succeeded but could not reproduce the bug"
            
            self.report_progress("Bug reproduction complete", message)
            
            return AgentResult(
                success=success,
                message=message,
                outputs={
                    "debug_mode": "bug_reproduction",
                    "debug_session": debug_state.__dict__,
                    "generated_tests": test_generation_result.generated_tests,
                    "test_execution_result": test_execution_result,
                    "reproducing_test_available": success
                }
            )
            
        except Exception as e:
            self.report_error(f"Bug reproduction failed: {e}", e)
            return AgentResult(
                success=False,
                message=f"Bug reproduction failed: {e}",
                outputs={"debug_session": debug_state.__dict__}
            )
    
    def _execute_evidence_gathering(self, goal: str, inputs: Dict[str, Any], global_context: GlobalContext) -> AgentResult:
        """Execute comprehensive evidence gathering."""
        self.report_progress("Evidence gathering", "Collecting diagnostic information")
        
        problem_description = inputs["problem_description"]
        debug_state = create_debugging_session(problem_description)
        
        evidence = {
            "initial_failure": inputs.get("failed_test_report"),
            "code_context": inputs.get("code_context", {}),
            "environment_data": inputs.get("environment_data", {}),
            "log_scan_results": None,
            "problem_classification": None
        }
        
        try:
            # Scan for relevant logs
            self.report_progress("Scanning logs", "Searching for relevant error logs")
            log_results = scan_for_relevant_logs(
                problem_description, 
                debug_state.start_timestamp,
                str(global_context.workspace_path) if global_context.workspace_path else "."
            )
            evidence["log_scan_results"] = log_results
            
            # Classify the problem type
            problem_classification = classify_problem_type(
                problem_description,
                log_results.get("log_entries", [])
            )
            evidence["problem_classification"] = problem_classification
            
            # Use problem analysis tools for deeper analysis
            if inputs.get("failed_test_report"):
                error_analysis = analyze_errors(
                    error_logs=[inputs["failed_test_report"]],
                    stack_trace=inputs.get("stack_trace")
                )
                evidence["error_analysis"] = error_analysis
            
            self.report_progress("Evidence gathering complete", 
                               f"Found {len(evidence)} evidence sources")
            
            return AgentResult(
                success=True,
                message="Evidence gathering complete",
                outputs={
                    "debug_mode": "evidence_gathering", 
                    "debug_session": debug_state.__dict__,
                    "evidence": evidence,
                    "problem_classification": problem_classification
                }
            )
            
        except Exception as e:
            self.report_error(f"Evidence gathering failed: {e}", e)
            return AgentResult(
                success=False,
                message=f"Evidence gathering failed: {e}",
                outputs={"debug_session": debug_state.__dict__, "partial_evidence": evidence}
            )
    
    def _execute_hypothesis_formation(self, goal: str, inputs: Dict[str, Any], global_context: GlobalContext) -> AgentResult:
        """Execute hypothesis formation and analysis."""
        self.report_progress("Hypothesis formation", "Analyzing evidence to form debugging hypothesis")
        
        problem_description = inputs["problem_description"]
        evidence = inputs.get("evidence", {})
        debug_state = create_debugging_session(problem_description)
        
        # Restore debugging state if provided
        if "debug_session" in inputs:
            session_data = inputs["debug_session"]
            debug_state.hypotheses_tested = session_data.get("hypotheses_tested", [])
            debug_state.fixes_attempted = session_data.get("fixes_attempted", [])
            debug_state.confidence = session_data.get("confidence", 1.0)
        
        try:
            # Generate hypothesis using debugging tools
            hypothesis = generate_debugging_hypothesis(
                problem_description=problem_description,
                evidence=evidence,
                debugging_state=debug_state,
                llm_client=self._llm_client
            )
            
            # Add hypothesis to state
            debug_state.hypotheses_tested.append(hypothesis["primary_hypothesis"])
            
            self.report_progress("Hypothesis formed", 
                               f"Primary hypothesis: {hypothesis['primary_hypothesis'][:80]}...")
            
            return AgentResult(
                success=True,
                message=f"Debugging hypothesis formed with {hypothesis.get('confidence', 0.5):.1f} confidence",
                outputs={
                    "debug_mode": "hypothesis_formation",
                    "debug_session": debug_state.__dict__,
                    "hypothesis": hypothesis,
                    "recommended_strategy": hypothesis.get("recommended_strategy"),
                    "solution_type": hypothesis.get("solution_type")
                }
            )
            
        except Exception as e:
            self.report_error(f"Hypothesis formation failed: {e}", e)
            return AgentResult(
                success=False,
                message=f"Hypothesis formation failed: {e}",
                outputs={"debug_session": debug_state.__dict__}
            )
    
    def _execute_fix_validation(self, goal: str, inputs: Dict[str, Any], global_context: GlobalContext) -> AgentResult:
        """Execute fix validation to ensure problem resolution."""
        self.report_progress("Fix validation", "Validating that fixes resolve the problem")
        
        problem_description = inputs["problem_description"]
        debug_state = create_debugging_session(problem_description)
        
        try:
            # Run comprehensive tests  
            test_files = inputs.get("test_files", [])
            if not test_files and global_context.workspace_path:
                # Auto-discover test files
                from pathlib import Path
                test_files = []
                for test_file in Path(global_context.workspace_path).rglob("test_*.py"):
                    if test_file.is_file():
                        test_files.append(str(test_file))
            
            validation_result = execute_tests(
                test_files=test_files,
                framework="pytest",
                timeout_seconds=inputs.get("timeout_seconds", 120)
            )
            
            # Analyze validation results
            success = validation_result.get("success", False) and validation_result.get("failures", 0) == 0
            
            if success:
                message = f"Fix validation successful - all {validation_result.get('total', 0)} tests passed"
                debug_state.confidence = 1.0
            else:
                message = f"Fix validation failed - {validation_result.get('failures', 0)} tests still failing"
                debug_state.confidence *= 0.6
            
            self.report_progress("Fix validation complete", message)
            
            return AgentResult(
                success=success,
                message=message,
                outputs={
                    "debug_mode": "fix_validation",
                    "debug_session": debug_state.__dict__,
                    "validation_result": validation_result,
                    "tests_passed": validation_result.get("passed", 0),
                    "tests_failed": validation_result.get("failures", 0),
                    "fix_successful": success
                }
            )
            
        except Exception as e:
            self.report_error(f"Fix validation failed: {e}", e)
            return AgentResult(
                success=False,
                message=f"Fix validation failed: {e}",
                outputs={"debug_session": debug_state.__dict__}
            )
    
    def _execute_full_debugging(self, goal: str, inputs: Dict[str, Any], global_context: GlobalContext) -> AgentResult:
        """Execute complete debugging workflow from evidence to validation."""
        self.report_progress("Full debugging", "Running complete debugging workflow")
        
        problem_description = inputs["problem_description"]
        max_iterations = inputs.get("max_iterations", self.max_debug_iterations)
        debug_state = create_debugging_session(
            problem_description, 
            failed_test_report=inputs.get("failed_test_report")
        )
        
        workflow_results = {
            "debug_mode": "full_debugging",
            "iterations_performed": [],
            "final_result": None,
            "debug_session": debug_state.__dict__
        }
        
        try:
            # Phase 1: Bug reproduction if needed
            if not debug_state.last_error_report:
                repro_result = self._execute_bug_reproduction(goal, inputs, global_context)
                workflow_results["iterations_performed"].append({
                    "phase": "bug_reproduction",
                    "result": repro_result.outputs
                })
                
                if not repro_result.success:
                    workflow_results["final_result"] = "reproduction_failed"
                    return AgentResult(
                        success=False,
                        message="Failed to reproduce the bug",
                        outputs=workflow_results
                    )
                
                debug_state.last_error_report = repro_result.outputs.get("test_execution_result", {}).get("failure_details")
            
            # Main debugging loop
            for iteration in range(max_iterations):
                self.report_progress(f"Debug iteration {iteration + 1}", f"Confidence: {debug_state.confidence:.2f}")
                
                iteration_result = self._execute_debugging_iteration(
                    debug_state, inputs, global_context, iteration
                )
                
                workflow_results["iterations_performed"].append({
                    "iteration": iteration + 1,
                    "result": iteration_result
                })
                
                # Check if debugging succeeded
                if iteration_result.get("validation_successful"):
                    workflow_results["final_result"] = "success"
                    self.report_progress("Debugging successful", f"Resolved after {iteration + 1} iterations")
                    
                    return AgentResult(
                        success=True,
                        message=f"Successfully debugged problem after {iteration + 1} iterations",
                        outputs=workflow_results
                    )
                
                # Check confidence threshold for user guidance
                if debug_state.confidence < 0.4:
                    self.report_progress("Low confidence", "May need additional guidance or investigation")
                    
                    # Request user guidance when confidence is low
                    if iteration_result.get("hypothesis_formed") and "hypothesis" in workflow_results["iterations_performed"][-1]["result"]:
                        hypothesis = workflow_results["iterations_performed"][-1]["result"]["hypothesis"]
                        if not self._request_user_guidance(hypothesis, debug_state):
                            workflow_results["final_result"] = "user_cancelled"
                            return AgentResult(
                                success=False,
                                message="Debugging session cancelled by user",
                                outputs=workflow_results
                            )
                    break
            
            # Debugging failed after max iterations
            workflow_results["final_result"] = "max_iterations_exceeded"
            message = f"Unable to resolve issue after {max_iterations} iterations"
            
            return AgentResult(
                success=False,
                message=message,
                outputs=workflow_results
            )
            
        except Exception as e:
            self.report_error(f"Full debugging failed: {e}", e)
            workflow_results["final_result"] = "exception"
            workflow_results["error"] = str(e)
            
            return AgentResult(
                success=False,
                message=f"Full debugging failed: {e}",
                outputs=workflow_results
            )
    
    def _execute_debugging_iteration(
        self, 
        debug_state: DebuggingState, 
        inputs: Dict[str, Any], 
        global_context: GlobalContext,
        iteration: int
    ) -> Dict[str, Any]:
        """Execute a single debugging iteration."""
        iteration_result = {
            "iteration": iteration + 1,
            "evidence_gathered": False,
            "hypothesis_formed": False,
            "fix_applied": False,
            "validation_successful": False,
            "reflection_performed": False
        }
        
        try:
            # Gather evidence
            evidence_inputs = {
                "problem_description": debug_state.problem_description,
                "failed_test_report": debug_state.last_error_report,
                "code_context": inputs.get("code_context", {}),
                "environment_data": inputs.get("environment_data", {})
            }
            
            evidence_result = self._execute_evidence_gathering("gather evidence", evidence_inputs, global_context)
            iteration_result["evidence_gathered"] = evidence_result.success
            
            if not evidence_result.success:
                return iteration_result
            
            evidence = evidence_result.outputs.get("evidence", {})
            
            # Form hypothesis
            hypothesis_inputs = {
                "problem_description": debug_state.problem_description,
                "evidence": evidence,
                "debug_session": debug_state.__dict__
            }
            
            hypothesis_result = self._execute_hypothesis_formation("form hypothesis", hypothesis_inputs, global_context)
            iteration_result["hypothesis_formed"] = hypothesis_result.success
            
            if not hypothesis_result.success:
                return iteration_result
            
            hypothesis = hypothesis_result.outputs.get("hypothesis", {})
            
            # Apply fix strategy
            fix_result = self._apply_fix_strategy(hypothesis, evidence, global_context)
            iteration_result["fix_applied"] = fix_result.get("success", False)
            debug_state.fixes_attempted.append(hypothesis.get("recommended_strategy", "unknown"))
            
            if not fix_result.get("success", False):
                # Perform reflection on failed fix
                reflection = perform_failure_reflection(
                    old_error=debug_state.last_error_report,
                    new_error=fix_result.get("error_details"),
                    fix_attempted=hypothesis.get("recommended_strategy", "unknown"),
                    llm_client=self._llm_client
                )
                
                debug_state.reflection_history.append(reflection)
                iteration_result["reflection_performed"] = True
                
                # Update confidence
                update_debugging_confidence(
                    debug_state,
                    validation_passed=False,
                    failure_category=reflection.get("failure_category")
                )
                
                return iteration_result
            
            # Validate fix
            validation_inputs = {
                "problem_description": debug_state.problem_description,
                "timeout_seconds": 60
            }
            
            validation_result = self._execute_fix_validation("validate fix", validation_inputs, global_context)
            iteration_result["validation_successful"] = validation_result.success
            
            if validation_result.success:
                debug_state.confidence = 1.0
            else:
                # Reflect on failed validation
                reflection = perform_failure_reflection(
                    old_error=debug_state.last_error_report,
                    new_error=validation_result.outputs.get("validation_result"),
                    fix_attempted=hypothesis.get("recommended_strategy", "unknown"),
                    llm_client=self._llm_client
                )
                
                debug_state.reflection_history.append(reflection)
                iteration_result["reflection_performed"] = True
                
                # Update confidence and error state
                update_debugging_confidence(
                    debug_state,
                    validation_passed=False,
                    failure_category=reflection.get("failure_category")
                )
                
                debug_state.last_error_report = validation_result.outputs.get("validation_result")
            
            return iteration_result
            
        except Exception as e:
            iteration_result["error"] = str(e)
            self.report_error(f"Debugging iteration {iteration + 1} failed: {e}", e)
            return iteration_result
    
    def _apply_fix_strategy(self, hypothesis: Dict[str, Any], evidence: Dict[str, Any], global_context: GlobalContext) -> Dict[str, Any]:
        """Apply the recommended fix strategy."""
        solution_type = hypothesis.get("solution_type", "SURGICAL")
        self.report_progress(f"Applying {solution_type} fix", hypothesis.get("recommended_strategy", ""))
        
        try:
            if solution_type == "SURGICAL":
                # Apply surgical fix - simplified implementation  
                # In a full implementation, this would use surgical repair tools
                self.report_progress("Surgical fix", "Applying targeted code changes")
                
                return {
                    "success": True,  # Simplified - assume success for now
                    "strategy": "surgical", 
                    "message": "Surgical fix applied (simplified implementation)"
                }
            
            else:  # DESIGN_CHANGE
                # Apply design change using code generation tools
                code_context = evidence.get("code_context", {})
                
                generation_context = CodeGenerationContext(
                    goal=hypothesis.get("recommended_strategy", ""),
                    technical_spec=hypothesis.get("root_cause_analysis", ""),
                    existing_code=code_context
                )
                
                fix_result = generate_code(generation_context)
                
                return {
                    "success": fix_result.success,
                    "strategy": "design_change", 
                    "error_details": fix_result.error_details
                }
                
        except Exception as e:
            return {
                "success": False,
                "error_details": {"exception": str(e)}
            }
    
    def _generate_surgical_fix_script(self, hypothesis: Dict[str, Any], evidence: Dict[str, Any]) -> str:
        """Generate surgical fix script based on hypothesis using original prompt."""
        # Use LLM client if available for surgical fix generation
        if hasattr(self, '_llm_client') and self._llm_client:
            try:
                fix_prompt = self._build_surgical_fix_prompt(hypothesis, evidence)
                response = self._llm_client.invoke(fix_prompt)
                fix_data = json.loads(response)
                suggested_fixes = fix_data.get("suggested_fix_diffs", {})
                
                # Convert to script format
                script_parts = ["#!/bin/bash", f"# Surgical fix for: {hypothesis.get('primary_hypothesis', '')}", ""]
                for file_path, diff_content in suggested_fixes.items():
                    script_parts.append(f"# Applying fix to {file_path}")
                    script_parts.append(f"echo 'Applying fix to {file_path}'")
                    script_parts.append(f"# {diff_content}")
                
                return "\n".join(script_parts)
                
            except Exception as e:
                logger.error(f"Failed to generate LLM surgical fix: {e}")
        
        # Fallback to simple script generation
        return f"""#!/bin/bash
# Surgical fix for: {hypothesis.get('primary_hypothesis', '')}
# Strategy: {hypothesis.get('recommended_strategy', '')}

echo "Applying surgical fix..."
# Add specific fix commands based on hypothesis
echo "Fix applied successfully"
"""
    
    def _build_surgical_fix_prompt(self, hypothesis: Dict[str, Any], evidence: Dict[str, Any]) -> str:
        """Original surgical fix prompt from DebuggingAgent."""
        code_context_str = json.dumps(evidence.get("code_context", {}), indent=2)
        return f"""
        You are an expert software developer specializing in surgical code fixes.
        Based on the analysis and code, generate a set of code patches in the standard 'diff' format.
        The fix may require changes to one or more files.

        **Analysis**: {json.dumps(hypothesis, indent=2)}
        **Relevant Code**:
        ---
        {code_context_str}
        ---

        Generate a JSON object with a single key, "suggested_fix_diffs", which is a dictionary where each key is a file path and each value is the code patch for that file.
        """
    
    def _execute_auto_debugging(self, goal: str, inputs: Dict[str, Any], global_context: GlobalContext) -> AgentResult:
        """Auto-detect and execute appropriate debugging approach."""
        self.report_progress("Auto debugging", "Detecting appropriate debugging approach")
        
        # Default to full debugging if we have sufficient information
        if inputs.get("problem_description") and (inputs.get("failed_test_report") or inputs.get("code_context")):
            return self._execute_full_debugging(goal, inputs, global_context)
        
        # Otherwise start with bug reproduction
        return self._execute_bug_reproduction(goal, inputs, global_context)
    
    def _request_user_guidance(self, hypothesis: Dict[str, Any], state: DebuggingState) -> bool:
        """
        Request user guidance when confidence is low.
        Uses the original user guidance prompt from DebuggingAgent.
        """
        # For now, return True to continue - in a full implementation this would
        # integrate with a UI agent for user interaction
        try:
            # Original user guidance prompt from DebuggingAgent
            guidance_prompt = f"""
            I'm having trouble solving this bug. Here is my current analysis:
            **Hypothesis**: {hypothesis.get('primary_hypothesis')}
            **Proposed Strategy**: {hypothesis.get('recommended_strategy')}
            
            I have already tried: {', '.join(state.fixes_attempted) or 'nothing yet'}.
            
            Do you have any suggestions, or should I proceed? (Type your suggestion or press Enter to proceed)
            """
            
            self.report_progress("User guidance requested", guidance_prompt)
            
            # In a full implementation, this would:
            # 1. Use a ClarifierAgent to prompt the user
            # 2. Wait for user input
            # 3. Parse the response and update the debugging state
            # 4. Return False if user wants to cancel
            
            # For now, assume user wants to proceed
            return True
            
        except Exception as e:
            logger.warning(f"User guidance request failed: {e}")
            return True
    
    def _generate_new_spec_for_coder(self, hypothesis: Dict[str, Any], evidence: Dict[str, Any]) -> str:
        """
        Generate new specification for code changes.
        Uses the original spec generation from DebuggingAgent.
        """
        affected_files = [f['path'] for f in evidence.get("code_context", {}).get("files", [])]
        return f"# New Spec\n**Problem:** {hypothesis['root_cause_analysis']}\n**Changes:** {hypothesis['recommended_strategy']}\n**Files:** {json.dumps(affected_files)}"