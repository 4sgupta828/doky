# agents/debugging.py
import json
import logging
from typing import Dict, Any, List
from datetime import datetime
import uuid
from pathlib import Path

# Foundational dependencies
from .base import BaseAgent
from core.context import GlobalContext
from core.models import AgentResponse, AgentResult, AgentExecutionError, TaskNode
from core.instruction_schemas import (
    InstructionScript, StructuredInstruction, InstructionType,
    create_fix_code_instruction, create_command_instruction, create_validation_instruction,
    ToolingInstruction, create_diagnostic_instruction
)
from utils.env_detector import get_python_version_command, get_pip_list_command

# Get a logger instance for this module
logger = logging.getLogger(__name__)


# --- Real LLM Integration Placeholder ---
class LLMClient:
    """A placeholder for a real LLM client (e.g., OpenAI, Gemini)."""
    def invoke(self, prompt: str) -> str:
        raise NotImplementedError("LLMClient.invoke must be implemented by a concrete class.")

# --- Agent Implementation ---

class DebuggingAgent(BaseAgent):
    """
    The team's expert troubleshooter and master investigator. This agent orchestrates
    multiple specialized agents to solve complex problems through:
    1. Evidence gathering (via ToolingAgent)
    2. Root cause analysis and hypothesis generation
    3. Smart fix strategy decisions (surgical fixes vs design changes via SpecAgent + CoderAgent)
    4. Validation and iteration (via TestRunner + TestGenerator)
    """

    def __init__(self, llm_client: Any = None, agent_registry: Dict[str, Any] = None):
        super().__init__(
            name="DebuggingAgent",
            description="Master troubleshooter that orchestrates other agents to solve complex problems end-to-end."
        )
        self.llm_client = llm_client or LLMClient()
        self.agent_registry = agent_registry or {}
        self.max_debug_iterations = 3
    
    # === FUNCTION-CALL INTERFACE ===
    
    def required_inputs(self) -> List[str]:
        """Required inputs for DebuggingAgent execution."""
        return ["failed_test_report", "code_context"]
    
    def optional_inputs(self) -> List[str]:
        """Optional inputs for DebuggingAgent execution."""
        return ["environment_data", "previous_attempts", "max_iterations"]
    
    def execute_v2(self, goal: str, inputs: Dict[str, Any], global_context: GlobalContext) -> AgentResult:
        """
        NEW INTERFACE: Debug failures with explicit inputs/outputs.
        
        This replaces the artifact hunting with direct data passing.
        """
        logger.info(f"DebuggingAgent executing with function-call interface: '{goal}'")
        
        # Fail-fast validation
        self.validate_inputs(inputs)
        
        # Extract explicit inputs
        failed_test_report = inputs["failed_test_report"]
        code_context = inputs["code_context"]
        environment_data = inputs.get("environment_data", {})
        max_iterations = inputs.get("max_iterations", self.max_debug_iterations)
        
        self.report_progress("Starting debug process", f"Goal: {goal[:80]}...")
        
        try:
            # Execute debugging with explicit inputs
            debug_result = self._debug_with_explicit_inputs(
                goal=goal,
                failed_test_report=failed_test_report,
                code_context=code_context,
                environment_data=environment_data,
                max_iterations=max_iterations,
                global_context=global_context
            )
            
            if debug_result["success"]:
                return self.create_result(
                    success=True,
                    message=f"Successfully debugged issue: {debug_result['solution_applied']}",
                    outputs={
                        "solution_applied": debug_result["solution_applied"],
                        "modified_files": debug_result.get("modified_files", []),
                        "iterations_used": debug_result["iterations_used"],
                        "root_cause": debug_result.get("root_cause", "Unknown"),
                        "validation_passed": debug_result.get("validation_passed", False)
                    }
                )
            else:
                return self.create_result(
                    success=False,
                    message=f"Debugging failed after {debug_result['iterations_used']} iterations",
                    outputs={
                        "iterations_used": debug_result["iterations_used"],
                        "attempted_solutions": debug_result.get("attempted_solutions", []),
                        "final_error": debug_result.get("final_error", "Unknown error")
                    },
                    error_details={
                        "debug_failed": True,
                        "iterations_exhausted": debug_result["iterations_used"] >= max_iterations
                    }
                )
                
        except Exception as e:
            error_msg = f"DebuggingAgent execution error: {e}"
            logger.error(error_msg, exc_info=True)
            return self.create_result(
                success=False,
                message=error_msg,
                error_details={"exception": str(e), "exception_type": type(e).__name__}
            )
    
    def _debug_with_explicit_inputs(self, goal: str, failed_test_report: Dict[str, Any], 
                                   code_context: str, environment_data: Dict[str, Any],
                                   max_iterations: int, global_context: GlobalContext) -> Dict[str, Any]:
        """
        Execute debugging process with explicit inputs (no artifact hunting).
        
        Returns:
            Dictionary with debugging results and metadata
        """
        self.report_thinking(f"Starting debugging with {max_iterations} max iterations")
        
        for iteration in range(max_iterations):
            logger.info(f"--- Debugging Iteration {iteration + 1}/{max_iterations} ---")
            self.report_progress(f"Debug iteration {iteration + 1}/{max_iterations}", "Analyzing and attempting fix")
            
            try:
                # Phase 1: Gather evidence (using ToolingAgent if available)
                evidence = self._gather_evidence_direct(failed_test_report, code_context, environment_data, global_context)
                
                # Phase 2: Analyze and hypothesize
                hypothesis = self._analyze_and_hypothesize_direct(evidence, failed_test_report, code_context)
                
                # Phase 3: Apply fix (using ToolingAgent)
                fix_result = self._apply_fix_direct(hypothesis, global_context)
                
                # Phase 4: Validate fix
                validation_result = self._validate_fix_direct(global_context)
                
                if validation_result["success"]:
                    logger.info(f"✅ Debugging succeeded after {iteration + 1} iterations")
                    return {
                        "success": True,
                        "iterations_used": iteration + 1,
                        "solution_applied": hypothesis.get("recommended_strategy", "Unknown solution"),
                        "modified_files": fix_result.get("modified_files", []),
                        "root_cause": hypothesis.get("primary_hypothesis", "Unknown"),
                        "validation_passed": True
                    }
                else:
                    logger.warning(f"❌ Fix attempt {iteration + 1} failed: {validation_result.get('message', 'Unknown error')}")
                    continue
                    
            except Exception as e:
                logger.error(f"Debugging iteration {iteration + 1} failed with exception: {e}")
                continue
        
        # All iterations exhausted
        return {
            "success": False,
            "iterations_used": max_iterations,
            "final_error": "Maximum debugging iterations reached without successful fix"
        }
    
    def _gather_evidence_direct(self, failed_test_report: Dict[str, Any], code_context: str, 
                               environment_data: Dict[str, Any], global_context: GlobalContext) -> Dict[str, Any]:
        """
        Gather evidence using direct function calls to ToolingAgent (no artifact hunting).
        """
        self.report_thinking("Gathering evidence with direct ToolingAgent calls")
        
        evidence = {
            "initial_failure": failed_test_report,
            "code_context": code_context,
            "environment_data": environment_data,
            "diagnostic_output": {}
        }
        
        # Call ToolingAgent directly if available
        if "ToolingAgent" in self.agent_registry:
            tooling_agent = self.agent_registry["ToolingAgent"]
            
            # Direct function call with explicit inputs
            diagnostic_result = self.call_agent_v2(
                target_agent=tooling_agent,
                goal="Gather diagnostic evidence for debugging",
                inputs={
                    "commands": [
                        get_python_version_command(),
                        get_pip_list_command(),
                        "ls -la",
                        "pytest --version",
                        "git status --porcelain"
                    ],
                    "purpose": "Gather comprehensive debugging evidence",
                    "timeout": 60
                },
                global_context=global_context
            )
            
            if diagnostic_result.success:
                evidence["diagnostic_output"] = diagnostic_result.outputs
                self.report_progress("Evidence gathered", f"Diagnostic commands executed successfully")
            else:
                logger.warning(f"Diagnostic evidence gathering failed: {diagnostic_result.message}")
        
        return evidence
    
    def _analyze_and_hypothesize_direct(self, evidence: Dict[str, Any], failed_test_report: Dict[str, Any], 
                                       code_context: str) -> Dict[str, Any]:
        """
        Analyze evidence and generate hypothesis using direct data (no artifact hunting).
        """
        self.report_thinking("Analyzing evidence and generating hypothesis with LLM")
        
        try:
            # Build comprehensive analysis prompt with direct data
            analysis_prompt = f"""
            You are an expert debugging agent analyzing a system failure. Based on the evidence below, 
            provide a structured analysis with hypotheses and recommended fix strategy.

            EVIDENCE:
            Failed Test Report: {json.dumps(failed_test_report, indent=2)}
            Code Context: {code_context[:1000]}...
            Diagnostic Output: {json.dumps(evidence.get('diagnostic_output', {}), indent=2)}

            Your analysis must be a valid JSON object with this structure:
            {{
                "root_cause_analysis": "Detailed explanation of what went wrong and why",
                "primary_hypothesis": "Most likely cause of the issue",
                "alternative_hypotheses": ["other possible causes"],
                "confidence_level": "high|medium|low",
                "error_category": "environment|dependencies|code|configuration|system",
                "solution_type": "SURGICAL|DESIGN_CHANGE",
                "complexity_assessment": "simple|moderate|complex",
                "recommended_strategy": "Specific approach to fix the issue",
                "risk_assessment": "Potential risks and side effects of the fix"
            }}
            """
            
            self.report_progress("AI analysis in progress", "Processing evidence with LLM for root cause analysis")
            
            response = self.llm_client.invoke(analysis_prompt)
            hypothesis = json.loads(response)
            
            logger.info(f"Generated hypothesis: {hypothesis.get('primary_hypothesis', 'Unknown')}")
            self.report_progress("Hypothesis generated", f"Root cause: {hypothesis.get('primary_hypothesis', 'Unknown')[:60]}...")
            
            return hypothesis
            
        except Exception as e:
            logger.error(f"Hypothesis generation failed: {e}")
            # Fallback basic hypothesis
            return {
                "root_cause_analysis": f"Analysis failed due to: {e}",
                "primary_hypothesis": "Unknown issue requiring manual investigation",
                "solution_type": "DESIGN_CHANGE",
                "complexity_assessment": "complex",
                "recommended_strategy": "Manual debugging required"
            }
    
    def _apply_fix_direct(self, hypothesis: Dict[str, Any], global_context: GlobalContext) -> Dict[str, Any]:
        """
        Apply fix using direct function calls (no artifact hunting).
        """
        solution_type = hypothesis.get("solution_type", "DESIGN_CHANGE")
        recommended_strategy = hypothesis.get("recommended_strategy", "Apply manual fix")
        
        self.report_thinking(f"Applying {solution_type} fix strategy using direct agent calls")
        
        # For now, simulate applying a fix (in full implementation, would call ScriptExecutorAgent or CodeGenerationAgent)
        if "ToolingAgent" in self.agent_registry:
            # Apply simple fixes via ToolingAgent
            tooling_agent = self.agent_registry["ToolingAgent"]
            
            fix_result = self.call_agent_v2(
                target_agent=tooling_agent,
                goal=f"Apply fix: {recommended_strategy}",
                inputs={
                    "commands": [
                        f"echo 'Applying fix: {recommended_strategy}'"
                    ],
                    "purpose": f"Apply {solution_type} fix for debugging"
                },
                global_context=global_context
            )
            
            if fix_result.success:
                return {
                    "success": True,
                    "modified_files": [],  # Would contain actual modified files
                    "fix_applied": recommended_strategy
                }
        
        # Fallback: manual fix recommendations
        return {
            "success": True,
            "modified_files": [],
            "fix_applied": "Manual fix recommendations created"
        }
    
    def _validate_fix_direct(self, global_context: GlobalContext) -> Dict[str, Any]:
        """
        Validate fix using direct TestRunnerAgent call (no artifact hunting).
        """
        self.report_thinking("Validating fix using direct TestRunnerAgent call")
        
        if "TestRunnerAgent" in self.agent_registry:
            test_runner = self.agent_registry["TestRunnerAgent"]
            
            # Direct function call for validation
            validation_result = self.call_agent_v2(
                target_agent=test_runner,
                goal="Validate that debugging fix resolved the issue",
                inputs={
                    "test_mode": "validation",
                    "run_all_tests": True,
                    "validation_mode": True  # Prevent circular calls
                },
                global_context=global_context
            )
            
            return {
                "success": validation_result.success,
                "message": validation_result.message,
                "test_results": validation_result.outputs
            }
        else:
            # No test runner available
            logger.warning("TestRunnerAgent not available for validation")
            return {
                "success": False,
                "message": "Cannot validate - TestRunnerAgent not available"
            }

    def _build_prompt(self, failed_test_report: Dict, code_context: str) -> str:
        """Constructs a detailed prompt to guide the LLM in debugging the code."""
        return f"""
        You are an expert Python debugger. Your task is to analyze a failed test report
        and the relevant source code to identify the root cause of the failure and
        propose a fix in the form of a code diff.

        **Failed Test Report:**
        ---
        {json.dumps(failed_test_report, indent=2)}
        ---

        **Relevant Source Code:**
        ---
        {code_context}
        ---

        **Instructions:**
        1.  **Analyze the Root Cause**: Carefully examine the error message, stack trace, and source code to pinpoint the exact line and reason for the failure.
        2.  **Formulate an Explanation**: Write a clear, concise explanation of the bug in Markdown format.
        3.  **Propose a Fix**: Generate a code patch in the standard 'diff' format to correct the bug. The diff should only contain the necessary changes to fix the issue.

        **Your output MUST be a single, valid JSON object with two keys:**
        1.  `root_cause_analysis`: A string containing the Markdown explanation of the bug.
        2.  `suggested_fix_diff`: A string containing the code patch in standard diff format.

        **JSON Output Format Example:**
        {{
            "root_cause_analysis": "### Root Cause Analysis\\n\\nThe `add` function fails on non-integer inputs because it does not perform type checking. The test failed when passing a string, which caused a `TypeError`.",
            "suggested_fix_diff": "--- a/src/calculator.py\\n+++ b/src/calculator.py\\n@@ -1,2 +1,4 @@\\n def add(a, b):\\n+    if not isinstance(a, int) or not isinstance(b, int):\\n+        raise TypeError(\\"Both inputs must be integers.\\")\\n     return a + b"
        }}

        Now, perform the debugging analysis.
        """

    def execute(self, goal: str, context: GlobalContext, current_task: TaskNode) -> AgentResponse:
        """Main debugging orchestration method implementing the 4-phase approach."""
        logger.info(f"DebuggingAgent executing with goal: '{goal}' - Starting comprehensive debugging process")
        
        # Report initial analysis and strategy to user
        self.report_progress("Initializing debug process", f"Analyzing: '{goal[:80]}...'")
        self.report_thinking(f"Starting comprehensive 4-phase debugging approach: Evidence → Analysis → Fix → Validation. Max {self.max_debug_iterations} iterations allowed.")
        
        for iteration in range(self.max_debug_iterations):
            logger.info(f"--- Debugging Iteration {iteration + 1}/{self.max_debug_iterations} ---")
            self.report_progress(f"Debug iteration {iteration + 1}/{self.max_debug_iterations}", "Running comprehensive debugging cycle")
            
            try:
                # Phase 1: Evidence Gathering
                logger.info("Phase 1: Gathering comprehensive evidence...")
                self.report_progress("Phase 1: Evidence gathering", "Collecting system context, logs, and failure data")
                evidence = self._gather_evidence(context, current_task)
                
                # Phase 2: Root Cause Analysis & Hypothesis Generation
                logger.info("Phase 2: Analyzing root cause and generating hypotheses...")
                self.report_progress("Phase 2: Root cause analysis", "Analyzing evidence and generating fix hypotheses")
                hypothesis = self._analyze_and_hypothesize(evidence, context, current_task)
                
                # Phase 3: Smart Fix Strategy Decision
                logger.info("Phase 3: Determining fix strategy...")
                self.report_progress("Phase 3: Fix strategy execution", f"Applying {hypothesis.get('solution_type', 'unknown')} fix strategy")
                fix_result = self._execute_fix_strategy(hypothesis, context, current_task)
                
                # Phase 4: Validation
                logger.info("Phase 4: Validating the fix...")
                self.report_progress("Phase 4: Fix validation", "Testing fix effectiveness with comprehensive validation")
                validation_result = self._validate_fix(context, current_task)
                
                if validation_result["success"]:
                    logger.info(f"✅ Debugging succeeded after {iteration + 1} iterations")
                    self.report_progress("Debugging successful!", f"Issue resolved after {iteration + 1} iterations using {hypothesis.get('solution_type', 'unknown')} approach")
                    
                    # Report final success analysis
                    self.report_thinking(f"Success! Applied {hypothesis.get('solution_type', 'unknown')} fix strategy. Validation confirmed the issue is resolved. Total iterations: {iteration + 1}.")
                    
                    # Display transparency summary for user
                    context.print_communication_summary()
                    
                    return AgentResponse(
                        success=True,
                        message=f"Successfully resolved issue after {iteration + 1} debugging iterations. Final solution: {hypothesis['solution_type']}",
                        artifacts_generated=fix_result.get("artifacts_generated", [])
                    )
                else:
                    logger.warning(f"❌ Fix attempt {iteration + 1} failed: {validation_result['message']}")
                    self.report_progress(f"Iteration {iteration + 1} failed", f"Fix validation failed: {validation_result.get('message', 'Unknown error')[:60]}...")
                    self.report_thinking(f"Iteration {iteration + 1} unsuccessful. Validation failed: {validation_result.get('message', 'Unknown')}. {'Will retry with refined approach.' if iteration + 1 < self.max_debug_iterations else 'Max iterations reached.'}")
                    
                    # Store failure info for next iteration
                    context.add_artifact(f"debug_iteration_{iteration}_failure.json", 
                                       json.dumps(validation_result), current_task.task_id)
                    
            except Exception as e:
                logger.error(f"Debugging iteration {iteration + 1} failed with exception: {e}")
                self.report_progress(f"Iteration {iteration + 1} exception", f"Unexpected error: {str(e)[:60]}...")
                self.report_thinking(f"Iteration {iteration + 1} encountered unexpected error: {e}. {'Will attempt next iteration.' if iteration + 1 < self.max_debug_iterations else 'Max iterations reached - debugging failed.'}")
                if iteration == self.max_debug_iterations - 1:
                    # Last iteration failed
                    break
                continue
        
        # All iterations failed - still show transparency summary
        self.report_progress("Debugging failed", f"Unable to resolve after {self.max_debug_iterations} iterations")
        self.report_thinking(f"Exhausted all {self.max_debug_iterations} debugging attempts. Issue requires manual intervention or different approach. All failure contexts have been documented for analysis.")
        
        context.print_communication_summary()
        
        return AgentResponse(
            success=False,
            message=f"Unable to resolve issue after {self.max_debug_iterations} debugging iterations. Manual intervention required.",
            artifacts_generated=[f"debug_iteration_{i}_failure.json" for i in range(self.max_debug_iterations)]
        )

    def _gather_evidence(self, context: GlobalContext, current_task: TaskNode) -> Dict[str, Any]:
        """Phase 1: Comprehensive evidence gathering using ToolingAgent and direct analysis."""
        self.report_thinking("Starting evidence gathering phase. Will collect failure reports, system context, code context, and environment data for comprehensive analysis.")
        
        evidence = {
            "initial_failure": {},
            "system_context": {},
            "code_context": {},
            "environment_context": {},
            "logs_and_traces": {}
        }
        
        try:
            # Get initial failure report
            failed_report = context.get_artifact("failed_test_report.json")
            if failed_report:
                evidence["initial_failure"] = json.loads(failed_report) if isinstance(failed_report, str) else failed_report
                self.report_progress("Failure report analyzed", f"Found test failure data: {evidence['initial_failure'].get('summary', {}).get('failed', 'unknown')} failed tests")
            else:
                self.report_thinking("No failed test report found. Will proceed with available evidence.")
            
            # Get code context
            code_context = context.get_artifact("targeted_code_context.json")
            if code_context and isinstance(code_context, dict) and "files" in code_context:
                evidence["code_context"]["targeted_code"] = code_context
                file_count = len(code_context["files"])
                total_chars = sum(len(file_data["content"]) for file_data in code_context["files"])
                self.report_progress("Code context loaded", f"Loaded {file_count} files with {total_chars} chars of relevant code context")
            else:
                self.report_thinking("No targeted code context found. Analysis will be based on available artifacts.")
            
            # Use ToolingAgent if available for deeper evidence gathering
            if "ToolingAgent" in self.agent_registry:
                # Create structured diagnostic instruction for evidence gathering
                diagnostic_commands = [
                    get_python_version_command(),
                    get_pip_list_command(),
                    "ls -la",
                    "pytest --version",
                    "git status --porcelain"
                ]
                
                diagnostic_instruction = create_diagnostic_instruction(
                    instruction_id=f"debug_evidence_{current_task.task_id}",
                    commands=diagnostic_commands,
                    purpose="Gather comprehensive debugging evidence for failed tests",
                    timeout=60,
                    expected_artifacts=None
                )
                
                # Store structured instruction in context
                context.add_artifact("tooling_instruction.json", diagnostic_instruction.model_dump_json(indent=2), current_task.task_id)
                
                # Log the structured communication for transparency
                self.log_communication(context, "ToolingAgent", "delegation", 
                                     f"Execute structured diagnostic: {diagnostic_instruction.purpose}", 
                                     {
                                         "instruction_type": "diagnostic", 
                                         "commands_count": len(diagnostic_instruction.commands),
                                         "instruction_id": diagnostic_instruction.instruction_id
                                     },
                                     current_task.task_id)
                
                tooling_agent = self.agent_registry["ToolingAgent"]
                
                # Use helper method to call ToolingAgent with progress tracker transfer
                tooling_result = self.call_agent_with_progress(
                    tooling_agent,
                    f"Execute diagnostic instruction: {diagnostic_instruction.instruction_id}",
                    context,
                    current_task,
                    f"tooling_{current_task.task_id}",
                    input_artifact_keys=["tooling_instruction.json"]
                )
                if tooling_result.success:
                    # Log successful response for transparency
                    self.log_communication(context, "ToolingAgent", "response", 
                                         f"Successfully executed diagnostic: {len(tooling_result.artifacts_generated or [])} artifacts",
                                         {"artifacts": tooling_result.artifacts_generated or [], "success": True},
                                         current_task.task_id)
                    self.report_progress("Structured diagnostics completed", f"ToolingAgent executed {len(diagnostic_instruction.commands)} diagnostic commands successfully")
                    
                    # Extract additional evidence from tooling agent results
                    for artifact_key in tooling_result.artifacts_generated or []:
                        artifact_content = context.get_artifact(artifact_key)
                        if artifact_content:
                            evidence["environment_context"][artifact_key] = artifact_content
                else:
                    # Log failed response for transparency
                    self.log_communication(context, "ToolingAgent", "error", 
                                         f"Failed to execute diagnostics: {tooling_result.message}",
                                         {"success": False, "error_message": tooling_result.message},
                                         current_task.task_id)
            
            # Add system context directly if ToolingAgent not available
            else:
                import platform
                import sys
                evidence["system_context"] = {
                    "platform": platform.platform(),
                    "python_version": sys.version,
                    "working_directory": context.workspace_path
                }
                
            # Report comprehensive evidence summary
            evidence_summary = {
                "failure_data": bool(evidence["initial_failure"]),
                "code_context": bool(evidence["code_context"]),
                "env_artifacts": len(evidence["environment_context"]),
                "system_info": bool(evidence["system_context"])
            }
            
            logger.info(f"Evidence gathered: {len(evidence)} categories")
            self.report_progress("Evidence collection complete", f"Gathered: failure={evidence_summary['failure_data']}, code={evidence_summary['code_context']}, env={evidence_summary['env_artifacts']} artifacts")
            self.report_intermediate_output("evidence_summary", evidence_summary)
            
            return evidence
            
        except Exception as e:
            logger.error(f"Evidence gathering failed: {e}")
            return evidence

    def _analyze_and_hypothesize(self, evidence: Dict[str, Any], context: GlobalContext, current_task: TaskNode) -> Dict[str, Any]:
        """Phase 2: Root cause analysis and hypothesis generation using LLM."""
        self.report_thinking("Starting root cause analysis using AI reasoning. Will analyze all evidence to generate hypotheses and determine optimal fix strategy.")
        
        try:
            # Build comprehensive analysis prompt
            analysis_prompt = f"""
            You are an expert debugging agent analyzing a complex system failure. Based on the comprehensive evidence below, 
            provide a structured analysis with hypotheses and recommended fix strategy.

            EVIDENCE:
            {json.dumps(evidence, indent=2)}

            Your analysis must be a valid JSON object with this structure:
            {{
                "root_cause_analysis": "Detailed explanation of what went wrong and why",
                "primary_hypothesis": "Most likely cause of the issue",
                "alternative_hypotheses": ["other possible causes"],
                "confidence_level": "high|medium|low",
                "error_category": "environment|dependencies|code|configuration|system",
                "solution_type": "SURGICAL|DESIGN_CHANGE", 
                "complexity_assessment": "simple|moderate|complex",
                "recommended_strategy": "Specific approach to fix the issue",
                "risk_assessment": "Potential risks and side effects of the fix"
            }}
            
            Focus on actionable insights and be specific about the fix strategy.
            """
            
            self.report_progress("AI analysis in progress", "Processing evidence with AI reasoning for root cause analysis")
            
            response = self.llm_client.invoke(analysis_prompt)
            hypothesis = json.loads(response)
            
            # Store analysis for reference
            context.add_artifact("debug_hypothesis.json", json.dumps(hypothesis, indent=2), current_task.task_id)
            
            logger.info(f"Generated hypothesis: {hypothesis.get('primary_hypothesis', 'Unknown')}")
            logger.info(f"Solution type: {hypothesis.get('solution_type', 'Unknown')}")
            
            # Report detailed hypothesis analysis
            self.report_progress("Hypothesis generated", f"Root cause: {hypothesis.get('primary_hypothesis', 'Unknown')[:60]}...")
            self.report_thinking(f"Analysis complete. Primary hypothesis: '{hypothesis.get('primary_hypothesis', 'Unknown')}'. Confidence: {hypothesis.get('confidence_level', 'unknown')}. Recommended strategy: {hypothesis.get('solution_type', 'unknown')}.")
            
            # Show intermediate analysis results
            analysis_summary = {
                "hypothesis": hypothesis.get('primary_hypothesis', 'Unknown')[:100],
                "confidence": hypothesis.get('confidence_level', 'unknown'),
                "solution_type": hypothesis.get('solution_type', 'unknown'),
                "complexity": hypothesis.get('complexity_assessment', 'unknown')
            }
            self.report_intermediate_output("hypothesis_analysis", analysis_summary)
            
            return hypothesis
            
        except Exception as e:
            logger.error(f"Hypothesis generation failed: {e}")
            self.report_thinking(f"AI analysis failed with error: {e}. Falling back to basic hypothesis for manual investigation.")
            self.report_progress("Analysis fallback", "AI analysis failed, using basic hypothesis for manual debugging")
            
            # Fallback basic hypothesis
            return {
                "root_cause_analysis": f"Analysis failed due to: {e}",
                "primary_hypothesis": "Unknown issue requiring manual investigation",
                "solution_type": "DESIGN_CHANGE",
                "complexity_assessment": "complex",
                "recommended_strategy": "Manual debugging required"
            }

    def _execute_fix_strategy(self, hypothesis: Dict[str, Any], context: GlobalContext, current_task: TaskNode) -> Dict[str, Any]:
        """Phase 3: Execute the chosen fix strategy by creating and executing structured repair scripts."""
        solution_type = hypothesis.get("solution_type", "DESIGN_CHANGE")
        
        self.report_thinking(f"Creating structured repair script for {solution_type} fix strategy. This will generate precise, executable instructions for the ScriptExecutorAgent.")
        
        try:
            if solution_type == "SURGICAL" and "ScriptExecutorAgent" in self.agent_registry:
                # Create surgical fix script
                logger.info("Creating surgical fix script for ScriptExecutorAgent")
                self.report_progress("Surgical fix approach", "Generating structured repair script for precise code changes")
                
                # Generate structured repair script
                repair_script = self._create_surgical_repair_script(hypothesis, context, current_task)
                
                # Log the delegation for transparency
                self.log_communication(context, "ScriptExecutorAgent", "delegation", 
                                     f"Execute surgical repair script: {repair_script.title}",
                                     {"solution_type": "SURGICAL", "script_id": repair_script.script_id, "instructions_count": len(repair_script.instructions)},
                                     current_task.task_id)
                
                # Store the script in context
                context.add_artifact("instruction_script.json", repair_script.model_dump_json(indent=2), current_task.task_id)
                
                # Execute via ScriptExecutorAgent with progress tracker transfer
                script_executor = self.agent_registry["ScriptExecutorAgent"]
                
                result = self.call_agent_with_progress(
                    script_executor,
                    f"Execute repair script: {repair_script.title}",
                    context,
                    current_task,
                    f"script_exec_{current_task.task_id}",
                    input_artifact_keys=["instruction_script.json"]
                )
                
                # Log the response for transparency
                response_type = "response" if result.success else "error"
                self.log_communication(context, "ScriptExecutorAgent", response_type, 
                                     result.message,
                                     {"success": result.success, "artifacts": result.artifacts_generated},
                                     current_task.task_id)
                
                # Report fix execution results
                if result.success:
                    self.report_progress("Surgical fix applied", f"ScriptExecutor successfully applied changes: {len(result.artifacts_generated or [])} files modified")
                else:
                    self.report_progress("Surgical fix failed", f"ScriptExecutor failed: {result.message[:60]}...")
                    
                return {"success": result.success, "artifacts_generated": result.artifacts_generated, "message": result.message}
                
            elif solution_type == "DESIGN_CHANGE" and "CodeGenerationAgent" in self.agent_registry:
                # Complex fix requiring structured design changes via CodeGenerationAgent
                logger.info("Creating structured design change request for CodeGenerationAgent")
                self.report_progress("Design change approach", "Generating evidence-based design change request")
                
                # Create structured design change request with evidence and reasoning
                design_change_request = self._create_design_change_request(hypothesis, context, current_task)
                
                # Log the delegation for transparency
                self.log_communication(context, "CodeGenerationAgent", "delegation", 
                                     f"Apply design changes: {design_change_request['design_problem']}",
                                     {
                                         "solution_type": "DESIGN_CHANGE", 
                                         "request_id": design_change_request['request_id'], 
                                         "files_to_modify": len(design_change_request['files_to_modify']),
                                         "change_scope": "design_level"
                                     },
                                     current_task.task_id)
                
                # Store the design change request in context
                context.add_artifact("design_change_request.json", json.dumps(design_change_request, indent=2), current_task.task_id)
                
                # Execute via CodeGenerationAgent with design update mode and progress tracker transfer
                code_agent = self.agent_registry["CodeGenerationAgent"]
                
                result = self.call_agent_with_progress(
                    code_agent,
                    f"Design update: {design_change_request['design_problem']}",
                    context,
                    current_task,
                    f"code_gen_{current_task.task_id}",
                    input_artifact_keys=["design_change_request.json"]
                )
                
                # Log the response for transparency
                response_type = "response" if result.success else "error"
                self.log_communication(context, "CodeGenerationAgent", response_type, 
                                     result.message,
                                     {"success": result.success, "artifacts": result.artifacts_generated},
                                     current_task.task_id)
                
                # Report fix execution results
                if result.success:
                    self.report_progress("Design change applied", f"CodeGenerationAgent successfully applied design changes: {len(result.artifacts_generated or [])} files modified")
                    self.report_thinking(f"Successfully applied evidence-based design changes. Modified files: {', '.join(result.artifacts_generated or [])}")
                else:
                    self.report_progress("Design change failed", f"CodeGenerationAgent failed: {result.message[:60]}...")
                    
                return {"success": result.success, "artifacts_generated": result.artifacts_generated, "message": result.message}
                    
            else:
                # Fallback: create manual fix recommendations
                logger.warning("Required agents not available, creating manual fix recommendations")
                self.report_progress("Manual fix mode", "Required agents unavailable - generating manual fix recommendations")
                self.report_thinking("Cannot execute automated fix because required agents (SpecAgent/CodeAgent) are not available. Creating detailed manual recommendations instead.")
                
                fix_recommendations = {
                    "hypothesis": hypothesis,
                    "manual_steps": [
                        f"1. Address root cause: {hypothesis.get('primary_hypothesis', 'Unknown')}",
                        f"2. Apply strategy: {hypothesis.get('recommended_strategy', 'Manual intervention')}",
                        "3. Test the fix thoroughly",
                        "4. Monitor for side effects"
                    ],
                    "risk_factors": hypothesis.get("risk_assessment", "Unknown risks")
                }
                
                context.add_artifact("manual_fix_recommendations.json", json.dumps(fix_recommendations, indent=2), current_task.task_id)
                self.report_intermediate_output("manual_fix_recommendations", fix_recommendations)
                return {"success": True, "artifacts_generated": ["manual_fix_recommendations.json"], "message": "Created manual fix recommendations"}
                
        except Exception as e:
            logger.error(f"Fix strategy execution failed: {e}")
            return {"success": False, "message": f"Fix execution failed: {e}"}

    def _validate_fix(self, context: GlobalContext, current_task: TaskNode) -> Dict[str, Any]:
        """Phase 4: Validate the fix using TestRunner and potentially TestGenerator."""
        self.report_thinking("Starting fix validation phase. Will run tests to confirm the issue is resolved and no regressions were introduced.")
        
        try:
            if "TestRunnerAgent" in self.agent_registry:
                logger.info("Validating fix using TestRunnerAgent")
                self.report_progress("Running validation tests", "TestRunnerAgent will verify fix effectiveness")
                
                # Log the delegation for transparency
                self.log_communication(context, "TestRunnerAgent", "delegation", 
                                     "Validate that the debugging fix resolved the issue",
                                     {"task_type": "validation", "purpose": "test fix"},
                                     current_task.task_id)
                
                test_runner = self.agent_registry["TestRunnerAgent"]
                
                # Use helper method to call TestRunnerAgent with progress tracker transfer  
                # TestRunnerAgent doesn't need specific input artifacts for validation
                result = self.call_agent_with_progress(
                    test_runner,
                    "Run tests to validate fix",
                    context,
                    current_task,
                    f"test_validate_{current_task.task_id}",
                    input_artifact_keys=[]
                )
                
                # Log the TestRunner response
                test_response_type = "response" if result.success else "error"
                self.log_communication(context, "TestRunnerAgent", test_response_type, 
                                     result.message,
                                     {"success": result.success, "artifacts": result.artifacts_generated},
                                     current_task.task_id)
                
                # Report validation results with detail
                if result.success:
                    self.report_progress("Validation successful", "All tests passed - fix verified effective")
                else:
                    self.report_progress("Validation failed", f"Tests still failing: {result.message[:60]}...")
                
                if result.success:
                    self.report_thinking("Fix validation successful! All tests are now passing, confirming the issue has been resolved without introducing regressions.")
                    return {"success": True, "message": "Fix validated successfully - tests now pass"}
                else:
                    # Tests still failing, try to enhance test coverage if TestGenerator available
                    if "TestGeneratorAgent" in self.agent_registry:
                        logger.info("Tests still failing, enhancing test coverage")
                        self.report_progress("Enhancing test coverage", "TestGeneratorAgent will create additional validation tests")
                        self.report_thinking("Validation failed - tests are still failing. Will generate additional tests to better understand the issue scope and improve validation coverage.")
                        
                        # Log the delegation for transparency
                        self.log_communication(context, "TestGeneratorAgent", "delegation", 
                                             "Generate additional tests to better validate the fix",
                                             {"reason": "tests_still_failing", "purpose": "enhance_coverage"},
                                             current_task.task_id)
                        
                        test_gen = self.agent_registry["TestGeneratorAgent"]
                        
                        # Use helper method to call TestGeneratorAgent with progress tracker transfer
                        # TestGeneratorAgent doesn't need specific input artifacts
                        gen_result = self.call_agent_with_progress(
                            test_gen,
                            "Generate targeted tests for debugging validation",
                            context,
                            current_task,
                            f"test_gen_{current_task.task_id}",
                            input_artifact_keys=[]
                        )
                        
                        # Log the TestGenerator response
                        gen_response_type = "response" if gen_result.success else "error"
                        self.log_communication(context, "TestGeneratorAgent", gen_response_type, 
                                             gen_result.message,
                                             {"success": gen_result.success, "artifacts": gen_result.artifacts_generated},
                                             current_task.task_id)
                        
                        # Report test generation results
                        if gen_result.success:
                            self.report_progress("Additional tests created", f"Generated {len(gen_result.artifacts_generated or [])} new test files for enhanced validation")
                        else:
                            self.report_progress("Test generation failed", f"Could not create additional tests: {gen_result.message[:60]}...")
                    
                    self.report_thinking(f"Validation failed even after attempting to enhance test coverage. The fix may be incomplete or the issue more complex than initially analyzed. Message: {result.message}")
                    return {"success": False, "message": f"Validation failed: {result.message}"}
            else:
                logger.warning("TestRunnerAgent not available for validation")
                self.report_progress("Validation unavailable", "TestRunnerAgent not available - cannot verify fix")
                self.report_thinking("Cannot perform automated validation because TestRunnerAgent is not available. Fix has been applied but effectiveness cannot be confirmed automatically.")
                return {"success": False, "message": "Cannot validate - TestRunnerAgent not available"}
                
        except Exception as e:
            logger.error(f"Fix validation failed: {e}")
            self.report_thinking(f"Validation phase encountered unexpected error: {e}. This indicates a system issue during test execution.")
            return {"success": False, "message": f"Validation error: {e}"}

    def _create_surgical_repair_script(self, hypothesis: Dict[str, Any], context: GlobalContext, current_task: TaskNode) -> InstructionScript:
        """Create a structured repair script for surgical fixes."""
        script_id = f"surgical_repair_{current_task.task_id}_{uuid.uuid4().hex[:8]}"
        
        # Analyze the hypothesis to generate specific repair instructions
        instructions = []
        
        # Get the failed test report and code context for analysis
        failed_report = context.get_artifact("failed_test_report.json")
        code_context = context.get_artifact("targeted_code_context.json")
        
        if failed_report and isinstance(failed_report, str):
            try:
                failed_report = json.loads(failed_report)
            except:
                pass
        
        # Extract specific fix instructions from the hypothesis
        recommended_strategy = hypothesis.get("recommended_strategy", "Apply general fix")
        primary_hypothesis = hypothesis.get("primary_hypothesis", "Unknown issue")
        
        # Create structured repair instructions based on the analysis
        if "function" in recommended_strategy.lower() or "method" in recommended_strategy.lower():
            # Function-level fix
            instructions.append(create_fix_code_instruction(
                instruction_id=f"fix_function_{len(instructions)+1}",
                file_path=self._extract_file_path(failed_report, code_context),
                issue_description=primary_hypothesis,
                fix_content=self._generate_fix_content(hypothesis, context),
                test_command=self._determine_test_command(context)
            ))
        else:
            # General code fix
            instructions.append(create_fix_code_instruction(
                instruction_id=f"fix_code_{len(instructions)+1}",
                file_path=self._extract_file_path(failed_report, code_context),
                issue_description=primary_hypothesis,
                fix_content=self._generate_fix_content(hypothesis, context),
                test_command=self._determine_test_command(context)
            ))
        
        # Add validation instruction
        test_command = self._determine_test_command(context)
        if test_command:
            instructions.append(create_validation_instruction(
                instruction_id="validate_fix",
                test_command=test_command,
                description="Validate that the fix resolves the issue"
            ))
        
        # Create the repair script
        return InstructionScript(
            script_id=script_id,
            title=f"Surgical Fix: {primary_hypothesis[:50]}...",
            description=f"Automated surgical repair for: {recommended_strategy}",
            created_by="DebuggingAgent",
            target_issue=primary_hypothesis,
            estimated_duration="1-3 minutes",
            instructions=instructions,
            backup_required=True,
            rollback_strategy="automatic_backup"
        )

    def _extract_file_path(self, failed_report: Dict, code_context: str) -> str:
        """Extract the most likely file path from failure context."""
        # Try to extract from failed report
        if failed_report:
            if isinstance(failed_report, dict):
                # Look for file paths in various places
                for key in ['file', 'filename', 'path', 'source_file']:
                    if key in failed_report:
                        return str(failed_report[key])
                
                # Check in error details
                if 'error' in failed_report:
                    error_str = str(failed_report['error'])
                    # Look for Python file patterns
                    import re
                    file_match = re.search(r'File "([^"]+\.py)"', error_str)
                    if file_match:
                        return file_match.group(1)
        
        # Fallback: assume main.py if no specific file found
        return "main.py"

    def _generate_fix_content(self, hypothesis: Dict[str, Any], context: GlobalContext) -> str:
        """Generate the actual fix content based on hypothesis."""
        # This would ideally use the LLM to generate specific fix code
        # For now, return a placeholder that includes the strategy
        strategy = hypothesis.get("recommended_strategy", "Fix the issue")
        
        return f"""# Generated fix based on: {strategy}
# TODO: Implement specific fix for: {hypothesis.get('primary_hypothesis', 'Unknown issue')}
# This is a placeholder - in practice, would generate actual fix code
pass"""

    def _determine_test_command(self, context: GlobalContext) -> str:
        """Determine appropriate test command based on context."""
        # Look for common test frameworks
        workspace_path = Path(context.workspace_path) if isinstance(context.workspace_path, str) else context.workspace_path
        
        # Check for pytest
        if (workspace_path / "pytest.ini").exists() or (workspace_path / "pyproject.toml").exists():
            return "pytest -v"
        
        # Check for unittest
        if (workspace_path / "tests").exists():
            return "python -m pytest tests/"
        
        # Fallback
        return "python -m pytest"

    def _create_design_change_request(self, hypothesis: Dict[str, Any], context: GlobalContext, current_task: TaskNode) -> Dict[str, Any]:
        """Create a structured design change request with evidence and reasoning."""
        # Get evidence from context
        failed_report = context.get_artifact("failed_test_report.json")
        code_context = context.get_artifact("targeted_code_context.json")
        
        if failed_report and isinstance(failed_report, str):
            try:
                failed_report = json.loads(failed_report)
            except:
                pass
        
        # Extract key information
        root_cause = hypothesis.get("primary_hypothesis", "Unknown design issue")
        recommended_strategy = hypothesis.get("recommended_strategy", "Apply design improvements")
        confidence = hypothesis.get("confidence_level", "medium")
        
        # Analyze what files need to be modified
        files_to_modify = self._identify_files_for_design_change(failed_report, code_context, hypothesis)
        
        # Create evidence-based reasoning
        evidence = {
            "failure_analysis": failed_report.get("error", "Unknown error") if failed_report else "No failure report",
            "affected_components": self._extract_affected_components(failed_report, code_context),
            "complexity_assessment": hypothesis.get("complexity_assessment", "moderate"),
            "risk_factors": hypothesis.get("risk_assessment", "Standard design change risks")
        }
        
        # Generate specific design recommendations
        design_recommendations = self._generate_design_recommendations(hypothesis, evidence)
        
        # Create the structured request
        design_change_request = {
            "request_id": f"design_change_{current_task.task_id}_{uuid.uuid4().hex[:8]}",
            "created_by": "DebuggingAgent",
            "timestamp": datetime.now().isoformat(),
            
            # Core analysis
            "root_cause_analysis": root_cause,
            "design_problem": f"Design issue identified: {recommended_strategy}",
            "confidence_level": confidence,
            
            # Evidence and reasoning
            "evidence": evidence,
            "analysis_summary": f"Based on debugging analysis, the issue '{root_cause}' requires design-level changes rather than simple fixes.",
            
            # Specific changes requested
            "recommended_changes": design_recommendations,
            "files_to_modify": files_to_modify,
            "change_scope": "design_level",  # vs "tactical"
            
            # Implementation guidance
            "preserve_functionality": True,
            "backward_compatibility": True,
            "testing_requirements": self._determine_test_command(context),
            
            # Quality and validation
            "validation_criteria": [
                "All existing tests continue to pass",
                "New design addresses the root cause",
                "Code maintainability is improved",
                "No performance regressions"
            ]
        }
        
        return design_change_request

    def _identify_files_for_design_change(self, failed_report: Dict, code_context: str, hypothesis: Dict) -> List[str]:
        """Identify which files need to be modified for the design change."""
        files = []
        
        # Extract from failure report
        if failed_report:
            if 'file' in failed_report:
                files.append(failed_report['file'])
            elif 'filename' in failed_report:
                files.append(failed_report['filename'])
        
        # Look for patterns in the strategy that indicate multiple files
        strategy = hypothesis.get("recommended_strategy", "").lower()
        
        if any(keyword in strategy for keyword in ['refactor', 'restructure', 'split', 'separate']):
            # Design changes often affect multiple files
            base_file = files[0] if files else "main.py"
            # Suggest related files that might need changes
            if base_file.endswith('.py'):
                base_name = base_file.replace('.py', '')
                files.extend([
                    f"{base_name}_utils.py",
                    f"{base_name}_config.py",
                    f"tests/test_{base_name}.py"
                ])
        
        # Fallback to main file if nothing found
        if not files:
            files = ["main.py"]
            
        return files

    def _extract_affected_components(self, failed_report: Dict, code_context: str) -> List[str]:
        """Extract the names of affected components from the context."""
        components = []
        
        if failed_report:
            if 'function' in failed_report:
                components.append(f"Function: {failed_report['function']}")
            if 'class' in failed_report:
                components.append(f"Class: {failed_report['class']}")
        
        # Extract function/class names from code context
        if code_context:
            import re
            # Find function definitions
            functions = re.findall(r'def\s+(\w+)\s*\(', code_context)
            components.extend([f"Function: {func}" for func in functions[:3]])  # Limit to first 3
            
            # Find class definitions
            classes = re.findall(r'class\s+(\w+)', code_context)
            components.extend([f"Class: {cls}" for cls in classes[:3]])  # Limit to first 3
        
        return components or ["Unknown component"]

    def _generate_design_recommendations(self, hypothesis: Dict, evidence: Dict) -> List[str]:
        """Generate specific, actionable design recommendations."""
        recommendations = []
        
        strategy = hypothesis.get("recommended_strategy", "").lower()
        root_cause = hypothesis.get("primary_hypothesis", "").lower()
        
        # Generate recommendations based on common design patterns
        if "validation" in root_cause or "input" in root_cause:
            recommendations.append("Implement proper input validation with clear error messages")
            recommendations.append("Add parameter type checking and boundary validation")
        
        if "error handling" in root_cause or "exception" in root_cause:
            recommendations.append("Implement comprehensive error handling strategy")
            recommendations.append("Add proper exception hierarchy and error propagation")
        
        if "separation" in strategy or "coupling" in root_cause:
            recommendations.append("Separate concerns into distinct modules or classes")
            recommendations.append("Reduce coupling between components")
        
        if "performance" in root_cause or "efficiency" in root_cause:
            recommendations.append("Optimize algorithm complexity and resource usage")
            recommendations.append("Add caching or memoization where appropriate")
        
        # Fallback recommendations
        if not recommendations:
            recommendations = [
                "Improve code structure and organization",
                "Add proper abstraction layers",
                "Enhance error handling and validation",
                "Improve code documentation and clarity"
            ]
        
        return recommendations


# --- Self-Testing Block ---
if __name__ == "__main__":
    import unittest
    from unittest.mock import MagicMock
    import shutil
    from pathlib import Path
    from utils.logger import setup_logger

    setup_logger(default_level=logging.INFO)

    class TestDebuggingAgent(unittest.TestCase):

        def setUp(self):
            self.test_workspace_path = "./temp_debugging_test_ws"
            if Path(self.test_workspace_path).exists():
                shutil.rmtree(self.test_workspace_path)
            
            self.mock_llm_client = MagicMock(spec=LLMClient)
            self.context = GlobalContext(workspace_path=self.test_workspace_path)
            self.agent = DebuggingAgent(llm_client=self.mock_llm_client)
            self.task = TaskNode(
                goal="Debug a failed test",
                assigned_agent="DebuggingAgent",
                input_artifact_keys=["failed_test_report.json", "targeted_code_context.txt"]
            )
            # Pre-populate context with necessary artifacts
            self.context.add_artifact("failed_test_report.json", {"summary": {"failed": 1}, "error": "TypeError"}, "task_test")
            self.context.add_artifact("targeted_code_context.txt", "def add(a, b): return a + b", "task_context")

        def tearDown(self):
            shutil.rmtree(self.test_workspace_path)

        def test_successful_debugging(self):
            """Tests the ideal case where the LLM returns a valid analysis and diff."""
            print("\n--- [Test Case 1: DebuggingAgent Success] ---")
            # Configure the mock LLM to return a valid debug analysis.
            mock_debug_output = json.dumps({
                "root_cause_analysis": "The function fails due to a TypeError.",
                "suggested_fix_diff": "--- a/file.py\n+++ b/file.py\n@@ -1 +1,2 @@\n+    if not isinstance(a, int): raise TypeError\n     return a + b"
            })
            self.mock_llm_client.invoke.return_value = mock_debug_output

            response = self.agent.execute(self.task.goal, self.context, self.task)

            self.mock_llm_client.invoke.assert_called_once()
            self.assertTrue(response.success)
            self.assertIn("root_cause_analysis.md", response.artifacts_generated)
            self.assertIn("suggested_fix.diff", response.artifacts_generated)
            
            analysis = self.context.get_artifact("root_cause_analysis.md")
            self.assertIn("TypeError", analysis)
            logger.info("✅ test_successful_debugging: PASSED")

        def test_failure_on_missing_artifacts(self):
            """Tests that the agent fails gracefully if prerequisites are not in the context."""
            print("\n--- [Test Case 2: DebuggingAgent Missing Artifacts] ---")
            empty_context = GlobalContext(workspace_path=self.test_workspace_path) # Use a fresh context
            
            response = self.agent.execute(self.task.goal, empty_context, self.task)

            self.assertFalse(response.success)
            self.assertIn("Missing required artifacts", response.message)
            self.mock_llm_client.invoke.assert_not_called()
            logger.info("✅ test_failure_on_missing_artifacts: PASSED")

        def test_llm_returns_invalid_json(self):
            """Tests how the agent handles a malformed JSON string from the LLM."""
            print("\n--- [Test Case 3: DebuggingAgent Invalid JSON] ---")
            self.mock_llm_client