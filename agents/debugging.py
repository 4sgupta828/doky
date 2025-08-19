# agents/debugging.py
import json
import logging
from typing import Dict, Any, List, Optional
import uuid
from dataclasses import dataclass, field

# Foundational dependencies
from .base import BaseAgent
from core.context import GlobalContext
from core.models import AgentResult, TaskNode
from core.instruction_schemas import (
    InstructionScript,
    create_fix_code_instruction,
    ToolingInstruction
)

# Get a logger instance for this module
logger = logging.getLogger(__name__)

@dataclass
class DebuggingState:
    """A state object to track the progress of a single debugging session."""
    session_id: str = field(default_factory=lambda: f"debug_{uuid.uuid4().hex[:8]}")
    problem_description: str = ""
    hypotheses_tested: List[str] = field(default_factory=list)
    fixes_attempted: List[str] = field(default_factory=list)
    validation_history: List[Dict] = field(default_factory=list)
    reflection_history: List[Dict] = field(default_factory=list) # Tracks reflections
    confidence: float = 1.0 # Starts high, degrades on failure
    last_error_report: Optional[Dict] = None

class DebuggingAgent(BaseAgent):
    """
    The team's expert troubleshooter. This agent orchestrates a stateful,
    iterative debugging session. It can proactively write tests to reproduce
    bugs, learn from failed attempts, and engage the user when it gets stuck.
    """

    def __init__(self, llm_client: Any = None, agent_registry: Dict[str, Any] = None):
        super().__init__(
            name="DebuggingAgent",
            description="Master troubleshooter that orchestrates other agents to solve complex problems."
        )
        self.llm_client = llm_client
        self.agent_registry = agent_registry or {}
        self.max_debug_iterations = 5

    def required_inputs(self) -> List[str]:
        """The agent can start with just a description of the problem."""
        return ["problem_description"]

    def optional_inputs(self) -> List[str]:
        return ["failed_test_report", "code_context", "environment_data", "max_iterations"]

    def execute_v2(self, goal: str, inputs: Dict[str, Any], global_context: GlobalContext) -> AgentResult:
        """Orchestrates the entire debugging process from triage to validation."""
        self.validate_inputs(inputs)

        state = DebuggingState(
            problem_description=inputs["problem_description"],
            last_error_report=inputs.get("failed_test_report")
        )
        max_iterations = inputs.get("max_iterations", self.max_debug_iterations)
        self.report_progress("Starting debug session", f"Goal: {state.problem_description[:80]}...")

        if not state.last_error_report:
            self.report_thinking("No initial test failure report. I must first write a test to reproduce the bug.")
            repro_result = self._reproduce_bug(state, inputs.get("code_context", {}), global_context)
            if not repro_result.success:
                return self.create_result(success=False, message=f"Failed to reproduce the bug: {repro_result.message}")
            state.last_error_report = repro_result.outputs.get("failed_test_report")

        for iteration in range(max_iterations):
            logger.info(f"--- Debugging Iteration {iteration + 1}/{max_iterations} (Confidence: {state.confidence:.2f}) ---")
            
            try:
                evidence = self._gather_evidence(state.last_error_report, inputs.get("code_context", {}), global_context)
                hypothesis = self._analyze_and_hypothesize(evidence, state)
                state.hypotheses_tested.append(hypothesis["primary_hypothesis"])

                if state.confidence < 0.4:
                    if not self._request_user_guidance(hypothesis, state, global_context):
                        return self.create_result(success=False, message="Debugging session cancelled by user.")

                fix_result = self._execute_fix_strategy(hypothesis, evidence, global_context)
                state.fixes_attempted.append(hypothesis["recommended_strategy"])
                if not fix_result.success:
                    self._reflect_and_adapt(state, validation_passed=False, new_error_report={"error": fix_result.message}, fix_attempted=hypothesis["recommended_strategy"])
                    continue

                validation_result = self._validate_fix(global_context)
                
                self._reflect_and_adapt(
                    state,
                    validation_passed=validation_result.success,
                    new_error_report=validation_result.outputs.get("failed_test_report"),
                    fix_attempted=hypothesis["recommended_strategy"]
                )

                if validation_result.success:
                    logger.info(f"✅ Debugging succeeded after {iteration + 1} iterations.")
                    return self.create_result(success=True, message=f"Successfully resolved: {state.problem_description}")
                else:
                    state.last_error_report = validation_result.outputs.get("failed_test_report", state.last_error_report)

            except Exception as e:
                logger.error(f"Debugging iteration failed with an exception: {e}", exc_info=True)
                self._reflect_and_adapt(state, validation_passed=False, new_error_report={"error": str(e)}, fix_attempted="N/A")
                continue

        return self.create_result(success=False, message=f"Unable to resolve issue after {max_iterations} iterations.")

    def _reproduce_bug(self, state: DebuggingState, code_context: Dict, context: GlobalContext) -> AgentResult:
        """Phase 0: Call TestGenerationAgent to create a failing test that reproduces the bug."""
        test_gen_agent = self.agent_registry.get("TestGenerationAgent")
        if not test_gen_agent:
            return self.create_result(success=False, message="TestGenerationAgent not available to reproduce bug.")

        self.report_progress("Reproducing Bug", "Delegating to TestGenerationAgent to create a failing test.")
        
        test_gen_result = self.call_agent_v2(
            target_agent=test_gen_agent,
            goal=f"Create a single Python test file that fails by demonstrating this bug: {state.problem_description}",
            inputs={ "code_to_test": {fi['path']: fi['content'] for fi in code_context.get("files", [])}, "spec": state.problem_description },
            global_context=context
        )

        if not test_gen_result.success: return test_gen_result
        
        test_runner = self.agent_registry.get("TestRunnerAgent")
        if not test_runner: return self.create_result(success=False, message="TestRunnerAgent not available to confirm bug reproduction.")
            
        return self.call_agent_v2(
            target_agent=test_runner,
            goal="Run the newly generated test to confirm the bug.",
            inputs={"specific_test_files": test_gen_result.outputs.get("artifacts_generated", [])},
            global_context=context
        )

    def _reflect_and_adapt(self, state: DebuggingState, validation_passed: bool, new_error_report: Optional[Dict], fix_attempted: str):
        """Phase 5: The meta-cognition step. Analyze the failure and update state."""
        if validation_passed:
            state.confidence = 1.0
            state.reflection_history.append({"status": "SUCCESS", "summary": "Fix was successful."})
            return

        self.report_thinking("Fix was not successful. Reflecting on the failure to guide the next attempt.")
        
        reflection_prompt = self._build_reflection_prompt(state.last_error_report, new_error_report, fix_attempted)
        try:
            reflection_str = self.llm_client.invoke(reflection_prompt)
            reflection = json.loads(reflection_str)
            state.reflection_history.append(reflection)
            self.report_intermediate_output("failure_reflection", reflection)

            if reflection["failure_category"] == "SAME_ERROR": state.confidence *= 0.6
            elif reflection["failure_category"] == "REGRESSION": state.confidence *= 0.5
            else: state.confidence *= 0.8
        
        except Exception as e:
            logger.error(f"Failed to perform reflection: {e}")
            state.confidence *= 0.7 # Default confidence degradation
            state.reflection_history.append({"status": "ERROR", "summary": f"Reflection failed: {e}"})

        state.last_error_report = new_error_report

    def _build_reflection_prompt(self, old_error: Optional[Dict], new_error: Optional[Dict], fix_attempted: str) -> str:
        """Builds a prompt for the LLM to analyze a failed fix attempt."""
        return f"""
        You are a senior software engineer performing a post-mortem on a failed bug fix attempt.
        Analyze the original error, the attempted fix, and the new error to guide the next debugging step.

        **Original Error Report:**
        {json.dumps(old_error, indent=2, default=str)}

        **Attempted Fix Strategy:**
        "{fix_attempted}"

        **New Error Report (after applying the fix):**
        {json.dumps(new_error, indent=2, default=str)}

        Analyze the situation and return a JSON object with this structure:
        {{
            "failure_category": "SAME_ERROR|NEW_ERROR|REGRESSION|INCOMPLETE_FIX|ENVIRONMENT_ISSUE",
            "analysis": "A brief explanation of why the fix likely failed.",
            "next_strategy_hint": "A high-level suggestion for what to try in the next iteration."
        }}
        """

    def _analyze_and_hypothesize(self, evidence: Dict, state: DebuggingState) -> Dict:
        """Phase 2: Analyze evidence and generate a hypothesis, now aware of past attempts and reflections."""
        self.report_thinking("Analyzing evidence, history, and reflections to form a new hypothesis.")
        
        prompt = self._build_analysis_prompt(evidence, state)
        response_str = self.llm_client.invoke(prompt)
        hypothesis = json.loads(response_str)
        
        self.report_intermediate_output("hypothesis_analysis", hypothesis)
        return hypothesis

    def _build_analysis_prompt(self, evidence: Dict, state: DebuggingState) -> str:
        """Constructs a prompt for the LLM to analyze the debugging evidence, including session state and reflections."""
        last_reflection = state.reflection_history[-1] if state.reflection_history else None
        
        return f"""
        You are an expert debugging agent. Your goal is to solve the following problem by generating a new hypothesis.
        Avoid repeating past mistakes by carefully considering the debugging history and the reflection on the last failed attempt.

        **OVERALL PROBLEM TO SOLVE**: {state.problem_description}

        **DEBUGGING HISTORY:**
        - Hypotheses Already Tested: {json.dumps(state.hypotheses_tested)}
        - Fixes Attempted: {json.dumps(state.fixes_attempted)}
        - Last Known Error: {json.dumps(state.last_error_report, default=str)}
        - Reflection on Last Failure: {json.dumps(last_reflection, default=str)}

        **CURRENT EVIDENCE:**
        {json.dumps(evidence, indent=2, default=str)}

        Your analysis must be a valid JSON object with this structure:
        {{
            "root_cause_analysis": "Detailed explanation of the bug, incorporating insights from the reflection.",
            "primary_hypothesis": "A NEW, UNTRIED hypothesis about the cause, guided by the reflection.",
            "solution_type": "SURGICAL|DESIGN_CHANGE",
            "recommended_strategy": "A NEW, UNTRIED approach to fix the issue."
        }}
        """

    def _gather_evidence(self, failed_report: Dict, code_context: Dict, env_data: Dict, context: GlobalContext) -> Dict:
        self.report_thinking("Gathering evidence...")
        evidence = {"initial_failure": failed_report, "code_context": code_context, "environment_data": env_data}
        tooling_agent = self.agent_registry.get("ToolingAgent")
        if tooling_agent:
            result = self.call_agent_v2(tooling_agent, "Gather diagnostics", {"commands": ["git status --porcelain"], "purpose": "Check repo status"}, context)
            if result.success: evidence["diagnostic_output"] = result.outputs
        return evidence

    def _execute_fix_strategy(self, hypothesis: Dict, evidence: Dict, context: GlobalContext) -> AgentResult:
        solution_type = hypothesis.get("solution_type", "DESIGN_CHANGE")
        self.report_thinking(f"Executing '{solution_type}' strategy.")
        if solution_type == "SURGICAL" and "ScriptExecutorAgent" in self.agent_registry:
            script = self._create_surgical_repair_script(hypothesis, evidence)
            context.add_artifact("instruction_script.json", script.model_dump_json(indent=2), "DebuggingAgent")
            return self.call_agent_v2(self.agent_registry["ScriptExecutorAgent"], f"Execute surgical fix", {"instruction_script": script.model_dump()}, context)
        else:
            new_spec = self._generate_new_spec_for_coder(hypothesis, evidence)
            context.add_artifact("technical_spec.md", new_spec, "DebuggingAgent")
            files = [f['path'] for f in evidence.get("code_context", {}).get("files", [])]
            return self.call_agent_v2(self.agent_registry["CoderAgent"], "Implement design change", {"technical_spec": new_spec, "files_to_generate": files}, context)

    def _validate_fix(self, context: GlobalContext) -> AgentResult:
        self.report_thinking("Validating fix...")
        test_runner = self.agent_registry.get("TestRunnerAgent")
        if not test_runner: return self.create_result(success=False, message="TestRunnerAgent not available.")
        return self.call_agent_v2(test_runner, "Validate fix", {"run_all_tests": True, "validation_mode": True}, context)

    def _build_surgical_fix_prompt(self, hypothesis: Dict, evidence: Dict) -> str:
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

    def _create_surgical_repair_script(self, hypothesis: Dict, evidence: Dict) -> InstructionScript:
        instructions = []
        fix_prompt = self._build_surgical_fix_prompt(hypothesis, evidence)
        try:
            fix_response_str = self.llm_client.invoke(fix_prompt)
            diff_map = json.loads(fix_response_str).get("suggested_fix_diffs", {})
            for i, (file_path, diff_content) in enumerate(diff_map.items()):
                instructions.append(create_fix_code_instruction(instruction_id=f"fix_{i+1}", file_path=file_path, issue_description=f"Apply fix to {file_path}", fix_content=diff_content))
        except Exception as e:
            logger.error(f"Failed to generate surgical fix: {e}")
            instructions.append(create_fix_code_instruction(instruction_id="fix_failed", file_path="error.log", issue_description="Error during fix generation", fix_content=f"Error: {e}"))
        return InstructionScript(script_id=f"repair_{uuid.uuid4().hex[:8]}", title=f"Surgical Fix for {hypothesis['primary_hypothesis']}", description=hypothesis["recommended_strategy"], created_by="DebuggingAgent", target_issue=hypothesis["root_cause_analysis"], instructions=instructions)

    def _generate_new_spec_for_coder(self, hypothesis: Dict, evidence: Dict) -> str:
        affected_files = [f['path'] for f in evidence.get("code_context", {}).get("files", [])]
        return f"# New Spec\n**Problem:** {hypothesis['root_cause_analysis']}\n**Changes:** {hypothesis['recommended_strategy']}\n**Files:** {json.dumps(affected_files)}"
    
    def _request_user_guidance(self, hypothesis: Dict, state: DebuggingState, context: GlobalContext) -> bool:
        ui_agent = self.agent_registry.get("ClarifierAgent")
        if not ui_agent or not hasattr(ui_agent, 'ui') or not hasattr(ui_agent.ui, 'prompt_for_input'):
            logger.warning("UI agent not available, cannot request user guidance.")
            return True

        prompt = f"""
        I'm having trouble solving this bug. Here is my current analysis:
        **Hypothesis**: {hypothesis.get('primary_hypothesis')}
        **Proposed Strategy**: {hypothesis.get('recommended_strategy')}
        
        I have already tried: {', '.join(state.fixes_attempted) or 'nothing yet'}.
        
        Do you have any suggestions, or should I proceed? (Type your suggestion or press Enter to proceed)
        """
        
        user_feedback = ui_agent.ui.prompt_for_input(prompt)
        if user_feedback and user_feedback.lower() not in ["", "yes", "proceed", "continue"]:
            state.hypotheses_tested.append("User provided new guidance.")
            if state.last_error_report:
                state.last_error_report['user_guidance'] = user_feedback
            state.confidence = 0.9
        elif user_feedback.lower() in ["cancel", "stop", "exit"]:
            return False
        return True
