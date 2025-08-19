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

        # Initialize the state for this debugging session
        state = DebuggingState(
            problem_description=inputs["problem_description"],
            last_error_report=inputs.get("failed_test_report")
        )

        max_iterations = inputs.get("max_iterations", self.max_debug_iterations)
        self.report_progress("Starting debug session", f"Goal: {state.problem_description[:80]}...")

        # --- NEW: Phase 0: Triage & Bug Reproduction ---
        if not state.last_error_report:
            self.report_thinking("No initial test failure report. I must first write a test to reproduce the bug.")
            repro_result = self._reproduce_bug(state, inputs.get("code_context", {}), global_context)
            if not repro_result.success:
                return self.create_result(success=False, message=f"Failed to reproduce the bug: {repro_result.message}")
            state.last_error_report = repro_result.outputs.get("failed_test_report")

        # --- Main Debugging Loop ---
        for iteration in range(max_iterations):
            logger.info(f"--- Debugging Iteration {iteration + 1}/{max_iterations} (Confidence: {state.confidence:.2f}) ---")
            
            try:
                # Phases 1 & 2: Gather Evidence & Hypothesize
                evidence = self._gather_evidence(state.last_error_report, inputs.get("code_context", {}), global_context)
                hypothesis = self._analyze_and_hypothesize(evidence, state)
                state.hypotheses_tested.append(hypothesis["primary_hypothesis"])

                # Meta-Cognition: Engage user if confidence is low
                if state.confidence < 0.4:
                    if not self._request_user_guidance(hypothesis, state, global_context):
                        return self.create_result(success=False, message="Debugging session cancelled by user.")

                # Phase 3: Execute Fix Strategy
                fix_result = self._execute_fix_strategy(hypothesis, evidence, global_context)
                state.fixes_attempted.append(hypothesis["recommended_strategy"])
                if not fix_result.success:
                    self._reflect_and_adapt(state, validation_passed=False, new_error_report={"error": fix_result.message})
                    continue

                # Phase 4: Validate Fix
                validation_result = self._validate_fix(global_context)
                
                # Phase 5: Reflect and Adapt for next loop
                self._reflect_and_adapt(
                    state,
                    validation_passed=validation_result.success,
                    new_error_report=validation_result.outputs.get("failed_test_report")
                )

                if validation_result.success:
                    logger.info(f"✅ Debugging succeeded after {iteration + 1} iterations.")
                    return self.create_result(success=True, message=f"Successfully resolved: {state.problem_description}")

            except Exception as e:
                logger.error(f"Debugging iteration failed with an exception: {e}", exc_info=True)
                self._reflect_and_adapt(state, validation_passed=False, new_error_report={"error": str(e)})
                continue

        return self.create_result(success=False, message=f"Unable to resolve issue after {max_iterations} iterations.")

    def _reproduce_bug(self, state: DebuggingState, code_context: Dict, context: GlobalContext) -> AgentResult:
        """Phase 0: Call TestGenerationAgent to create a failing test that reproduces the bug."""
        test_gen_agent = self.agent_registry.get("TestGenerationAgent")
        if not test_gen_agent:
            return self.create_result(success=False, message="TestGenerationAgent not available to reproduce bug.")

        self.report_progress("Reproducing Bug", "Delegating to TestGenerationAgent to create a failing test.")
        
        # Call TestGenerationAgent
        test_gen_result = self.call_agent_v2(
            target_agent=test_gen_agent,
            goal=f"Create a single Python test file that fails by demonstrating this bug: {state.problem_description}",
            inputs={
                "code_to_test": {file_info['path']: file_info['content'] for file_info in code_context.get("files", [])},
                "spec": state.problem_description
            },
            global_context=context
        )

        if not test_gen_result.success:
            return test_gen_result
        
        # Now, run the newly created test to get a failure report
        test_runner = self.agent_registry.get("TestRunnerAgent")
        if not test_runner:
            return self.create_result(success=False, message="TestRunnerAgent not available to confirm bug reproduction.")
            
        return self.call_agent_v2(
            target_agent=test_runner,
            goal="Run the newly generated test to confirm the bug.",
            inputs={"specific_test_files": test_gen_result.outputs.get("artifacts_generated", [])},
            global_context=context
        )

    def _reflect_and_adapt(self, state: DebuggingState, validation_passed: bool, new_error_report: Optional[Dict]):
        """Phase 5: The meta-cognition step. Update state based on the outcome."""
        if validation_passed:
            state.confidence = 1.0
        else:
            # Compare JSON strings for a simple but effective check
            last_error_str = json.dumps(state.last_error_report, sort_keys=True)
            new_error_str = json.dumps(new_error_report, sort_keys=True)

            if new_error_str == last_error_str:
                state.confidence *= 0.6
                self.report_thinking("The fix had no effect. I am likely on the wrong track.")
            else:
                state.confidence *= 0.8
                self.report_thinking("The error has changed. This provides new information for the next attempt.")
        
        state.last_error_report = new_error_report

    def _gather_evidence(self, failed_report: Dict, code_context: Dict, env_data: Dict, context: GlobalContext) -> Dict:
        """Phase 1: Gather comprehensive evidence."""
        self.report_thinking("Gathering evidence: collecting failure reports, system context, and environment data.")
        
        evidence = {
            "initial_failure": failed_report,
            "code_context": code_context,
            "environment_data": env_data
        }

        tooling_agent = self.agent_registry.get("ToolingAgent")
        if tooling_agent:
            diagnostic_result = self.call_agent_v2(
                target_agent=tooling_agent,
                goal="Gather diagnostic evidence for debugging",
                inputs={
                    "commands": ["git status --porcelain", "ls -laR"],
                    "purpose": "Gather repository status and file structure."
                },
                global_context=context
            )
            if diagnostic_result.success:
                evidence["diagnostic_output"] = diagnostic_result.outputs
        
        return evidence

    def _analyze_and_hypothesize(self, evidence: Dict, state: DebuggingState) -> Dict:
        """Phase 2: Analyze evidence and generate a hypothesis, now aware of past attempts."""
        self.report_thinking("Analyzing evidence and past attempts to form a new hypothesis.")
        
        prompt = self._build_analysis_prompt(evidence, state)
        response_str = self.llm_client.invoke(prompt)
        hypothesis = json.loads(response_str)
        
        self.report_intermediate_output("hypothesis_analysis", hypothesis)
        return hypothesis

    def _build_analysis_prompt(self, evidence: Dict, state: DebuggingState) -> str:
        """Constructs a prompt for the LLM to analyze the debugging evidence, including session state."""
        return f"""
        You are an expert debugging agent. Your goal is to solve the following problem.
        Analyze the provided evidence to determine the root cause and the best solution strategy.
        You are in a multi-turn debugging session. Avoid repeating past mistakes.

        **OVERALL PROBLEM TO SOLVE**: {state.problem_description}

        **DEBUGGING HISTORY (What has been tried and failed):**
        - Hypotheses Tested: {json.dumps(state.hypotheses_tested)}
        - Fixes Attempted: {json.dumps(state.fixes_attempted)}
        - Last Known Error: {json.dumps(state.last_error_report, default=str)}

        **CURRENT EVIDENCE:**
        {json.dumps(evidence, indent=2, default=str)}

        Your analysis must be a valid JSON object with this structure:
        {{
            "root_cause_analysis": "Detailed explanation of the bug, considering the history and the overall problem.",
            "primary_hypothesis": "A NEW, UNTRIED hypothesis about the most likely cause.",
            "solution_type": "SURGICAL|DESIGN_CHANGE",
            "recommended_strategy": "A NEW, UNTRIED approach to fix the issue."
        }}
        - Use "SURGICAL" for small, targeted fixes.
        - Use "DESIGN_CHANGE" for larger refactoring or architectural changes.
        """

    def _execute_fix_strategy(self, hypothesis: Dict, evidence: Dict, context: GlobalContext) -> AgentResult:
        """Phase 3: Decide on a fix strategy and delegate to the appropriate agent."""
        solution_type = hypothesis.get("solution_type", "DESIGN_CHANGE")
        self.report_thinking(f"Deciding on fix strategy. Analysis suggests a '{solution_type}' approach.")

        if solution_type == "SURGICAL" and "ScriptExecutorAgent" in self.agent_registry:
            script = self._create_surgical_repair_script(hypothesis, evidence)
            context.add_artifact("instruction_script.json", script.model_dump_json(indent=2), "DebuggingAgent")
            
            return self.call_agent_v2(
                target_agent=self.agent_registry["ScriptExecutorAgent"],
                goal=f"Execute surgical repair script: {script.title}",
                inputs={"instruction_script": script.model_dump()},
                global_context=context
            )
        else: # Default to a design change / refactoring approach
            self.report_thinking("A design change is required. Generating a new technical specification for the CoderAgent.")
            new_spec = self._generate_new_spec_for_coder(hypothesis, evidence)
            context.add_artifact("technical_spec.md", new_spec, "DebuggingAgent")
            
            # Extract the list of file paths from the code_context
            files_to_generate = [file_info['path'] for file_info in evidence.get("code_context", {}).get("files", [])]

            return self.call_agent_v2(
                target_agent=self.agent_registry["CoderAgent"],
                goal="Implement the required design change based on the new specification.",
                inputs={
                    "technical_spec": new_spec,
                    "files_to_generate": files_to_generate
                },
                global_context=context
            )

    def _validate_fix(self, context: GlobalContext) -> AgentResult:
        """Phase 4: Validate the fix by running tests."""
        self.report_thinking("Validating the applied fix by re-running the test suite.")
        
        test_runner = self.agent_registry.get("TestRunnerAgent")
        if not test_runner:
            return self.create_result(success=False, message="TestRunnerAgent not available for validation.")
            
        return self.call_agent_v2(
            target_agent=test_runner,
            goal="Validate that the debugging fix resolved the issue",
            inputs={"run_all_tests": True, "validation_mode": True},
            global_context=context
        )

    def _build_surgical_fix_prompt(self, hypothesis: Dict, evidence: Dict) -> str:
        """Constructs a prompt for the LLM to generate a code fix as a diff."""
        code_context_str = json.dumps(evidence.get("code_context", {}), indent=2)
        
        return f"""
        You are an expert software developer specializing in surgical code fixes.
        Based on the following analysis and code, generate a code patch in the standard 'diff' format.

        **Root Cause Analysis**: {hypothesis.get('root_cause_analysis')}
        **Primary Hypothesis**: {hypothesis.get('primary_hypothesis')}
        **Recommended Strategy**: {hypothesis.get('recommended_strategy')}

        **Relevant Code:**
        ---
        {code_context_str}
        ---

        Generate a JSON object with a single key, "suggested_fix_diff", containing the code patch as a string.
        The diff must be in the standard unified diff format.

        Example:
        {{
            "suggested_fix_diff": "--- a/src/main.py\\n+++ b/src/main.py\\n@@ -1,2 +1,4 @@\\n def my_func():\\n+    # This is the fix\\n     return True"
        }}
        """

    def _create_surgical_repair_script(self, hypothesis: Dict, evidence: Dict) -> InstructionScript:
        """Creates a structured script for a targeted fix, now with an LLM-generated diff."""
        file_to_fix = list(evidence.get("code_context", {}).get("files", [{}])[0].keys())[0]
        
        # --- NEW: Call LLM to generate the fix ---
        fix_prompt = self._build_surgical_fix_prompt(hypothesis, evidence)
        try:
            fix_response_str = self.llm_client.invoke(fix_prompt)
            fix_data = json.loads(fix_response_str)
            fix_content = fix_data.get("suggested_fix_diff", "# LLM failed to generate a valid diff.")
        except Exception as e:
            logger.error(f"Failed to generate surgical fix from LLM: {e}")
            fix_content = f"# Error generating fix: {e}"
        # --- End of new logic ---

        instruction = create_fix_code_instruction(
            instruction_id="fix_001",
            file_path=file_to_fix,
            issue_description=hypothesis["primary_hypothesis"],
            fix_content=fix_content # Use the LLM-generated diff
        )
        
        return InstructionScript(
            script_id=f"repair_{uuid.uuid4().hex[:8]}",
            title=f"Surgical Fix for {hypothesis['primary_hypothesis']}",
            description=hypothesis["recommended_strategy"],
            created_by="DebuggingAgent",
            target_issue=hypothesis["root_cause_analysis"],
            instructions=[instruction]
        )

    def _generate_new_spec_for_coder(self, hypothesis: Dict, evidence: Dict) -> str:
        """Generates a new technical specification for a design change."""
        # Extract file paths from the evidence
        affected_files = [file_info['path'] for file_info in evidence.get("code_context", {}).get("files", [])]

        return f"""
        # New Technical Specification for Design Change
        **Problem:** {hypothesis['root_cause_analysis']}
        **Required Changes:** {hypothesis['recommended_strategy']}
        **Affected Files:** {json.dumps(affected_files)}
        
        Please refactor the code in the affected files to address the above problem.
        """
    
    def _request_user_guidance(self, hypothesis: Dict, state: DebuggingState, context: GlobalContext) -> bool:
        """Pause the loop and ask the user for help."""
        # This requires the UI agent to be available
        ui_agent = self.agent_registry.get("ClarifierAgent") # Using Clarifier as a proxy for UI
        if not ui_agent:
            logger.warning("UI agent not available, cannot request user guidance.")
            return True # Continue without guidance

        prompt = f"""
        I'm having trouble solving this bug. Here is my current analysis:
        
        **Hypothesis**: {hypothesis.get('primary_hypothesis')}
        **Proposed Strategy**: {hypothesis.get('recommended_strategy')}
        
        I have already tried the following fixes:
        - {chr(10).join(f'- {fix}' for fix in state.fixes_attempted)}
        
        Do you have any suggestions, or should I proceed with my proposed strategy? (Type your suggestion or press Enter to proceed)
        """
        
        # In a real system, we'd use a dedicated UI method.
        # For now, we simulate this by calling the ClarifierAgent's prompt method if it exists.
        if hasattr(ui_agent, 'ui') and hasattr(ui_agent.ui, 'prompt_for_input'):
            user_feedback = ui_agent.ui.prompt_for_input(prompt)
            if user_feedback and user_feedback.lower() not in ["", "yes", "proceed", "continue"]:
                # User provided guidance, inject it into the next analysis
                state.hypotheses_tested.append("User provided new guidance.")
                state.last_error_report['user_guidance'] = user_feedback
                state.confidence = 0.9 # Boost confidence based on user feedback
            elif user_feedback.lower() in ["cancel", "stop", "exit"]:
                return False
        return True
