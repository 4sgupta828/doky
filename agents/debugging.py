# agents/debugging.py
import json
import logging
from typing import Dict, Any

# Foundational dependencies
from .base import BaseAgent
from core.context import GlobalContext
from core.models import AgentResponse, TaskNode

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
        
        for iteration in range(self.max_debug_iterations):
            logger.info(f"--- Debugging Iteration {iteration + 1}/{self.max_debug_iterations} ---")
            
            try:
                # Phase 1: Evidence Gathering
                logger.info("Phase 1: Gathering comprehensive evidence...")
                evidence = self._gather_evidence(context, current_task)
                
                # Phase 2: Root Cause Analysis & Hypothesis Generation
                logger.info("Phase 2: Analyzing root cause and generating hypotheses...")
                hypothesis = self._analyze_and_hypothesize(evidence, context, current_task)
                
                # Phase 3: Smart Fix Strategy Decision
                logger.info("Phase 3: Determining fix strategy...")
                fix_result = self._execute_fix_strategy(hypothesis, context, current_task)
                
                # Phase 4: Validation
                logger.info("Phase 4: Validating the fix...")
                validation_result = self._validate_fix(context, current_task)
                
                if validation_result["success"]:
                    logger.info(f"✅ Debugging succeeded after {iteration + 1} iterations")
                    
                    # Display transparency summary for user
                    context.print_communication_summary()
                    
                    return AgentResponse(
                        success=True,
                        message=f"Successfully resolved issue after {iteration + 1} debugging iterations. Final solution: {hypothesis['solution_type']}",
                        artifacts_generated=fix_result.get("artifacts_generated", [])
                    )
                else:
                    logger.warning(f"❌ Fix attempt {iteration + 1} failed: {validation_result['message']}")
                    # Store failure info for next iteration
                    context.add_artifact(f"debug_iteration_{iteration}_failure.json", 
                                       json.dumps(validation_result), current_task.task_id)
                    
            except Exception as e:
                logger.error(f"Debugging iteration {iteration + 1} failed with exception: {e}")
                if iteration == self.max_debug_iterations - 1:
                    # Last iteration failed
                    break
                continue
        
        # All iterations failed - still show transparency summary
        context.print_communication_summary()
        
        return AgentResponse(
            success=False,
            message=f"Unable to resolve issue after {self.max_debug_iterations} debugging iterations. Manual intervention required.",
            artifacts_generated=[f"debug_iteration_{i}_failure.json" for i in range(self.max_debug_iterations)]
        )

    def _gather_evidence(self, context: GlobalContext, current_task: TaskNode) -> Dict[str, Any]:
        """Phase 1: Comprehensive evidence gathering using ToolingAgent and direct analysis."""
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
            
            # Get code context
            code_context = context.get_artifact("targeted_code_context.txt")
            if code_context:
                evidence["code_context"]["targeted_code"] = code_context
            
            # Use ToolingAgent if available for deeper evidence gathering
            if "ToolingAgent" in self.agent_registry:
                # Log the communication for transparency
                self.log_communication(context, "ToolingAgent", "delegation", 
                                     "Gather comprehensive debugging evidence", 
                                     {"task_type": "evidence_gathering", "artifacts_available": list(context.artifacts.keys())},
                                     current_task.task_id)
                
                tooling_agent = self.agent_registry["ToolingAgent"]
                # Create a subtask for evidence gathering
                evidence_task = TaskNode(
                    goal="Gather comprehensive debugging evidence",
                    assigned_agent="ToolingAgent",
                    input_artifact_keys=list(context.artifacts.keys())
                )
                
                tooling_result = tooling_agent.execute("Analyze system and environment for debugging", context, evidence_task)
                if tooling_result.success:
                    # Log successful response for transparency
                    self.log_communication(context, "ToolingAgent", "response", 
                                         f"Successfully gathered evidence: {len(tooling_result.artifacts_generated or [])} artifacts",
                                         {"artifacts": tooling_result.artifacts_generated or [], "success": True},
                                         current_task.task_id)
                    # Extract additional evidence from tooling agent results
                    for artifact_key in tooling_result.artifacts_generated or []:
                        artifact_content = context.get_artifact(artifact_key)
                        if artifact_content:
                            evidence["environment_context"][artifact_key] = artifact_content
                else:
                    # Log failed response for transparency
                    self.log_communication(context, "ToolingAgent", "error", 
                                         f"Failed to gather evidence: {tooling_result.message}",
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
                
            logger.info(f"Evidence gathered: {len(evidence)} categories")
            return evidence
            
        except Exception as e:
            logger.error(f"Evidence gathering failed: {e}")
            return evidence

    def _analyze_and_hypothesize(self, evidence: Dict[str, Any], context: GlobalContext, current_task: TaskNode) -> Dict[str, Any]:
        """Phase 2: Root cause analysis and hypothesis generation using LLM."""
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
            
            response = self.llm_client.invoke(analysis_prompt)
            hypothesis = json.loads(response)
            
            # Store analysis for reference
            context.add_artifact("debug_hypothesis.json", json.dumps(hypothesis, indent=2), current_task.task_id)
            
            logger.info(f"Generated hypothesis: {hypothesis.get('primary_hypothesis', 'Unknown')}")
            logger.info(f"Solution type: {hypothesis.get('solution_type', 'Unknown')}")
            
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

    def _execute_fix_strategy(self, hypothesis: Dict[str, Any], context: GlobalContext, current_task: TaskNode) -> Dict[str, Any]:
        """Phase 3: Execute the chosen fix strategy via appropriate agents."""
        solution_type = hypothesis.get("solution_type", "DESIGN_CHANGE")
        
        try:
            if solution_type == "SURGICAL" and "CodeGenerationAgent" in self.agent_registry:
                # Direct surgical fix via CodeGenerationAgent
                logger.info("Executing surgical fix via CodeGenerationAgent")
                
                # Log the delegation for transparency
                self.log_communication(context, "CodeGenerationAgent", "delegation", 
                                     f"Apply surgical fix: {hypothesis.get('recommended_strategy', 'Fix the issue')}",
                                     {"solution_type": "SURGICAL", "strategy": hypothesis.get('recommended_strategy')},
                                     current_task.task_id)
                
                coder_agent = self.agent_registry["CodeGenerationAgent"]
                fix_task = TaskNode(
                    goal=f"Apply surgical fix: {hypothesis.get('recommended_strategy', 'Fix the issue')}",
                    assigned_agent="CodeGenerationAgent",
                    input_artifact_keys=["debug_hypothesis.json"]
                )
                
                result = coder_agent.execute(hypothesis["recommended_strategy"], context, fix_task)
                
                # Log the response for transparency
                response_type = "response" if result.success else "error"
                self.log_communication(context, "CodeGenerationAgent", response_type, 
                                     result.message,
                                     {"success": result.success, "artifacts": result.artifacts_generated},
                                     current_task.task_id)
                return {"success": result.success, "artifacts_generated": result.artifacts_generated, "message": result.message}
                
            elif solution_type == "DESIGN_CHANGE" and "SpecGenerationAgent" in self.agent_registry and "CodeGenerationAgent" in self.agent_registry:
                # Complex fix requiring specification first
                logger.info("Executing design change via SpecAgent -> CoderAgent")
                
                # Log the delegation to SpecAgent for transparency
                self.log_communication(context, "SpecGenerationAgent", "delegation", 
                                     f"Create specification for: {hypothesis.get('recommended_strategy', 'Fix the issue')}",
                                     {"solution_type": "DESIGN_CHANGE", "hypothesis": hypothesis.get('primary_hypothesis')},
                                     current_task.task_id)
                
                spec_agent = self.agent_registry["SpecGenerationAgent"]
                coder_agent = self.agent_registry["CodeGenerationAgent"]
                
                # First, generate specification
                spec_task = TaskNode(
                    goal=f"Create specification for: {hypothesis.get('recommended_strategy', 'Fix the issue')}",
                    assigned_agent="SpecGenerationAgent",
                    input_artifact_keys=["debug_hypothesis.json"]
                )
                
                spec_result = spec_agent.execute(f"Create spec to resolve: {hypothesis['primary_hypothesis']}", context, spec_task)
                
                # Log the SpecAgent response
                spec_response_type = "response" if spec_result.success else "error"
                self.log_communication(context, "SpecGenerationAgent", spec_response_type, 
                                     spec_result.message,
                                     {"success": spec_result.success, "artifacts": spec_result.artifacts_generated},
                                     current_task.task_id)
                
                if spec_result.success:
                    # Log the delegation to CodeGenerationAgent for transparency
                    self.log_communication(context, "CodeGenerationAgent", "delegation", 
                                         "Implement the debugging fix specification",
                                         {"input_artifacts": spec_result.artifacts_generated or [], "chain_source": "SpecGenerationAgent"},
                                         current_task.task_id)
                    
                    # Then implement the specification
                    code_task = TaskNode(
                        goal="Implement the debugging fix specification",
                        assigned_agent="CodeGenerationAgent", 
                        input_artifact_keys=spec_result.artifacts_generated or []
                    )
                    
                    code_result = coder_agent.execute("Implement the specification to fix the issue", context, code_task)
                    
                    # Log the CodeGenerationAgent response
                    code_response_type = "response" if code_result.success else "error"
                    self.log_communication(context, "CodeGenerationAgent", code_response_type, 
                                         code_result.message,
                                         {"success": code_result.success, "artifacts": code_result.artifacts_generated},
                                         current_task.task_id)
                    return {"success": code_result.success, "artifacts_generated": code_result.artifacts_generated, "message": code_result.message}
                else:
                    return {"success": False, "message": f"Specification generation failed: {spec_result.message}"}
                    
            else:
                # Fallback: create manual fix recommendations
                logger.warning("Required agents not available, creating manual fix recommendations")
                
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
                return {"success": True, "artifacts_generated": ["manual_fix_recommendations.json"], "message": "Created manual fix recommendations"}
                
        except Exception as e:
            logger.error(f"Fix strategy execution failed: {e}")
            return {"success": False, "message": f"Fix execution failed: {e}"}

    def _validate_fix(self, context: GlobalContext, current_task: TaskNode) -> Dict[str, Any]:
        """Phase 4: Validate the fix using TestRunner and potentially TestGenerator."""
        try:
            if "TestRunnerAgent" in self.agent_registry:
                logger.info("Validating fix using TestRunnerAgent")
                
                # Log the delegation for transparency
                self.log_communication(context, "TestRunnerAgent", "delegation", 
                                     "Validate that the debugging fix resolved the issue",
                                     {"task_type": "validation", "purpose": "test fix"},
                                     current_task.task_id)
                
                test_runner = self.agent_registry["TestRunnerAgent"]
                validation_task = TaskNode(
                    goal="Validate that the debugging fix resolved the issue",
                    assigned_agent="TestRunnerAgent"
                )
                
                result = test_runner.execute("Run tests to validate fix", context, validation_task)
                
                # Log the TestRunner response
                test_response_type = "response" if result.success else "error"
                self.log_communication(context, "TestRunnerAgent", test_response_type, 
                                     result.message,
                                     {"success": result.success, "artifacts": result.artifacts_generated},
                                     current_task.task_id)
                
                if result.success:
                    return {"success": True, "message": "Fix validated successfully - tests now pass"}
                else:
                    # Tests still failing, try to enhance test coverage if TestGenerator available
                    if "TestGeneratorAgent" in self.agent_registry:
                        logger.info("Tests still failing, enhancing test coverage")
                        
                        # Log the delegation for transparency
                        self.log_communication(context, "TestGeneratorAgent", "delegation", 
                                             "Generate additional tests to better validate the fix",
                                             {"reason": "tests_still_failing", "purpose": "enhance_coverage"},
                                             current_task.task_id)
                        
                        test_gen = self.agent_registry["TestGeneratorAgent"]
                        gen_task = TaskNode(
                            goal="Generate additional tests to better validate the fix",
                            assigned_agent="TestGeneratorAgent"
                        )
                        
                        gen_result = test_gen.execute("Generate targeted tests for debugging validation", context, gen_task)
                        
                        # Log the TestGenerator response
                        gen_response_type = "response" if gen_result.success else "error"
                        self.log_communication(context, "TestGeneratorAgent", gen_response_type, 
                                             gen_result.message,
                                             {"success": gen_result.success, "artifacts": gen_result.artifacts_generated},
                                             current_task.task_id)
                    
                    return {"success": False, "message": f"Validation failed: {result.message}"}
            else:
                logger.warning("TestRunnerAgent not available for validation")
                return {"success": False, "message": "Cannot validate - TestRunnerAgent not available"}
                
        except Exception as e:
            logger.error(f"Fix validation failed: {e}")
            return {"success": False, "message": f"Validation error: {e}"}


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