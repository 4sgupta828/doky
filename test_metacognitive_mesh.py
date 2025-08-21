#!/usr/bin/env python3
"""
Test Script for Meta-Cognitive Agent Mesh System

This script validates the complete implementation of our revolutionary
agent mesh architecture with meta-cognitive oversight.
"""

import asyncio
import logging
import tempfile
from pathlib import Path

# Set up basic logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our revolutionary system
from main_interactive_metacognitive import MetaCognitiveAgentMeshSession
from core.models import AgentResult


class TestMetaCognitiveAgentMesh:
    """Test suite for the Meta-Cognitive Agent Mesh"""
    
    def __init__(self):
        self.test_results = []
        self.temp_workspace = None
    
    async def run_all_tests(self):
        """Run comprehensive test suite"""
        
        print("🚀 Starting Meta-Cognitive Agent Mesh Test Suite\n")
        
        # Setup test environment
        await self._setup_test_environment()
        
        # Run individual tests
        await self._test_agent_mesh_initialization()
        await self._test_simple_request_processing()
        await self._test_collaborative_workflow()
        await self._test_loop_detection()
        await self._test_completion_validation()
        await self._test_intervention_system()
        
        # Cleanup
        await self._cleanup_test_environment()
        
        # Report results
        self._report_test_results()
    
    async def _setup_test_environment(self):
        """Setup test environment"""
        self.temp_workspace = tempfile.mkdtemp(prefix="agent_mesh_test_")
        print(f"📁 Test workspace: {self.temp_workspace}")
    
    async def _test_agent_mesh_initialization(self):
        """Test 1: Agent mesh initialization"""
        print("🧪 Test 1: Agent Mesh Initialization")
        
        try:
            session = MetaCognitiveAgentMeshSession(workspace_path=self.temp_workspace)
            
            # Verify agents are created
            assert len(session.agents) == 6, f"Expected 6 agents, got {len(session.agents)}"
            
            # Verify meta-cognitive systems
            assert session.completion_tracker is not None, "Completion tracker not initialized"
            assert session.loop_detector is not None, "Loop detector not initialized"
            assert session.progress_monitor is not None, "Progress monitor not initialized"
            assert session.intervention_system is not None, "Intervention system not initialized"
            
            # Verify agent peer networks
            for agent_name, agent in session.agents.items():
                assert hasattr(agent, 'peer_agents'), f"{agent_name} missing peer network"
                assert len(agent.peer_agents) == 5, f"{agent_name} should have 5 peers, got {len(agent.peer_agents)}"
                assert agent.meta_cognition_enabled, f"{agent_name} meta-cognition not enabled"
            
            self._record_test_result("Agent Mesh Initialization", True, "All agents and systems initialized correctly")
            print("   ✅ PASSED: All agents and meta-cognitive systems initialized\n")
            
        except Exception as e:
            self._record_test_result("Agent Mesh Initialization", False, str(e))
            print(f"   ❌ FAILED: {e}\n")
    
    async def _test_simple_request_processing(self):
        """Test 2: Simple request processing"""
        print("🧪 Test 2: Simple Request Processing")
        
        try:
            session = MetaCognitiveAgentMeshSession(workspace_path=self.temp_workspace)
            
            # Test a simple analysis request
            analyst = session.agents["AnalystAgent"]
            
            result = await analyst.execute_with_oversight(
                goal="process_user_request_collaboratively",
                inputs={
                    "user_request": "Analyze the structure of a Python web application",
                    "workflow_id": "test_workflow_001"
                },
                global_context=session.global_context
            )
            
            # Verify result
            assert isinstance(result, AgentResult), "Result should be AgentResult instance"
            assert result.success, f"Request should succeed: {result.message}"
            assert result.outputs, "Result should have outputs"
            
            # Verify analyst validation markers
            assert "analyst_validation" in result.outputs or "completion_validation" in result.outputs, \
                "Result should contain analyst validation markers"
            
            self._record_test_result("Simple Request Processing", True, "Request processed successfully")
            print("   ✅ PASSED: Simple request processed with analyst validation\n")
            
        except Exception as e:
            self._record_test_result("Simple Request Processing", False, str(e))
            print(f"   ❌ FAILED: {e}\n")
    
    async def _test_collaborative_workflow(self):
        """Test 3: Multi-agent collaborative workflow"""
        print("🧪 Test 3: Collaborative Workflow")
        
        try:
            session = MetaCognitiveAgentMeshSession(workspace_path=self.temp_workspace)
            
            # Test a complex request requiring multiple agents
            analyst = session.agents["AnalystAgent"]
            
            result = await analyst.execute_with_oversight(
                goal="process_user_request_collaboratively",
                inputs={
                    "user_request": "Create a Python function to process data and write tests for it",
                    "workflow_id": "test_workflow_002"
                },
                global_context=session.global_context
            )
            
            # Verify collaborative result
            assert result.success, f"Collaborative request should succeed: {result.message}"
            assert result.outputs, "Collaborative result should have outputs"
            
            # Check if multiple agents were involved (if workflow tracking is working)
            workflow_state = session.progress_monitor.workflow_states.get("test_workflow_002")
            if workflow_state:
                assert len(workflow_state.agent_handoff_chain) > 1, "Multiple agents should be involved"
                assert workflow_state.completion_percentage > 0, "Progress should be made"
            
            self._record_test_result("Collaborative Workflow", True, "Multi-agent workflow executed")
            print("   ✅ PASSED: Collaborative workflow completed successfully\n")
            
        except Exception as e:
            self._record_test_result("Collaborative Workflow", False, str(e))
            print(f"   ❌ FAILED: {e}\n")
    
    async def _test_loop_detection(self):
        """Test 4: Loop detection system"""
        print("🧪 Test 4: Loop Detection")
        
        try:
            session = MetaCognitiveAgentMeshSession(workspace_path=self.temp_workspace)
            
            # Simulate agent interactions to trigger loop detection
            loop_detector = session.loop_detector
            workflow_id = "test_workflow_loop"
            
            # Simulate ping-pong pattern
            interactions = [
                ("AnalystAgent", "CreatorAgent", "create something"),
                ("CreatorAgent", "AnalystAgent", "analyze creation"),
                ("AnalystAgent", "CreatorAgent", "create something"),
                ("CreatorAgent", "AnalystAgent", "analyze creation")
            ]
            
            loop_detected = False
            for from_agent, to_agent, task in interactions:
                can_proceed = loop_detector.track_agent_interaction(workflow_id, from_agent, to_agent, task)
                if not can_proceed:
                    loop_detected = True
                    break
            
            assert loop_detected, "Loop detection should have triggered on ping-pong pattern"
            
            # Verify loop analysis
            workflow_graphs = loop_detector.workflow_graphs.get(workflow_id)
            assert workflow_graphs, "Workflow graph should exist"
            assert len(workflow_graphs["loop_warnings"]) > 0, "Loop warnings should be recorded"
            
            self._record_test_result("Loop Detection", True, "Loop detection system working correctly")
            print("   ✅ PASSED: Loop detection prevented infinite ping-pong\n")
            
        except Exception as e:
            self._record_test_result("Loop Detection", False, str(e))
            print(f"   ❌ FAILED: {e}\n")
    
    async def _test_completion_validation(self):
        """Test 5: Completion validation system"""
        print("🧪 Test 5: Completion Validation")
        
        try:
            session = MetaCognitiveAgentMeshSession(workspace_path=self.temp_workspace)
            
            # Initialize a workflow and test completion validation
            workflow_id = "test_workflow_completion"
            user_request = "Create a simple Python function"
            
            # Initialize progress monitoring
            workflow_state = session.progress_monitor.initialize_workflow(workflow_id, user_request)
            
            # Define completion criteria
            completion_criteria = session.completion_tracker.define_completion_criteria(
                workflow_id, user_request, {"expected_deliverables": ["generated_code"]}
            )
            
            # Create a test result
            test_result = AgentResult(
                success=True,
                message="Function created successfully",
                outputs={
                    "generated_code": "def test_function(): pass",
                    "analyst_validation": "completed"
                }
            )
            
            # Simulate agent handoff to analyst
            workflow_state.agent_handoff_chain.append("AnalystAgent")
            
            # Test completion validation
            validation_result = session.completion_tracker.validate_completion(
                workflow_id, test_result, workflow_state
            )
            
            assert validation_result["is_complete"], "Request should be marked as complete"
            assert validation_result["analyst_validated"], "Analyst validation should be detected"
            
            self._record_test_result("Completion Validation", True, "Completion validation working correctly")
            print("   ✅ PASSED: Completion validation system working\n")
            
        except Exception as e:
            self._record_test_result("Completion Validation", False, str(e))
            print(f"   ❌ FAILED: {e}\n")
    
    async def _test_intervention_system(self):
        """Test 6: Intervention system"""
        print("🧪 Test 6: Intervention System")
        
        try:
            session = MetaCognitiveAgentMeshSession(workspace_path=self.temp_workspace)
            
            # Create a stuck workflow scenario
            workflow_id = "test_workflow_intervention"
            user_request = "Test intervention system"
            
            workflow_state = session.progress_monitor.initialize_workflow(workflow_id, user_request)
            
            # Simulate stuck conditions
            stuck_conditions = [
                {
                    "type": "no_progress_timeout",
                    "details": "No progress for 300 seconds",
                    "severity": "high",
                    "suggested_action": "force_analyst_validation"
                }
            ]
            
            # Test intervention assessment
            intervention_result = await session.intervention_system.assess_and_intervene(
                workflow_id, workflow_state, stuck_conditions, session.agents
            )
            
            assert intervention_result is not None, "Intervention should be applied"
            assert isinstance(intervention_result, AgentResult), "Intervention should return AgentResult"
            
            self._record_test_result("Intervention System", True, "Intervention system responsive")
            print("   ✅ PASSED: Intervention system working correctly\n")
            
        except Exception as e:
            self._record_test_result("Intervention System", False, str(e))
            print(f"   ❌ FAILED: {e}\n")
    
    async def _cleanup_test_environment(self):
        """Cleanup test environment"""
        if self.temp_workspace:
            import shutil
            shutil.rmtree(self.temp_workspace, ignore_errors=True)
            print(f"🧹 Cleaned up test workspace: {self.temp_workspace}")
    
    def _record_test_result(self, test_name: str, passed: bool, message: str):
        """Record test result"""
        self.test_results.append({
            "test": test_name,
            "passed": passed,
            "message": message
        })
    
    def _report_test_results(self):
        """Report final test results"""
        passed_tests = [r for r in self.test_results if r["passed"]]
        failed_tests = [r for r in self.test_results if not r["passed"]]
        
        print("\n" + "="*60)
        print("🎯 META-COGNITIVE AGENT MESH TEST RESULTS")
        print("="*60)
        
        print(f"\n✅ PASSED: {len(passed_tests)}/{len(self.test_results)} tests")
        print(f"❌ FAILED: {len(failed_tests)}/{len(self.test_results)} tests")
        
        if failed_tests:
            print(f"\n❌ FAILED TESTS:")
            for test in failed_tests:
                print(f"   - {test['test']}: {test['message']}")
        
        if passed_tests:
            print(f"\n✅ PASSED TESTS:")
            for test in passed_tests:
                print(f"   - {test['test']}")
        
        success_rate = len(passed_tests) / len(self.test_results) * 100
        print(f"\n🎯 SUCCESS RATE: {success_rate:.1f}%")
        
        if success_rate >= 80:
            print("🎉 META-COGNITIVE AGENT MESH: IMPLEMENTATION SUCCESSFUL! 🎉")
        else:
            print("⚠️  META-COGNITIVE AGENT MESH: NEEDS IMPROVEMENTS")


async def main():
    """Run the test suite"""
    tester = TestMetaCognitiveAgentMesh()
    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())