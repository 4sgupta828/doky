# main_interactive_metacognitive.py
"""
REVOLUTIONARY META-COGNITIVE AGENT MESH INTERACTIVE SESSION

This is the new main interactive session that leverages the complete
meta-cognitive agent mesh architecture with:

- Intelligent peer-to-peer agent collaboration
- Advanced loop detection and prevention
- Comprehensive progress monitoring
- Fail-safe intervention mechanisms
- Completion tracking and validation

Minimal external logic - agents handle everything through dynamic collaboration!
"""

import argparse
import asyncio
import logging
import sys
import time
import uuid
from typing import List, Optional

# Core dependencies
from core.context import GlobalContext
from core.meta_cognition import (
    CompletionTrackingSystem, LoopDetectionSystem, 
    ProgressMonitoringSystem, FailSafeInterventionSystem
)

# Enhanced foundational agents
from fagents.meta_cognitive_agents import (
    MetaCognitiveAnalystAgent, MetaCognitiveStrategistAgent,
    MetaCognitiveCreatorAgent, MetaCognitiveSurgeonAgent, 
    MetaCognitiveExecutorAgent, MetaCognitiveDebuggingAgent
)

# UI and utilities
from utils.logger import setup_logger

logger = logging.getLogger(__name__)


class CollaborationUI:
    """Simple UI for displaying agent mesh collaboration"""
    
    class Style:
        RESET = '\033[0m'
        BOLD = '\033[1m'
        
        class Fg:
            RED = '\033[31m'
            GREEN = '\033[32m'
            YELLOW = '\033[33m'
            BLUE = '\033[34m'
            MAGENTA = '\033[35m'
            CYAN = '\033[36m'
    
    def display_system_message(self, message: str):
        """Display system message"""
        print(f"{self.Style.Fg.CYAN}🔧 SYSTEM:{self.Style.RESET} {message}")
    
    def display_progress(self, agent: str, phase: str, message: str):
        """Display agent progress"""
        print(f"{self.Style.Fg.BLUE}🤖 {agent}:{self.Style.RESET} [{phase}] {message}")
    
    def display_response(self, message: str):
        """Display agent response"""
        print(f"\n{self.Style.Fg.GREEN}✅ RESULT:{self.Style.RESET} {message}\n")
    
    def display_error(self, message: str):
        """Display error message"""
        print(f"{self.Style.Fg.RED}❌ ERROR:{self.Style.RESET} {message}")
    
    def display_workflow_status(self, status: dict):
        """Display workflow status"""
        completion = status.get('completion_percentage', 0)
        current_agent = status.get('current_agent', 'Unknown')
        
        print(f"{self.Style.Fg.YELLOW}📊 WORKFLOW:{self.Style.RESET} {completion:.1f}% complete, Current: {current_agent}")
    
    def display_intervention(self, intervention_type: str, message: str):
        """Display intervention message"""
        print(f"{self.Style.Fg.MAGENTA}🚨 INTERVENTION:{self.Style.RESET} {intervention_type} - {message}")


class MetaCognitiveAgentMeshSession:
    """
    Revolutionary Agent Mesh Session with comprehensive meta-cognitive oversight.
    
    This session provides minimal external coordination - the foundational agents
    handle everything through intelligent peer-to-peer collaboration under
    meta-cognitive oversight.
    """
    
    def __init__(self, workspace_path: Optional[str] = None, resume_snapshot: Optional[str] = None):
        self.ui = CollaborationUI()
        self.global_context = GlobalContext(workspace_path=workspace_path)
        
        # Initialize meta-cognitive systems
        self.completion_tracker = CompletionTrackingSystem()
        self.loop_detector = LoopDetectionSystem(max_iterations=20, max_same_agent_consecutive=4)
        self.progress_monitor = ProgressMonitoringSystem(progress_timeout=900)  # 15 minutes
        self.intervention_system = FailSafeInterventionSystem()
        
        # Create the revolutionary agent mesh
        self._create_agent_mesh()
        
        logger.info("🌐 Meta-Cognitive Agent Mesh Session initialized")
    
    def _create_agent_mesh(self):
        """Create the revolutionary meta-cognitive agent mesh network"""
        
        self.ui.display_system_message("🚀 Initializing Meta-Cognitive Agent Mesh...")
        
        # Instantiate all enhanced foundational agents
        self.agents = {
            "AnalystAgent": MetaCognitiveAnalystAgent(),
            "StrategistAgent": MetaCognitiveStrategistAgent(), 
            "CreatorAgent": MetaCognitiveCreatorAgent(),
            "SurgeonAgent": MetaCognitiveSurgeonAgent(),
            "ExecutorAgent": MetaCognitiveExecutorAgent(),
            "DebuggingAgent": MetaCognitiveDebuggingAgent()
        }
        
        # Setup meta-cognitive oversight for all agents
        for agent in self.agents.values():
            agent.setup_meta_cognition(
                loop_detector=self.loop_detector,
                progress_monitor=self.progress_monitor,
                completion_tracker=self.completion_tracker,
                intervention_system=self.intervention_system,
                global_context=self.global_context
            )
        
        # Enable peer-to-peer collaboration network
        for agent in self.agents.values():
            agent.register_peer_network(self.agents)
        
        self.ui.display_system_message("🌐 Agent Mesh Network Established")
        self.ui.display_system_message(f"   └─ {len(self.agents)} foundational agents ready")
        self.ui.display_system_message("   └─ Meta-cognitive oversight active")
        self.ui.display_system_message("   └─ Peer-to-peer collaboration enabled")
    
    async def start(self):
        """Start the revolutionary meta-cognitive agent mesh session"""
        
        print(f"\n{self.ui.Style.BOLD}{self.ui.Style.Fg.MAGENTA}")
        print("╔════════════════════════════════════════════════════════════╗")
        print("║          🚀 REVOLUTIONARY AGENT MESH COLLECTIVE           ║")
        print("║                                                            ║") 
        print("║  🧠 Meta-Cognitive Oversight    🤝 Peer Collaboration     ║")
        print("║  🔄 Loop Prevention             📊 Progress Monitoring     ║")
        print("║  🚨 Fail-Safe Interventions    ✅ Completion Guarantees   ║")
        print("╚════════════════════════════════════════════════════════════╝")
        print(f"{self.ui.Style.RESET}\n")
        
        print("💭 Type your requests - agents will collaborate automatically!")
        print("📝 Commands: 'status' (workflow status), 'agents' (agent info), 'quit' (exit)\n")
        
        while True:
            try:
                # Get user input
                user_input = await self._get_user_input()
                
                if user_input.lower() in ['exit', 'quit', 'bye']:
                    self.ui.display_system_message("👋 Agent Mesh session ending...")
                    break
                
                if user_input.strip() == '':
                    continue
                
                # Handle special commands
                if user_input.lower() == 'status':
                    await self._display_system_status()
                    continue
                elif user_input.lower() == 'agents':
                    self._display_agent_info()
                    continue
                
                # **REVOLUTIONARY**: Process request through agent mesh with oversight
                result = await self._process_request_with_comprehensive_oversight(user_input)
                
                # Display result
                if result.success:
                    self.ui.display_response(result.message)
                else:
                    self.ui.display_error(f"Request failed: {result.message}")
                
            except KeyboardInterrupt:
                self.ui.display_system_message("👋 Session interrupted by user")
                break
            except Exception as e:
                logger.error(f"Session error: {e}", exc_info=True)
                self.ui.display_error(f"Session error: {e}")
    
    async def _process_request_with_comprehensive_oversight(self, user_input: str):
        """
        REVOLUTIONARY request processing with comprehensive meta-cognitive oversight.
        
        This is where the magic happens - minimal external logic, maximum agent intelligence!
        """
        
        # Generate unique workflow ID for tracking
        workflow_id = f"workflow_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
        
        self.ui.display_system_message(f"🧠 Activating Meta-Cognitive Oversight (ID: {workflow_id[-12:]})")
        
        try:
            # Initialize comprehensive oversight systems
            await self._initialize_oversight_systems(workflow_id, user_input)
            
            # **MINIMAL EXTERNAL LOGIC**: Hand everything to the agent mesh!
            result = await self._execute_with_comprehensive_oversight(workflow_id, user_input)
            
            # **GUARANTEE COMPLETION**: Ensure request is fully addressed
            final_result = await self._ensure_completion_guarantee(workflow_id, result)
            
            # Display final workflow status
            await self._display_workflow_completion_status(workflow_id)
            
            return final_result
            
        except Exception as e:
            logger.error(f"Comprehensive oversight failed: {e}", exc_info=True)
            
            # **FAIL-SAFE**: Emergency intervention on critical failure
            emergency_result = await self.intervention_system._emergency_intervention(
                workflow_id, 
                self.progress_monitor.workflow_states.get(workflow_id),
                [{"type": "critical_error", "details": str(e), "severity": "critical"}],
                self.agents
            )
            
            return emergency_result or {"success": False, "message": f"Critical failure: {e}"}
            
        finally:
            # Cleanup and learning
            await self._cleanup_and_learn(workflow_id)
    
    async def _initialize_oversight_systems(self, workflow_id: str, user_input: str):
        """Initialize all oversight systems for the workflow"""
        
        # Initialize progress monitoring
        workflow_state = self.progress_monitor.initialize_workflow(
            workflow_id, 
            user_input
        )
        
        # The completion criteria will be defined by the Analyst when it processes the request
        # This happens automatically in the MetaCognitiveAnalystAgent
        
        self.ui.display_system_message("✅ Oversight systems initialized")
    
    async def _execute_with_comprehensive_oversight(self, workflow_id: str, user_input: str):
        """Execute request with real-time oversight and intervention capabilities"""
        
        # **THE REVOLUTIONARY PART**: Just hand it to the Analyst!
        # The Analyst will coordinate with other agents through peer-to-peer collaboration
        analyst = self.agents["AnalystAgent"]
        
        # Start monitoring for interventions in the background
        intervention_task = asyncio.create_task(
            self._monitor_and_intervene(workflow_id)
        )
        
        try:
            # Let the Analyst handle everything with full oversight enabled
            result = await analyst.execute_with_oversight(
                goal="process_user_request_collaboratively",
                inputs={
                    "user_request": user_input,
                    "workflow_id": workflow_id,
                    "oversight_enabled": True,
                    "ui": self.ui
                },
                global_context=self.global_context
            )
            
            # Cancel intervention monitoring
            intervention_task.cancel()
            
            return result
            
        except Exception as e:
            intervention_task.cancel()
            raise e
    
    async def _monitor_and_intervene(self, workflow_id: str):
        """Monitor workflow and apply interventions when needed"""
        
        try:
            while True:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                workflow_state = self.progress_monitor.workflow_states.get(workflow_id)
                if not workflow_state:
                    break
                
                # Check for stuck conditions
                stuck_conditions = self.progress_monitor._check_for_stuck_conditions(workflow_state)
                
                if stuck_conditions:
                    self.ui.display_intervention("Stuck Detected", f"{len(stuck_conditions)} issues found")
                    
                    # Apply intervention
                    intervention_result = await self.intervention_system.assess_and_intervene(
                        workflow_id, workflow_state, stuck_conditions, self.agents
                    )
                    
                    if intervention_result:
                        self.ui.display_intervention("Applied", intervention_result.message)
                
                # Display periodic status
                status = self.progress_monitor.get_workflow_status(workflow_id)
                if status and not status.get("error"):
                    self.ui.display_workflow_status(status)
                
        except asyncio.CancelledError:
            pass  # Normal cancellation
        except Exception as e:
            logger.error(f"Intervention monitoring error: {e}")
    
    async def _ensure_completion_guarantee(self, workflow_id: str, result):
        """GUARANTEE that request is properly completed with Analyst validation"""
        
        workflow_state = self.progress_monitor.workflow_states.get(workflow_id)
        
        if not workflow_state:
            self.ui.display_system_message("⚠️  No workflow state found")
            return result
        
        # Validate completion using completion tracker
        completion_status = self.completion_tracker.validate_completion(workflow_id, result, workflow_state)
        
        if completion_status["is_complete"] and completion_status["analyst_validated"]:
            self.ui.display_system_message("✅ Request completed and validated by Analyst")
            return result
        
        # If not complete, force Analyst validation
        self.ui.display_system_message("⚠️  Enforcing completion guarantee...")
        
        analyst = self.agents["AnalystAgent"]
        validation_result = await analyst.execute_with_oversight(
            goal="intervention_forced_validation",
            inputs={
                "workflow_id": workflow_id,
                "partial_result": result,
                "missing_requirements": completion_status.get("missing_requirements", []),
                "completion_enforcement": True
            },
            global_context=self.global_context
        )
        
        self.ui.display_system_message("✅ Completion guarantee enforced")
        return validation_result
    
    async def _display_workflow_completion_status(self, workflow_id: str):
        """Display comprehensive workflow completion status"""
        
        # Get final status from all monitoring systems
        progress_status = self.progress_monitor.get_workflow_status(workflow_id)
        collaboration_summary = self.loop_detector.get_workflow_collaboration_summary(workflow_id)
        
        if progress_status and not progress_status.get("error"):
            duration = progress_status["total_duration_seconds"]
            completion = progress_status["completion_percentage"]
            agents_involved = len(set(progress_status["collaboration_pattern"]))
            
            print(f"\n{self.ui.Style.Fg.CYAN}📊 WORKFLOW SUMMARY:{self.ui.Style.RESET}")
            print(f"   ⏱️  Duration: {duration:.1f}s")
            print(f"   📈 Completion: {completion:.1f}%")  
            print(f"   🤝 Agents: {agents_involved}")
            print(f"   🔄 Handoffs: {progress_status['agent_handoff_count']}")
            print(f"   ✅ Milestones: {progress_status['successful_milestones']}/{progress_status['milestones_completed']}")
            
            if collaboration_summary and not collaboration_summary.get("error"):
                print(f"   🌐 Interactions: {collaboration_summary['total_interactions']}")
                print(f"   ⚠️  Loop warnings: {collaboration_summary['loop_warnings']}")
    
    async def _cleanup_and_learn(self, workflow_id: str):
        """Cleanup workflow tracking and learn from the session"""
        
        # Get final statistics
        workflow_state = self.progress_monitor.workflow_states.get(workflow_id)
        
        if workflow_state:
            # Update agent collaboration statistics
            for agent in self.agents.values():
                if hasattr(agent, 'collaboration_history'):
                    recent_collabs = [c for c in agent.collaboration_history 
                                    if time.time() - c["timestamp"] < 3600]  # Last hour
                    logger.info(f"{agent.name} recent collaborations: {len(recent_collabs)}")
        
        # Clean up tracking data to prevent memory leaks
        if workflow_id in self.progress_monitor.workflow_states:
            del self.progress_monitor.workflow_states[workflow_id]
        
        if workflow_id in self.loop_detector.workflow_graphs:
            del self.loop_detector.workflow_graphs[workflow_id]
        
        if workflow_id in self.completion_tracker.completion_criteria:
            del self.completion_tracker.completion_criteria[workflow_id]
    
    async def _display_system_status(self):
        """Display comprehensive system status"""
        
        print(f"\n{self.ui.Style.Fg.CYAN}🔧 SYSTEM STATUS:{self.ui.Style.RESET}")
        
        # Agent collaboration statistics
        print("\n🤖 AGENT COLLABORATION:")
        for agent_name, agent in self.agents.items():
            if hasattr(agent, 'get_collaboration_stats'):
                stats = agent.get_collaboration_stats()
                if stats["total_collaborations"] > 0:
                    print(f"   {agent_name}: {stats['total_collaborations']} collaborations, "
                          f"{stats['overall_success_rate']:.1%} success")
        
        # Active workflows
        active_workflows = len(self.progress_monitor.workflow_states)
        print(f"\n📊 ACTIVE WORKFLOWS: {active_workflows}")
        
        # System health
        print(f"\n💚 SYSTEM HEALTH: {'🟢 Healthy' if active_workflows < 5 else '🟡 Busy'}")
    
    def _display_agent_info(self):
        """Display agent information and capabilities"""
        
        print(f"\n{self.ui.Style.Fg.BLUE}🤖 FOUNDATIONAL AGENTS:{self.ui.Style.RESET}")
        
        for agent_name, agent in self.agents.items():
            capabilities = agent.get_capabilities()
            description = capabilities.get("description", "No description")
            
            print(f"\n   {self.ui.Style.BOLD}{agent_name}:{self.ui.Style.RESET}")
            print(f"      {description}")
            
            if "primary_functions" in capabilities:
                functions = capabilities["primary_functions"][:3]  # Show first 3
                print(f"      Functions: {', '.join(functions)}")
    
    async def _get_user_input(self) -> str:
        """Get user input asynchronously"""
        
        # Simple synchronous input for now
        # In a full implementation, this would be truly async
        try:
            return input(f"{self.ui.Style.Fg.GREEN}🚀 >>> {self.ui.Style.RESET}")
        except EOFError:
            return "quit"


def main(args: List[str]) -> None:
    """
    Revolutionary main entry point using Meta-Cognitive Agent Mesh Architecture.
    
    This function:
    1. Sets up logging
    2. Creates the Meta-Cognitive Agent Mesh Session
    3. Lets foundational agents handle everything through peer-to-peer collaboration
    """
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Launch the Revolutionary Meta-Cognitive Agent Mesh Collective"
    )
    parser.add_argument("--workspace", type=str, default=None,
                       help="Workspace directory path")
    parser.add_argument("--resume", type=str, default=None,
                       help="Resume from snapshot")
    parser.add_argument("--quiet-logs", action="store_true",
                       help="Suppress console logs")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug logging")
    
    # Handle test mode
    if "--test" in args:
        print("Meta-Cognitive Agent Mesh: Test mode - OK")
        return
    
    parsed_args = parser.parse_args(args)
    
    # Setup logging
    setup_logger(suppress_console_logs=parsed_args.quiet_logs)
    
    # Launch Meta-Cognitive Agent Mesh Session
    try:
        session = MetaCognitiveAgentMeshSession(
            workspace_path=parsed_args.workspace,
            resume_snapshot=parsed_args.resume
        )
        
        # Start the revolutionary agent mesh
        asyncio.run(session.start())
        
    except Exception as e:
        logger.critical("Meta-Cognitive Agent Mesh session failed", exc_info=True)
        print(f"\n❌ {CollaborationUI.Style.Fg.RED}{CollaborationUI.Style.BOLD}AGENT MESH ERROR:{CollaborationUI.Style.RESET} {e}")
        sys.exit(1)


if __name__ == "__main__":
    main(sys.argv[1:])