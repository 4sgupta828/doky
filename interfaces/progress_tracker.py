# interfaces/progress_tracker.py
import logging
from typing import Any, List, Optional, Dict
from dataclasses import dataclass, field
from datetime import datetime

# Get a logger instance for this module
logger = logging.getLogger(__name__)

@dataclass
class ProgressStep:
    """Represents a single step in an agent's progress."""
    step_name: str
    details: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    status: str = "in_progress"  # "in_progress", "completed", "failed"
    output: Optional[Any] = None
    error: Optional[str] = None

@dataclass
class AgentProgress:
    """Tracks the complete progress of an agent's execution."""
    agent_name: str
    task_id: str
    goal: str
    started_at: datetime = field(default_factory=datetime.now)
    current_step: Optional[str] = None
    steps: List[ProgressStep] = field(default_factory=list)
    thinking_logs: List[str] = field(default_factory=list)
    intermediate_outputs: Dict[str, Any] = field(default_factory=dict)
    
    def add_step(self, step_name: str, details: str = None) -> ProgressStep:
        """Adds a new progress step and marks it as current."""
        step = ProgressStep(step_name=step_name, details=details)
        self.steps.append(step)
        self.current_step = step_name
        return step
    
    def complete_step(self, output: Any = None):
        """Marks the current step as completed."""
        if self.steps:
            self.steps[-1].status = "completed"
            if output is not None:
                self.steps[-1].output = output
    
    def fail_step(self, error: str):
        """Marks the current step as failed."""
        if self.steps:
            self.steps[-1].status = "failed"
            self.steps[-1].error = error
    
    def add_thinking(self, thought: str):
        """Records the agent's thought process."""
        self.thinking_logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {thought}")
    
    def add_intermediate_output(self, key: str, content: Any):
        """Records intermediate outputs like partial code, specs, etc."""
        self.intermediate_outputs[key] = content

class ProgressTracker:
    """
    A centralized progress tracking system that agents can use to report
    their progress, thoughts, and intermediate outputs for UI visibility.
    """
    
    def __init__(self, ui_interface=None):
        """Initialize the progress tracker with optional UI interface."""
        self.ui_interface = ui_interface
        self.active_progresses: Dict[str, AgentProgress] = {}
        self.completed_progresses: List[AgentProgress] = []
    
    def start_agent_progress(self, agent_name: str, task_id: str, goal: str) -> AgentProgress:
        """Start tracking progress for an agent."""
        progress = AgentProgress(
            agent_name=agent_name, 
            task_id=task_id, 
            goal=goal
        )
        self.active_progresses[task_id] = progress
        logger.info(f"Started progress tracking for {agent_name} on task {task_id}")
        return progress
    
    def report_progress(self, task_id: str, step: str, details: str = None):
        """Report a progress step for a task."""
        if task_id not in self.active_progresses:
            logger.warning(f"No active progress found for task {task_id}")
            return
        
        progress = self.active_progresses[task_id]
        progress.add_step(step, details)
        
        # Display to UI if available
        if self.ui_interface and hasattr(self.ui_interface, 'display_agent_progress'):
            self.ui_interface.display_agent_progress(progress.agent_name, step, details)
        
        logger.debug(f"{progress.agent_name} progress: {step}")
    
    def report_thinking(self, task_id: str, thought: str):
        """Report the agent's thought process."""
        if task_id not in self.active_progresses:
            logger.warning(f"No active progress found for task {task_id}")
            return
        
        progress = self.active_progresses[task_id]
        progress.add_thinking(thought)
        
        # Display to UI if available
        if self.ui_interface and hasattr(self.ui_interface, 'display_agent_thinking'):
            self.ui_interface.display_agent_thinking(progress.agent_name, thought)
        
        logger.debug(f"{progress.agent_name} thinking: {thought}")
    
    def report_intermediate_output(self, task_id: str, output_type: str, content: Any):
        """Report intermediate outputs like generated code, specs, etc."""
        if task_id not in self.active_progresses:
            logger.warning(f"No active progress found for task {task_id}")
            return
        
        progress = self.active_progresses[task_id]
        progress.add_intermediate_output(output_type, content)
        
        # Display to UI if available
        if self.ui_interface and hasattr(self.ui_interface, 'display_intermediate_output'):
            self.ui_interface.display_intermediate_output(
                progress.agent_name, output_type, content
            )
        
        logger.debug(f"{progress.agent_name} generated {output_type}")
    
    def complete_step(self, task_id: str, output: Any = None):
        """Mark the current step as completed."""
        if task_id not in self.active_progresses:
            return
        
        progress = self.active_progresses[task_id]
        progress.complete_step(output)
    
    def fail_step(self, task_id: str, error: str, troubleshooting_steps: List[str] = None):
        """Mark the current step as failed with troubleshooting suggestions."""
        if task_id not in self.active_progresses:
            return
        
        progress = self.active_progresses[task_id]
        progress.fail_step(error)
        
        # Display failure analysis to UI if available
        if self.ui_interface and hasattr(self.ui_interface, 'display_failure_analysis'):
            self.ui_interface.display_failure_analysis(
                progress.agent_name, error, troubleshooting_steps
            )
        
        logger.error(f"{progress.agent_name} step failed: {error}")
    
    def finish_agent_progress(self, task_id: str, success: bool = True):
        """Mark agent progress as complete and move to completed list."""
        if task_id not in self.active_progresses:
            return
        
        progress = self.active_progresses[task_id]
        
        # Complete or fail the current step if any
        if progress.steps and progress.steps[-1].status == "in_progress":
            if success:
                progress.complete_step()
            else:
                progress.fail_step("Task failed")
        
        # Move to completed
        self.completed_progresses.append(progress)
        del self.active_progresses[task_id]
        
        logger.info(f"Finished progress tracking for {progress.agent_name} on task {task_id}")
    
    def get_progress(self, task_id: str) -> Optional[AgentProgress]:
        """Get current progress for a task."""
        return self.active_progresses.get(task_id)
    
    def get_all_active_progresses(self) -> Dict[str, AgentProgress]:
        """Get all active progresses."""
        return self.active_progresses.copy()


# --- Self-Testing Block ---
if __name__ == "__main__":
    import unittest
    from unittest.mock import MagicMock
    from utils.logger import setup_logger
    
    setup_logger(default_level=logging.INFO)
    
    class TestProgressTracker(unittest.TestCase):
        
        def setUp(self):
            self.mock_ui = MagicMock()
            self.tracker = ProgressTracker(ui_interface=self.mock_ui)
        
        def test_progress_tracking_flow(self):
            """Test the complete progress tracking flow."""
            print("\n--- [Test Case 1: Complete Progress Flow] ---")
            
            # Start tracking
            progress = self.tracker.start_agent_progress("TestAgent", "task_1", "Test goal")
            self.assertEqual(progress.agent_name, "TestAgent")
            self.assertEqual(progress.task_id, "task_1")
            
            # Report progress
            self.tracker.report_progress("task_1", "Analyzing requirements", "Looking at user input")
            self.assertEqual(len(progress.steps), 1)
            self.assertEqual(progress.current_step, "Analyzing requirements")
            
            # Report thinking
            self.tracker.report_thinking("task_1", "I need to break this down into components")
            self.assertEqual(len(progress.thinking_logs), 1)
            
            # Report intermediate output
            self.tracker.report_intermediate_output("task_1", "spec", "# API Specification\n...")
            self.assertIn("spec", progress.intermediate_outputs)
            
            # Complete step
            self.tracker.complete_step("task_1", "Requirements analyzed")
            self.assertEqual(progress.steps[-1].status, "completed")
            
            # Finish progress
            self.tracker.finish_agent_progress("task_1", success=True)
            self.assertEqual(len(self.tracker.completed_progresses), 1)
            self.assertNotIn("task_1", self.tracker.active_progresses)
            
            logger.info("✅ test_progress_tracking_flow: PASSED")
        
        def test_ui_integration(self):
            """Test that UI methods are called correctly."""
            print("\n--- [Test Case 2: UI Integration] ---")
            
            # Start tracking
            self.tracker.start_agent_progress("TestAgent", "task_1", "Test goal")
            
            # Report progress - should call UI
            self.tracker.report_progress("task_1", "Testing", "Running tests")
            self.mock_ui.display_agent_progress.assert_called_once_with("TestAgent", "Testing", "Running tests")
            
            # Report thinking - should call UI
            self.tracker.report_thinking("task_1", "Thinking about approach")
            self.mock_ui.display_agent_thinking.assert_called_once_with("TestAgent", "Thinking about approach")
            
            # Report intermediate output - should call UI
            self.tracker.report_intermediate_output("task_1", "code", "def test(): pass")
            self.mock_ui.display_intermediate_output.assert_called_once_with("TestAgent", "code", "def test(): pass")
            
            # Fail step - should call UI
            self.tracker.fail_step("task_1", "Test failed", ["Check syntax", "Run lint"])
            self.mock_ui.display_failure_analysis.assert_called_once_with(
                "TestAgent", "Test failed", ["Check syntax", "Run lint"]
            )
            
            logger.info("✅ test_ui_integration: PASSED")
    
    unittest.main(argv=['first-arg-is-ignored'], exit=False)