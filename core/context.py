# core/context.py
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.models import TaskGraph  # Depends on Tier 1 models

# Get a logger instance for this module.
logger = logging.getLogger(__name__)

class WorkspaceManager:
    """
    This is an abstract base class that defines the interface for managing the
    state of the code workspace. The GitWorkspaceManager is a concrete implementation
    of this contract. This abstract definition allows for future flexibility, such as
    a virtual in-memory workspace for testing.
    """
    def get_file_content(self, path: str) -> Optional[str]: raise NotImplementedError
    def write_file_content(self, path: str, content: str, task_id: str): raise NotImplementedError
    def list_files(self, path: str = ".") -> List[str]: raise NotImplementedError
    def revert_changes(self, task_id: str): raise NotImplementedError
    def get_diff(self) -> str: raise NotImplementedError


class GlobalContext:
    """
    The single source of truth for the entire mission.
    It holds the project plan (TaskGraph), all generated data (artifacts), and manages
    the workspace. Encapsulating state in this object makes the system highly debuggable,
    as the entire state of the mission can be inspected or even serialized at any point.
    """

    def __init__(self, workspace_path: Optional[str] = None):
        """
        Initializes the GlobalContext.

        Args:
            workspace_path: The file path for the coding workspace. If None, 
                          auto-generates a timestamped directory in /Users/sgupta/
        """
        # Import here to avoid circular import
        from utils.git_manager import GitWorkspaceManager
        
        # Track if workspace was auto-generated
        workspace_was_auto_generated = workspace_path is None
        
        # Auto-generate workspace path if not provided
        if workspace_path is None:
            from config import config
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            workspace_path = os.path.join(config.workspace_root_dir, f"ws_{timestamp}")
            logger.info(f"Auto-generating workspace directory: {workspace_path}")
        
        self.workspace_path = workspace_path  # Store for user communication
        self.workspace_was_auto_generated = workspace_was_auto_generated
        
        self.task_graph = TaskGraph()
        self.workspace: WorkspaceManager = GitWorkspaceManager(workspace_path)
        self.artifacts: Dict[str, Any] = {}
        self.mission_log: List[Dict[str, Any]] = []
        
        # Initialize session data directory
        from config import config
        self.session_dir = config.get_workspace_session_dir(workspace_path)
        
        self.log_event(
            "context_initialized",
            {"workspace_path": workspace_path, "message": "GlobalContext is ready."}
        )

    def add_artifact(self, key: str, value: Any, source_task_id: str):
        """
        Adds a new artifact (data, spec, code, etc.) to the central repository.
        This method is the primary way agents share information.

        Args:
            key: A unique, descriptive key for the artifact (e.g., "technical_spec.md").
            value: The data to be stored. Can be any serializable type.
            source_task_id: The ID of the task that produced this artifact.
        """
        if key in self.artifacts:
            logger.warning(f"Artifact key '{key}' already exists. It will be overwritten by task '{source_task_id}'.")
        
        self.artifacts[key] = value
        self.log_event(
            "artifact_added",
            {"key": key, "source_task_id": source_task_id, "type": str(type(value).__name__)}
        )

    def get_artifact(self, key: str, default: Any = None) -> Any:
        """
        Safely retrieves an artifact by its key.

        Args:
            key: The key of the artifact to retrieve.
            default: The value to return if the key is not found.

        Returns:
            The artifact's value, or the default value if not found.
        """
        if key not in self.artifacts:
            logger.warning(f"Artifact with key '{key}' not found. Returning default value.")
        return self.artifacts.get(key, default)
    
    def log_event(self, event_type: str, details: Dict[str, Any]):
        """
        Records a significant event in the mission log for transparency and debugging.

        Args:
            event_type: A string identifier for the type of event (e.g., "task_started").
            details: A dictionary containing relevant data about the event.
        """
        log_entry = {"event": event_type, "details": details}
        self.mission_log.append(log_entry)
        logger.debug(f"Logged event: {log_entry}")

    def save_snapshot(self, file_path: Optional[str] = None):
        """
        Saves the current state of the GlobalContext to a JSON file.
        This is an invaluable debugging tool, allowing developers to capture the exact
        state of a mission at the moment of failure for later analysis.

        Args:
            file_path: The path to save the snapshot file. If None, saves to session directory.
        """
        if file_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = str(self.session_dir / f"snapshot_{timestamp}.json")
        
        logger.info(f"Saving debug snapshot to: {file_path}")
        snapshot_path = Path(file_path)
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            state_data = {
                # Pydantic's model_dump is used for safe serialization to JSON-compatible types.
                "task_graph": self.task_graph.model_dump(mode='json'),
                "artifacts": {k: str(v)[:500] + '...' if isinstance(v, str) else type(v).__name__ for k, v in self.artifacts.items()},
                "mission_log": self.mission_log,
                "workspace_path": self.workspace_path,
                "session_dir": str(self.session_dir),
                "timestamp": datetime.now().isoformat()
            }
            with open(snapshot_path, 'w', encoding='utf-8') as f:
                json.dump(state_data, f, indent=2)
            logger.info("Debug snapshot saved successfully.")
        except TypeError as e:
            logger.error(f"Failed to serialize context to JSON. Check for non-serializable artifacts. Error: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"An unexpected error occurred while saving snapshot: {e}", exc_info=True)
    
    def save_session_memory(self, memory_data: Dict[str, Any]):
        """
        Saves session-specific memory/data to the session directory.
        
        Args:
            memory_data: Dictionary of memory data to save
        """
        memory_file = self.session_dir / "session_memory.json"
        
        try:
            # Load existing memory if it exists
            existing_memory = {}
            if memory_file.exists():
                with open(memory_file, 'r', encoding='utf-8') as f:
                    existing_memory = json.load(f)
            
            # Merge with new memory data
            existing_memory.update(memory_data)
            existing_memory["last_updated"] = datetime.now().isoformat()
            
            # Save updated memory
            with open(memory_file, 'w', encoding='utf-8') as f:
                json.dump(existing_memory, f, indent=2)
                
            logger.info(f"Session memory saved to: {memory_file}")
            
        except Exception as e:
            logger.error(f"Failed to save session memory: {e}", exc_info=True)
    
    def load_session_memory(self) -> Dict[str, Any]:
        """
        Loads session-specific memory/data from the session directory.
        
        Returns:
            Dictionary of loaded memory data, empty dict if none exists
        """
        memory_file = self.session_dir / "session_memory.json"
        
        if not memory_file.exists():
            return {}
        
        try:
            with open(memory_file, 'r', encoding='utf-8') as f:
                memory_data = json.load(f)
            logger.info(f"Session memory loaded from: {memory_file}")
            return memory_data
        except Exception as e:
            logger.error(f"Failed to load session memory: {e}", exc_info=True)
            return {}


# --- Self-Testing Block ---
# This block validates the functionality of the GlobalContext.
# To run this test: `python core/context.py`
if __name__ == "__main__":
    import shutil
    from utils.logger import setup_logger
    from core.models import TaskNode

    # Setup a clean testing environment
    TEST_WORKSPACE = "./temp_context_test_ws"
    if Path(TEST_WORKSPACE).exists():
        shutil.rmtree(TEST_WORKSPACE)
    
    setup_logger(default_level=logging.DEBUG)

    print("\n--- Testing GlobalContext ---")

    # 1. Test Initialization
    print("\n[1] Testing context initialization...")
    try:
        context = GlobalContext(workspace_path=TEST_WORKSPACE)
        assert Path(TEST_WORKSPACE).exists()
        assert len(context.mission_log) == 1  # Initial log event
        logger.info("Context initialization test passed.")
    except Exception as e:
        logger.error("Context initialization failed.", exc_info=True)
        exit()

    # 2. Test Artifact Management
    print("\n[2] Testing artifact management...")
    task1 = TaskNode(goal="Create a spec", assigned_agent="SpecGen")
    spec_content = {"api_version": "v1", "endpoints": ["/users"]}
    context.add_artifact(
        key="tech_spec.json",
        value=spec_content,
        source_task_id=task1.task_id
    )
    
    retrieved_spec = context.get_artifact("tech_spec.json")
    assert retrieved_spec == spec_content
    logger.info("Artifact addition and retrieval successful.")

    # Test retrieving a non-existent artifact
    non_existent = context.get_artifact("non_existent.key", default="not_found")
    assert non_existent == "not_found"
    logger.info("Retrieving non-existent artifact with default value successful.")
    
    # Test overwriting an artifact
    context.add_artifact("tech_spec.json", {"api_version": "v2"}, "task_002")
    assert context.get_artifact("tech_spec.json")["api_version"] == "v2"
    logger.info("Artifact overwriting successful.")

    # 3. Test Event Logging
    print("\n[3] Testing event logging...")
    initial_log_count = len(context.mission_log)
    context.log_event("test_event", {"data": "some_value"})
    assert len(context.mission_log) == initial_log_count + 1
    assert context.mission_log[-1]["event"] == "test_event"
    logger.info("Event logging successful.")

    # 4. Test Workspace Interaction (via the manager)
    print("\n[4] Testing workspace interaction...")
    try:
        task2 = TaskNode(goal="Write code", assigned_agent="Coder")
        context.workspace.write_file_content("app.py", "print('hello')", task2.task_id)
        file_list = context.workspace.list_files()
        assert "app.py" in file_list
        logger.info("Workspace interaction test passed.")
    except Exception as e:
        logger.error("Workspace interaction test failed.", exc_info=True)

    # 5. Test State Snapshot
    print("\n[5] Testing state snapshot...")
    snapshot_file = Path("./temp_context_test_ws/snapshot.json")
    try:
        context.save_snapshot(str(snapshot_file))
        assert snapshot_file.exists()
        with open(snapshot_file, 'r') as f:
            data = json.load(f)
            assert "task_graph" in data
            assert "tech_spec.json" in data["artifacts"]
        logger.info("State snapshot test passed.")
    except Exception as e:
        logger.error("State snapshot test failed.", exc_info=True)

    # Clean up the test directory
    shutil.rmtree(TEST_WORKSPACE)
    logger.info(f"Cleaned up test directory: {TEST_WORKSPACE}")

    print("\n--- All GlobalContext Tests Passed Successfully ---")