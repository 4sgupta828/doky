# core/context.py
from .models import TaskGraph
from typing import Any, Dict
import logging
import json

# from utils.git_manager import GitWorkspaceManager

class WorkspaceManager:
    """
    An abstract interface for managing the state of the code workspace.
    This could be implemented using Git, a virtual file system, or other versioning tools.
    Its purpose is to enable safe, reversible file operations, which is key for backtracking.
    """
    def get_file_content(self, path: str) -> str: pass
    def write_file_content(self, path: str, content: str, task_id: str): pass
    def list_files(self, path: str = ".") -> list[str]: pass
    def revert_changes(self, task_id: str): pass
    def get_diff(self) -> str: pass


class GlobalContext:
    """
    The single source of truth for the entire mission.
    It holds the project plan (TaskGraph), all generated data (artifacts), and manages
    the workspace. Encapsulating state in this object makes the system highly debuggable,
    as the entire state of the mission can be inspected or even serialized at any point.
    """

    def __init__(self):
        self.task_graph = TaskGraph()
        # self.workspace = GitWorkspaceManager("./mission_workspace")
        self.artifacts: Dict[str, Any] = {}
        self.mission_log: List[Dict[str, Any]] = []
        logging.info("GlobalContext initialized.")

    def add_artifact(self, key: str, value: Any, source_task_id: str):
        """
        Adds a new artifact (data, spec, code, etc.) to the central repository.

        Args:
            key: A unique, descriptive key for the artifact (e.g., "technical_spec.md").
            value: The data to be stored.
            source_task_id: The ID of the task that produced this artifact.
        """
        logging.info(f"Artifact '{key}' added by Task '{source_task_id}'.")
        self.artifacts[key] = value
        self.log_event(
            "artifact_added", 
            {"key": key, "source": source_task_id, "type": str(type(value))}
        )

    def get_artifact(self, key: str) -> Any:
        """Retrieves an artifact by its key."""
        return self.artifacts.get(key)
    
    def log_event(self, event_type: str, details: Dict):
        """Records a significant event in the mission log for transparency and debugging."""
        self.mission_log.append({"event": event_type, "details": details})

    def save_snapshot(self, path: str):
        """
        Saves the current state of the GlobalContext to a file.
        This is an invaluable debugging tool, allowing developers to capture the exact
        state of a mission at the moment of failure for later analysis.
        """
        # state_data = {
        #     "task_graph": self.task_graph, # Requires serialization for dataclasses
        #     "artifacts": self.artifacts,   # Requires serialization for complex objects
        #     "mission_log": self.mission_log
        # }
        # with open(path, 'w') as f:
        #     json.dump(state_data, f, indent=2)
        logging.info(f"Debug snapshot saved to {path}")