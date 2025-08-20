# agents/file_system.py
import logging
from typing import Dict, Any, List

# Foundational dependencies
from .base import BaseAgent
from core.context import GlobalContext
from core.models import AgentResult, TaskNode
from tools.file_system_tools import FileSystemTools

# Get a logger instance for this module
logger = logging.getLogger(__name__)


class FileSystemAgent(BaseAgent):
    """
    Infrastructure Tier: Direct file system manipulation.
    
    This agent handles all direct file system operations. It is the "hands"
    of the system for interacting with the workspace files.
    
    Responsibilities:
    - File discovery and filtering
    - File content reading and writing
    - Directory operations (create, delete, copy, move)
    - Path resolution and validation
    - Safe file operations with backup/rollback
    
    This agent does NOT execute external shell commands. For that, use the ToolingAgent.
    """

    def __init__(self):
        super().__init__(
            name="FileSystemAgent",
            description="Handles all direct file system operations like reading, writing, and discovering files."
        )

    def required_inputs(self) -> List[str]:
        """Required inputs for FileSystemAgent execution."""
        return ["operation"]  # read, write, discover, create, delete, copy, move

    def optional_inputs(self) -> List[str]:
        """Optional inputs for FileSystemAgent execution."""
        return [
            "target_path",
            "destination_path",
            "content", 
            "patterns",
            "exclude_patterns",
            "recursive",
            "file_types",
            "max_depth",
            "backup_enabled",
            "encoding"
        ]

    def execute_v2(self, goal: str, inputs: Dict[str, Any], global_context: GlobalContext) -> AgentResult:
        """
        Performs file system operations based on the specified inputs.
        """
        logger.info(f"FileSystemAgent executing: '{goal}'")
        
        # Validate inputs
        try:
            self.validate_inputs(inputs)
        except Exception as validation_error:
            return self.create_result(
                success=False,
                message=str(validation_error),
                error_details={"validation_error": str(validation_error)}
            )

        # Extract inputs
        operation = inputs["operation"]
        target_path = inputs.get("target_path", ".")
        content = inputs.get("content")
        patterns = inputs.get("patterns", [])
        exclude_patterns = inputs.get("exclude_patterns", [])
        recursive = inputs.get("recursive", True)
        file_types = inputs.get("file_types", [])
        max_depth = inputs.get("max_depth", 10)
        backup_enabled = inputs.get("backup_enabled", False)
        encoding = inputs.get("encoding", "utf-8")

        try:
            self.report_progress("Starting file operation", f"Operation: {operation} on {target_path}")

            # Route to appropriate operation handler using FileSystemTools
            if operation == "read":
                result = FileSystemTools.read_file(target_path, encoding, global_context)
            elif operation == "write":
                result = FileSystemTools.write_file(target_path, content, encoding, backup_enabled, global_context)
            elif operation == "discover":
                result = FileSystemTools.discover_files(
                    target_path, patterns, exclude_patterns, recursive, 
                    file_types, max_depth, global_context
                )
            elif operation == "create":
                result = FileSystemTools.create_path(target_path, content, encoding, global_context)
            elif operation == "delete":
                result = FileSystemTools.delete_path(target_path, backup_enabled, global_context)
            elif operation == "copy":
                result = FileSystemTools.copy_path(target_path, inputs.get("destination_path"), global_context)
            elif operation == "move":
                result = FileSystemTools.move_path(target_path, inputs.get("destination_path"), global_context)
            else:
                return self.create_result(
                    success=False,
                    message=f"Unsupported file operation: {operation}"
                )

            self.report_progress("File operation complete", result["message"])

            return self.create_result(
                success=result["success"],
                message=result["message"],
                outputs=result.get("outputs", {})
            )

        except Exception as e:
            error_msg = f"FileSystemAgent execution failed: {e}"
            logger.error(error_msg, exc_info=True)
            return self.create_result(
                success=False,
                message=error_msg,
                error_details={"exception": str(e)}
            )

