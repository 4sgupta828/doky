# agents/file_system.py
import logging
import os
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional

# Foundational dependencies
from .base import BaseAgent
from core.context import GlobalContext
from core.models import AgentResult, TaskNode

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

            # Route to appropriate operation handler
            if operation == "read":
                result = self._read_operation(target_path, encoding, global_context)
            elif operation == "write":
                result = self._write_operation(target_path, content, encoding, backup_enabled, global_context)
            elif operation == "discover":
                result = self._discover_operation(
                    target_path, patterns, exclude_patterns, recursive, 
                    file_types, max_depth, global_context
                )
            elif operation == "create":
                result = self._create_operation(target_path, content, encoding, global_context)
            elif operation == "delete":
                result = self._delete_operation(target_path, backup_enabled, global_context)
            elif operation == "copy":
                result = self._copy_operation(target_path, inputs.get("destination_path"), global_context)
            elif operation == "move":
                result = self._move_operation(target_path, inputs.get("destination_path"), global_context)
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

    def _read_operation(self, target_path: str, encoding: str, context: GlobalContext) -> Dict[str, Any]:
        """Read file or files from the file system."""
        try:
            path = self._resolve_path(target_path, context)
            
            if path.is_file():
                content = path.read_text(encoding=encoding)
                return {
                    "success": True, "message": f"Successfully read file: {path}",
                    "outputs": { "file_path": str(path), "content": content }
                }
            elif path.is_dir():
                files_content = {str(p): p.read_text(encoding=encoding) for p in path.rglob("*") if p.is_file()}
                return {
                    "success": True, "message": f"Successfully read {len(files_content)} files from directory: {path}",
                    "outputs": { "directory_path": str(path), "files_content": files_content }
                }
            else:
                return {"success": False, "message": f"Path does not exist: {path}"}
        except Exception as e:
            return {"success": False, "message": f"Read operation failed: {e}"}

    def _write_operation(self, target_path: str, content: str, encoding: str, 
                        backup_enabled: bool, context: GlobalContext) -> Dict[str, Any]:
        """Write content to a file."""
        try:
            if content is None:
                return {"success": False, "message": "No content provided for write operation"}

            path = self._resolve_path(target_path, context)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            if backup_enabled and path.exists():
                shutil.copy2(path, path.with_suffix(f"{path.suffix}.backup"))
            
            path.write_text(content, encoding=encoding)
            
            return {"success": True, "message": f"Successfully wrote to: {path}", "outputs": {"file_path": str(path)}}
        except Exception as e:
            return {"success": False, "message": f"Write operation failed: {e}"}

    def _discover_operation(self, target_path: str, patterns: List[str], exclude_patterns: List[str],
                           recursive: bool, file_types: List[str], max_depth: int, 
                           context: GlobalContext) -> Dict[str, Any]:
        """Discover files based on patterns and criteria."""
        try:
            path = self._resolve_path(target_path, context)
            if not path.exists():
                return {"success": False, "message": f"Discovery path does not exist: {path}"}

            discovered_files = []
            if path.is_file():
                if self._matches_criteria(path, patterns, exclude_patterns, file_types):
                    discovered_files.append(str(path))
            elif path.is_dir():
                discovered_files = self._discover_files_in_directory(
                    path, patterns, exclude_patterns, recursive, file_types, max_depth
                )

            return {"success": True, "message": f"Discovered {len(discovered_files)} files.", "outputs": {"discovered_files": discovered_files}}
        except Exception as e:
            return {"success": False, "message": f"Discovery failed: {e}"}

    def _create_operation(self, target_path: str, content: Optional[str], encoding: str, 
                         context: GlobalContext) -> Dict[str, Any]:
        """Create a new file or directory."""
        try:
            path = self._resolve_path(target_path, context)
            if path.exists():
                return {"success": False, "message": f"Path already exists: {path}"}

            if content is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding=encoding)
                return {"success": True, "message": f"Successfully created file: {path}", "outputs": {"created_path": str(path), "type": "file"}}
            else:
                path.mkdir(parents=True, exist_ok=True)
                return {"success": True, "message": f"Successfully created directory: {path}", "outputs": {"created_path": str(path), "type": "directory"}}
        except Exception as e:
            return {"success": False, "message": f"Create operation failed: {e}"}

    def _delete_operation(self, target_path: str, backup_enabled: bool, 
                         context: GlobalContext) -> Dict[str, Any]:
        """Delete a file or directory."""
        try:
            path = self._resolve_path(target_path, context)
            if not path.exists():
                return {"success": False, "message": f"Path does not exist: {path}"}

            if backup_enabled:
                backup_path = path.with_suffix(f"{path.suffix}.deleted.backup")
                if path.is_file(): shutil.copy2(path, backup_path)
                else: shutil.copytree(path, backup_path)

            if path.is_file(): path.unlink()
            else: shutil.rmtree(path)

            return {"success": True, "message": f"Successfully deleted: {path}"}
        except Exception as e:
            return {"success": False, "message": f"Delete operation failed: {e}"}

    def _copy_operation(self, source_path: str, destination_path: str, 
                       context: GlobalContext) -> Dict[str, Any]:
        """Copy a file or directory."""
        try:
            if not destination_path:
                return {"success": False, "message": "No destination path provided"}
            src = self._resolve_path(source_path, context)
            dest = self._resolve_path(destination_path, context)
            if not src.exists():
                return {"success": False, "message": f"Source path does not exist: {src}"}
            
            if src.is_file():
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dest)
            else:
                shutil.copytree(src, dest)
            
            return {"success": True, "message": f"Successfully copied from {src} to {dest}"}
        except Exception as e:
            return {"success": False, "message": f"Copy operation failed: {e}"}

    def _move_operation(self, source_path: str, destination_path: str, 
                       context: GlobalContext) -> Dict[str, Any]:
        """Move a file or directory."""
        try:
            if not destination_path:
                return {"success": False, "message": "No destination path provided"}
            src = self._resolve_path(source_path, context)
            dest = self._resolve_path(destination_path, context)
            if not src.exists():
                return {"success": False, "message": f"Source path does not exist: {src}"}
            
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dest))
            
            return {"success": True, "message": f"Successfully moved from {src} to {dest}"}
        except Exception as e:
            return {"success": False, "message": f"Move operation failed: {e}"}

    def _resolve_path(self, target_path: str, context: GlobalContext) -> Path:
        """Resolve target path relative to workspace or as absolute path."""
        path = Path(target_path)
        return path if path.is_absolute() else Path(context.workspace_path) / path

    def _matches_criteria(self, file_path: Path, patterns: List[str], exclude_patterns: List[str], 
                         file_types: List[str]) -> bool:
        """Check if a file matches the discovery criteria."""
        if any(p in str(file_path) for p in exclude_patterns): return False
        if patterns and not any(file_path.match(p) for p in patterns): return False
        if file_types and file_path.suffix.lower() not in [f".{ft.lstrip('.')}" for ft in file_types]: return False
        return True

    def _discover_files_in_directory(self, directory: Path, patterns: List[str], exclude_patterns: List[str],
                                   recursive: bool, file_types: List[str], max_depth: int) -> List[str]:
        """Discover files in a directory based on criteria."""
        discovered = []
        
        def scan(current_dir: Path, depth: int):
            if depth > max_depth: return
            for item in current_dir.iterdir():
                if item.is_file() and self._matches_criteria(item, patterns, exclude_patterns, file_types):
                    discovered.append(str(item))
                elif item.is_dir() and recursive:
                    scan(item, depth + 1)
        
        scan(directory, 0)
        return discovered
