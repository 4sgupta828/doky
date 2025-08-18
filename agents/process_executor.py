# agents/process_executor.py
import logging
import os
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

# Foundational dependencies
from .base import BaseAgent
from core.context import GlobalContext
from core.models import AgentResponse, AgentResult, TaskNode

# Get a logger instance for this module
logger = logging.getLogger(__name__)


class ProcessExecutorAgent(BaseAgent):
    """
    Infrastructure Tier: Command execution and process management.
    
    This agent handles system command execution and process management.
    
    Responsibilities:
    - Command execution and process management
    - File discovery and filtering
    - File content reading and writing
    - Directory operations
    - Path resolution and validation
    - File pattern matching
    - Safe file operations with backup/rollback
    """

    def __init__(self):
        super().__init__(
            name="ProcessExecutorAgent",
            description="Executes commands and handles file system operations."
        )

    def required_inputs(self) -> List[str]:
        """Required inputs for FileSystemAgent execution."""
        return ["operation"]  # read, write, discover, create, delete, copy, move

    def optional_inputs(self) -> List[str]:
        """Optional inputs for FileSystemAgent execution."""
        return [
            "target_path",
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
        NEW INTERFACE: Perform file system operations.
        """
        logger.info(f"FileSystemAgent executing: '{goal}'")
        
        # Validate inputs with graceful handling
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
                # Read single file
                with open(path, 'r', encoding=encoding) as f:
                    content = f.read()
                
                return {
                    "success": True,
                    "message": f"Successfully read file: {path}",
                    "outputs": {
                        "file_path": str(path),
                        "content": content,
                        "size_bytes": len(content.encode(encoding)),
                        "line_count": content.count('\n') + 1 if content else 0
                    }
                }
            
            elif path.is_dir():
                # Read all files in directory
                files_content = {}
                for file_path in path.rglob("*"):
                    if file_path.is_file() and self._is_text_file(file_path):
                        try:
                            with open(file_path, 'r', encoding=encoding) as f:
                                files_content[str(file_path)] = f.read()
                        except Exception as e:
                            logger.warning(f"Could not read {file_path}: {e}")
                
                return {
                    "success": True,
                    "message": f"Successfully read {len(files_content)} files from directory: {path}",
                    "outputs": {
                        "directory_path": str(path),
                        "files_content": files_content,
                        "files_count": len(files_content)
                    }
                }
            
            else:
                return {
                    "success": False,
                    "message": f"Path does not exist: {path}"
                }

        except Exception as e:
            return {
                "success": False,
                "message": f"Read operation failed: {e}",
                "exception": str(e)
            }

    def _write_operation(self, target_path: str, content: str, encoding: str, 
                        backup_enabled: bool, context: GlobalContext) -> Dict[str, Any]:
        """Write content to a file."""
        try:
            if not content:
                return {
                    "success": False,
                    "message": "No content provided for write operation"
                }

            path = self._resolve_path(target_path, context)
            
            # Create backup if enabled and file exists
            backup_path = None
            if backup_enabled and path.exists():
                backup_path = path.with_suffix(f"{path.suffix}.backup")
                shutil.copy2(path, backup_path)
            
            # Ensure parent directory exists
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write content
            with open(path, 'w', encoding=encoding) as f:
                f.write(content)
            
            return {
                "success": True,
                "message": f"Successfully wrote content to: {path}",
                "outputs": {
                    "file_path": str(path),
                    "content_size": len(content),
                    "backup_created": backup_path is not None,
                    "backup_path": str(backup_path) if backup_path else None
                }
            }

        except Exception as e:
            return {
                "success": False,
                "message": f"Write operation failed: {e}",
                "exception": str(e)
            }

    def _discover_operation(self, target_path: str, patterns: List[str], exclude_patterns: List[str],
                           recursive: bool, file_types: List[str], max_depth: int, 
                           context: GlobalContext) -> Dict[str, Any]:
        """Discover files based on patterns and criteria."""
        try:
            path = self._resolve_path(target_path, context)
            
            if not path.exists():
                return {
                    "success": False,
                    "message": f"Discovery path does not exist: {path}"
                }

            discovered_files = []
            
            if path.is_file():
                # Single file - check if it matches criteria
                if self._matches_criteria(path, patterns, exclude_patterns, file_types):
                    discovered_files.append(str(path))
            
            elif path.is_dir():
                # Directory - discover files
                discovered_files = self._discover_files_in_directory(
                    path, patterns, exclude_patterns, recursive, file_types, max_depth
                )

            return {
                "success": True,
                "message": f"Discovered {len(discovered_files)} files in {path}",
                "outputs": {
                    "discovered_files": discovered_files,
                    "discovery_path": str(path),
                    "file_count": len(discovered_files),
                    "patterns_used": patterns,
                    "excluded_patterns": exclude_patterns
                }
            }

        except Exception as e:
            return {
                "success": False,
                "message": f"Discovery operation failed: {e}",
                "exception": str(e)
            }

    def _create_operation(self, target_path: str, content: str, encoding: str, 
                         context: GlobalContext) -> Dict[str, Any]:
        """Create a new file or directory."""
        try:
            path = self._resolve_path(target_path, context)
            
            if path.exists():
                return {
                    "success": False,
                    "message": f"Path already exists: {path}"
                }

            if content is not None:
                # Create file with content
                path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(path, 'w', encoding=encoding) as f:
                    f.write(content)
                
                return {
                    "success": True,
                    "message": f"Successfully created file: {path}",
                    "outputs": {
                        "created_path": str(path),
                        "type": "file",
                        "content_size": len(content)
                    }
                }
            else:
                # Create directory
                path.mkdir(parents=True, exist_ok=True)
                
                return {
                    "success": True,
                    "message": f"Successfully created directory: {path}",
                    "outputs": {
                        "created_path": str(path),
                        "type": "directory"
                    }
                }

        except Exception as e:
            return {
                "success": False,
                "message": f"Create operation failed: {e}",
                "exception": str(e)
            }

    def _delete_operation(self, target_path: str, backup_enabled: bool, 
                         context: GlobalContext) -> Dict[str, Any]:
        """Delete a file or directory."""
        try:
            path = self._resolve_path(target_path, context)
            
            if not path.exists():
                return {
                    "success": False,
                    "message": f"Path does not exist: {path}"
                }

            # Create backup if enabled
            backup_path = None
            if backup_enabled:
                backup_path = path.with_suffix(f"{path.suffix}.deleted.backup")
                if path.is_file():
                    shutil.copy2(path, backup_path)
                else:
                    shutil.copytree(path, backup_path)

            # Delete the path
            if path.is_file():
                path.unlink()
                deleted_type = "file"
            else:
                shutil.rmtree(path)
                deleted_type = "directory"

            return {
                "success": True,
                "message": f"Successfully deleted {deleted_type}: {path}",
                "outputs": {
                    "deleted_path": str(path),
                    "type": deleted_type,
                    "backup_created": backup_path is not None,
                    "backup_path": str(backup_path) if backup_path else None
                }
            }

        except Exception as e:
            return {
                "success": False,
                "message": f"Delete operation failed: {e}",
                "exception": str(e)
            }

    def _copy_operation(self, source_path: str, destination_path: str, 
                       context: GlobalContext) -> Dict[str, Any]:
        """Copy a file or directory."""
        try:
            if not destination_path:
                return {
                    "success": False,
                    "message": "No destination path provided for copy operation"
                }

            src_path = self._resolve_path(source_path, context)
            dest_path = self._resolve_path(destination_path, context)
            
            if not src_path.exists():
                return {
                    "success": False,
                    "message": f"Source path does not exist: {src_path}"
                }

            if src_path.is_file():
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_path, dest_path)
                copy_type = "file"
            else:
                shutil.copytree(src_path, dest_path)
                copy_type = "directory"

            return {
                "success": True,
                "message": f"Successfully copied {copy_type} from {src_path} to {dest_path}",
                "outputs": {
                    "source_path": str(src_path),
                    "destination_path": str(dest_path),
                    "type": copy_type
                }
            }

        except Exception as e:
            return {
                "success": False,
                "message": f"Copy operation failed: {e}",
                "exception": str(e)
            }

    def _move_operation(self, source_path: str, destination_path: str, 
                       context: GlobalContext) -> Dict[str, Any]:
        """Move a file or directory."""
        try:
            if not destination_path:
                return {
                    "success": False,
                    "message": "No destination path provided for move operation"
                }

            src_path = self._resolve_path(source_path, context)
            dest_path = self._resolve_path(destination_path, context)
            
            if not src_path.exists():
                return {
                    "success": False,
                    "message": f"Source path does not exist: {src_path}"
                }

            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src_path), str(dest_path))
            
            move_type = "file" if dest_path.is_file() else "directory"

            return {
                "success": True,
                "message": f"Successfully moved {move_type} from {src_path} to {dest_path}",
                "outputs": {
                    "source_path": str(src_path),
                    "destination_path": str(dest_path),
                    "type": move_type
                }
            }

        except Exception as e:
            return {
                "success": False,
                "message": f"Move operation failed: {e}",
                "exception": str(e)
            }

    def _resolve_path(self, target_path: str, context: GlobalContext) -> Path:
        """Resolve target path relative to workspace or as absolute path."""
        path = Path(target_path)
        
        if path.is_absolute():
            return path
        else:
            return context.workspace_path / path

    def _is_text_file(self, file_path: Path) -> bool:
        """Check if a file is likely a text file."""
        text_extensions = {'.txt', '.py', '.js', '.html', '.css', '.md', '.json', '.xml', '.yaml', '.yml', '.ini', '.cfg'}
        return file_path.suffix.lower() in text_extensions

    def _matches_criteria(self, file_path: Path, patterns: List[str], exclude_patterns: List[str], 
                         file_types: List[str]) -> bool:
        """Check if a file matches the discovery criteria."""
        # Check exclude patterns first
        for exclude_pattern in exclude_patterns:
            if exclude_pattern in str(file_path):
                return False

        # Check include patterns
        if patterns:
            matches_pattern = any(pattern in str(file_path) or file_path.match(pattern) for pattern in patterns)
            if not matches_pattern:
                return False

        # Check file types
        if file_types:
            matches_type = file_path.suffix.lower() in [f".{ft}" if not ft.startswith('.') else ft for ft in file_types]
            if not matches_type:
                return False

        return True

    def _discover_files_in_directory(self, directory: Path, patterns: List[str], exclude_patterns: List[str],
                                   recursive: bool, file_types: List[str], max_depth: int) -> List[str]:
        """Discover files in a directory based on criteria."""
        discovered = []
        
        def scan_directory(current_dir: Path, current_depth: int):
            if current_depth > max_depth:
                return
            
            try:
                items = current_dir.iterdir() if recursive else current_dir.glob("*")
                
                for item in items:
                    if item.is_file():
                        if self._matches_criteria(item, patterns, exclude_patterns, file_types):
                            discovered.append(str(item))
                    elif item.is_dir() and recursive and current_depth < max_depth:
                        scan_directory(item, current_depth + 1)
                        
            except PermissionError:
                logger.warning(f"Permission denied accessing: {current_dir}")
            except Exception as e:
                logger.warning(f"Error scanning {current_dir}: {e}")

        scan_directory(directory, 0)
        return discovered

    # Legacy execute method for backward compatibility
    def execute(self, goal: str, context: GlobalContext, current_task: TaskNode) -> AgentResponse:
        """Legacy execute method for backward compatibility."""
        # Simple operation detection based on goal
        if "read" in goal.lower():
            operation = "read"
        elif "write" in goal.lower() or "create" in goal.lower():
            operation = "write"
        elif "find" in goal.lower() or "discover" in goal.lower():
            operation = "discover"
        else:
            operation = "discover"  # Default to discover

        inputs = {
            "operation": operation,
            "target_path": str(context.workspace_path)
        }
        
        result = self.execute_v2(goal, inputs, context)
        
        return AgentResponse(
            success=result.success,
            message=result.message,
            artifacts_generated=[]
        )