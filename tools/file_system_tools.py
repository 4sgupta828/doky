# tools/file_system_tools.py
import logging
import os
import shutil
from pathlib import Path
from typing import Dict, Any, List, Optional

from core.context import GlobalContext

# Get a logger instance for this module
logger = logging.getLogger(__name__)


class FileSystemTools:
    """
    Atomic file system operations following the principle of structured tools.
    
    This module provides low-level file system operations that can be used
    by agents or other components. Each method is atomic and focused on a
    single responsibility.
    """

    @staticmethod
    def read_file(target_path: str, encoding: str = "utf-8", context: GlobalContext = None) -> Dict[str, Any]:
        """Read file or files from the file system."""
        try:
            path = FileSystemTools._resolve_path(target_path, context)
            
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

    @staticmethod
    def write_file(target_path: str, content: str, encoding: str = "utf-8", 
                   backup_enabled: bool = False, context: GlobalContext = None) -> Dict[str, Any]:
        """Write content to a file."""
        try:
            if content is None:
                return {"success": False, "message": "No content provided for write operation"}

            path = FileSystemTools._resolve_path(target_path, context)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            if backup_enabled and path.exists():
                shutil.copy2(path, path.with_suffix(f"{path.suffix}.backup"))
            
            path.write_text(content, encoding=encoding)
            
            return {"success": True, "message": f"Successfully wrote to: {path}", "outputs": {"file_path": str(path)}}
        except Exception as e:
            return {"success": False, "message": f"Write operation failed: {e}"}

    @staticmethod
    def discover_files(target_path: str, patterns: List[str] = None, exclude_patterns: List[str] = None,
                      recursive: bool = True, file_types: List[str] = None, max_depth: int = 10, 
                      context: GlobalContext = None) -> Dict[str, Any]:
        """Discover files based on patterns and criteria."""
        try:
            patterns = patterns or []
            exclude_patterns = exclude_patterns or []
            file_types = file_types or []
            
            path = FileSystemTools._resolve_path(target_path, context)
            if not path.exists():
                return {"success": False, "message": f"Discovery path does not exist: {path}"}

            discovered_files = []
            if path.is_file():
                if FileSystemTools._matches_criteria(path, patterns, exclude_patterns, file_types):
                    discovered_files.append(str(path))
            elif path.is_dir():
                discovered_files = FileSystemTools._discover_files_in_directory(
                    path, patterns, exclude_patterns, recursive, file_types, max_depth
                )

            return {"success": True, "message": f"Discovered {len(discovered_files)} files.", "outputs": {"discovered_files": discovered_files}}
        except Exception as e:
            return {"success": False, "message": f"Discovery failed: {e}"}

    @staticmethod
    def create_path(target_path: str, content: Optional[str] = None, encoding: str = "utf-8", 
                    context: GlobalContext = None) -> Dict[str, Any]:
        """Create a new file or directory."""
        try:
            path = FileSystemTools._resolve_path(target_path, context)
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

    @staticmethod
    def delete_path(target_path: str, backup_enabled: bool = False, 
                    context: GlobalContext = None) -> Dict[str, Any]:
        """Delete a file or directory."""
        try:
            path = FileSystemTools._resolve_path(target_path, context)
            if not path.exists():
                return {"success": False, "message": f"Path does not exist: {path}"}

            if backup_enabled:
                backup_path = path.with_suffix(f"{path.suffix}.deleted.backup")
                if path.is_file(): 
                    shutil.copy2(path, backup_path)
                else: 
                    shutil.copytree(path, backup_path)

            if path.is_file(): 
                path.unlink()
            else: 
                shutil.rmtree(path)

            return {"success": True, "message": f"Successfully deleted: {path}"}
        except Exception as e:
            return {"success": False, "message": f"Delete operation failed: {e}"}

    @staticmethod
    def copy_path(source_path: str, destination_path: str, 
                  context: GlobalContext = None) -> Dict[str, Any]:
        """Copy a file or directory."""
        try:
            if not destination_path:
                return {"success": False, "message": "No destination path provided"}
            
            src = FileSystemTools._resolve_path(source_path, context)
            dest = FileSystemTools._resolve_path(destination_path, context)
            
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

    @staticmethod
    def move_path(source_path: str, destination_path: str, 
                  context: GlobalContext = None) -> Dict[str, Any]:
        """Move a file or directory."""
        try:
            if not destination_path:
                return {"success": False, "message": "No destination path provided"}
            
            src = FileSystemTools._resolve_path(source_path, context)
            dest = FileSystemTools._resolve_path(destination_path, context)
            
            if not src.exists():
                return {"success": False, "message": f"Source path does not exist: {src}"}
            
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dest))
            
            return {"success": True, "message": f"Successfully moved from {src} to {dest}"}
        except Exception as e:
            return {"success": False, "message": f"Move operation failed: {e}"}

    @staticmethod
    def _resolve_path(target_path: str, context: GlobalContext = None) -> Path:
        """Resolve target path relative to workspace or as absolute path."""
        path = Path(target_path)
        if path.is_absolute():
            return path
        
        if context and hasattr(context, 'workspace_path'):
            return Path(context.workspace_path) / path
        else:
            return Path.cwd() / path

    @staticmethod
    def _matches_criteria(file_path: Path, patterns: List[str], exclude_patterns: List[str], 
                         file_types: List[str]) -> bool:
        """Check if a file matches the discovery criteria."""
        if any(p in str(file_path) for p in exclude_patterns): 
            return False
        if patterns and not any(file_path.match(p) for p in patterns): 
            return False
        if file_types and file_path.suffix.lower() not in [f".{ft.lstrip('.')}" for ft in file_types]: 
            return False
        return True

    @staticmethod
    def _discover_files_in_directory(directory: Path, patterns: List[str], exclude_patterns: List[str],
                                   recursive: bool, file_types: List[str], max_depth: int) -> List[str]:
        """Discover files in a directory based on criteria."""
        discovered = []
        
        def scan(current_dir: Path, depth: int):
            if depth > max_depth: 
                return
            try:
                for item in current_dir.iterdir():
                    if item.is_file() and FileSystemTools._matches_criteria(item, patterns, exclude_patterns, file_types):
                        discovered.append(str(item))
                    elif item.is_dir() and recursive:
                        scan(item, depth + 1)
            except PermissionError:
                logger.warning(f"Permission denied accessing: {current_dir}")
        
        scan(directory, 0)
        return discovered