# tools/execution/filesystem_tools.py
"""
Filesystem tools for file operations, directory management, and path handling.
Extracted from FileSystemAgent to provide atomic, reusable filesystem capabilities.
"""

import logging
import os
import shutil
import stat
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

class FileOperation(Enum):
    """Types of file operations."""
    READ = "read"
    WRITE = "write"
    CREATE = "create"
    DELETE = "delete"
    COPY = "copy"
    MOVE = "move"
    DISCOVER = "discover"
    BACKUP = "backup"

class FileType(Enum):
    """File types for filtering."""
    PYTHON = "python"
    JAVASCRIPT = "javascript"
    TEXT = "text"
    JSON = "json"
    YAML = "yaml"
    MARKDOWN = "markdown"
    CONFIG = "config"
    ALL = "all"

@dataclass
class FileInfo:
    """Information about a file or directory."""
    path: str
    name: str
    size: int
    is_directory: bool
    is_file: bool
    created_time: datetime
    modified_time: datetime
    permissions: str
    extension: str = ""
    
    @classmethod
    def from_path(cls, path: Path) -> 'FileInfo':
        """Create FileInfo from a Path object."""
        stat_result = path.stat()
        return cls(
            path=str(path.absolute()),
            name=path.name,
            size=stat_result.st_size,
            is_directory=path.is_dir(),
            is_file=path.is_file(),
            created_time=datetime.fromtimestamp(stat_result.st_ctime),
            modified_time=datetime.fromtimestamp(stat_result.st_mtime),
            permissions=stat.filemode(stat_result.st_mode),
            extension=path.suffix
        )

@dataclass
class FilesystemContext:
    """Context for filesystem operations."""
    working_directory: str = "."
    backup_enabled: bool = False
    create_directories: bool = True
    overwrite_existing: bool = False
    preserve_permissions: bool = True
    encoding: str = "utf-8"

@dataclass
class FilesystemResult:
    """Result of a filesystem operation."""
    success: bool
    message: str
    operation: FileOperation
    paths_affected: List[str] = None
    content: Optional[str] = None
    file_info: Optional[FileInfo] = None
    error_details: Optional[str] = None
    
    def __post_init__(self):
        if self.paths_affected is None:
            self.paths_affected = []

def read_file(
    file_path: str,
    encoding: str = "utf-8",
    working_directory: str = "."
) -> FilesystemResult:
    """Read content from a file."""
    full_path = Path(working_directory) / file_path
    
    logger.info(f"Reading file: {full_path}")
    
    if not full_path.exists():
        return FilesystemResult(
            success=False,
            message=f"File does not exist: {file_path}",
            operation=FileOperation.READ,
            error_details=f"File not found: {full_path.absolute()}"
        )
    
    if not full_path.is_file():
        return FilesystemResult(
            success=False,
            message=f"Path is not a file: {file_path}",
            operation=FileOperation.READ,
            error_details=f"Path is a directory: {full_path.absolute()}"
        )
    
    try:
        content = full_path.read_text(encoding=encoding)
        file_info = FileInfo.from_path(full_path)
        
        return FilesystemResult(
            success=True,
            message=f"Successfully read file: {file_path} ({len(content)} characters)",
            operation=FileOperation.READ,
            paths_affected=[str(full_path)],
            content=content,
            file_info=file_info
        )
        
    except UnicodeDecodeError as e:
        return FilesystemResult(
            success=False,
            message=f"Failed to decode file with {encoding} encoding",
            operation=FileOperation.READ,
            error_details=f"Encoding error: {str(e)}"
        )
        
    except Exception as e:
        return FilesystemResult(
            success=False,
            message=f"Failed to read file: {str(e)}",
            operation=FileOperation.READ,
            error_details=str(e)
        )

def write_file(
    file_path: str,
    content: str,
    encoding: str = "utf-8",
    backup_enabled: bool = False,
    create_directories: bool = True,
    working_directory: str = "."
) -> FilesystemResult:
    """Write content to a file."""
    full_path = Path(working_directory) / file_path
    
    logger.info(f"Writing file: {full_path}")
    
    paths_affected = []
    
    try:
        # Create parent directories if needed
        if create_directories:
            full_path.parent.mkdir(parents=True, exist_ok=True)
            if not full_path.parent.exists():
                return FilesystemResult(
                    success=False,
                    message=f"Failed to create parent directories for: {file_path}",
                    operation=FileOperation.WRITE,
                    error_details=f"Could not create: {full_path.parent}"
                )
        
        # Create backup if file exists and backup is enabled
        backup_path = None
        if backup_enabled and full_path.exists():
            backup_path = create_backup(full_path)
            if backup_path:
                paths_affected.append(str(backup_path))
        
        # Write the file
        full_path.write_text(content, encoding=encoding)
        paths_affected.append(str(full_path))
        
        file_info = FileInfo.from_path(full_path)
        
        backup_msg = f" (backup created: {backup_path.name})" if backup_path else ""
        return FilesystemResult(
            success=True,
            message=f"Successfully wrote file: {file_path} ({len(content)} characters){backup_msg}",
            operation=FileOperation.WRITE,
            paths_affected=paths_affected,
            file_info=file_info
        )
        
    except Exception as e:
        return FilesystemResult(
            success=False,
            message=f"Failed to write file: {str(e)}",
            operation=FileOperation.WRITE,
            error_details=str(e)
        )

def create_path(
    path: str,
    content: Optional[str] = None,
    encoding: str = "utf-8",
    working_directory: str = "."
) -> FilesystemResult:
    """Create a file or directory."""
    full_path = Path(working_directory) / path
    
    logger.info(f"Creating path: {full_path}")
    
    if full_path.exists():
        return FilesystemResult(
            success=False,
            message=f"Path already exists: {path}",
            operation=FileOperation.CREATE,
            error_details=f"Path exists: {full_path.absolute()}"
        )
    
    try:
        if content is not None:
            # Create file with content
            full_path.parent.mkdir(parents=True, exist_ok=True)
            full_path.write_text(content, encoding=encoding)
            operation_desc = f"file with {len(content)} characters"
        else:
            # Create directory
            full_path.mkdir(parents=True, exist_ok=True)
            operation_desc = "directory"
        
        file_info = FileInfo.from_path(full_path)
        
        return FilesystemResult(
            success=True,
            message=f"Successfully created {operation_desc}: {path}",
            operation=FileOperation.CREATE,
            paths_affected=[str(full_path)],
            content=content,
            file_info=file_info
        )
        
    except Exception as e:
        return FilesystemResult(
            success=False,
            message=f"Failed to create path: {str(e)}",
            operation=FileOperation.CREATE,
            error_details=str(e)
        )

def delete_path(
    path: str,
    backup_enabled: bool = False,
    working_directory: str = "."
) -> FilesystemResult:
    """Delete a file or directory."""
    full_path = Path(working_directory) / path
    
    logger.info(f"Deleting path: {full_path}")
    
    if not full_path.exists():
        return FilesystemResult(
            success=False,
            message=f"Path does not exist: {path}",
            operation=FileOperation.DELETE,
            error_details=f"Path not found: {full_path.absolute()}"
        )
    
    paths_affected = []
    
    try:
        # Create backup if enabled
        backup_path = None
        if backup_enabled:
            backup_path = create_backup(full_path)
            if backup_path:
                paths_affected.append(str(backup_path))
        
        # Delete the path
        if full_path.is_file():
            full_path.unlink()
            operation_desc = "file"
        else:
            shutil.rmtree(full_path)
            operation_desc = "directory"
        
        paths_affected.append(str(full_path))
        
        backup_msg = f" (backup created: {backup_path.name})" if backup_path else ""
        return FilesystemResult(
            success=True,
            message=f"Successfully deleted {operation_desc}: {path}{backup_msg}",
            operation=FileOperation.DELETE,
            paths_affected=paths_affected
        )
        
    except Exception as e:
        return FilesystemResult(
            success=False,
            message=f"Failed to delete path: {str(e)}",
            operation=FileOperation.DELETE,
            error_details=str(e)
        )

def copy_path(
    source_path: str,
    destination_path: str,
    working_directory: str = ".",
    preserve_permissions: bool = True
) -> FilesystemResult:
    """Copy a file or directory."""
    source_full = Path(working_directory) / source_path
    dest_full = Path(working_directory) / destination_path
    
    logger.info(f"Copying {source_full} to {dest_full}")
    
    if not source_full.exists():
        return FilesystemResult(
            success=False,
            message=f"Source path does not exist: {source_path}",
            operation=FileOperation.COPY,
            error_details=f"Source not found: {source_full.absolute()}"
        )
    
    try:
        # Create parent directory of destination if needed
        dest_full.parent.mkdir(parents=True, exist_ok=True)
        
        if source_full.is_file():
            if preserve_permissions:
                shutil.copy2(source_full, dest_full)  # Preserves metadata
            else:
                shutil.copy(source_full, dest_full)   # Just content
            operation_desc = "file"
        else:
            shutil.copytree(source_full, dest_full)
            operation_desc = "directory"
        
        file_info = FileInfo.from_path(dest_full)
        
        return FilesystemResult(
            success=True,
            message=f"Successfully copied {operation_desc}: {source_path} → {destination_path}",
            operation=FileOperation.COPY,
            paths_affected=[str(source_full), str(dest_full)],
            file_info=file_info
        )
        
    except Exception as e:
        return FilesystemResult(
            success=False,
            message=f"Failed to copy path: {str(e)}",
            operation=FileOperation.COPY,
            error_details=str(e)
        )

def move_path(
    source_path: str,
    destination_path: str,
    working_directory: str = "."
) -> FilesystemResult:
    """Move/rename a file or directory."""
    source_full = Path(working_directory) / source_path
    dest_full = Path(working_directory) / destination_path
    
    logger.info(f"Moving {source_full} to {dest_full}")
    
    if not source_full.exists():
        return FilesystemResult(
            success=False,
            message=f"Source path does not exist: {source_path}",
            operation=FileOperation.MOVE,
            error_details=f"Source not found: {source_full.absolute()}"
        )
    
    try:
        # Create parent directory of destination if needed
        dest_full.parent.mkdir(parents=True, exist_ok=True)
        
        # Move the path
        shutil.move(str(source_full), str(dest_full))
        
        operation_desc = "file" if dest_full.is_file() else "directory"
        file_info = FileInfo.from_path(dest_full)
        
        return FilesystemResult(
            success=True,
            message=f"Successfully moved {operation_desc}: {source_path} → {destination_path}",
            operation=FileOperation.MOVE,
            paths_affected=[str(source_full), str(dest_full)],
            file_info=file_info
        )
        
    except Exception as e:
        return FilesystemResult(
            success=False,
            message=f"Failed to move path: {str(e)}",
            operation=FileOperation.MOVE,
            error_details=str(e)
        )

def discover_files(
    search_path: str = ".",
    patterns: List[str] = None,
    exclude_patterns: List[str] = None,
    recursive: bool = True,
    file_types: List[str] = None,
    max_depth: int = 10,
    working_directory: str = "."
) -> FilesystemResult:
    """Discover files based on patterns and filters."""
    full_path = Path(working_directory) / search_path
    
    logger.info(f"Discovering files in: {full_path}")
    
    if not full_path.exists():
        return FilesystemResult(
            success=False,
            message=f"Search path does not exist: {search_path}",
            operation=FileOperation.DISCOVER,
            error_details=f"Path not found: {full_path.absolute()}"
        )
    
    if patterns is None:
        patterns = ["*"]
    
    if exclude_patterns is None:
        exclude_patterns = [
            "__pycache__",
            "*.pyc",
            "*.pyo",
            ".git",
            ".svn",
            "node_modules",
            ".pytest_cache",
            ".mypy_cache"
        ]
    
    if file_types is None:
        file_types = ["all"]
    
    try:
        discovered_files = []
        
        # Convert file types to extensions
        type_extensions = {
            FileType.PYTHON.value: [".py", ".pyx", ".pyi"],
            FileType.JAVASCRIPT.value: [".js", ".jsx", ".ts", ".tsx"],
            FileType.TEXT.value: [".txt", ".text"],
            FileType.JSON.value: [".json"],
            FileType.YAML.value: [".yaml", ".yml"],
            FileType.MARKDOWN.value: [".md", ".markdown"],
            FileType.CONFIG.value: [".cfg", ".conf", ".config", ".ini", ".toml"]
        }
        
        # Get all allowed extensions
        allowed_extensions = set()
        if "all" in file_types:
            allowed_extensions = None  # Allow all
        else:
            for file_type in file_types:
                if file_type in type_extensions:
                    allowed_extensions.update(type_extensions[file_type])
        
        # Search for files
        for pattern in patterns:
            if recursive:
                matches = full_path.rglob(pattern)
            else:
                matches = full_path.glob(pattern)
            
            for match in matches:
                # Skip if max depth exceeded
                if recursive and max_depth > 0:
                    try:
                        relative_path = match.relative_to(full_path)
                        depth = len(relative_path.parts)
                        if depth > max_depth:
                            continue
                    except ValueError:
                        continue
                
                # Skip directories unless specifically searching for them
                if not match.is_file():
                    continue
                
                # Apply exclusion patterns
                skip = False
                for exclude_pattern in exclude_patterns:
                    if exclude_pattern in str(match):
                        skip = True
                        break
                if skip:
                    continue
                
                # Apply file type filter
                if allowed_extensions is not None and match.suffix not in allowed_extensions:
                    continue
                
                discovered_files.append(str(match.relative_to(Path(working_directory))))
        
        # Remove duplicates and sort
        discovered_files = sorted(list(set(discovered_files)))
        
        return FilesystemResult(
            success=True,
            message=f"Discovered {len(discovered_files)} files matching criteria",
            operation=FileOperation.DISCOVER,
            paths_affected=discovered_files,
            content="\n".join(discovered_files) if discovered_files else "No files found"
        )
        
    except Exception as e:
        return FilesystemResult(
            success=False,
            message=f"Failed to discover files: {str(e)}",
            operation=FileOperation.DISCOVER,
            error_details=str(e)
        )

def create_backup(file_path: Path) -> Optional[Path]:
    """Create a backup of a file or directory."""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = file_path.with_suffix(f"{file_path.suffix}.backup_{timestamp}")
        
        if file_path.is_file():
            shutil.copy2(file_path, backup_path)
        else:
            shutil.copytree(file_path, backup_path)
        
        logger.info(f"Created backup: {backup_path}")
        return backup_path
        
    except Exception as e:
        logger.error(f"Failed to create backup: {e}")
        return None

def get_file_info(
    file_path: str,
    working_directory: str = "."
) -> FilesystemResult:
    """Get detailed information about a file or directory."""
    full_path = Path(working_directory) / file_path
    
    if not full_path.exists():
        return FilesystemResult(
            success=False,
            message=f"Path does not exist: {file_path}",
            operation=FileOperation.READ,
            error_details=f"Path not found: {full_path.absolute()}"
        )
    
    try:
        file_info = FileInfo.from_path(full_path)
        
        # Additional info for directories
        additional_info = {}
        if full_path.is_dir():
            try:
                items = list(full_path.iterdir())
                additional_info["directory_items"] = len(items)
                additional_info["subdirectories"] = len([item for item in items if item.is_dir()])
                additional_info["files"] = len([item for item in items if item.is_file()])
            except Exception:
                additional_info["directory_items"] = "Unknown"
        
        return FilesystemResult(
            success=True,
            message=f"File info retrieved for: {file_path}",
            operation=FileOperation.READ,
            paths_affected=[str(full_path)],
            file_info=file_info,
            content=str(additional_info) if additional_info else None
        )
        
    except Exception as e:
        return FilesystemResult(
            success=False,
            message=f"Failed to get file info: {str(e)}",
            operation=FileOperation.READ,
            error_details=str(e)
        )

def ensure_directory_exists(
    directory_path: str,
    working_directory: str = "."
) -> FilesystemResult:
    """Ensure a directory exists, creating it if necessary."""
    full_path = Path(working_directory) / directory_path
    
    if full_path.exists():
        if full_path.is_dir():
            return FilesystemResult(
                success=True,
                message=f"Directory already exists: {directory_path}",
                operation=FileOperation.CREATE,
                paths_affected=[str(full_path)]
            )
        else:
            return FilesystemResult(
                success=False,
                message=f"Path exists but is not a directory: {directory_path}",
                operation=FileOperation.CREATE,
                error_details=f"Path is a file: {full_path.absolute()}"
            )
    
    try:
        full_path.mkdir(parents=True, exist_ok=True)
        file_info = FileInfo.from_path(full_path)
        
        return FilesystemResult(
            success=True,
            message=f"Directory created: {directory_path}",
            operation=FileOperation.CREATE,
            paths_affected=[str(full_path)],
            file_info=file_info
        )
        
    except Exception as e:
        return FilesystemResult(
            success=False,
            message=f"Failed to create directory: {str(e)}",
            operation=FileOperation.CREATE,
            error_details=str(e)
        )

def read_multiple_files(
    file_paths: List[str],
    encoding: str = "utf-8",
    working_directory: str = "."
) -> Dict[str, FilesystemResult]:
    """Read multiple files and return results for each."""
    results = {}
    
    for file_path in file_paths:
        result = read_file(file_path, encoding, working_directory)
        results[file_path] = result
    
    return results

def write_multiple_files(
    files_content: Dict[str, str],
    encoding: str = "utf-8",
    backup_enabled: bool = False,
    create_directories: bool = True,
    working_directory: str = "."
) -> Dict[str, FilesystemResult]:
    """Write multiple files and return results for each."""
    results = {}
    
    for file_path, content in files_content.items():
        result = write_file(
            file_path, content, encoding, backup_enabled, create_directories, working_directory
        )
        results[file_path] = result
    
    return results

def get_directory_size(
    directory_path: str,
    working_directory: str = "."
) -> FilesystemResult:
    """Calculate the total size of a directory and its contents."""
    full_path = Path(working_directory) / directory_path
    
    if not full_path.exists():
        return FilesystemResult(
            success=False,
            message=f"Directory does not exist: {directory_path}",
            operation=FileOperation.READ,
            error_details=f"Path not found: {full_path.absolute()}"
        )
    
    if not full_path.is_dir():
        return FilesystemResult(
            success=False,
            message=f"Path is not a directory: {directory_path}",
            operation=FileOperation.READ,
            error_details=f"Path is a file: {full_path.absolute()}"
        )
    
    try:
        total_size = 0
        file_count = 0
        dir_count = 0
        
        for item in full_path.rglob("*"):
            if item.is_file():
                total_size += item.stat().st_size
                file_count += 1
            elif item.is_dir():
                dir_count += 1
        
        size_info = {
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "file_count": file_count,
            "directory_count": dir_count
        }
        
        return FilesystemResult(
            success=True,
            message=f"Directory size calculated: {size_info['total_size_mb']} MB ({file_count} files, {dir_count} dirs)",
            operation=FileOperation.READ,
            paths_affected=[str(full_path)],
            content=str(size_info)
        )
        
    except Exception as e:
        return FilesystemResult(
            success=False,
            message=f"Failed to calculate directory size: {str(e)}",
            operation=FileOperation.READ,
            error_details=str(e)
        )