#!/usr/bin/env python3
"""
Snapshot discovery utilities for resume functionality.
Provides automatic discovery of snapshot files for session resumption.
"""

import logging
from pathlib import Path
from typing import Optional, List, Tuple
from datetime import datetime

from core.context import GlobalContext

logger = logging.getLogger(__name__)


def discover_latest_snapshot(workspace_path: Optional[str] = None) -> Optional[str]:
    """
    Automatically discover the most recent snapshot file for a workspace.
    
    Args:
        workspace_path: Path to workspace directory. If None, uses current directory.
        
    Returns:
        Path to the most recent snapshot file, or None if no snapshots found.
    """
    if workspace_path is None:
        workspace_path = str(Path.cwd())
    
    try:
        snapshots = GlobalContext.list_available_snapshots(workspace_path)
        if snapshots:
            latest_snapshot = snapshots[0]  # Already sorted by modification time (newest first)
            logger.info(f"Auto-discovered latest snapshot: {latest_snapshot}")
            return latest_snapshot
        else:
            logger.info(f"No snapshots found in workspace: {workspace_path}")
            return None
            
    except Exception as e:
        logger.error(f"Error discovering snapshots: {e}")
        return None


def discover_snapshots_with_details(workspace_path: Optional[str] = None) -> List[Tuple[str, datetime, str]]:
    """
    Discover all snapshots with their details for user selection.
    
    Args:
        workspace_path: Path to workspace directory. If None, uses current directory.
        
    Returns:
        List of tuples containing (path, modification_time, relative_age_string)
    """
    if workspace_path is None:
        workspace_path = str(Path.cwd())
    
    try:
        snapshots = GlobalContext.list_available_snapshots(workspace_path)
        detailed_snapshots = []
        
        now = datetime.now()
        
        for snapshot_path in snapshots:
            snapshot_file = Path(snapshot_path)
            if snapshot_file.exists():
                mod_time = datetime.fromtimestamp(snapshot_file.stat().st_mtime)
                age_delta = now - mod_time
                
                # Format relative age
                if age_delta.days > 0:
                    age_str = f"{age_delta.days} days ago"
                elif age_delta.seconds > 3600:
                    hours = age_delta.seconds // 3600
                    age_str = f"{hours} hours ago"
                elif age_delta.seconds > 60:
                    minutes = age_delta.seconds // 60
                    age_str = f"{minutes} minutes ago"
                else:
                    age_str = "just now"
                
                detailed_snapshots.append((snapshot_path, mod_time, age_str))
        
        return detailed_snapshots
        
    except Exception as e:
        logger.error(f"Error discovering snapshot details: {e}")
        return []


def resolve_resume_argument(resume_arg: str, workspace_path: Optional[str] = None) -> Optional[str]:
    """
    Resolve the --resume argument to a specific snapshot file path.
    
    Args:
        resume_arg: The resume argument value. Can be:
            - "auto" or "latest": Auto-discover the most recent snapshot
            - A specific file path: Use as-is if it exists
            - Empty string when flag used without value: Auto-discover
            
        workspace_path: Workspace path for auto-discovery
        
    Returns:
        Resolved snapshot file path, or None if no valid snapshot found.
    """
    if resume_arg is None:
        # No resume requested
        return None
    
    if resume_arg == "" or resume_arg.lower() in ["auto", "latest"]:
        # Auto-discovery mode
        logger.info("Auto-discovering latest snapshot for resume...")
        return discover_latest_snapshot(workspace_path)
    
    # Explicit path provided
    snapshot_path = Path(resume_arg)
    if snapshot_path.exists() and snapshot_path.is_file():
        logger.info(f"Using specified snapshot file: {resume_arg}")
        return str(snapshot_path)
    else:
        logger.error(f"Specified snapshot file does not exist: {resume_arg}")
        
        # Fallback to auto-discovery
        logger.info("Falling back to auto-discovery...")
        return discover_latest_snapshot(workspace_path)


def validate_snapshot_file(snapshot_path: str) -> bool:
    """
    Validate that a snapshot file is readable and contains valid data.
    
    Args:
        snapshot_path: Path to the snapshot file
        
    Returns:
        True if the snapshot file is valid, False otherwise.
    """
    try:
        import json
        
        snapshot_file = Path(snapshot_path)
        if not snapshot_file.exists():
            logger.error(f"Snapshot file does not exist: {snapshot_path}")
            return False
        
        # Try to parse the JSON
        with open(snapshot_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Basic validation - check for required fields
        required_fields = ['task_graph', 'mission_log', 'workspace_path', 'timestamp']
        for field in required_fields:
            if field not in data:
                logger.error(f"Invalid snapshot file - missing field '{field}': {snapshot_path}")
                return False
        
        logger.info(f"Snapshot file validation passed: {snapshot_path}")
        return True
        
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in snapshot file {snapshot_path}: {e}")
        return False
    except Exception as e:
        logger.error(f"Error validating snapshot file {snapshot_path}: {e}")
        return False


def display_available_snapshots(workspace_path: Optional[str] = None) -> None:
    """
    Display available snapshots for user information.
    
    Args:
        workspace_path: Workspace path to search for snapshots
    """
    snapshots = discover_snapshots_with_details(workspace_path)
    
    if not snapshots:
        print("No snapshots found in the workspace.")
        return
    
    print(f"\n📁 Available snapshots ({len(snapshots)}):")
    print("─" * 80)
    
    for i, (path, mod_time, age_str) in enumerate(snapshots, 1):
        snapshot_name = Path(path).name
        size_kb = Path(path).stat().st_size // 1024
        print(f"{i:2d}. {snapshot_name}")
        print(f"    Created: {mod_time.strftime('%Y-%m-%d %H:%M:%S')} ({age_str})")
        print(f"    Size: {size_kb} KB")
        print(f"    Path: {path}")
        print()