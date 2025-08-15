#!/usr/bin/env python3
"""
Utility script to list available snapshots and resume sessions.
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

from core.context import GlobalContext
from utils.logger import setup_logger

def list_snapshots(workspace_path: str):
    """List all available snapshots for a workspace."""
    snapshots = GlobalContext.list_available_snapshots(workspace_path)
    
    if not snapshots:
        print(f"No snapshots found for workspace: {workspace_path}")
        return
    
    print(f"\nAvailable snapshots for {workspace_path}:")
    print("=" * 60)
    
    for i, snapshot_path in enumerate(snapshots, 1):
        snapshot_file = Path(snapshot_path)
        
        # Get file modification time
        mtime = datetime.fromtimestamp(snapshot_file.stat().st_mtime)
        
        print(f"{i:2}. {snapshot_file.name}")
        print(f"    Created: {mtime.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"    Path: {snapshot_path}")
        print()

def main():
    setup_logger()
    
    parser = argparse.ArgumentParser(
        description="List and resume from session snapshots"
    )
    parser.add_argument(
        "workspace_path",
        help="Path to the workspace to search for snapshots"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Interactive mode to select and resume from a snapshot"
    )
    
    args = parser.parse_args()
    
    if not Path(args.workspace_path).exists():
        print(f"Error: Workspace path does not exist: {args.workspace_path}")
        sys.exit(1)
    
    snapshots = GlobalContext.list_available_snapshots(args.workspace_path)
    
    if not snapshots:
        print(f"No snapshots found for workspace: {args.workspace_path}")
        sys.exit(1)
    
    if args.resume:
        # Interactive resume mode
        list_snapshots(args.workspace_path)
        
        try:
            choice = input("\nEnter snapshot number to resume from (or 'q' to quit): ").strip()
            
            if choice.lower() == 'q':
                print("Exiting.")
                return
            
            snapshot_index = int(choice) - 1
            if 0 <= snapshot_index < len(snapshots):
                selected_snapshot = snapshots[snapshot_index]
                print(f"\nTo resume from this snapshot, run:")
                print(f"python main_interactive.py --resume '{selected_snapshot}'")
            else:
                print("Invalid selection.")
                sys.exit(1)
                
        except (ValueError, KeyboardInterrupt):
            print("\nOperation cancelled.")
            sys.exit(1)
    else:
        # Just list snapshots
        list_snapshots(args.workspace_path)

if __name__ == "__main__":
    main()