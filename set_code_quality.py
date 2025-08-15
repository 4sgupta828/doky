#!/usr/bin/env python3
"""
Utility script to set global code quality preferences for the session.
"""

import argparse
from agents.coder import CodeQuality
from core.context import GlobalContext

def main():
    parser = argparse.ArgumentParser(
        description="Set global code quality preference for the current session"
    )
    parser.add_argument(
        "quality",
        choices=["fast", "decent", "production"],
        help="Code quality level to set as default"
    )
    parser.add_argument(
        "--workspace",
        help="Workspace path (if not provided, uses current directory)"
    )
    
    args = parser.parse_args()
    
    # Map string to enum
    quality_map = {
        "fast": CodeQuality.FAST,
        "decent": CodeQuality.DECENT, 
        "production": CodeQuality.PRODUCTION
    }
    
    quality_level = quality_map[args.quality]
    
    # Load or create context
    if args.workspace:
        context = GlobalContext(workspace_path=args.workspace)
    else:
        context = GlobalContext()
    
    # Set the preference
    context.code_quality_preference = quality_level
    
    # Save the preference (could be enhanced to persist in a config file)
    context.add_artifact(
        "code_quality_preference.txt", 
        quality_level.value, 
        "user_preference"
    )
    
    print(f"✅ Code quality preference set to: {quality_level.value.upper()}")
    print(f"   Workspace: {context.workspace_path}")
    print(f"   This setting will be used for all code generation tasks.")

if __name__ == "__main__":
    main()