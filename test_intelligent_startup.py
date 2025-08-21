#!/usr/bin/env python3
"""
Test script to verify the intelligent interactive system starts correctly.
"""

import sys
from pathlib import Path
import logging

# Add project root to Python path
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

def test_intelligent_session_startup():
    """Test that the intelligent session can initialize without errors."""
    
    print("🧪 Testing Intelligent Interactive System Startup...")
    
    try:
        from main_interactive_intelligent import IntelligentInteractiveSession
        from utils.logger import setup_logger
        from pathlib import Path
        
        # Setup minimal logging
        setup_logger(suppress_console_logs=True)
        
        print("✅ Successfully imported required modules")
        
        # Test session initialization
        session = IntelligentInteractiveSession(
            workspace_path=str(Path.cwd()),
            llm_client=None  # Use fallback routing
        )
        
        print("✅ Successfully initialized IntelligentInteractiveSession")
        print(f"   • Workspace: {session.global_context.workspace_path}")
        print(f"   • LLM Client: {'Connected' if session.llm_client else 'Fallback Mode'}")
        print(f"   • Active Workflows: {len(session.active_workflows)}")
        print(f"   • Session History: {len(session.session_history)}")
        
        # Test workflow coordinator
        if hasattr(session, 'workflow_coordinator') and session.workflow_coordinator:
            print("✅ WorkflowCoordinator is available")
        else:
            print("❌ WorkflowCoordinator not properly initialized")
            return False
        
        # Test UI
        if hasattr(session, 'ui') and session.ui:
            print("✅ CollaborationUI is available")
        else:
            print("❌ CollaborationUI not properly initialized") 
            return False
        
        print("\n🎉 All startup tests passed! The intelligent system is ready to use.")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure all required modules are available:")
        print("   • fagents/workflow_coordinator.py")
        print("   • fagents/inter_agent_router.py")
        print("   • fagents/routing.py")
        return False
        
    except Exception as e:
        print(f"❌ Startup test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def show_usage_instructions():
    """Show how to run the system."""
    
    print("\n" + "="*60)
    print("HOW TO RUN THE INTELLIGENT SYSTEM")
    print("="*60)
    
    print("\n🚀 BASIC USAGE:")
    print("   python main_interactive_intelligent.py")
    
    print("\n🔧 WITH OPTIONS:")
    print("   python main_interactive_intelligent.py --workspace ./my_project")
    print("   python main_interactive_intelligent.py --quiet-logs")
    
    print("\n💬 EXAMPLE SESSION:")
    print("   💬 You: Create a Python calculator with tests")
    print("   🤖 SYSTEM: ✅ Goal completed successfully in 3 hops")
    print("   🤖 SYSTEM: 🔄 Path: AnalystAgent → CreatorAgent → ExecutorAgent")
    
    print("\n🎯 INTERACTIVE COMMANDS:")
    print("   help      - Show available commands")
    print("   status    - Show session status")
    print("   workflows - List active workflows")
    print("   history   - Show execution history")
    print("   quit      - Exit the system")


if __name__ == "__main__":
    print("🧠 Intelligent Foundation Agent System - Startup Test")
    
    success = test_intelligent_session_startup()
    show_usage_instructions()
    
    if success:
        print(f"\n{'='*60}")
        print("✅ STARTUP TEST SUCCESSFUL")
        print("="*60)
        print("The intelligent interactive system is ready to use!")
        print("\nRun: python main_interactive_intelligent.py")
    else:
        print(f"\n{'='*60}")
        print("❌ STARTUP TEST FAILED")
        print("="*60)
        print("Please check the error messages above and ensure all dependencies are available.")
    
    sys.exit(0 if success else 1)