"""
Enhanced input handler with command history and arrow key navigation support.
"""

import os
import sys
from typing import List, Optional

# Try to import readline for enhanced input features
try:
    import readline
    READLINE_AVAILABLE = True
except ImportError:
    READLINE_AVAILABLE = False

class CommandHistory:
    """Manages command history for the interactive session."""
    
    def __init__(self, max_history: int = 1000):
        """Initialize the command history manager.
        
        Args:
            max_history: Maximum number of commands to keep in history
        """
        self.max_history = max_history
        self.history: List[str] = []
        self.history_file: Optional[str] = None
        
        if READLINE_AVAILABLE:
            self._setup_readline()
    
    def _setup_readline(self):
        """Configure readline for enhanced input experience."""
        # Enable tab completion (basic)
        readline.parse_and_bind("tab: complete")
        
        # Enable history navigation with arrow keys
        readline.parse_and_bind('"\\e[A": previous-history')  # Up arrow
        readline.parse_and_bind('"\\e[B": next-history')      # Down arrow
        readline.parse_and_bind('"\\e[C": forward-char')      # Right arrow
        readline.parse_and_bind('"\\e[D": backward-char')     # Left arrow
        
        # Enable common editing shortcuts
        readline.parse_and_bind('"\\C-a": beginning-of-line') # Ctrl+A
        readline.parse_and_bind('"\\C-e": end-of-line')       # Ctrl+E
        readline.parse_and_bind('"\\C-k": kill-line')         # Ctrl+K
        readline.parse_and_bind('"\\C-u": unix-line-discard') # Ctrl+U
        
        # Set history length
        readline.set_history_length(self.max_history)
    
    def set_history_file(self, file_path: str):
        """Set the file path for persistent history storage.
        
        Args:
            file_path: Path to the history file
        """
        self.history_file = file_path
        self.load_history()
    
    def load_history(self):
        """Load command history from file."""
        if not self.history_file or not os.path.exists(self.history_file):
            return
        
        try:
            with open(self.history_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line and line not in self.history:
                        self.history.append(line)
                        if READLINE_AVAILABLE:
                            readline.add_history(line)
        except Exception:
            # Silently ignore history loading errors
            pass
    
    def save_history(self):
        """Save command history to file."""
        if not self.history_file:
            return
        
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
            
            # Save recent history to file
            recent_history = self.history[-self.max_history:]
            with open(self.history_file, 'w', encoding='utf-8') as f:
                for command in recent_history:
                    f.write(f"{command}\n")
        except Exception:
            # Silently ignore history saving errors
            pass
    
    def add_command(self, command: str):
        """Add a command to the history.
        
        Args:
            command: The command to add to history
        """
        if not command or not command.strip():
            return
        
        command = command.strip()
        
        # Don't add duplicate consecutive commands
        if self.history and self.history[-1] == command:
            return
        
        # Don't add sensitive commands to history
        if self._is_sensitive_command(command):
            return
        
        self.history.append(command)
        
        # Limit history size
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
        
        # Add to readline history
        if READLINE_AVAILABLE:
            readline.add_history(command)
    
    def _is_sensitive_command(self, command: str) -> bool:
        """Check if a command contains sensitive information.
        
        Args:
            command: Command to check
            
        Returns:
            True if the command appears to contain sensitive information
        """
        sensitive_keywords = ['password', 'token', 'key', 'secret', 'api_key']
        command_lower = command.lower()
        return any(keyword in command_lower for keyword in sensitive_keywords)
    
    def get_recent_commands(self, count: int = 10) -> List[str]:
        """Get the most recent commands.
        
        Args:
            count: Number of recent commands to return
            
        Returns:
            List of recent commands
        """
        return self.history[-count:] if self.history else []

class EnhancedInput:
    """Enhanced input handler with history and line editing support."""
    
    def __init__(self, history_file: Optional[str] = None):
        """Initialize the enhanced input handler.
        
        Args:
            history_file: Optional path to persistent history file
        """
        self.history = CommandHistory()
        
        if history_file:
            self.history.set_history_file(history_file)
        
        # Show a one-time message about enhanced features
        if READLINE_AVAILABLE and not getattr(self, '_features_shown', False):
            self._show_features_info()
            self._features_shown = True
    
    def _show_features_info(self):
        """Show information about available input features."""
        print("\n💡 Enhanced input features enabled:")
        print("   • Use ↑/↓ arrows to navigate command history")
        print("   • Use ←/→ arrows to move cursor within line")
        print("   • Ctrl+A: beginning of line, Ctrl+E: end of line")
        print("   • Ctrl+K: clear to end, Ctrl+U: clear entire line")
    
    def prompt(self, prompt_text: str) -> str:
        """Prompt for user input with enhanced features.
        
        Args:
            prompt_text: The prompt text to display
            
        Returns:
            User input string
        """
        try:
            if READLINE_AVAILABLE:
                user_input = input(prompt_text)
            else:
                # Fallback to basic input with a notice
                if not getattr(self, '_fallback_shown', False):
                    print("💡 Note: Install 'readline' for enhanced input features (history, arrow keys)")
                    self._fallback_shown = True
                user_input = input(prompt_text)
            
            # Add to history if it's a meaningful command
            self.history.add_command(user_input)
            
            return user_input.strip()
        
        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully
            print("\n")
            raise
        except EOFError:
            # Handle Ctrl+D as exit
            return "exit"
    
    def save_history(self):
        """Save the command history."""
        self.history.save_history()
    
    def get_history(self) -> List[str]:
        """Get the command history.
        
        Returns:
            List of historical commands
        """
        return self.history.history.copy()

# Global instance for the session
_global_input_handler: Optional[EnhancedInput] = None

def get_input_handler(history_file: Optional[str] = None) -> EnhancedInput:
    """Get or create the global input handler instance.
    
    Args:
        history_file: Optional history file path (used only on first call)
        
    Returns:
        The global EnhancedInput instance
    """
    global _global_input_handler
    if _global_input_handler is None:
        _global_input_handler = EnhancedInput(history_file)
    return _global_input_handler

def enhanced_input(prompt_text: str) -> str:
    """Enhanced input function with history support.
    
    Args:
        prompt_text: The prompt text to display
        
    Returns:
        User input string
    """
    handler = get_input_handler()
    return handler.prompt(prompt_text)