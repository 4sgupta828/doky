#!/usr/bin/env python3
"""
Configuration management for the Sovereign Agent Collective.
Handles environment variables, default values, and workspace settings.
"""

import os
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

class Config:
    """
    Central configuration manager for the agent collective.
    Reads from environment variables with sensible defaults.
    """
    
    def __init__(self):
        # Load environment variables from .env if available
        self._load_env_file()
    
    def _load_env_file(self):
        """Load environment variables from .env file if it exists."""
        env_file = Path(".env")
        if env_file.exists():
            try:
                with open(env_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            # Only set if not already in environment
                            if key.strip() not in os.environ:
                                os.environ[key.strip()] = value.strip()
            except Exception as e:
                logger.warning(f"Could not load .env file: {e}")
    
    @property
    def workspace_root_dir(self) -> str:
        """
        Root directory for auto-generated workspaces.
        
        Checks:
        1. WORKSPACE_ROOT_DIR environment variable
        2. Defaults to user's home directory
        """
        # First check environment variable
        root_dir = os.getenv("WORKSPACE_ROOT_DIR")
        if root_dir:
            return root_dir
            
        # Default to user's home directory
        home_dir = Path.home()
        default_root = str(home_dir)
        
        logger.debug(f"Using default workspace root: {default_root}")
        return default_root
    
    @property
    def session_data_dir_name(self) -> str:
        """
        Name of the hidden directory for session data within workspaces.
        """
        return os.getenv("SESSION_DATA_DIR", ".doky")
    
    @property
    def max_context_tokens(self) -> int:
        """
        Maximum tokens allowed in LLM context.
        """
        return int(os.getenv("MAX_CONTEXT_TOKENS", "50000"))
    
    @property
    def llm_max_response_tokens(self) -> int:
        """
        Default maximum tokens for LLM responses.
        """
        return int(os.getenv("LLM_MAX_RESPONSE_TOKENS", "16000"))
    
    @property
    def llm_schema_max_response_tokens(self) -> int:
        """
        Maximum tokens for LLM schema-based responses (function calling).
        Needs to be higher for complex structured outputs.
        """
        return int(os.getenv("LLM_SCHEMA_MAX_RESPONSE_TOKENS", "16000"))
    
    @property
    def llm_default_temperature(self) -> float:
        """
        Default temperature for LLM responses.
        """
        return float(os.getenv("LLM_DEFAULT_TEMPERATURE", "0.1"))
    
    @property
    def llm_token_estimation_ratio(self) -> int:
        """
        Characters per token ratio for token estimation.
        """
        return int(os.getenv("LLM_TOKEN_ESTIMATION_RATIO", "4"))
    
    @property
    def openai_api_key(self) -> Optional[str]:
        """OpenAI API key from environment."""
        return os.getenv("OPENAI_API_KEY")
    
    @property
    def anthropic_api_key(self) -> Optional[str]:
        """Anthropic API key from environment."""
        return os.getenv("ANTHROPIC_API_KEY")
    
    def get_workspace_session_dir(self, workspace_path: str) -> Path:
        """
        Get the path to session data directory within a workspace.
        
        Args:
            workspace_path: Path to the workspace directory
            
        Returns:
            Path to the session data directory (e.g., workspace/.doky/)
        """
        workspace = Path(workspace_path)
        session_dir = workspace / self.session_data_dir_name
        
        # Create the directory if it doesn't exist
        session_dir.mkdir(parents=True, exist_ok=True)
        
        return session_dir

# Global config instance
config = Config()

if __name__ == "__main__":
    """Test the configuration system."""
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("=== Configuration Test ===")
    print(f"Workspace root dir: {config.workspace_root_dir}")
    print(f"Session data dir name: {config.session_data_dir_name}")
    print(f"Max context tokens: {config.max_context_tokens}")
    print(f"LLM max response tokens: {config.llm_max_response_tokens}")
    print(f"LLM schema max response tokens: {config.llm_schema_max_response_tokens}")
    print(f"LLM default temperature: {config.llm_default_temperature}")
    print(f"LLM token estimation ratio: {config.llm_token_estimation_ratio}")
    print(f"Has OpenAI key: {config.openai_api_key is not None}")
    print(f"Has Anthropic key: {config.anthropic_api_key is not None}")
    
    # Test workspace session dir creation
    test_workspace = "/tmp/test_workspace"
    session_dir = config.get_workspace_session_dir(test_workspace)
    print(f"Test session dir: {session_dir}")
    print(f"Session dir exists: {session_dir.exists()}")
    
    # Clean up
    import shutil
    if Path(test_workspace).exists():
        shutil.rmtree(test_workspace)