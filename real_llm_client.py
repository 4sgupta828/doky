"""
Real LLM Client implementation for the Sovereign Agent Collective.
This implements actual LLM integration instead of placeholder code.
"""

import os
import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class RealLLMClient:
    """
    A real LLM client that can integrate with OpenAI, Anthropic, or other providers.
    This replaces the placeholder LLMClient classes throughout the codebase.
    """
    
    def __init__(self, provider: str = "openai", model: Optional[str] = None):
        """
        Initialize the LLM client with a specific provider.
        
        Args:
            provider: The LLM provider ("openai", "anthropic", etc.)
            model: The specific model to use (defaults to reasonable choices)
        """
        self.provider = provider.lower()
        self.model = model
        self._client = None
        
        # Set default models based on provider
        if not self.model:
            if self.provider == "openai":
                self.model = "gpt-4o"  # Default to GPT-4o
            elif self.provider == "anthropic":
                self.model = "claude-3-haiku-20240307"  # Fast and efficient
            else:
                raise ValueError(f"Unsupported provider: {provider}")
        
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the actual LLM client based on the provider."""
        try:
            if self.provider == "openai":
                import openai
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    raise ValueError("OPENAI_API_KEY environment variable not set")
                self._client = openai.OpenAI(api_key=api_key)
                
            elif self.provider == "anthropic":
                import anthropic
                api_key = os.getenv("ANTHROPIC_API_KEY")
                if not api_key:
                    raise ValueError("ANTHROPIC_API_KEY environment variable not set")
                self._client = anthropic.Anthropic(api_key=api_key)
                
        except ImportError as e:
            logger.error(f"Missing required package for {self.provider}: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize {self.provider} client: {e}")
            raise
    
    def invoke(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.1) -> str:
        """
        Invoke the LLM with a prompt and return the response.
        
        Args:
            prompt: The input prompt for the LLM
            max_tokens: Maximum tokens in the response
            temperature: Sampling temperature (0.0 to 1.0)
            
        Returns:
            The LLM's text response
        """
        try:
            if self.provider == "openai":
                response = self._client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                return response.choices[0].message.content.strip()
                
            elif self.provider == "anthropic":
                response = self._client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text.strip()
                
        except Exception as e:
            logger.error(f"LLM invocation failed: {e}")
            raise

# Convenience function to create a client
def create_llm_client(provider: str = None, model: str = None) -> RealLLMClient:
    """
    Create an LLM client, trying different providers based on available API keys.
    """
    if provider:
        return RealLLMClient(provider=provider, model=model)
    
    # Auto-detect based on available API keys - prefer Anthropic due to quota
    if os.getenv("ANTHROPIC_API_KEY"):
        logger.info("Using Anthropic client (ANTHROPIC_API_KEY found)")
        return RealLLMClient(provider="anthropic", model=model)
    elif os.getenv("OPENAI_API_KEY"):
        logger.info("Using OpenAI client (OPENAI_API_KEY found)")
        return RealLLMClient(provider="openai", model=model)
    else:
        raise ValueError(
            "No API keys found. Please set OPENAI_API_KEY or ANTHROPIC_API_KEY environment variable."
        )


if __name__ == "__main__":
    """Test the LLM client with a simple query."""
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    try:
        client = create_llm_client()
        print(f"Testing {client.provider} client with model {client.model}")
        
        test_prompt = "Write a simple Python function that adds two numbers together. Return only the code."
        response = client.invoke(test_prompt, max_tokens=200)
        
        print(f"Response: {response}")
        print("✅ LLM client test successful!")
        
    except Exception as e:
        print(f"❌ LLM client test failed: {e}")
        print("\nMake sure you have set either OPENAI_API_KEY or ANTHROPIC_API_KEY")
        print("And installed the required packages: pip install openai anthropic")
        sys.exit(1)