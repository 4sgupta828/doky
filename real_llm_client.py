"""
Real LLM Client implementation for the Sovereign Agent Collective.
This implements actual LLM integration instead of placeholder code.
"""

import os
import json
import logging
import time
import inspect
from typing import Optional

logger = logging.getLogger(__name__)

class ContextTooLargeError(Exception):
    """Raised when the context exceeds the configured limits."""
    pass

class RealLLMClient:
    """
    A real LLM client that can integrate with OpenAI, Anthropic, or other providers.
    This replaces the placeholder LLMClient classes throughout the codebase.
    """
    
    def __init__(self, provider: str = "openai", model: Optional[str] = None, max_context_tokens: int = 50000):
        """
        Initialize the LLM client with a specific provider.
        
        Args:
            provider: The LLM provider ("openai", "anthropic", etc.)
            model: The specific model to use (defaults to reasonable choices)
            max_context_tokens: Maximum tokens allowed in prompt context (default: 50K)
        """
        self.provider = provider.lower()
        self.model = model
        self._client = None
        self.max_context_tokens = max_context_tokens
        
        # Set default models based on provider
        if not self.model:
            if self.provider == "openai":
                self.model = "gpt-5"  # Default to GPT-5
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
    
    def _estimate_token_count(self, text: str) -> int:
        """
        Estimate token count for a given text.
        Uses configurable characters per token ratio.
        """
        from config import config
        return len(text) // config.llm_token_estimation_ratio
    
    def _validate_context_size(self, prompt: str) -> None:
        """
        Validate that the prompt doesn't exceed context limits.
        
        Args:
            prompt: The input prompt to validate
            
        Raises:
            ContextTooLargeError: If prompt exceeds limits
        """
        estimated_tokens = self._estimate_token_count(prompt)
        
        if estimated_tokens > self.max_context_tokens:
            # Analyze what's making the context large
            lines = prompt.split('\n')
            
            current_section = "unknown"
            section_sizes = {}
            
            for line in lines:
                if line.strip().startswith('**') and line.strip().endswith(':**'):
                    current_section = line.strip().replace('*', '').replace(':', '').lower()
                    section_sizes[current_section] = 0
                
                if current_section in section_sizes:
                    section_sizes[current_section] += len(line)
            
            # Sort sections by size
            large_sections = sorted(
                [(section, size) for section, size in section_sizes.items() if size > 5000],
                key=lambda x: x[1], 
                reverse=True
            )
            
            error_details = f"""
Context size limit exceeded!
- Estimated tokens: {estimated_tokens:,}
- Max allowed: {self.max_context_tokens:,}
- Prompt length: {len(prompt):,} characters

Large sections contributing to context:
"""
            for section, size in large_sections[:5]:  # Show top 5
                error_details += f"- {section}: {size:,} characters ({size//4:,} est. tokens)\n"
            
            if not large_sections:
                error_details += "- Unable to identify specific sections\n"
            
            error_details += f"\nSuggestions:\n- Reduce existing code context\n- Simplify technical specifications\n- Break down the task into smaller pieces"
            
            raise ContextTooLargeError(error_details)
    
    def _get_caller_info(self) -> str:
        """Get information about the caller (agent) making the LLM request."""
        frame = inspect.currentframe()
        try:
            # Go up the call stack to find the calling agent
            while frame:
                frame = frame.f_back
                if frame and frame.f_code.co_filename.endswith('.py'):
                    filename = frame.f_code.co_filename.split('/')[-1]
                    if 'agent' in filename.lower() or filename in ['orchestrator.py', 'planner.py']:
                        class_name = frame.f_locals.get('self', {})..__class__.__name__ if 'self' in frame.f_locals else 'Unknown'
                        function_name = frame.f_code.co_name
                        return f"{class_name}.{function_name} ({filename})"
            return "Unknown caller"
        except:
            return "Unknown caller"
        finally:
            del frame

    def invoke(self, prompt: str, max_tokens: int = None, temperature: float = None) -> str:
        """
        Invoke the LLM with a prompt and return the response.
        
        Args:
            prompt: The input prompt for the LLM
            max_tokens: Maximum tokens in the response
            temperature: Sampling temperature (0.0 to 1.0)
            
        Returns:
            The LLM's text response
            
        Raises:
            ContextTooLargeError: If prompt exceeds context limits
        """
        # Use config defaults if not provided
        from config import config
        if max_tokens is None:
            max_tokens = config.llm_max_response_tokens
        if temperature is None:
            temperature = config.llm_default_temperature
            
        # Validate context size before making the call
        self._validate_context_size(prompt)
        
        # Get caller information and start timing
        caller_info = self._get_caller_info()
        start_time = time.time()
        
        logger.info(f"LLM request started - Caller: {caller_info}, Model: {self.model}, Tokens: {max_tokens}")
        
        try:
            if self.provider == "openai":
                # Use max_completion_tokens for newer models like GPT-5
                if "gpt-5" in self.model.lower():
                    # GPT-5 only supports default temperature (1) and max_completion_tokens
                    response = self._client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        max_completion_tokens=max_tokens
                    )
                else:
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
            duration = time.time() - start_time
            logger.error(f"LLM request failed - Caller: {caller_info}, Duration: {duration:.2f}s, Error: {e}")
            raise
        finally:
            duration = time.time() - start_time
            logger.info(f"LLM request completed - Caller: {caller_info}, Duration: {duration:.2f}s")

    def invoke_with_schema(self, prompt: str, schema: dict, max_tokens: int = None, temperature: float = None) -> str:
        """
        Invoke the LLM with function calling to guarantee JSON response matching schema.
        
        Args:
            prompt: The input prompt for the LLM
            schema: JSON schema for the expected response structure
            max_tokens: Maximum tokens in the response
            temperature: Sampling temperature (0.0 to 1.0)
            
        Returns:
            The LLM's JSON response as a string
            
        Raises:
            ContextTooLargeError: If prompt exceeds context limits
        """
        # Use config defaults if not provided
        from config import config
        if max_tokens is None:
            max_tokens = config.llm_schema_max_response_tokens
        if temperature is None:
            temperature = config.llm_default_temperature
            
        logger.debug(f"invoke_with_schema called with max_tokens={max_tokens}, temperature={temperature}")
            
        # Validate context size before making the call
        self._validate_context_size(prompt)
        
        # Get caller information and start timing
        caller_info = self._get_caller_info()
        start_time = time.time()
        
        logger.info(f"LLM schema request started - Caller: {caller_info}, Model: {self.model}, Tokens: {max_tokens}")
        
        try:
            if self.provider == "openai":
                # Use tools API for GPT-5, legacy functions API for older models
                if "gpt-5" in self.model.lower():
                    # GPT-5 uses tools API
                    tool_def = {
                        "type": "function",
                        "function": {
                            "name": "respond",
                            "description": "Respond with the requested structured data",
                            "parameters": schema
                        }
                    }
                    
                    response = self._client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        tools=[tool_def],
                        tool_choice={"type": "function", "function": {"name": "respond"}},
                        max_completion_tokens=max_tokens
                    )
                    
                    # Debug logging
                    logger.debug(f"OpenAI response finish reason: {response.choices[0].finish_reason}")
                    logger.debug(f"Message content: {response.choices[0].message.content}")
                    logger.debug(f"Tool calls present: {bool(response.choices[0].message.tool_calls)}")
                    if response.choices[0].message.tool_calls:
                        logger.debug(f"First tool call arguments: {response.choices[0].message.tool_calls[0].function.arguments}")
                        logger.debug(f"Tool call function name: {response.choices[0].message.tool_calls[0].function.name}")
                    logger.debug(f"Prompt token count: {response.usage.prompt_tokens if response.usage else 'unknown'}")
                    logger.debug(f"Completion token count: {response.usage.completion_tokens if response.usage else 'unknown'}")
                    logger.debug(f"Model used: {self.model}")
                    logger.debug(f"Max completion tokens requested: {max_tokens}")
                    
                    # Extract tool call from response
                    tool_calls = response.choices[0].message.tool_calls
                    if not tool_calls:
                        logger.error(f"No tool calls in response. Full message: {response.choices[0].message}")
                        logger.error(f"Finish reason: {response.choices[0].finish_reason}")
                        
                        # Check if it's a content policy or other issue
                        if response.choices[0].finish_reason == "content_filter":
                            raise ValueError("Request was filtered by content policy")
                        elif response.choices[0].finish_reason == "length":
                            raise ValueError("Response was cut off due to length limits")
                        elif response.choices[0].message.content:
                            # Model returned content instead of tool call - try to extract JSON
                            content = response.choices[0].message.content.strip()
                            logger.warning(f"Model returned content instead of tool call, attempting to extract JSON: {content[:200]}...")
                            
                            # Try to find JSON in the content
                            import re
                            json_match = re.search(r'\{.*\}', content, re.DOTALL)
                            if json_match:
                                logger.info("Found JSON in content, using it as response")
                                return json_match.group(0)
                            else:
                                raise ValueError(f"Model returned content instead of tool call and no JSON found: {content[:200]}...")
                        else:
                            raise ValueError("No tool calls returned by OpenAI model")
                    return tool_calls[0].function.arguments
                    
                else:
                    # Legacy function calling for older models
                    function_def = {
                        "name": "respond",
                        "description": "Respond with the requested structured data",
                        "parameters": schema
                    }
                    
                    response = self._client.chat.completions.create(
                        model=self.model,
                        messages=[{"role": "user", "content": prompt}],
                        functions=[function_def],
                        function_call={"name": "respond"},
                        max_tokens=max_tokens,
                        temperature=temperature
                    )
                    
                    function_call = response.choices[0].message.function_call
                    if function_call is None:
                        raise ValueError("No function call returned by OpenAI model")
                    return function_call.arguments
                
            elif self.provider == "anthropic":
                # Anthropic tool calling
                tool_def = {
                    "name": "respond",
                    "description": "Respond with the requested structured data",
                    "input_schema": schema
                }
                
                response = self._client.messages.create(
                    model=self.model,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    tools=[tool_def],
                    tool_choice={"type": "tool", "name": "respond"},
                    messages=[{"role": "user", "content": prompt}]
                )
                
                # Extract tool use from response
                for content_block in response.content:
                    if hasattr(content_block, 'type') and content_block.type == 'tool_use':
                        return json.dumps(content_block.input)
                
                # Fallback if no tool use found
                raise ValueError("No tool use found in Anthropic response")
                
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"LLM schema request failed - Caller: {caller_info}, Duration: {duration:.2f}s, Error: {e}")
            raise
        finally:
            duration = time.time() - start_time
            logger.info(f"LLM schema request completed - Caller: {caller_info}, Duration: {duration:.2f}s")

# Convenience function to create a client
def create_llm_client(provider: str = None, model: str = None, max_context_tokens: int = None) -> RealLLMClient:
    """
    Create an LLM client, trying different providers based on available API keys.
    """
    # Use config for max_context_tokens if not provided
    if max_context_tokens is None:
        from config import config
        max_context_tokens = config.max_context_tokens
    
    if provider:
        return RealLLMClient(provider=provider, model=model, max_context_tokens=max_context_tokens)
    
    # Auto-detect based on available API keys - prefer Anthropic due to quota
    if os.getenv("ANTHROPIC_API_KEY"):
        logger.info("Using Anthropic client (ANTHROPIC_API_KEY found)")
        return RealLLMClient(provider="anthropic", model=model, max_context_tokens=max_context_tokens)
    elif os.getenv("OPENAI_API_KEY"):
        logger.info("Using OpenAI client (OPENAI_API_KEY found)")
        return RealLLMClient(provider="openai", model=model, max_context_tokens=max_context_tokens)
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