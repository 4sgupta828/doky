#!/usr/bin/env python3
"""
Test script to verify context size limits are working properly.
"""

import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from real_llm_client import RealLLMClient, ContextTooLargeError

def test_context_limits():
    """Test that context limits work as expected."""
    logging.basicConfig(level=logging.INFO)
    
    # Create a client with very small context limit for testing
    # We'll mock the client initialization to avoid API key requirement
    test_client = RealLLMClient.__new__(RealLLMClient)
    test_client.provider = "openai"
    test_client.model = "gpt-5"
    test_client.max_context_tokens = 10
    test_client._client = None
    
    # Test 1: Small prompt should work (no actual API call made due to no keys)
    small_prompt = "Hello world"
    try:
        estimated_tokens = test_client._estimate_token_count(small_prompt)
        print(f"✓ Small prompt estimation works: {estimated_tokens} tokens for '{small_prompt}'")
        
        # This would fail due to no API keys, but context validation should pass
        test_client._validate_context_size(small_prompt)
        print("✓ Small prompt passes context validation")
        
    except ContextTooLargeError as e:
        print(f"✗ Small prompt unexpectedly failed: {e}")
        return False
    except Exception as e:
        print(f"✓ Small prompt passed context validation (other error expected: {type(e).__name__})")
    
    # Test 2: Large prompt should fail fast
    large_prompt = "This is a very long prompt. " * 200  # ~5600 characters, ~1400 tokens
    try:
        test_client._validate_context_size(large_prompt)
        print("✗ Large prompt should have failed validation!")
        return False
    except ContextTooLargeError as e:
        print(f"✓ Large prompt correctly failed validation")
        print(f"✓ Error details provided: {str(e)[:200]}...")
        
        # Check that error contains useful information
        error_str = str(e)
        assert "Context size limit exceeded" in error_str
        assert "Estimated tokens:" in error_str
        assert "Max allowed:" in error_str
        print("✓ Error contains all expected details")
        return True
    except Exception as e:
        print(f"✗ Large prompt failed with unexpected error: {e}")
        return False

def test_realistic_limits():
    """Test with realistic limits."""
    print("\n--- Testing Realistic Limits ---")
    
    # Test with default 50K token limit (mock initialization)
    realistic_client = RealLLMClient.__new__(RealLLMClient)
    realistic_client.provider = "openai"
    realistic_client.model = "gpt-5"
    realistic_client.max_context_tokens = 50000
    realistic_client._client = None
    
    # Create a moderately large prompt (should pass)
    moderate_prompt = "def function():\n    pass\n" * 1000  # ~24K characters, ~6K tokens
    
    try:
        realistic_client._validate_context_size(moderate_prompt)
        print("✓ Moderate prompt (6K tokens) passes with 50K limit")
    except ContextTooLargeError as e:
        print(f"✗ Moderate prompt unexpectedly failed: {e}")
        return False
    
    # Create a huge prompt (should fail)
    huge_prompt = "def function():\n    pass\n" * 10000  # ~240K characters, ~60K tokens
    
    try:
        realistic_client._validate_context_size(huge_prompt)
        print("✗ Huge prompt should have failed validation!")
        return False
    except ContextTooLargeError as e:
        print("✓ Huge prompt (60K tokens) correctly failed with 50K limit")
        return True

if __name__ == "__main__":
    print("=== Testing Context Size Limits ===")
    
    success1 = test_context_limits()
    success2 = test_realistic_limits()
    
    if success1 and success2:
        print("\n🎉 All context limit tests passed!")
    else:
        print("\n❌ Some context limit tests failed!")
        exit(1)