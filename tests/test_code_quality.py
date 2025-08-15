#!/usr/bin/env python3
"""
Test script for code quality level detection and prompt generation.
"""

import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.coder import CodeGenerationAgent, CodeQuality
from core.context import GlobalContext
from utils.logger import setup_logger

setup_logger(default_level=logging.INFO)

def test_quality_detection():
    """Test that quality level detection works correctly."""
    print("🔍 Testing Code Quality Level Detection")
    
    agent = CodeGenerationAgent()
    context = GlobalContext()
    
    test_cases = [
        ("Create a quick prototype for user login", CodeQuality.FAST),
        ("Build a production-ready authentication system", CodeQuality.PRODUCTION),
        ("Implement a user management module", CodeQuality.FAST),  # default
        ("Generate a rapid MVP for testing", CodeQuality.FAST),
        ("Create enterprise-grade logging system", CodeQuality.PRODUCTION),
    ]
    
    print("\nQuality Detection Results:")
    print("=" * 60)
    
    all_passed = True
    for goal, expected in test_cases:
        detected = agent._detect_quality_level(goal, context)
        status = "✅" if detected == expected else "❌"
        print(f"{status} '{goal[:40]}...' -> {detected.value.upper()} (expected {expected.value.upper()})")
        
        if detected != expected:
            all_passed = False
    
    return all_passed

def test_prompt_differences():
    """Test that different quality levels produce different prompts."""
    print("\n🔍 Testing Prompt Generation Differences")
    
    agent = CodeGenerationAgent()
    spec = "Create a simple calculator with add and subtract functions"
    files = ["calculator.py"]
    existing_code = {}
    
    prompts = {}
    for quality in CodeQuality:
        prompt = agent._build_prompt(files, spec, existing_code, quality)
        prompts[quality] = prompt
        print(f"\n{quality.value.upper()} Quality Prompt Length: {len(prompt)} characters")
        
        # Show quality-specific instruction snippet
        lines = prompt.split('\n')
        quality_section = []
        in_quality_section = False
        for line in lines:
            if "Quality-Specific Instructions:" in line:
                in_quality_section = True
            elif in_quality_section and line.strip().startswith("**"):
                break
            elif in_quality_section and line.strip():
                quality_section.append(line.strip())
        
        print(f"Key Instructions: {quality_section[:2]}")  # Show first 2 instructions
    
    # Verify prompts are different
    if len(set(prompts.values())) == len(prompts):
        print("\n✅ All quality levels produce unique prompts")
        return True
    else:
        print("\n❌ Some quality levels produce identical prompts")
        return False

def main():
    print("🚀 Testing Code Quality Level System")
    
    detection_passed = test_quality_detection()
    prompt_passed = test_prompt_differences()
    
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    print(f"Quality Detection:  {'✅ PASS' if detection_passed else '❌ FAIL'}")
    print(f"Prompt Differences: {'✅ PASS' if prompt_passed else '❌ FAIL'}")
    print("="*60)
    
    if detection_passed and prompt_passed:
        print("🎉 All code quality tests PASSED!")
        print("\nUsage Examples:")
        print("- 'Create a quick prototype...' -> FAST quality")
        print("- 'Build production-ready...' -> PRODUCTION quality") 
        print("- 'Implement a module...' -> FAST quality (default)")
    else:
        print("⚠️ Some tests failed.")

if __name__ == "__main__":
    main()