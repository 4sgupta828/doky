#!/usr/bin/env python3
"""
Test the increased display limits for agent I/O transparency.
"""

import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from interfaces.collaboration_ui import CollaborationUI
from utils.content_trimmer import ContentTrimmer


def test_increased_limits():
    """Test that the increased limits show more content."""
    
    print("🧪 Testing Increased I/O Display Limits")
    print("=" * 80)
    
    ui = CollaborationUI()
    trimmer = ContentTrimmer()
    
    # Create various types of longer content
    long_text = "This is a long text sample. " * 60  # ~1680 characters
    long_code = """def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n-1)

def prime_check(n):
    if n < 2:
        return False
    for i in range(2, int(n**0.5) + 1):
        if n % i == 0:
            return False
    return True

""" * 20  # Long code sample
    
    large_dict = {f"key_{i}": f"value_{i} with some detailed content here" for i in range(50)}
    large_list = [f"Item {i} with detailed description and content" for i in range(40)]
    
    print("\n📋 Testing Content Trimming with New Limits:")
    print("─" * 60)
    
    # Test cases
    test_cases = [
        ("Long Text", long_text, "text"),
        ("Long Code", long_code, "code"), 
        ("Large Dict", large_dict, "dict"),
        ("Large List", large_list, "list")
    ]
    
    for name, content, content_type in test_cases:
        result = trimmer.trim_content(content, content_type)
        print(f"\n🔍 {name}:")
        print(f"   Original length: {result.original_length}")
        print(f"   Display length: {result.display_length}")
        print(f"   Truncated: {result.was_truncated}")
        if result.was_truncated:
            print(f"   Truncation info: {result.truncation_info}")
        print(f"   Content type: {result.content_type}")
    
    print("\n" + "─" * 60)
    print("\n📤 Testing Agent I/O Display with Long Content:")
    
    # Test agent input display with long content
    ui.display_agent_input(
        "TestAgent",
        "Process a large dataset and generate comprehensive analysis with detailed recommendations",
        {
            "dataset_description": long_text[:500] + "...",
            "processing_config": large_dict,
            "analysis_steps": large_list[:10],
            "code_template": long_code[:1000] + "..."
        },
        "Agent selected based on its capability to handle large-scale data processing tasks"
    )
    
    # Test agent output display with long content
    ui.display_agent_output(
        "TestAgent", 
        True,
        "Successfully processed dataset and generated comprehensive analysis with 47 key findings",
        {
            "analysis_summary": long_text[:800],
            "generated_code": long_code[:1500],
            "recommendations": large_list[:15],
            "metrics": large_dict
        }
    )
    
    print("\n✅ Increased limits test completed!")
    print("📈 Summary of improvements:")
    print("   • Text content: 400 → 1200 characters, 15 → 40 lines")
    print("   • JSON content: 600 → 1500 characters, 20 → 50 lines") 
    print("   • Code content: 800 → 2000 characters, 30 → 80 lines")
    print("   • Dict summaries: 300 → 800 characters, 10 → 25 items")
    print("   • List summaries: 200 → 600 characters, 10 → 20 items")
    print("   • LLM prompts: 200 → 800 characters")
    print("   • LLM responses: 300 → 1000 characters")


if __name__ == "__main__":
    test_increased_limits()