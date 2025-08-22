"""
Test module for prompts/planner.py

Generated tests for: add some tests
"""

import pytest
import sys
from pathlib import Path

# Add source directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from prompts.planner import build_intent_analysis_prompt, build_planning_prompt


def test_build_intent_analysis_prompt_basic():
    """Test basic functionality of build_intent_analysis_prompt."""
    # TODO: Add specific test cases for build_intent_analysis_prompt
    result = build_intent_analysis_prompt()
    assert result is not None


def test_build_intent_analysis_prompt_edge_cases():
    """Test edge cases for build_intent_analysis_prompt."""
    # TODO: Add edge case tests for build_intent_analysis_prompt
    with pytest.raises(Exception):
        build_intent_analysis_prompt(None)


def test_build_planning_prompt_basic():
    """Test basic functionality of build_planning_prompt."""
    # TODO: Add specific test cases for build_planning_prompt
    result = build_planning_prompt()
    assert result is not None


def test_build_planning_prompt_edge_cases():
    """Test edge cases for build_planning_prompt."""
    # TODO: Add edge case tests for build_planning_prompt
    with pytest.raises(Exception):
        build_planning_prompt(None)

