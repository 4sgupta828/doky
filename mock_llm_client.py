# mock_llm_client.py
"""
Mock LLM Client for testing the intelligent routing system.

This provides a simple implementation that returns realistic routing decisions
for testing purposes, so we can verify the LLM routing system works correctly.
"""

import json
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)


class MockLLMClient:
    """
    Mock LLM client that provides realistic routing decisions for testing.
    
    This client analyzes the prompts and returns contextually appropriate
    routing decisions to test the intelligent routing system.
    """
    
    def __init__(self, enable_logging: bool = True):
        self.enable_logging = enable_logging
        self.call_count = 0
    
    def invoke(self, prompt: str) -> str:
        """
        Mock invoke method that analyzes the prompt and returns appropriate routing decisions.
        
        Args:
            prompt: The routing prompt from the LLM router
            
        Returns:
            JSON string with routing decision
        """
        self.call_count += 1
        
        if self.enable_logging:
            logger.info(f"MockLLMClient call #{self.call_count}")
        
        prompt_lower = prompt.lower()
        
        # Determine agent type from prompt
        if "routing intelligence for the AnalystAgent" in prompt:
            return self._route_analyst_agent(prompt, prompt_lower)
        elif "routing intelligence for the CreatorAgent" in prompt:
            return self._route_creator_agent(prompt, prompt_lower)
        elif "routing intelligence for the ExecutorAgent" in prompt:
            return self._route_executor_agent(prompt, prompt_lower)
        elif "routing intelligence for the SurgeonAgent" in prompt:
            return self._route_surgeon_agent(prompt, prompt_lower)
        elif "routing intelligence for the StrategistAgent" in prompt:
            return self._route_strategist_agent(prompt, prompt_lower)
        elif "Inter-Agent Routing Intelligence" in prompt:
            return self._route_inter_agent(prompt, prompt_lower)
        else:
            # Default routing decision
            return self._default_routing_decision()
    
    def _route_analyst_agent(self, prompt: str, prompt_lower: str) -> str:
        """Route for AnalystAgent based on goal analysis."""
        
        goal = self._extract_goal_from_prompt(prompt)
        goal_lower = goal.lower() if goal else ""
        
        if any(word in goal_lower for word in ["security", "vulnerability", "audit"]):
            return json.dumps({
                "decision": "quality_analysis",
                "confidence": 0.92,
                "reasoning": "Goal involves security concerns, requiring comprehensive security scanning and vulnerability assessment",
                "recommended_inputs": {"security_scan": True, "vulnerability_check": True}
            })
        elif any(word in goal_lower for word in ["problem", "error", "debug", "issue", "bug"]):
            return json.dumps({
                "decision": "problem_analysis", 
                "confidence": 0.95,
                "reasoning": "User is experiencing issues that need diagnosis and root cause analysis",
                "recommended_inputs": {"analysis_depth": "comprehensive", "include_stack_trace": True}
            })
        elif any(word in goal_lower for word in ["analyze", "check", "validate", "review"]):
            return json.dumps({
                "decision": "code_analysis",
                "confidence": 0.88,
                "reasoning": "Goal requires code analysis and validation to understand structure and correctness",
                "recommended_inputs": {"check_imports": True, "validation_level": "strict"}
            })
        else:
            return json.dumps({
                "decision": "comprehensive_analysis",
                "confidence": 0.85,
                "reasoning": "Goal requires comprehensive analysis to understand requirements and context",
                "recommended_inputs": {"analysis_scope": "full"}
            })
    
    def _route_creator_agent(self, prompt: str, prompt_lower: str) -> str:
        """Route for CreatorAgent based on creation intent."""
        
        goal = self._extract_goal_from_prompt(prompt)
        goal_lower = goal.lower() if goal else ""
        
        if any(word in goal_lower for word in ["test", "testing", "unit test", "test suite"]):
            return json.dumps({
                "decision": "tests",
                "confidence": 0.93,
                "reasoning": "User needs test creation to validate code functionality and ensure quality",
                "recommended_inputs": {"test_framework": "pytest", "test_type": "unit"}
            })
        elif any(word in goal_lower for word in ["documentation", "readme", "docs", "document"]):
            return json.dumps({
                "decision": "documentation",
                "confidence": 0.90,
                "reasoning": "User wants documentation generation to explain and document the codebase",
                "recommended_inputs": {"doc_type": "README", "target_audience": "developers"}
            })
        elif any(word in goal_lower for word in ["spec", "specification", "requirements"]):
            return json.dumps({
                "decision": "specification",
                "confidence": 0.87,
                "reasoning": "User needs technical specification creation from requirements",
                "recommended_inputs": {"spec_format": "detailed", "include_examples": True}
            })
        else:
            return json.dumps({
                "decision": "code",
                "confidence": 0.91,
                "reasoning": "Goal involves code creation and implementation of functionality",
                "recommended_inputs": {"code_quality": "production", "include_comments": True}
            })
    
    def _route_executor_agent(self, prompt: str, prompt_lower: str) -> str:
        """Route for ExecutorAgent based on execution needs."""
        
        goal = self._extract_goal_from_prompt(prompt)
        goal_lower = goal.lower() if goal else ""
        
        if any(word in goal_lower for word in ["test", "run test", "execute test"]):
            return json.dumps({
                "decision": "test_execution",
                "confidence": 0.94,
                "reasoning": "Goal requires running tests to validate implementation and ensure quality",
                "recommended_inputs": {"test_framework": "pytest", "coverage": True}
            })
        elif any(word in goal_lower for word in ["validate", "check", "verify"]):
            return json.dumps({
                "decision": "code_validation",
                "confidence": 0.89,
                "reasoning": "Goal requires validation of code correctness and functionality",
                "recommended_inputs": {"validation_type": "comprehensive", "check_syntax": True}
            })
        elif any(word in goal_lower for word in ["build", "compile", "deploy"]):
            return json.dumps({
                "decision": "shell_execution",
                "confidence": 0.86,
                "reasoning": "Goal requires shell operations for building or deployment",
                "recommended_inputs": {"environment": "production", "safety_checks": True}
            })
        else:
            return json.dumps({
                "decision": "file_operations",
                "confidence": 0.82,
                "reasoning": "Goal requires file system operations and management",
                "recommended_inputs": {"operation_type": "safe", "backup": True}
            })
    
    def _route_surgeon_agent(self, prompt: str, prompt_lower: str) -> str:
        """Route for SurgeonAgent based on modification needs."""
        
        goal = self._extract_goal_from_prompt(prompt)
        goal_lower = goal.lower() if goal else ""
        
        if any(word in goal_lower for word in ["requirements", "dependencies", "pip", "package"]):
            return json.dumps({
                "decision": "requirements_management",
                "confidence": 0.91,
                "reasoning": "Goal involves dependency and requirements management",
                "recommended_inputs": {"backup_existing": True, "version_strategy": "latest"}
            })
        elif any(word in goal_lower for word in ["config", "configuration", "settings"]):
            return json.dumps({
                "decision": "configuration_management", 
                "confidence": 0.88,
                "reasoning": "Goal involves configuration file management and updates",
                "recommended_inputs": {"backup_original": True, "validate_syntax": True}
            })
        elif any(word in goal_lower for word in ["fix", "repair", "patch", "modify"]):
            return json.dumps({
                "decision": "code_modification",
                "confidence": 0.93,
                "reasoning": "Goal requires precise code modifications and surgical fixes",
                "recommended_inputs": {"backup_enabled": True, "validation_mode": "strict"}
            })
        else:
            return json.dumps({
                "decision": "precise_repair",
                "confidence": 0.85,
                "reasoning": "Goal requires systematic surgical operations with validation",
                "recommended_inputs": {"multi_step": True, "validation_required": True}
            })
    
    def _route_strategist_agent(self, prompt: str, prompt_lower: str) -> str:
        """Route for StrategistAgent based on planning needs."""
        
        goal = self._extract_goal_from_prompt(prompt)
        goal_lower = goal.lower() if goal else ""
        
        if any(word in goal_lower for word in ["plan", "planning", "strategy", "decompose"]):
            return json.dumps({
                "decision": "task_planning",
                "confidence": 0.92,
                "reasoning": "Goal requires task decomposition and strategic planning",
                "recommended_inputs": {"planning_quality": "DECENT", "include_dependencies": True}
            })
        elif any(word in goal_lower for word in ["orchestrate", "coordinate", "execute workflow"]):
            return json.dumps({
                "decision": "workflow_orchestration",
                "confidence": 0.89,
                "reasoning": "Goal requires workflow orchestration and multi-agent coordination",
                "recommended_inputs": {"orchestration_mode": "adaptive", "parallel_execution": True}
            })
        else:
            return json.dumps({
                "decision": "task_planning",
                "confidence": 0.85,
                "reasoning": "Goal requires strategic task planning and decomposition",
                "recommended_inputs": {"complexity_handling": "advanced"}
            })
    
    def _route_inter_agent(self, prompt: str, prompt_lower: str) -> str:
        """Route for inter-agent coordination."""
        
        # Extract current hop information
        current_hop = 1
        if "current hop:" in prompt_lower:
            try:
                hop_part = prompt_lower.split("current hop:")[1].split("/")[0].strip()
                current_hop = int(hop_part)
            except:
                current_hop = 1
        
        # Check execution history for context
        if current_hop == 1:
            # Start with analysis
            return json.dumps({
                "agent_name": "AnalystAgent",
                "confidence": 0.90,
                "reasoning": "Starting workflow with comprehensive analysis to understand requirements and context",
                "goal_for_agent": "Analyze and understand the user's request to determine implementation approach",
                "recommended_inputs": {"analysis_depth": "comprehensive"},
                "is_completion": False,
                "completion_summary": ""
            })
        elif current_hop == 2:
            # Move to creation after analysis
            return json.dumps({
                "agent_name": "CreatorAgent",
                "confidence": 0.94,
                "reasoning": "Analysis complete, now creating implementation based on requirements",
                "goal_for_agent": "Create implementation based on analysis results",
                "recommended_inputs": {"based_on_analysis": True, "quality": "production"},
                "is_completion": False,
                "completion_summary": ""
            })
        elif current_hop == 3:
            # Move to execution/testing
            return json.dumps({
                "agent_name": "ExecutorAgent",
                "confidence": 0.91,
                "reasoning": "Implementation ready, now validating and testing the created solution",
                "goal_for_agent": "Execute tests and validate the implementation",
                "recommended_inputs": {"comprehensive_testing": True, "validation": True},
                "is_completion": False,
                "completion_summary": ""
            })
        else:
            # Final validation
            return json.dumps({
                "agent_name": "AnalystAgent",
                "confidence": 0.87,
                "reasoning": "Final validation to ensure user's goal has been achieved",
                "goal_for_agent": "Validate completion and provide final assessment",
                "recommended_inputs": {"validation_type": "completion", "goal_assessment": True},
                "is_completion": True,
                "completion_summary": "Implementation created, tested, and validated successfully"
            })
    
    def _extract_goal_from_prompt(self, prompt: str) -> str:
        """Extract the user goal from the prompt."""
        lines = prompt.split('\n')
        for line in lines:
            if 'Goal:' in line or 'USER\'S GOAL:' in line:
                return line.split(':', 1)[1].strip().strip('"')
        return ""
    
    def _default_routing_decision(self) -> str:
        """Provide a default routing decision."""
        return json.dumps({
            "decision": "code_analysis",
            "confidence": 0.70,
            "reasoning": "Default routing decision for unrecognized prompt pattern",
            "recommended_inputs": {}
        })


def create_mock_llm_client(enable_logging: bool = True) -> MockLLMClient:
    """Create a mock LLM client for testing."""
    return MockLLMClient(enable_logging=enable_logging)