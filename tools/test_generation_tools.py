# tools/test_generation_tools.py
import json
import logging
from typing import Dict, Any, List, Literal, Optional
from enum import Enum
from pathlib import Path

# Get a logger instance for this module
logger = logging.getLogger(__name__)


class TestQuality(Enum):
    """Defines different test quality levels for speed vs thoroughness trade-offs."""
    FAST = "fast"          # Quick, basic tests - prioritizes speed
    DECENT = "decent"      # Balanced approach - good coverage, reasonable detail (default)
    PRODUCTION = "production"  # Comprehensive, thorough tests with full edge case coverage


class TestGenerationTools:
    """
    Agent-agnostic test generation utilities.
    
    This module provides low-level test generation utilities that can be used
    by any agent or component. It contains no agent-specific logic, prompts,
    or LLM interactions - only pure data processing and analysis functions.
    """

    @staticmethod
    def determine_test_type(goal_text: str) -> Literal["unit", "integration"]:
        """Analyzes goal text to determine what kind of test to write."""
        if "integration" in goal_text.lower():
            return "integration"
        # Default to unit tests for specificity and speed
        return "unit"

    @staticmethod
    def detect_quality_level(goal_text: str, context_quality: Optional[TestQuality] = None) -> TestQuality:
        """Detects the desired test quality level from goal text and context."""
        goal_lower = goal_text.lower()
        
        # Check for explicit quality keywords in the goal
        if any(keyword in goal_lower for keyword in ['fast', 'quick', 'basic', 'simple', 'minimal']):
            logger.debug("Detected FAST test quality level from goal keywords")
            return TestQuality.FAST
        elif any(keyword in goal_lower for keyword in ['decent', 'thorough', 'comprehensive', 'good', 'complete']):
            logger.debug("Detected DECENT test quality level from goal keywords")
            return TestQuality.DECENT
        elif any(keyword in goal_lower for keyword in ['production', 'exhaustive', 'full', 'robust', 'enterprise']):
            logger.debug("Detected PRODUCTION test quality level from goal keywords")
            return TestQuality.PRODUCTION
        
        # Use context preference if provided
        if context_quality:
            return context_quality
            
        # Default to FAST for speed optimization
        logger.debug("Using default FAST test quality level")
        return TestQuality.FAST

    @staticmethod
    def get_quality_config(quality: TestQuality, test_type: Literal["unit", "integration"]) -> Dict[str, Any]:
        """Returns quality-specific configuration data."""
        if test_type == "unit":
            quality_configs = {
                TestQuality.FAST: {
                    "description": "basic unit tests with essential coverage",
                    "complexity_level": "simple",
                    "coverage_target": "main_paths",
                    "mock_strategy": "simple",
                    "test_categories": ["success_cases", "obvious_errors"]
                },
                TestQuality.DECENT: {
                    "description": "comprehensive unit tests with good coverage", 
                    "complexity_level": "thorough",
                    "coverage_target": "good_coverage",
                    "mock_strategy": "effective",
                    "test_categories": ["success_cases", "edge_cases", "error_conditions"]
                },
                TestQuality.PRODUCTION: {
                    "description": "exhaustive unit tests with complete coverage",
                    "complexity_level": "comprehensive",
                    "coverage_target": "complete_paths",
                    "mock_strategy": "sophisticated", 
                    "test_categories": ["success_cases", "edge_cases", "error_conditions", "boundary_values", "performance"]
                }
            }
        else:  # integration
            quality_configs = {
                TestQuality.FAST: {
                    "description": "basic integration tests for key workflows",
                    "complexity_level": "simple",
                    "coverage_target": "main_workflows",
                    "test_scope": "happy_path",
                    "test_categories": ["key_workflows", "basic_endpoints"]
                },
                TestQuality.DECENT: {
                    "description": "thorough integration tests with good workflow coverage",
                    "complexity_level": "thorough", 
                    "coverage_target": "component_interactions",
                    "test_scope": "end_to_end",
                    "test_categories": ["user_flows", "api_testing", "error_scenarios"]
                },
                TestQuality.PRODUCTION: {
                    "description": "exhaustive integration tests with complete workflow coverage",
                    "complexity_level": "comprehensive",
                    "coverage_target": "system_interactions", 
                    "test_scope": "complete_workflows",
                    "test_categories": ["end_to_end_workflows", "edge_cases", "failure_modes", "concurrent_access", "performance"]
                }
            }
        return quality_configs[quality]

    @staticmethod
    def discover_code_files(file_list: List[str] = None, workspace_path: str = None) -> Dict[str, str]:
        """
        Discover code files to test from a file list or workspace.
        Returns a dictionary mapping file paths to their content.
        """
        code_to_test = {}
        
        if file_list:
            # Use provided file list (e.g., from manifest)
            for file_path in file_list:
                if not file_path.startswith("tests/") and not file_path.startswith("test_"):
                    try:
                        if workspace_path:
                            full_path = Path(workspace_path) / file_path
                        else:
                            full_path = Path(file_path)
                        
                        if full_path.exists() and full_path.suffix == '.py':
                            content = full_path.read_text(encoding='utf-8')
                            if content.strip():  # Only include non-empty files
                                code_to_test[file_path] = content
                                logger.debug(f"Added file to test: {file_path}")
                    except Exception as e:
                        logger.warning(f"Failed to read file {file_path}: {e}")
        
        elif workspace_path:
            # Discover Python files in workspace
            workspace = Path(workspace_path)
            if workspace.exists():
                try:
                    for py_file in workspace.rglob("*.py"):
                        rel_path = py_file.relative_to(workspace)
                        rel_path_str = str(rel_path)
                        
                        # Skip test files and common directories
                        if (not rel_path_str.startswith("tests/") and 
                            not rel_path_str.startswith("test_") and
                            "/__pycache__/" not in rel_path_str and
                            "/.venv/" not in rel_path_str and
                            "/venv/" not in rel_path_str):
                            
                            try:
                                content = py_file.read_text(encoding='utf-8')
                                if content.strip():  # Only include non-empty files
                                    code_to_test[rel_path_str] = content
                                    logger.debug(f"Discovered file to test: {rel_path_str}")
                            except Exception as e:
                                logger.warning(f"Failed to read discovered file {rel_path_str}: {e}")
                except Exception as e:
                    logger.warning(f"Failed to discover files in workspace: {e}")
        
        return code_to_test

    @staticmethod
    def generate_test_file_paths(code_files: Dict[str, str], test_type: Literal["unit", "integration"], 
                                base_directory: str = "tests/") -> List[str]:
        """Generate appropriate test file paths based on source code files and test type."""
        test_paths = []
        
        for file_path in code_files.keys():
            # Convert source file path to test file path
            path_obj = Path(file_path)
            
            # Remove file extension and get the stem
            base_name = path_obj.stem
            
            # Create test file name based on type
            if test_type == "integration":
                test_filename = f"test_{base_name}_integration.py"
            else:
                test_filename = f"test_{base_name}.py"
            
            # Place in base directory (default: tests/)
            test_path = f"{base_directory.rstrip('/')}/{test_filename}"
            test_paths.append(test_path)
        
        return test_paths

    @staticmethod
    def parse_test_response(response_str: str) -> Dict[str, str]:
        """Parse and validate test generation response."""
        try:
            # Try to parse as JSON
            response_data = json.loads(response_str)
            
            # Validate it's a dictionary
            if not isinstance(response_data, dict):
                raise ValueError("Response is not a dictionary")
            
            # Validate all values are strings (test code content)
            for key, value in response_data.items():
                if not isinstance(value, str):
                    raise ValueError(f"Test content for '{key}' is not a string")
                if not value.strip():
                    logger.warning(f"Empty test content for '{key}'")
            
            return response_data
            
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON response: {e}")
        except Exception as e:
            raise ValueError(f"Failed to parse response: {e}")

    @staticmethod
    def validate_test_files(test_files: Dict[str, str]) -> Dict[str, Any]:
        """Validate generated test files for common issues."""
        validation_results = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "statistics": {
                "total_files": len(test_files),
                "total_lines": 0,
                "files_with_imports": 0,
                "files_with_test_functions": 0,
                "average_lines_per_file": 0
            }
        }
        
        for file_path, content in test_files.items():
            # Check file path format
            if not file_path.startswith("tests/"):
                validation_results["warnings"].append(f"Test file '{file_path}' not in 'tests/' directory")
            
            if not file_path.endswith(".py"):
                validation_results["errors"].append(f"Test file '{file_path}' is not a Python file")
                validation_results["valid"] = False
                continue
            
            # Basic content analysis
            lines = content.split('\n')
            validation_results["statistics"]["total_lines"] += len(lines)
            
            # Check for imports
            if any(line.strip().startswith(('import ', 'from ')) for line in lines):
                validation_results["statistics"]["files_with_imports"] += 1
            else:
                validation_results["warnings"].append(f"No imports found in '{file_path}'")
            
            # Check for test functions
            if any(line.strip().startswith('def test_') for line in lines):
                validation_results["statistics"]["files_with_test_functions"] += 1
            else:
                validation_results["errors"].append(f"No test functions found in '{file_path}'")
                validation_results["valid"] = False
            
            # Check for pytest patterns
            if 'assert' not in content and 'pytest' not in content:
                validation_results["warnings"].append(f"No assertions or pytest usage detected in '{file_path}'")
        
        # Calculate average lines per file
        if len(test_files) > 0:
            validation_results["statistics"]["average_lines_per_file"] = (
                validation_results["statistics"]["total_lines"] / len(test_files)
            )
        
        return validation_results

    @staticmethod
    def analyze_source_complexity(source_code: Dict[str, str]) -> Dict[str, Any]:
        """Analyze source code complexity to inform test generation strategy."""
        analysis = {
            "total_files": len(source_code),
            "total_lines": 0,
            "functions_count": 0,
            "classes_count": 0,
            "complexity_indicators": {
                "has_async": False,
                "has_decorators": False,
                "has_exceptions": False,
                "has_imports": False,
                "has_complex_logic": False
            },
            "recommended_quality": TestQuality.DECENT.value
        }
        
        for file_path, content in source_code.items():
            lines = content.split('\n')
            analysis["total_lines"] += len(lines)
            
            # Count functions and classes
            for line in lines:
                stripped = line.strip()
                if stripped.startswith('def '):
                    analysis["functions_count"] += 1
                elif stripped.startswith('class '):
                    analysis["classes_count"] += 1
                elif stripped.startswith('async def'):
                    analysis["complexity_indicators"]["has_async"] = True
                elif stripped.startswith('@'):
                    analysis["complexity_indicators"]["has_decorators"] = True
                elif any(keyword in stripped for keyword in ['try:', 'except:', 'raise']):
                    analysis["complexity_indicators"]["has_exceptions"] = True
                elif stripped.startswith(('import ', 'from ')):
                    analysis["complexity_indicators"]["has_imports"] = True
                elif any(keyword in stripped for keyword in ['if ', 'for ', 'while ', 'with ']):
                    analysis["complexity_indicators"]["has_complex_logic"] = True
        
        # Recommend quality level based on complexity
        complexity_score = sum([
            analysis["functions_count"] > 10,
            analysis["classes_count"] > 3,
            analysis["complexity_indicators"]["has_async"],
            analysis["complexity_indicators"]["has_decorators"],
            analysis["complexity_indicators"]["has_exceptions"],
            analysis["complexity_indicators"]["has_complex_logic"]
        ])
        
        if complexity_score >= 4:
            analysis["recommended_quality"] = TestQuality.PRODUCTION.value
        elif complexity_score >= 2:
            analysis["recommended_quality"] = TestQuality.DECENT.value
        else:
            analysis["recommended_quality"] = TestQuality.FAST.value
        
        return analysis

    @staticmethod
    def create_test_metadata(test_config: Dict[str, Any], source_analysis: Dict[str, Any], 
                           validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create comprehensive metadata about the test generation process."""
        return {
            "generation_timestamp": None,  # To be filled by caller
            "test_configuration": {
                "test_type": test_config.get("test_type"),
                "quality_level": test_config.get("quality", {}).get("value") if isinstance(test_config.get("quality"), TestQuality) else test_config.get("quality"),
                "output_directory": test_config.get("output_directory", "tests/")
            },
            "source_analysis": source_analysis,
            "validation_results": validation_results,
            "generation_summary": {
                "files_generated": validation_results["statistics"]["total_files"],
                "total_test_lines": validation_results["statistics"]["total_lines"],
                "validation_passed": validation_results["valid"],
                "warnings_count": len(validation_results["warnings"]),
                "errors_count": len(validation_results["errors"])
            }
        }