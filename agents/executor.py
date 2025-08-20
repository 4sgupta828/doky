# agents/executor.py
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

# Foundational dependencies
from .base import BaseAgent
from core.context import GlobalContext
from core.models import AgentResult

# Execution tools
from tools.test_tools import (
    TestExecutionContext, run_test_suite, TestFramework, TestExecutionResult
)
from tools.code_validator import (
    ValidationContext, comprehensive_validation, ValidationType, ValidationSummary
)
from tools.shell import (
    ShellExecutionContext, execute_commands, ShellCommand, ExecutionMode, ExecutionResult
)
from tools.file_system_tools import (
    read_file, write_file, create_path, delete_path, copy_path, move_path, 
    discover_files, FilesystemResult, FileOperation
)
from tools.execution.environment_tools import (
    EnvironmentContext, execute_environment_operation, EnvironmentOperation, EnvironmentResult
)
from tools.execution.dependency_tools import (
    DependencyContext, execute_dependency_operation, DependencyOperation, PackageManager, DependencyResult
)

# Get a logger instance for this module
logger = logging.getLogger(__name__)


class ExecutorAgent(BaseAgent):
    """
    FOUNDATIONAL AGENT 5: EXECUTOR
    
    The ExecutorAgent is a foundational agent that consolidates all execution-related
    capabilities into a single, powerful interface. It combines test execution, 
    code validation, shell operations, filesystem operations, environment management,
    and dependency management into orthogonal, composable operations.
    
    This agent represents the execution tier of the foundational agent architecture,
    providing comprehensive execution capabilities through atomic, reusable tools.
    
    Capabilities:
    - Test execution (pytest, unittest, nose, custom frameworks)
    - Code validation (syntax, imports, execution, linting, type checking)
    - Shell command execution with safety validation
    - Filesystem operations with backup and recovery
    - Environment management and virtual environments
    - Package management (pip, conda)
    
    Design Principles:
    - Atomic tools for maximum reusability
    - Comprehensive error handling and logging
    - Type-safe operations with dataclasses
    - Orthogonal capabilities that compose cleanly
    - Defensive programming with validation and backups
    """

    def __init__(self):
        super().__init__(
            name="ExecutorAgent",
            description="Foundational execution agent combining test execution, validation, shell operations, filesystem operations, environment management, and dependency management."
        )

    def required_inputs(self) -> List[str]:
        """Required inputs for ExecutorAgent execution."""
        return ["operation_type"]

    def optional_inputs(self) -> List[str]:
        """Optional inputs for ExecutorAgent execution."""
        return [
            # Test execution inputs
            "test_target", "test_framework", "test_patterns", "exclude_patterns", 
            "coverage_enabled", "output_format",
            
            # Validation inputs
            "code_files", "validation_types", "strict_mode", "test_script",
            
            # Shell execution inputs
            "commands", "execution_mode", "ignore_errors", "shell_mode",
            
            # Filesystem inputs
            "file_operation", "source_path", "target_path", "content", 
            "file_patterns", "create_backup", "recursive",
            
            # Environment inputs
            "environment_operation", "venv_name", "python_version", 
            "requirements_file", "additional_packages", "environment_vars", "force_recreate",
            
            # Dependency inputs
            "dependency_operation", "packages", "package_manager", "upgrade_packages", 
            "force_reinstall", "index_url", "extra_index_urls",
            
            # Common inputs
            "working_directory", "python_executable", "timeout_seconds"
        ]

    def execute_v2(self, goal: str, inputs: Dict[str, Any], global_context: GlobalContext) -> AgentResult:
        """
        Execute comprehensive execution operations using atomic tools.
        """
        logger.info(f"ExecutorAgent executing: '{goal}'")
        
        try:
            self.validate_inputs(inputs)
        except Exception as validation_error:
            return self.create_result(
                success=False,
                message=str(validation_error),
                error_details={"validation_error": str(validation_error)}
            )

        operation_type = inputs["operation_type"]
        working_directory = inputs.get("working_directory", str(global_context.workspace_path))
        
        try:
            self.report_progress(f"Starting {operation_type} operation", goal)
            
            if operation_type == "test_execution":
                result = self._execute_test_operation(inputs, working_directory)
            elif operation_type == "validation":
                result = self._execute_validation_operation(inputs, working_directory)
            elif operation_type == "shell_execution":
                result = self._execute_shell_operation(inputs, working_directory)
            elif operation_type == "filesystem":
                result = self._execute_filesystem_operation(inputs, working_directory)
            elif operation_type == "environment":
                result = self._execute_environment_operation(inputs, working_directory)
            elif operation_type == "dependency":
                result = self._execute_dependency_operation(inputs, working_directory)
            else:
                return self.create_result(
                    success=False,
                    message=f"Unknown operation type: {operation_type}",
                    error_details={
                        "supported_operations": [
                            "test_execution", "validation", "shell_execution", 
                            "filesystem", "environment", "dependency"
                        ]
                    }
                )

            if result.success:
                self.report_progress(f"{operation_type} operation completed", result.message)
            else:
                self.report_progress(f"{operation_type} operation failed", result.message)

            return self.create_result(
                success=result.success,
                message=result.message,
                outputs=self._convert_result_to_outputs(result),
                error_details=result.error_details if hasattr(result, 'error_details') else None
            )

        except Exception as e:
            error_msg = f"ExecutorAgent execution failed: {e}"
            logger.error(error_msg, exc_info=True)
            return self.create_result(
                success=False,
                message=error_msg,
                error_details={"exception": str(e)}
            )

    def _execute_test_operation(self, inputs: Dict[str, Any], working_directory: str) -> TestExecutionResult:
        """Execute test execution operations."""
        test_framework = inputs.get("test_framework", "auto")
        if test_framework == "auto":
            test_framework = TestFramework.AUTO
        else:
            test_framework = TestFramework(test_framework)
        
        context = TestExecutionContext(
            test_target=inputs.get("test_target"),
            test_framework=test_framework,
            python_executable=inputs.get("python_executable"),
            working_directory=working_directory,
            timeout_seconds=inputs.get("timeout_seconds", 300),
            test_patterns=inputs.get("test_patterns", []),
            exclude_patterns=inputs.get("exclude_patterns", []),
            output_format=inputs.get("output_format", "json"),
            coverage_enabled=inputs.get("coverage_enabled", False),
            additional_args=inputs.get("additional_args", [])
        )
        
        return run_test_suite(context)

    def _execute_validation_operation(self, inputs: Dict[str, Any], working_directory: str) -> ValidationSummary:
        """Execute code validation operations."""
        validation_types_input = inputs.get("validation_types", ["syntax_check", "import_check"])
        validation_types = [ValidationType(vtype) for vtype in validation_types_input]
        
        context = ValidationContext(
            code_files=inputs.get("code_files", {}),
            working_directory=working_directory,
            python_executable=inputs.get("python_executable"),
            timeout_seconds=inputs.get("timeout_seconds", 30),
            test_script=inputs.get("test_script"),
            validation_types=validation_types,
            strict_mode=inputs.get("strict_mode", False)
        )
        
        return comprehensive_validation(context)

    def _execute_shell_operation(self, inputs: Dict[str, Any], working_directory: str) -> ExecutionResult:
        """Execute shell command operations."""
        commands_input = inputs.get("commands", [])
        commands = []
        
        for cmd in commands_input:
            if isinstance(cmd, str):
                commands.append(ShellCommand(command=cmd))
            elif isinstance(cmd, dict):
                commands.append(ShellCommand(
                    command=cmd["command"],
                    timeout=cmd.get("timeout", 60),
                    expected_return_codes=cmd.get("expected_return_codes", [0])
                ))
        
        execution_mode = inputs.get("execution_mode", "sequential")
        context = ShellExecutionContext(
            commands=commands,
            working_directory=working_directory,
            execution_mode=ExecutionMode(execution_mode),
            global_timeout=inputs.get("timeout_seconds", 600),
            environment_vars=inputs.get("environment_vars"),
            ignore_errors=inputs.get("ignore_errors", False),
            capture_output=True,
            shell=inputs.get("shell_mode", False)
        )
        
        return execute_commands(context)

    def _execute_filesystem_operation(self, inputs: Dict[str, Any], working_directory: str) -> FilesystemResult:
        """Execute filesystem operations."""
        file_operation = inputs.get("file_operation", "read")
        
        if file_operation == "read":
            return read_file(
                file_path=inputs.get("source_path", ""),
                working_directory=working_directory
            )
        elif file_operation == "write":
            return write_file(
                file_path=inputs.get("target_path", ""),
                content=inputs.get("content", ""),
                backup_enabled=inputs.get("create_backup", False),
                working_directory=working_directory
            )
        elif file_operation == "create":
            return create_path(
                path=inputs.get("target_path", ""),
                content=inputs.get("content"),
                working_directory=working_directory
            )
        elif file_operation == "delete":
            return delete_path(
                path=inputs.get("source_path", ""),
                backup_enabled=inputs.get("create_backup", False),
                working_directory=working_directory
            )
        elif file_operation == "copy":
            return copy_path(
                source_path=inputs.get("source_path", ""),
                destination_path=inputs.get("target_path", ""),
                working_directory=working_directory
            )
        elif file_operation == "move":
            return move_path(
                source_path=inputs.get("source_path", ""),
                destination_path=inputs.get("target_path", ""),
                working_directory=working_directory
            )
        elif file_operation == "discover":
            return discover_files(
                search_path=inputs.get("source_path", "."),
                patterns=inputs.get("file_patterns", ["*"]),
                recursive=inputs.get("recursive", True),
                working_directory=working_directory
            )
        else:
            # Create a failed result for unknown operations
            return FilesystemResult(
                success=False,
                message=f"Unknown filesystem operation: {file_operation}",
                operation=FileOperation.READ,  # Default value
                error_details=f"Supported operations: read, write, create, delete, copy, move, discover"
            )

    def _execute_environment_operation(self, inputs: Dict[str, Any], working_directory: str) -> EnvironmentResult:
        """Execute environment management operations."""
        environment_operation = inputs.get("environment_operation", "analyze_env")
        
        context = EnvironmentContext(
            operation=EnvironmentOperation(environment_operation),
            workspace_path=Path(working_directory),
            venv_name=inputs.get("venv_name", "venv"),
            python_version=inputs.get("python_version"),
            requirements_file=inputs.get("requirements_file"),
            additional_packages=inputs.get("additional_packages", []),
            environment_vars=inputs.get("environment_vars", {}),
            force_recreate=inputs.get("force_recreate", False),
            timeout_seconds=inputs.get("timeout_seconds", 300),
            validate_packages=inputs.get("validate_packages", [])
        )
        
        return execute_environment_operation(context)

    def _execute_dependency_operation(self, inputs: Dict[str, Any], working_directory: str) -> DependencyResult:
        """Execute dependency management operations."""
        dependency_operation = inputs.get("dependency_operation", "list")
        package_manager = inputs.get("package_manager", "pip")
        
        context = DependencyContext(
            operation=DependencyOperation(dependency_operation),
            packages=inputs.get("packages", []),
            requirements_file=inputs.get("requirements_file"),
            package_manager=PackageManager(package_manager),
            working_directory=working_directory,
            python_executable=inputs.get("python_executable"),
            upgrade_packages=inputs.get("upgrade_packages", False),
            force_reinstall=inputs.get("force_reinstall", False),
            index_url=inputs.get("index_url"),
            extra_index_urls=inputs.get("extra_index_urls", []),
            timeout_seconds=inputs.get("timeout_seconds", 300)
        )
        
        return execute_dependency_operation(context)

    def _convert_result_to_outputs(self, result) -> Dict[str, Any]:
        """Convert execution result to agent outputs format."""
        outputs = {
            "success": result.success,
            "message": result.message,
        }
        
        # Add type-specific outputs
        if isinstance(result, TestExecutionResult):
            outputs.update({
                "framework_used": result.framework.value if result.framework else None,
                "test_files_discovered": result.test_files_discovered,
                "test_files_executed": result.test_files_executed,
                "tests_run": result.tests_run,
                "tests_passed": result.tests_passed,
                "tests_failed": result.tests_failed,
                "tests_skipped": result.tests_skipped,
                "test_output": result.output,
                "execution_time": result.execution_time,
                "coverage_percentage": result.coverage_percentage
            })
        
        elif isinstance(result, ValidationSummary):
            outputs.update({
                "overall_success": result.overall_success,
                "total_files": result.total_files,
                "passed_validations": result.passed_validations,
                "failed_validations": result.failed_validations,
                "warnings": result.warnings,
                "execution_time": result.execution_time,
                "summary_message": result.summary_message,
                "validation_reports": len(result.reports) if result.reports else 0
            })
        
        elif isinstance(result, ExecutionResult):
            outputs.update({
                "commands_executed": len(result.command_results),
                "successful_commands": sum(1 for cmd in result.command_results if cmd.success),
                "failed_commands": sum(1 for cmd in result.command_results if not cmd.success),
                "total_execution_time": result.total_execution_time,
                "command_results": [
                    {
                        "command": cmd.command,
                        "success": cmd.success,
                        "exit_code": cmd.exit_code,
                        "output": cmd.stdout,
                        "execution_time": cmd.execution_time
                    } for cmd in result.command_results
                ]
            })
        
        elif isinstance(result, FilesystemResult):
            outputs.update({
                "operation": result.operation.value,
                "paths_affected": result.paths_affected or [],
                "content": result.content,
                "file_info": result.file_info.__dict__ if result.file_info else None
            })
        
        elif isinstance(result, EnvironmentResult):
            outputs.update({
                "operation": result.operation.value,
                "venv_path": str(result.venv_path) if result.venv_path else None,
                "python_executable": str(result.python_executable) if result.python_executable else None,
                "pip_executable": str(result.pip_executable) if result.pip_executable else None,
                "installed_packages": result.installed_packages,
                "environment_vars": result.environment_vars,
                "analysis_results": result.analysis_results,
                "validation_details": result.validation_details
            })
        
        elif isinstance(result, DependencyResult):
            outputs.update({
                "operation": result.operation.value,
                "package_manager": result.package_manager.value,
                "packages_affected": result.packages_affected,
                "commands_executed": result.commands_executed,
                "output": result.output,
                "conflicts_found": result.conflicts_found,
                "requirements_content": result.requirements_content
            })
        
        return outputs