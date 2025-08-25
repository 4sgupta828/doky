# fagents/executor.py
import logging
import os
import subprocess
import sys
import time
import tempfile
from pathlib import Path
from typing import Dict, Any, List, Optional

from .base import FoundationalAgent
from core.context import GlobalContext
from core.models import AgentResult

# Import atomic tools for execution capabilities
from tools.test_tools import detect_test_framework, discover_test_files, execute_tests, parse_test_results
from tools.file_system_tools import read_file, write_file, discover_files, create_path, delete_path, copy_path, move_path
from tools.environment_tools import EnvironmentTools
from tools.shell import execute_shell_command

# LLM-based routing
from .routing import LLMRouter, create_routing_context, RoutingDecision

logger = logging.getLogger(__name__)


class FileSystemToolsWrapper:
    """Wrapper to maintain the original FileSystemTools interface."""
    
    @staticmethod
    def read_file(target_path, encoding, global_context):
        result = read_file(target_path, encoding, str(global_context.workspace_path))
        return {
            "success": result.success,
            "message": result.message,
            "outputs": {"content": result.content} if result.content else {}
        }
    
    @staticmethod
    def write_file(target_path, content, encoding, backup_enabled, global_context):
        # backup_enabled is available for future use if needed
        result = write_file(target_path, content, encoding, str(global_context.workspace_path))
        return {
            "success": result.success,
            "message": result.message,
            "outputs": {"file_path": target_path} if result.success else {}
        }
    
    @staticmethod
    def discover_files(target_path, patterns, exclude_patterns, recursive, file_types, max_depth, global_context):
        result = discover_files(target_path, patterns, exclude_patterns, recursive, file_types, max_depth, str(global_context.workspace_path))
        return {
            "success": result.success,
            "message": result.message,
            "outputs": {"discovered_files": result.paths_affected} if result.success else {}
        }
    
    @staticmethod
    def create_path(target_path, content, encoding, global_context):
        result = create_path(target_path, content, encoding, str(global_context.workspace_path))
        return {
            "success": result.success,
            "message": result.message,
            "outputs": {"created_path": target_path} if result.success else {}
        }
    
    @staticmethod
    def delete_path(target_path, backup_enabled, global_context):
        # backup_enabled is available for future use if needed
        result = delete_path(target_path, str(global_context.workspace_path))
        return {
            "success": result.success,
            "message": result.message,
            "outputs": {"deleted_path": target_path} if result.success else {}
        }
    
    @staticmethod
    def copy_path(target_path, destination_path, global_context):
        result = copy_path(target_path, destination_path, str(global_context.workspace_path))
        return {
            "success": result.success,
            "message": result.message,
            "outputs": {"copied_from": target_path, "copied_to": destination_path} if result.success else {}
        }
    
    @staticmethod
    def move_path(target_path, destination_path, global_context):
        result = move_path(target_path, destination_path, str(global_context.workspace_path))
        return {
            "success": result.success,
            "message": result.message,
            "outputs": {"moved_from": target_path, "moved_to": destination_path} if result.success else {}
        }


class ExecutorAgent(FoundationalAgent):
    """
    Foundational Executor Agent - The system's execution specialist.
    
    This agent combines all execution-related capabilities from:
    - TestRunnerAgent (test execution)
    - ExecutionValidatorAgent (validation)
    - ToolingAgent (shell operations)
    - FileSystemAgent (file operations)
    - EnvironmentModifierAgent (environment setup)
    - DependencyModifierAgent (package management)
    
    The Executor agent is responsible for:
    - Running tests and validating results
    - Executing shell commands and build operations
    - Managing files and directories
    - Setting up environments and installing dependencies
    - Validating code execution and functionality
    
    All operations use atomic, structured tools and maintain clean separation
    from the original agents directory.
    """

    def __init__(self, llm_client: Any = None):
        super().__init__(
            name="ExecutorAgent",
            description="Foundational agent for execution operations: tests, builds, file management, environment setup, and validation"
        )
        self._llm_client = llm_client
        self.router = LLMRouter(llm_client)

    def get_capabilities(self) -> Dict[str, str]:
        """Return the capabilities of this foundational agent."""
        return {
            "test_execution": "Run test suites with framework detection and result analysis",
            "code_validation": "Validate code syntax, imports, and execution",
            "shell_operations": "Execute shell commands for build, deployment, and tooling",
            "file_management": "Read, write, discover, and manage files and directories",
            "environment_setup": "Create virtual environments and install dependencies",
            "dependency_management": "Install, update, and manage package dependencies",
            "execution_validation": "Validate that code executes correctly and passes tests"
        }

    def execute(self, goal: str, inputs: Dict[str, Any], global_context: GlobalContext) -> AgentResult:
        """
        Execute operations based on the goal and inputs.
        
        The goal determines which execution capability to use:
        - "run_tests": Execute test suites
        - "validate_code": Validate code execution
        - "execute_command": Run shell commands
        - "manage_files": File operations
        - "setup_environment": Environment and dependency management
        - Multi-step orchestration for complex workflows
        """
        logger.info(f"ExecutorAgent executing: '{goal}'")
        
        try:
            # ENHANCED: Check if InterAgentRouter flagged this for orchestration
            if inputs.get("orchestration_required"):
                logger.info("InterAgentRouter flagged orchestration_required=True - delegating to workflow orchestration tools")
                return self._delegate_to_workflow_orchestration(goal, inputs, global_context)
            
            # Try LLM-based routing first if available
            if self._llm_client:
                try:
                    workspace_files = global_context.workspace.list_files() if global_context.workspace else []
                    routing_context = create_routing_context(
                        agent_type="ExecutorAgent",
                        goal=goal,
                        inputs=inputs,
                        workspace_files=workspace_files,
                        available_capabilities=list(self.get_capabilities().keys())
                    )
                    
                    routing_result = self.router.route_request(routing_context)
                    
                    # Map routing decision to handler
                    operation_mapping = {
                        RoutingDecision.TEST_EXECUTION: self._handle_test_execution,
                        RoutingDecision.CODE_VALIDATION: self._handle_code_validation,
                        RoutingDecision.SHELL_EXECUTION: self._handle_shell_execution,
                        RoutingDecision.FILE_OPERATIONS: self._handle_file_operations,
                        RoutingDecision.ENVIRONMENT_SETUP: self._handle_environment_setup,
                    }
                    
                    handler = operation_mapping.get(routing_result.decision)
                    if handler:
                        logger.info(f"LLM routing: {routing_result.decision.value} (confidence: {routing_result.confidence:.2f})")
                        logger.info(f"LLM routing reasoning: {routing_result.reasoning}")
                        return handler(goal, inputs, global_context)
                except RuntimeError as e:
                    logger.warning(f"LLM routing failed: {e}. Falling back to intelligent routing.")
            
            # Fallback to intelligent routing
            return self._handle_intelligent_routing(goal, inputs, global_context)

        except Exception as e:
            error_msg = f"ExecutorAgent execution failed: {e}"
            logger.error(error_msg, exc_info=True)
            return AgentResult(
                success=False,
                message=error_msg,
                error_details={"exception": str(e), "goal": goal}
            )

    def _handle_test_execution(self, goal: str, inputs: Dict[str, Any], global_context: GlobalContext) -> AgentResult:
        """Handle test execution operations."""
        logger.info("Handling test execution")
        
        # Extract test-specific inputs
        test_target = inputs.get("test_target") or inputs.get("specific_test_files")
        test_framework = inputs.get("test_framework", "auto")
        python_executable = inputs.get("python_executable")
        working_directory = inputs.get("working_directory", str(global_context.workspace_path))
        timeout_seconds = inputs.get("timeout_seconds", 300)
        test_patterns = inputs.get("test_patterns", [])
        exclude_patterns = inputs.get("exclude_patterns", [])
        output_format = inputs.get("output_format", "json")
        coverage_enabled = inputs.get("coverage_enabled", False)
        additional_args = inputs.get("additional_args", [])

        try:
            # Step 0: Standardize project if needed (new step)
            standardization_result = self._ensure_project_standardization(working_directory)
            if not standardization_result.success:
                logger.warning(f"Project standardization failed: {standardization_result.message}")
                # Continue anyway - standardization is optional
            
            # Step 1: Detect test framework
            framework_result = detect_test_framework(
                requested_framework=test_framework, 
                test_target=test_target, 
                working_directory=working_directory
            )
            
            if not framework_result["success"]:
                return AgentResult(
                    success=False,
                    message=f"Test framework detection failed: {framework_result['message']}",
                    error_details=framework_result
                )

            detected_framework = framework_result["framework"]

            # Step 2: Check if test framework is available and auto-install if needed
            availability_check = self._check_framework_availability(detected_framework, python_executable)
            
            if not availability_check["available"]:
                logger.info(f"Test framework {detected_framework} not available: {availability_check['message']}")
                logger.info(f"Attempting to auto-install {detected_framework}...")
                
                install_result = self._auto_install_test_framework(
                    framework=detected_framework,
                    python_executable=python_executable,
                    working_directory=working_directory,
                    global_context=global_context
                )
                
                if not install_result["success"]:
                    return AgentResult(
                        success=False,
                        message=f"Cannot run tests: {detected_framework} framework is not available and auto-installation failed. {install_result['message']}",
                        error_details={
                            "framework_detection": framework_result,
                            "availability_check": availability_check,
                            "installation_attempt": install_result
                        }
                    )
                else:
                    logger.info(f"Successfully auto-installed {detected_framework}: {install_result['message']}")
            else:
                logger.info(f"Test framework {detected_framework} is available: {availability_check['message']}")

            # Step 3: Discover test files
            discovery_result = discover_test_files(
                test_target=test_target or ".",
                framework=detected_framework,
                working_directory=working_directory,
                test_patterns=test_patterns if test_patterns else None,
                exclude_patterns=exclude_patterns if exclude_patterns else None
            )
            
            if not discovery_result["success"]:
                return AgentResult(
                    success=False,
                    message=f"Test discovery failed: {discovery_result['message']}",
                    error_details=discovery_result
                )

            test_files = discovery_result["test_files"]
            if not test_files:
                return AgentResult(
                    success=True, 
                    message="No test files found to run.",
                    outputs={"test_files_discovered": 0}
                )

            # Step 4: Execute tests
            execution_result = execute_tests(
                test_files=test_files,
                framework=detected_framework,
                python_executable=python_executable,
                working_directory=working_directory,
                timeout_seconds=timeout_seconds,
                coverage_enabled=coverage_enabled,
                additional_args=additional_args,
                output_format=output_format
            )

            # Step 5: Parse and analyze results
            analysis_result = parse_test_results(execution_result, detected_framework)

            # Combine results
            final_result = {
                "test_framework": detected_framework,
                "framework_availability_check": availability_check,
                "framework_auto_installed": not availability_check["available"],
                "test_files_discovered": len(test_files),
                "test_files_executed": len(execution_result.get("executed_files", [])),
                "execution_details": execution_result,
                "analysis": analysis_result,
                "success": execution_result.get("success", False) and analysis_result.get("success", False)
            }

            message = self._create_test_summary_message(final_result, analysis_result)

            return AgentResult(
                success=final_result["success"],
                message=message,
                outputs=final_result
            )

        except Exception as e:
            return AgentResult(
                success=False,
                message=f"Test execution failed: {e}",
                error_details={"exception": str(e)}
            )

    def _handle_code_validation(self, goal: str, inputs: Dict[str, Any], global_context: GlobalContext) -> AgentResult:
        """Handle code validation operations."""
        logger.info("Handling code validation")
        
        code_files = inputs.get('code_files', {})
        test_script = inputs.get('test_script')
        output_dir = inputs.get('output_directory', str(global_context.workspace_path))
        timeout_seconds = inputs.get('timeout_seconds', 30)
        
        if not code_files:
            return AgentResult(
                success=False,
                message="No code files provided for validation",
                error_details={"required_input": "code_files"}
            )

        # Initialize validation results
        validation_results = {
            'syntax_check': False,
            'import_check': False,
            'test_execution': False,
            'overall_success': False,
            'details': [],
            'errors': [],
            'warnings': []
        }

        try:
            # Step 1: Syntax validation
            syntax_valid, syntax_details = self._validate_syntax(code_files)
            validation_results['syntax_check'] = syntax_valid
            validation_results['details'].extend(syntax_details)

            # Step 2: Import validation (only if syntax is valid)
            if syntax_valid:
                import_valid, import_details = self._validate_imports(code_files, output_dir)
                validation_results['import_check'] = import_valid
                validation_results['details'].extend(import_details)
            else:
                validation_results['details'].append("Import validation skipped due to syntax errors")

            # Step 3: Test execution (if test script available and imports are valid)
            if test_script and os.path.exists(test_script) and validation_results.get('import_check', False):
                test_result, test_details = self._execute_validation_tests(test_script, output_dir, timeout_seconds)
                validation_results['test_execution'] = test_result
                validation_results['details'].extend(test_details)
            elif test_script:
                validation_results['details'].append(f"Test execution skipped: script={bool(test_script)}, imports_valid={validation_results.get('import_check', False)}")
            else:
                validation_results['details'].append("Test execution skipped: no test script provided")

            # Determine overall success
            overall_success = syntax_valid and validation_results.get('import_check', True)
            if test_script and os.path.exists(test_script):
                overall_success = overall_success and validation_results.get('test_execution', False)

            validation_results['overall_success'] = overall_success

            # Generate summary message
            passed_checks = sum([
                validation_results['syntax_check'],
                validation_results.get('import_check', True),
                validation_results.get('test_execution', True) if test_script else True
            ])
            total_checks = 2 + (1 if test_script and os.path.exists(test_script) else 0)

            if overall_success:
                message = f"Validation PASSED - {passed_checks}/{total_checks} checks successful"
            else:
                message = f"Validation FAILED - {passed_checks}/{total_checks} checks passed"

            return AgentResult(
                success=True,  # The validation process completed successfully
                message=message,
                outputs=validation_results
            )

        except Exception as e:
            validation_results['errors'].append(str(e))
            return AgentResult(
                success=False,
                message=f"Validation execution failed: {e}",
                error_details=validation_results
            )

    def _handle_shell_execution(self, goal: str, inputs: Dict[str, Any], global_context: GlobalContext) -> AgentResult:
        """Handle shell command execution operations."""
        logger.info("Handling shell command execution")
        
        commands = inputs.get("commands", [])
        purpose = inputs.get("purpose", "Shell command execution")
        working_directory = inputs.get("working_directory", str(global_context.workspace_path))
        # timeout = inputs.get("timeout", 120)  # Available if needed for future use
        environment_vars = inputs.get("environment_vars")
        ignore_errors = inputs.get("ignore_errors", False)

        if not commands:
            return AgentResult(
                success=False,
                message="No commands provided for execution. InterAgentRouter should provide structured commands in inputs when routing to ExecutorAgent for shell execution.",
                error_details={"required_input": "commands", "goal": goal, "hint": "InterAgentRouter should extract commands from LLM goal and provide them in structured format"}
            )

        try:
            command_results = []
            failed_commands = []
            combined_output = ""
            start_time = time.time()

            for i, command in enumerate(commands):
                logger.info(f"Executing command {i+1}/{len(commands)}: {command[:50]}...")
                
                # Execute command using atomic tool
                tool_output = execute_shell_command(
                    command=command, 
                    working_dir=working_directory,
                    env=environment_vars
                )
                
                # Store the original command in the result
                tool_output['command'] = command
                command_results.append(tool_output)
                
                combined_output += f"=== Command: {command} ===\n"
                combined_output += f"Exit code: {tool_output['exit_code']}\n"
                combined_output += f"STDOUT:\n{tool_output['stdout']}\n"
                combined_output += f"STDERR:\n{tool_output['stderr']}\n\n"
                
                if tool_output['exit_code'] != 0:
                    failed_commands.append(command)
                    if not ignore_errors:
                        logger.error(f"Command failed, stopping execution: {command}")
                        break

            total_duration = time.time() - start_time
            success = len(failed_commands) == 0

            message = f"Executed {len(commands)} commands. Success: {success}"
            if failed_commands:
                message += f". Failed commands: {len(failed_commands)}"

            return AgentResult(
                success=success,
                message=message,
                outputs={
                    "commands_executed": [res['command'] for res in command_results],
                    "command_results": command_results,
                    "combined_output": combined_output,
                    "duration_seconds": total_duration,
                    "failed_commands": failed_commands,
                    "purpose": purpose
                }
            )

        except Exception as e:
            return AgentResult(
                success=False,
                message=f"Shell execution failed: {e}",
                error_details={"exception": str(e)}
            )

    def _handle_file_operations(self, goal: str, inputs: Dict[str, Any], global_context: GlobalContext) -> AgentResult:
        """Handle file system operations."""
        logger.info("Handling file operations")
        
        operation = inputs.get("operation", "read")
        target_path = inputs.get("target_path", ".")
        content = inputs.get("content")
        patterns = inputs.get("patterns", [])
        exclude_patterns = inputs.get("exclude_patterns", [])
        recursive = inputs.get("recursive", True)
        file_types = inputs.get("file_types", [])
        max_depth = inputs.get("max_depth", 10)
        backup_enabled = inputs.get("backup_enabled", False)
        encoding = inputs.get("encoding", "utf-8")

        try:
            # Route to appropriate file operation using FileSystemToolsWrapper
            if operation == "read":
                result = FileSystemToolsWrapper.read_file(target_path, encoding, global_context)
            elif operation == "write":
                result = FileSystemToolsWrapper.write_file(target_path, content, encoding, backup_enabled, global_context)
            elif operation == "discover":
                result = FileSystemToolsWrapper.discover_files(
                    target_path, patterns, exclude_patterns, recursive, 
                    file_types, max_depth, global_context
                )
            elif operation == "create":
                result = FileSystemToolsWrapper.create_path(target_path, content, encoding, global_context)
            elif operation == "delete":
                result = FileSystemToolsWrapper.delete_path(target_path, backup_enabled, global_context)
            elif operation == "copy":
                result = FileSystemToolsWrapper.copy_path(target_path, inputs.get("destination_path"), global_context)
            elif operation == "move":
                result = FileSystemToolsWrapper.move_path(target_path, inputs.get("destination_path"), global_context)
            else:
                return AgentResult(
                    success=False,
                    message=f"Unsupported file operation: {operation}",
                    error_details={"supported_operations": ["read", "write", "discover", "create", "delete", "copy", "move"]}
                )

            return AgentResult(
                success=result["success"],
                message=result["message"],
                outputs=result.get("outputs", {})
            )

        except Exception as e:
            return AgentResult(
                success=False,
                message=f"File operation failed: {e}",
                error_details={"exception": str(e), "operation": operation}
            )

    def _handle_environment_setup(self, goal: str, inputs: Dict[str, Any], global_context: GlobalContext) -> AgentResult:
        """Handle environment and dependency management operations."""
        logger.info("Handling environment setup and dependency management")
        
        # CRITICAL FIX: If commands are provided in inputs, this should be shell execution, not pure environment setup
        if inputs.get("commands"):
            logger.info("Commands detected in environment setup inputs - routing to shell execution")
            return self._handle_shell_execution(goal, inputs, global_context)
        
        # Check if this is a Git repo initialization request
        if "init" in goal.lower() and "repo" in goal.lower():
            return self._handle_git_repo_setup(inputs, global_context)
        
        # Check if this is a dependency operation vs environment setup
        operation = inputs.get("operation")
        if operation:
            # This is a dependency management operation
            return self._handle_dependency_management(inputs, global_context)
        else:
            # This is an environment setup operation
            return self._handle_environment_creation(inputs, global_context)

    def _handle_dependency_management(self, inputs: Dict[str, Any], global_context: GlobalContext) -> AgentResult:
        """Handle package dependency operations."""
        operation = inputs["operation"]
        packages = inputs.get("packages", [])
        requirements_file = inputs.get("requirements_file")
        package_manager = inputs.get("package_manager", "pip")
        working_directory = inputs.get("working_directory", str(global_context.workspace_path))
        python_executable = inputs.get("python_executable", sys.executable)
        upgrade_packages = inputs.get("upgrade_packages", False)
        force_reinstall = inputs.get("force_reinstall", False)
        index_url = inputs.get("index_url")
        extra_index_urls = inputs.get("extra_index_urls", [])

        try:
            if operation == "install":
                result = self._install_packages(
                    packages, requirements_file, package_manager, 
                    working_directory, python_executable, upgrade_packages, 
                    force_reinstall, index_url, extra_index_urls
                )
            elif operation == "uninstall":
                result = self._uninstall_packages(
                    packages, package_manager, working_directory, python_executable
                )
            elif operation == "list":
                result = self._list_packages(package_manager, working_directory, python_executable)
            elif operation == "freeze":
                result = self._freeze_requirements(
                    requirements_file, package_manager, working_directory, python_executable
                )
            elif operation == "update":
                result = self._update_packages(
                    packages, package_manager, working_directory, python_executable
                )
            elif operation == "check":
                result = self._check_dependencies(
                    package_manager, working_directory, python_executable
                )
            else:
                return AgentResult(
                    success=False,
                    message=f"Unknown dependency operation: {operation}",
                    error_details={"supported_operations": ["install", "uninstall", "list", "freeze", "update", "check"]}
                )

            return AgentResult(
                success=result["success"],
                message=result["message"],
                outputs=result["outputs"]
            )

        except Exception as e:
            return AgentResult(
                success=False,
                message=f"Dependency management failed: {e}",
                error_details={"exception": str(e), "operation": operation}
            )

    def _handle_environment_creation(self, inputs: Dict[str, Any], global_context: GlobalContext) -> AgentResult:
        """Handle virtual environment creation and setup."""
        workspace_path = Path(inputs.get("workspace_path", global_context.workspace_path))
        venv_name = inputs.get("venv_name", "venv")
        requirements_file = inputs.get("requirements_file", "requirements.txt")
        python_version = inputs.get("python_version")
        additional_packages = inputs.get("additional_packages", [])
        environment_vars = inputs.get("environment_vars", {})
        force_recreate = inputs.get("force_recreate", False)

        try:
            # Step 1: Create or validate virtual environment using tools
            venv_result = EnvironmentTools.setup_virtual_environment(
                workspace_path, venv_name, python_version, force_recreate
            )
            
            if not venv_result["success"]:
                return AgentResult(
                    success=False,
                    message=f"Virtual environment setup failed: {venv_result['message']}",
                    error_details=venv_result
                )

            venv_path = venv_result["venv_path"]
            venv_python = venv_result["python_executable"]
            venv_pip = venv_result["pip_executable"]

            # Step 2: Install dependencies using tools
            deps_result = EnvironmentTools.install_dependencies(
                workspace_path, venv_pip, requirements_file, additional_packages
            )
            
            if not deps_result["success"]:
                return AgentResult(
                    success=False,
                    message=f"Dependency installation failed: {deps_result.get('message', 'Unknown error')}",
                    error_details=deps_result
                )

            # Step 3: Validate environment using tools
            validation_result = EnvironmentTools.validate_environment(venv_python, additional_packages)
            
            if not validation_result["success"]:
                return AgentResult(
                    success=False,
                    message=f"Environment validation failed: {validation_result['message']}",
                    error_details=validation_result
                )

            # Step 4: Set up environment variables if provided
            self._setup_environment_variables(environment_vars)

            return AgentResult(
                success=True,
                message=f"Python environment successfully set up at {venv_path}",
                outputs={
                    "venv_path": str(venv_path),
                    "python_executable": str(venv_python),
                    "pip_executable": str(venv_pip),
                    "installed_packages": deps_result.get("installed_packages", []),
                    "environment_ready": True,
                    "environment_vars": environment_vars,
                    "validation_details": validation_result.get("details", [])
                }
            )

        except Exception as e:
            return AgentResult(
                success=False,
                message=f"Environment setup failed: {e}",
                error_details={"exception": str(e)}
            )

    def _handle_intelligent_routing(self, goal: str, inputs: Dict[str, Any], global_context: GlobalContext) -> AgentResult:
        """Intelligently route operations based on goal and inputs."""
        logger.info("Using intelligent routing for execution")
        
        # Analyze goal keywords for automatic command generation
        goal_lower = goal.lower()
        
        # Handle Git repository setup requests
        if ("init" in goal_lower and "repo" in goal_lower) or ("git" in goal_lower and "init" in goal_lower):
            return self._handle_git_repo_setup(inputs, global_context)
        
        # Handle environment/repo setup requests
        if "fix" in goal_lower and "environment" in goal_lower:
            # Generate default environment setup commands
            setup_commands = [
                "git init",
                "git config user.name 'Agent User'",
                "git config user.email 'agent@example.com'",
                "git config --global --add safe.directory '*'"
            ]
            inputs["commands"] = setup_commands
            return self._handle_shell_execution(goal, inputs, global_context)
        
        # Analyze inputs to determine operation type
        if inputs.get("commands") or inputs.get("command"):
            return self._handle_shell_execution(goal, inputs, global_context)
        elif inputs.get("code_files") or "validate" in goal.lower():
            return self._handle_code_validation(goal, inputs, global_context)
        elif inputs.get("test_target") or inputs.get("specific_test_files") or "test" in goal.lower():
            return self._handle_test_execution(goal, inputs, global_context)
        elif inputs.get("operation") and inputs["operation"] in ["install", "uninstall", "list", "freeze", "update", "check"]:
            return self._handle_environment_setup(goal, inputs, global_context)
        elif inputs.get("target_path") or inputs.get("operation") in ["read", "write", "discover", "create", "delete", "copy", "move"]:
            return self._handle_file_operations(goal, inputs, global_context)
        elif inputs.get("workspace_path") or inputs.get("venv_name"):
            return self._handle_environment_setup(goal, inputs, global_context)
        else:
            return AgentResult(
                success=False,
                message=f"Could not determine execution type for goal '{goal}'. Please provide specific inputs.",
                error_details={
                    "goal": goal,
                    "available_inputs": list(inputs.keys()),
                    "suggested_inputs": [
                        "commands (for shell execution)",
                        "code_files (for validation)",
                        "test_target (for testing)",
                        "operation (for file/dependency operations)",
                        "workspace_path (for environment setup)"
                    ]
                }
            )

    # Helper methods extracted from original agents

    def _create_test_summary_message(self, final_result: Dict[str, Any], analysis_result: Dict[str, Any]) -> str:
        """Create a summary message for test execution."""
        framework = final_result.get("test_framework", "unknown")
        files_count = final_result.get("test_files_discovered", 0)
        
        if final_result["success"]:
            if analysis_result.get("passed_tests"):
                passed = analysis_result.get("passed_tests", 0)
                total = analysis_result.get("total_tests", 0)
                return f"Test suite passed: {passed}/{total} tests successful using {framework} ({files_count} files)"
            else:
                return f"Test execution successful using {framework}: {analysis_result.get('summary', 'Tests passed')}"
        else:
            if analysis_result.get("failed_tests"):
                failed = analysis_result.get("failed_tests", 0)
                total = analysis_result.get("total_tests", 0)
                return f"Test suite failed: {failed}/{total} tests failed using {framework} ({files_count} files)"
            else:
                return f"Test execution failed using {framework}: {analysis_result.get('summary', 'Tests failed')}"

    def _validate_syntax(self, code_files: Dict[str, str]) -> tuple[bool, List[str]]:
        """Validate Python syntax for all files."""
        details = []
        syntax_errors = []
        
        for file_path, content in code_files.items():
            try:
                compile(content, file_path, 'exec')
                details.append(f"✓ Syntax valid: {file_path}")
            except SyntaxError as e:
                error_msg = f"✗ Syntax error in {file_path} line {e.lineno}: {e.msg}"
                details.append(error_msg)
                syntax_errors.append(error_msg)
            except Exception as e:
                error_msg = f"✗ Compilation error in {file_path}: {e}"
                details.append(error_msg)
                syntax_errors.append(error_msg)
        
        is_valid = len(syntax_errors) == 0
        
        if is_valid:
            details.append(f"✓ All {len(code_files)} files have valid Python syntax")
        else:
            details.append(f"✗ {len(syntax_errors)} files have syntax errors")
        
        return is_valid, details

    def _validate_imports(self, code_files: Dict[str, str], output_dir: str) -> tuple[bool, List[str]]:
        """Validate that imports can be resolved."""
        details = []
        import_errors = []
        
        original_cwd = os.getcwd()
        original_path = sys.path.copy()
        
        try:
            os.chdir(output_dir)
            if output_dir not in sys.path:
                sys.path.insert(0, output_dir)
            
            for file_path, content in code_files.items():
                try:
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as temp_file:
                        temp_file.write(content)
                        temp_file_path = temp_file.name
                    
                    try:
                        import importlib.util
                        spec = importlib.util.spec_from_file_location("temp_module", temp_file_path)
                        if spec and spec.loader:
                            details.append(f"✓ Imports resolve: {file_path}")
                        else:
                            details.append(f"⚠ Could not create module spec for: {file_path}")
                    finally:
                        try:
                            os.unlink(temp_file_path)
                        except:
                            pass
                    
                except ImportError as e:
                    error_msg = f"✗ Import error in {file_path}: {e}"
                    details.append(error_msg)
                    import_errors.append(error_msg)
                except Exception as e:
                    details.append(f"⚠ Import check warning for {file_path}: {e}")
            
            is_valid = len(import_errors) == 0
            
            if is_valid:
                details.append(f"✓ Import validation passed for {len(code_files)} files")
            else:
                details.append(f"✗ {len(import_errors)} files have import issues")
            
            return is_valid, details
            
        finally:
            os.chdir(original_cwd)
            sys.path = original_path

    def _execute_validation_tests(self, test_script: str, output_dir: str, timeout_seconds: int) -> tuple[bool, List[str]]:
        """Execute test script for validation."""
        details = []
        
        if not os.path.exists(test_script):
            error_msg = f"✗ Test script not found: {test_script}"
            details.append(error_msg)
            return False, details
        
        try:
            result = subprocess.run(
                [sys.executable, test_script],
                cwd=output_dir,
                capture_output=True,
                text=True,
                timeout=timeout_seconds
            )
            
            if result.stdout:
                details.append(f"Test output: {result.stdout.strip()}")
            
            if result.stderr:
                details.append(f"Test errors: {result.stderr.strip()}")
            
            if result.returncode == 0:
                details.append("✓ Test script executed successfully")
                return True, details
            else:
                details.append(f"✗ Test script failed with return code {result.returncode}")
                return False, details
                
        except subprocess.TimeoutExpired:
            error_msg = f"✗ Test execution timed out after {timeout_seconds} seconds"
            details.append(error_msg)
            return False, details
        except Exception as e:
            error_msg = f"✗ Test execution failed: {e}"
            details.append(error_msg)
            return False, details

    def _install_packages(self, packages: List[str], requirements_file: Optional[str], 
                         package_manager: str, working_directory: str, python_executable: str,
                         upgrade_packages: bool, force_reinstall: bool, index_url: Optional[str],
                         extra_index_urls: List[str]) -> Dict[str, Any]:
        """Install packages using the specified package manager."""
        
        if not packages and not requirements_file:
            return {
                "success": False,
                "message": "No packages or requirements file specified",
                "outputs": {}
            }

        commands_executed = []
        results = []

        try:
            if package_manager == "pip":
                cmd = [python_executable, "-m", "pip", "install"]
                
                if upgrade_packages:
                    cmd.append("--upgrade")
                if force_reinstall:
                    cmd.extend(["--force-reinstall", "--no-deps"])
                if index_url:
                    cmd.extend(["--index-url", index_url])
                for extra_url in extra_index_urls:
                    cmd.extend(["--extra-index-url", extra_url])

                # Install from requirements file
                if requirements_file:
                    req_path = Path(working_directory) / requirements_file
                    if req_path.exists():
                        install_cmd = cmd + ["-r", str(req_path)]
                        result = subprocess.run(
                            install_cmd, 
                            capture_output=True, 
                            text=True, 
                            cwd=working_directory,
                            timeout=300
                        )
                        commands_executed.append(" ".join(install_cmd))
                        results.append({
                            "command": " ".join(install_cmd),
                            "returncode": result.returncode,
                            "stdout": result.stdout,
                            "stderr": result.stderr
                        })

                # Install individual packages
                if packages:
                    install_cmd = cmd + packages
                    result = subprocess.run(
                        install_cmd, 
                        capture_output=True, 
                        text=True, 
                        cwd=working_directory,
                        timeout=300
                    )
                    commands_executed.append(" ".join(install_cmd))
                    results.append({
                        "command": " ".join(install_cmd),
                        "returncode": result.returncode,
                        "stdout": result.stdout,
                        "stderr": result.stderr
                    })

            # Check if all installations succeeded
            all_success = all(r["returncode"] == 0 for r in results)
            
            if all_success:
                return {
                    "success": True,
                    "message": f"Successfully installed packages using {package_manager}",
                    "outputs": {
                        "package_manager": package_manager,
                        "commands_executed": commands_executed,
                        "results": results,
                        "packages_installed": packages,
                        "requirements_file": requirements_file
                    }
                }
            else:
                failed_results = [r for r in results if r["returncode"] != 0]
                return {
                    "success": False,
                    "message": f"Package installation failed with {package_manager}",
                    "outputs": {
                        "package_manager": package_manager,
                        "commands_executed": commands_executed,
                        "results": results,
                        "failed_results": failed_results
                    }
                }

        except Exception as e:
            return {
                "success": False,
                "message": f"Package installation error: {e}",
                "outputs": {"error": str(e), "commands_executed": commands_executed}
            }

    def _uninstall_packages(self, packages: List[str], package_manager: str,
                           working_directory: str, python_executable: str) -> Dict[str, Any]:
        """Uninstall packages using the specified package manager."""
        
        if not packages:
            return {
                "success": False,
                "message": "No packages specified for uninstallation",
                "outputs": {}
            }

        try:
            if package_manager == "pip":
                cmd = [python_executable, "-m", "pip", "uninstall", "-y"] + packages
            else:
                return {
                    "success": False,
                    "message": f"Unsupported package manager: {package_manager}",
                    "outputs": {}
                }

            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                cwd=working_directory,
                timeout=120
            )

            if result.returncode == 0:
                return {
                    "success": True,
                    "message": f"Successfully uninstalled packages: {', '.join(packages)}",
                    "outputs": {
                        "packages_uninstalled": packages,
                        "command": " ".join(cmd),
                        "stdout": result.stdout,
                        "stderr": result.stderr
                    }
                }
            else:
                return {
                    "success": False,
                    "message": f"Failed to uninstall packages: {result.stderr}",
                    "outputs": {
                        "command": " ".join(cmd),
                        "returncode": result.returncode,
                        "stdout": result.stdout,
                        "stderr": result.stderr
                    }
                }

        except Exception as e:
            return {
                "success": False,
                "message": f"Uninstallation error: {e}",
                "outputs": {"error": str(e)}
            }

    def _list_packages(self, package_manager: str, working_directory: str, 
                      python_executable: str) -> Dict[str, Any]:
        """List installed packages."""
        
        try:
            if package_manager == "pip":
                cmd = [python_executable, "-m", "pip", "list"]
            else:
                return {
                    "success": False,
                    "message": f"Unsupported package manager: {package_manager}",
                    "outputs": {}
                }

            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                cwd=working_directory,
                timeout=60
            )

            if result.returncode == 0:
                return {
                    "success": True,
                    "message": f"Successfully listed packages with {package_manager}",
                    "outputs": {
                        "package_list": result.stdout,
                        "command": " ".join(cmd)
                    }
                }
            else:
                return {
                    "success": False,
                    "message": f"Failed to list packages: {result.stderr}",
                    "outputs": {
                        "command": " ".join(cmd),
                        "stderr": result.stderr
                    }
                }

        except Exception as e:
            return {
                "success": False,
                "message": f"Package listing error: {e}",
                "outputs": {"error": str(e)}
            }

    def _freeze_requirements(self, requirements_file: Optional[str], package_manager: str,
                           working_directory: str, python_executable: str) -> Dict[str, Any]:
        """Generate requirements file from installed packages."""
        
        try:
            if package_manager == "pip":
                cmd = [python_executable, "-m", "pip", "freeze"]
            else:
                return {
                    "success": False,
                    "message": f"Freeze operation not supported for {package_manager}",
                    "outputs": {}
                }

            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                cwd=working_directory,
                timeout=60
            )

            if result.returncode == 0:
                # Write to file if specified
                if requirements_file:
                    req_path = Path(working_directory) / requirements_file
                    req_path.write_text(result.stdout)
                    
                return {
                    "success": True,
                    "message": f"Successfully generated requirements {'to ' + requirements_file if requirements_file else ''}",
                    "outputs": {
                        "requirements_content": result.stdout,
                        "requirements_file": requirements_file,
                        "command": " ".join(cmd)
                    }
                }
            else:
                return {
                    "success": False,
                    "message": f"Failed to freeze requirements: {result.stderr}",
                    "outputs": {
                        "command": " ".join(cmd),
                        "stderr": result.stderr
                    }
                }

        except Exception as e:
            return {
                "success": False,
                "message": f"Requirements freeze error: {e}",
                "outputs": {"error": str(e)}
            }

    def _update_packages(self, packages: List[str], package_manager: str,
                        working_directory: str, python_executable: str) -> Dict[str, Any]:
        """Update packages to latest versions."""
        
        if not packages:
            return {
                "success": False,
                "message": "No packages specified for update",
                "outputs": {}
            }

        try:
            if package_manager == "pip":
                cmd = [python_executable, "-m", "pip", "install", "--upgrade"] + packages
            else:
                return {
                    "success": False,
                    "message": f"Unsupported package manager: {package_manager}",
                    "outputs": {}
                }

            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                cwd=working_directory,
                timeout=300
            )

            if result.returncode == 0:
                return {
                    "success": True,
                    "message": f"Successfully updated packages: {', '.join(packages)}",
                    "outputs": {
                        "packages_updated": packages,
                        "command": " ".join(cmd),
                        "stdout": result.stdout,
                        "stderr": result.stderr
                    }
                }
            else:
                return {
                    "success": False,
                    "message": f"Failed to update packages: {result.stderr}",
                    "outputs": {
                        "command": " ".join(cmd),
                        "returncode": result.returncode,
                        "stderr": result.stderr
                    }
                }

        except Exception as e:
            return {
                "success": False,
                "message": f"Package update error: {e}",
                "outputs": {"error": str(e)}
            }

    def _check_dependencies(self, package_manager: str, working_directory: str,
                           python_executable: str) -> Dict[str, Any]:
        """Check for dependency conflicts and issues."""
        
        try:
            if package_manager == "pip":
                cmd = [python_executable, "-m", "pip", "check"]
            else:
                return {
                    "success": False,
                    "message": f"Dependency check not supported for {package_manager}",
                    "outputs": {}
                }

            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                cwd=working_directory,
                timeout=60
            )

            if result.returncode == 0:
                return {
                    "success": True,
                    "message": "No dependency conflicts found",
                    "outputs": {
                        "conflicts_found": False,
                        "command": " ".join(cmd),
                        "stdout": result.stdout
                    }
                }
            else:
                return {
                    "success": False,
                    "message": f"Dependency conflicts detected: {result.stdout}",
                    "outputs": {
                        "conflicts_found": True,
                        "conflicts": result.stdout,
                        "command": " ".join(cmd)
                    }
                }

        except Exception as e:
            return {
                "success": False,
                "message": f"Dependency check error: {e}",
                "outputs": {"error": str(e)}
            }

    def _setup_environment_variables(self, environment_vars: Dict[str, str]) -> Dict[str, Any]:
        """Set up environment variables for the current process."""
        if not environment_vars:
            return {"success": True, "message": "No environment variables to set"}

        for key, value in environment_vars.items():
            os.environ[key] = value
            logger.info(f"Set environment variable {key}")

        return {
            "success": True,
            "message": f"Set {len(environment_vars)} environment variables"
        }


    def _handle_git_repo_setup(self, inputs: Dict[str, Any], global_context: GlobalContext) -> AgentResult:
        """Handle Git repository initialization and setup."""
        logger.info("Handling Git repository setup")
        
        workspace_path = inputs.get("workspace_path", str(global_context.workspace_path))
        user_name = inputs.get("git_user_name", "Agent User")
        user_email = inputs.get("git_user_email", "agent@example.com")
        
        # Commands to set up Git repo environment
        commands = [
            "git init",
            f"git config user.name '{user_name}'",
            f"git config user.email '{user_email}'",
            "git config --global --add safe.directory '*'"
        ]
        
        try:
            command_results = []
            failed_commands = []
            
            for command in commands:
                logger.info(f"Executing: {command}")
                
                tool_output = execute_shell_command(
                    command=command, 
                    working_dir=workspace_path
                )
                
                tool_output['command'] = command
                command_results.append(tool_output)
                
                if tool_output['exit_code'] != 0:
                    failed_commands.append(command)
                    logger.error(f"Git setup command failed: {command}")
                    # Don't stop on safe.directory failure as it might already be set
                    if "safe.directory" not in command:
                        break
            
            success = len(failed_commands) == 0 or (len(failed_commands) == 1 and "safe.directory" in failed_commands[0])
            
            if success:
                message = f"Git repository successfully initialized at {workspace_path}"
            else:
                message = f"Git repository setup failed. Failed commands: {failed_commands}"
            
            return AgentResult(
                success=success,
                message=message,
                outputs={
                    "workspace_path": workspace_path,
                    "commands_executed": [res['command'] for res in command_results],
                    "command_results": command_results,
                    "failed_commands": failed_commands,
                    "git_configured": success
                }
            )
            
        except Exception as e:
            return AgentResult(
                success=False,
                message=f"Git repository setup failed: {e}",
                error_details={"exception": str(e)}
            )

    # ====== WORKFLOW ORCHESTRATION DELEGATION ======
    
    
    def _delegate_to_workflow_orchestration(self, goal: str, inputs: Dict[str, Any], 
                                          global_context: GlobalContext) -> AgentResult:
        """Delegate complex multi-step workflows to existing orchestration tools."""
        logger.info(f"Delegating multi-step workflow to orchestration tools: {goal[:100]}...")
        
        try:
            # Import orchestration tools
            from tools.workflow_orchestration_tools import (
                orchestrate_workflow, create_orchestration_context, OrchestrationMode
            )
            from core.models import TaskGraph, TaskNode
            
            commands = inputs.get("commands", [])
            working_directory = inputs.get("working_directory", str(global_context.workspace_path))
            purpose = inputs.get("purpose", "Multi-step workflow execution")
            
            if not commands:
                return AgentResult(
                    success=False,
                    message="Multi-step workflow detected but no commands provided",
                    error_details={"goal": goal, "inputs_keys": list(inputs.keys())}
                )
            
            # Convert commands to TaskGraph
            task_graph = self._create_task_graph_from_commands(commands, goal, working_directory)
            
            # Create orchestration context
            orchestration_context = create_orchestration_context(
                workflow_id=f"executor_workflow_{int(time.time())}",
                orchestration_mode="sequential",  # Commands are typically sequential
                max_parallel_tasks=1,  # ExecutorAgent workflows are sequential
                error_handling="stop",  # Stop on first failure for executor tasks
                global_context=global_context
            )
            
            # Create mini agent registry for shell execution
            shell_agent_registry = {
                "ShellExecutor": self  # Use ExecutorAgent itself for shell execution
            }
            
            # Execute workflow using existing orchestration tools
            orchestration_result = orchestrate_workflow(
                task_graph=task_graph,
                agent_registry=shell_agent_registry,
                context=orchestration_context
            )
            
            # Convert orchestration result to AgentResult
            success = orchestration_result.success
            message = self._create_orchestration_message(orchestration_result, purpose)
            
            outputs = {
                "orchestration_type": "delegated_workflow",
                "workflow_id": orchestration_result.workflow_id,
                "total_steps": orchestration_result.total_steps,
                "completed_steps": orchestration_result.completed_steps,
                "failed_steps": orchestration_result.failed_steps,
                "duration_seconds": orchestration_result.total_duration_seconds,
                "final_outputs": orchestration_result.final_outputs,
                "purpose": purpose,
                "tool_used": "workflow_orchestration_tools"
            }
            
            return AgentResult(
                success=success,
                message=message,
                outputs=outputs
            )
            
        except ImportError as e:
            logger.warning(f"Workflow orchestration tools not available: {e}. Falling back to simple shell execution.")
            # Fallback to simple shell execution if orchestration tools unavailable
            return self._handle_shell_execution(goal, inputs, global_context)
        except Exception as e:
            logger.error(f"Workflow orchestration delegation failed: {e}", exc_info=True)
            return AgentResult(
                success=False,
                message=f"Workflow orchestration failed: {e}",
                error_details={
                    "exception": str(e),
                    "goal": goal,
                    "fallback": "Consider using simple shell execution"
                }
            )
    
    def _check_framework_availability(self, framework: str, python_executable: str = None) -> Dict[str, Any]:
        """Check if a test framework is available in the current environment."""
        if python_executable is None:
            python_executable = sys.executable
        
        try:
            # Framework dependency mapping
            framework_modules = {
                "pytest": "pytest",
                "unittest": "unittest",  # built-in, always available
                "nose": "nose"
            }
            
            if framework not in framework_modules:
                return {
                    "available": False,
                    "message": f"Unknown test framework: {framework}",
                    "module": None
                }
            
            module_name = framework_modules[framework]
            
            # unittest is built-in, always available
            if framework == "unittest":
                return {
                    "available": True,
                    "message": f"{framework} is built-in and available",
                    "module": module_name
                }
            
            # Check if module can be imported
            import subprocess
            result = subprocess.run(
                [python_executable, "-c", f"import {module_name}; print('SUCCESS')"],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            available = result.returncode == 0 and "SUCCESS" in result.stdout
            
            return {
                "available": available,
                "message": f"{framework} {'is available' if available else 'is not installed'}",
                "module": module_name,
                "check_details": {
                    "returncode": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }
            }
            
        except Exception as e:
            return {
                "available": False,
                "message": f"Error checking {framework} availability: {e}",
                "module": framework_modules.get(framework),
                "error": str(e)
            }

    def _auto_install_test_framework(self, framework: str, python_executable: str = None, 
                                   working_directory: str = ".", global_context = None) -> Dict[str, Any]:
        """Automatically install missing test framework."""
        if python_executable is None:
            python_executable = sys.executable
        
        try:
            # Framework package mapping
            framework_packages = {
                "pytest": ["pytest"],
                "nose": ["nose"],
                "unittest": []  # built-in, no installation needed
            }
            
            if framework not in framework_packages:
                return {
                    "success": False,
                    "message": f"Unknown test framework for installation: {framework}"
                }
            
            packages = framework_packages[framework]
            if not packages:
                return {
                    "success": True,
                    "message": f"{framework} is built-in, no installation needed"
                }
            
            logger.info(f"Auto-installing test framework: {framework} (packages: {packages})")
            
            # Use existing dependency management method
            dependency_inputs = {
                "operation": "install",
                "packages": packages,
                "python_executable": python_executable,
                "working_directory": working_directory
            }
            install_result = self._handle_dependency_management(dependency_inputs, global_context)
            
            if install_result.success:
                # Verify installation worked
                availability_check = self._check_framework_availability(framework, python_executable)
                if availability_check["available"]:
                    return {
                        "success": True,
                        "message": f"Successfully installed and verified {framework}",
                        "packages_installed": packages,
                        "verification": availability_check
                    }
                else:
                    return {
                        "success": False,
                        "message": f"Installation completed but {framework} still not available",
                        "packages_installed": packages,
                        "verification_failed": availability_check
                    }
            else:
                return {
                    "success": False,
                    "message": f"Failed to install {framework}: {install_result.message}",
                    "install_details": install_result.error_details
                }
                
        except Exception as e:
            logger.error(f"Error auto-installing {framework}: {e}", exc_info=True)
            return {
                "success": False,
                "message": f"Exception during {framework} installation: {e}",
                "error": str(e)
            }

    def _create_task_graph_from_commands(self, commands: List[str], goal: str, 
                                       working_directory: str) -> 'TaskGraph':
        """Convert list of commands into a TaskGraph for orchestration."""
        from core.models import TaskGraph, TaskNode
        
        # Create task nodes for each command
        nodes = {}
        
        for i, command in enumerate(commands):
            task_id = f"cmd_{i+1}"
            
            # Create task node
            task_node = TaskNode(
                task_id=task_id,
                goal=f"Execute: {command}",
                assigned_agent="ShellExecutor",
                dependencies=[] if i == 0 else [f"cmd_{i}"]  # Sequential dependency
            )
            
            nodes[task_id] = task_node
        
        return TaskGraph(nodes=nodes)
    
    def _create_orchestration_message(self, orchestration_result, purpose: str) -> str:
        """Create message from orchestration result."""
        if orchestration_result.success:
            return (f"✅ Multi-step workflow completed successfully: {orchestration_result.completed_steps}/"
                   f"{orchestration_result.total_steps} steps completed in "
                   f"{orchestration_result.total_duration_seconds:.2f}s for {purpose}")
        else:
            return (f"⚠️ Multi-step workflow completed with issues: {orchestration_result.completed_steps}/"
                   f"{orchestration_result.total_steps} steps completed, "
                   f"{orchestration_result.failed_steps} failed. Errors: {orchestration_result.error_summary}")

    def _ensure_project_standardization(self, working_directory: str) -> AgentResult:
        """
        Ensure the project has standard configuration files for proper test detection.
        This prevents the need for generic auto-detection scripts.
        """
        try:
            from tools.project_standardization_tools import standardize_project
            
            logger.info(f"Checking project standardization in {working_directory}")
            
            # Attempt to standardize the project
            standardization_result = standardize_project(working_directory)
            
            if standardization_result.success:
                if standardization_result.content.get("already_standardized"):
                    message = f"✅ Project already standardized as {standardization_result.content['project_type']}"
                else:
                    generated_files = standardization_result.content.get("generated_files", [])
                    message = f"✅ Generated {len(generated_files)} standard config files: {generated_files}"
                
                return AgentResult(
                    success=True,
                    message=message,
                    outputs=standardization_result.content
                )
            else:
                return AgentResult(
                    success=False,
                    message=f"Project standardization failed: {standardization_result.message}",
                    error_details=standardization_result.error_details
                )
                
        except Exception as e:
            logger.error(f"Project standardization error: {e}")
            return AgentResult(
                success=False,
                message=f"Project standardization error: {e}",
                error_details={"exception": str(e)}
            )
