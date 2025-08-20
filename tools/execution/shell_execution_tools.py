# tools/execution/shell_execution_tools.py
"""
Shell execution tools for running commands, managing processes, and executing shell operations.
Extracted from ToolingAgent to provide atomic, reusable shell execution capabilities.
"""

import logging
import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

class CommandType(Enum):
    """Types of shell commands."""
    BUILD = "build"
    TEST = "test"
    INSTALL = "install"
    DEPLOY = "deploy"
    LINT = "lint"
    FORMAT = "format"
    CUSTOM = "custom"

class ExecutionMode(Enum):
    """Execution modes for commands."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"  # Stop on first failure

@dataclass
class ShellCommand:
    """Represents a shell command with metadata."""
    command: str
    command_type: CommandType = CommandType.CUSTOM
    description: str = ""
    timeout: int = 120
    environment_vars: Optional[Dict[str, str]] = None
    working_directory: Optional[str] = None
    expected_exit_codes: List[int] = None
    
    def __post_init__(self):
        if self.expected_exit_codes is None:
            self.expected_exit_codes = [0]

@dataclass
class CommandResult:
    """Result of executing a shell command."""
    command: str
    exit_code: int
    stdout: str
    stderr: str
    execution_time: float
    success: bool
    working_directory: str
    environment_vars: Optional[Dict[str, str]] = None

@dataclass
class ShellExecutionContext:
    """Context for shell execution operations."""
    commands: List[ShellCommand]
    working_directory: str = "."
    execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL
    global_timeout: int = 600
    environment_vars: Optional[Dict[str, str]] = None
    ignore_errors: bool = False
    capture_output: bool = True
    shell: bool = False

@dataclass
class ExecutionResult:
    """Result of executing multiple shell commands."""
    success: bool
    message: str
    commands_executed: List[str]
    command_results: List[CommandResult]
    total_execution_time: float
    failed_commands: List[str] = None
    
    def __post_init__(self):
        if self.failed_commands is None:
            self.failed_commands = []

def execute_command(
    command: str,
    working_directory: str = ".",
    timeout: int = 120,
    environment_vars: Optional[Dict[str, str]] = None,
    expected_exit_codes: List[int] = None,
    capture_output: bool = True,
    shell: bool = False
) -> CommandResult:
    """Execute a single shell command."""
    if expected_exit_codes is None:
        expected_exit_codes = [0]
    
    start_time = time.time()
    
    logger.info(f"Executing command: {command}")
    logger.debug(f"Working directory: {working_directory}")
    
    # Setup environment
    env = os.environ.copy()
    if environment_vars:
        env.update(environment_vars)
    
    # Ensure working directory exists
    Path(working_directory).mkdir(parents=True, exist_ok=True)
    
    try:
        # Split command if not using shell
        if not shell:
            cmd_parts = shlex.split(command)
        else:
            cmd_parts = command
        
        # Execute command
        process = subprocess.run(
            cmd_parts,
            cwd=working_directory,
            env=env,
            capture_output=capture_output,
            text=True,
            timeout=timeout,
            shell=shell
        )
        
        execution_time = time.time() - start_time
        success = process.returncode in expected_exit_codes
        
        result = CommandResult(
            command=command,
            exit_code=process.returncode,
            stdout=process.stdout or "",
            stderr=process.stderr or "",
            execution_time=execution_time,
            success=success,
            working_directory=working_directory,
            environment_vars=environment_vars
        )
        
        if success:
            logger.info(f"Command completed successfully in {execution_time:.2f}s")
        else:
            logger.error(f"Command failed with exit code {process.returncode}")
        
        return result
        
    except subprocess.TimeoutExpired:
        execution_time = time.time() - start_time
        logger.error(f"Command timed out after {timeout} seconds")
        
        return CommandResult(
            command=command,
            exit_code=-1,
            stdout="",
            stderr=f"Command timed out after {timeout} seconds",
            execution_time=execution_time,
            success=False,
            working_directory=working_directory,
            environment_vars=environment_vars
        )
        
    except Exception as e:
        execution_time = time.time() - start_time
        logger.error(f"Command execution error: {e}")
        
        return CommandResult(
            command=command,
            exit_code=-1,
            stdout="",
            stderr=f"Execution error: {str(e)}",
            execution_time=execution_time,
            success=False,
            working_directory=working_directory,
            environment_vars=environment_vars
        )

def execute_commands(context: ShellExecutionContext) -> ExecutionResult:
    """Execute multiple shell commands based on execution context."""
    start_time = time.time()
    command_results = []
    failed_commands = []
    commands_executed = []
    
    logger.info(f"Executing {len(context.commands)} commands in {context.execution_mode.value} mode")
    
    if context.execution_mode == ExecutionMode.SEQUENTIAL:
        # Execute commands one by one
        for i, shell_cmd in enumerate(context.commands):
            logger.info(f"Executing command {i+1}/{len(context.commands)}: {shell_cmd.description or shell_cmd.command[:50]}")
            
            # Use command-specific or global settings
            working_dir = shell_cmd.working_directory or context.working_directory
            env_vars = shell_cmd.environment_vars or context.environment_vars
            
            result = execute_command(
                command=shell_cmd.command,
                working_directory=working_dir,
                timeout=shell_cmd.timeout,
                environment_vars=env_vars,
                expected_exit_codes=shell_cmd.expected_exit_codes,
                capture_output=context.capture_output,
                shell=context.shell
            )
            
            command_results.append(result)
            commands_executed.append(shell_cmd.command)
            
            if not result.success:
                failed_commands.append(shell_cmd.command)
                if not context.ignore_errors:
                    logger.error(f"Command failed, stopping execution: {shell_cmd.command}")
                    break
    
    elif context.execution_mode == ExecutionMode.PARALLEL:
        # Execute commands in parallel (simplified version)
        import concurrent.futures
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            
            for shell_cmd in context.commands:
                working_dir = shell_cmd.working_directory or context.working_directory
                env_vars = shell_cmd.environment_vars or context.environment_vars
                
                future = executor.submit(
                    execute_command,
                    shell_cmd.command,
                    working_dir,
                    shell_cmd.timeout,
                    env_vars,
                    shell_cmd.expected_exit_codes,
                    context.capture_output,
                    context.shell
                )
                futures.append((future, shell_cmd.command))
            
            for future, command in futures:
                try:
                    result = future.result(timeout=context.global_timeout)
                    command_results.append(result)
                    commands_executed.append(command)
                    
                    if not result.success:
                        failed_commands.append(command)
                        
                except concurrent.futures.TimeoutError:
                    failed_commands.append(command)
                    command_results.append(CommandResult(
                        command=command,
                        exit_code=-1,
                        stdout="",
                        stderr="Global timeout exceeded",
                        execution_time=context.global_timeout,
                        success=False,
                        working_directory=context.working_directory
                    ))
    
    total_time = time.time() - start_time
    overall_success = len(failed_commands) == 0
    
    # Create summary message
    if overall_success:
        message = f"Successfully executed {len(commands_executed)} commands in {total_time:.2f}s"
    else:
        message = f"Execution completed with {len(failed_commands)} failures out of {len(commands_executed)} commands"
    
    return ExecutionResult(
        success=overall_success,
        message=message,
        commands_executed=commands_executed,
        command_results=command_results,
        total_execution_time=total_time,
        failed_commands=failed_commands
    )

def execute_build_command(
    build_command: str,
    working_directory: str = ".",
    build_tool: str = "auto",
    timeout: int = 300
) -> CommandResult:
    """Execute a build command with build-specific optimizations."""
    # Auto-detect build tool if not specified
    if build_tool == "auto":
        build_tool = detect_build_tool(working_directory)
    
    # Add build-specific environment variables
    env_vars = {}
    
    if build_tool == "npm":
        env_vars["NODE_ENV"] = "production"
    elif build_tool == "python":
        env_vars["PYTHONPATH"] = working_directory
    elif build_tool == "maven":
        env_vars["MAVEN_OPTS"] = "-Xmx1024m"
    
    return execute_command(
        command=build_command,
        working_directory=working_directory,
        timeout=timeout,
        environment_vars=env_vars,
        expected_exit_codes=[0]
    )

def execute_test_command(
    test_command: str,
    working_directory: str = ".",
    test_framework: str = "auto",
    timeout: int = 600
) -> CommandResult:
    """Execute a test command with test-specific settings."""
    # Add test-specific environment variables
    env_vars = {
        "CI": "true",
        "TESTING": "true"
    }
    
    if test_framework == "pytest":
        env_vars["PYTEST_CURRENT_TEST"] = "true"
    elif test_framework == "jest":
        env_vars["NODE_ENV"] = "test"
    
    return execute_command(
        command=test_command,
        working_directory=working_directory,
        timeout=timeout,
        environment_vars=env_vars,
        expected_exit_codes=[0, 1]  # Tests may fail but still be valid execution
    )

def detect_build_tool(working_directory: str) -> str:
    """Detect the build tool based on project files."""
    working_path = Path(working_directory)
    
    # Check for various build tool indicators
    if (working_path / "package.json").exists():
        return "npm"
    elif (working_path / "pom.xml").exists():
        return "maven"
    elif (working_path / "build.gradle").exists():
        return "gradle"
    elif (working_path / "setup.py").exists() or (working_path / "pyproject.toml").exists():
        return "python"
    elif (working_path / "Makefile").exists():
        return "make"
    elif (working_path / "Cargo.toml").exists():
        return "cargo"
    else:
        return "generic"

def validate_command_safety(command: str, safety_overrides: List[str] = None) -> Tuple[bool, str]:
    """Validate that a command is safe to execute."""
    if safety_overrides is None:
        safety_overrides = []
    
    # List of potentially dangerous commands
    dangerous_patterns = [
        "rm -rf /",
        "rm -rf /*",
        ":(){ :|:& };:",  # Fork bomb
        "mkfs",
        "fdisk",
        "dd if=",
        "format",
        "del /s /q",
        "> /dev/sda",
        "shutdown",
        "reboot",
        "halt",
    ]
    
    # Check for dangerous patterns
    command_lower = command.lower()
    for pattern in dangerous_patterns:
        if pattern in command_lower and pattern not in safety_overrides:
            return False, f"Potentially dangerous command detected: {pattern}"
    
    # Check for suspicious file operations
    if "rm -rf" in command_lower and not any(override in command_lower for override in safety_overrides):
        return False, "Recursive file deletion detected"
    
    return True, "Command appears safe"

def create_build_commands(
    working_directory: str,
    build_tool: str = "auto",
    build_mode: str = "production"
) -> List[ShellCommand]:
    """Create appropriate build commands based on project type."""
    if build_tool == "auto":
        build_tool = detect_build_tool(working_directory)
    
    commands = []
    
    if build_tool == "npm":
        commands.extend([
            ShellCommand(
                command="npm install",
                command_type=CommandType.INSTALL,
                description="Install npm dependencies",
                timeout=300
            ),
            ShellCommand(
                command=f"npm run build{'--production' if build_mode == 'production' else ''}",
                command_type=CommandType.BUILD,
                description="Build npm project",
                timeout=600
            )
        ])
    
    elif build_tool == "python":
        commands.extend([
            ShellCommand(
                command="pip install -e .",
                command_type=CommandType.INSTALL,
                description="Install Python package in editable mode",
                timeout=300
            ),
            ShellCommand(
                command="python setup.py build",
                command_type=CommandType.BUILD,
                description="Build Python package",
                timeout=300
            )
        ])
    
    elif build_tool == "maven":
        commands.append(
            ShellCommand(
                command=f"mvn clean {'package' if build_mode == 'production' else 'compile'}",
                command_type=CommandType.BUILD,
                description="Build Maven project",
                timeout=600
            )
        )
    
    elif build_tool == "make":
        commands.append(
            ShellCommand(
                command="make",
                command_type=CommandType.BUILD,
                description="Build using Makefile",
                timeout=600
            )
        )
    
    return commands

def create_test_commands(
    working_directory: str,
    test_framework: str = "auto"
) -> List[ShellCommand]:
    """Create appropriate test commands based on project type."""
    commands = []
    working_path = Path(working_directory)
    
    # Auto-detect test framework
    if test_framework == "auto":
        if (working_path / "pytest.ini").exists() or "pytest" in (working_path / "requirements.txt").read_text() if (working_path / "requirements.txt").exists() else "":
            test_framework = "pytest"
        elif (working_path / "package.json").exists():
            try:
                import json
                package_data = json.loads((working_path / "package.json").read_text())
                if "jest" in package_data.get("devDependencies", {}):
                    test_framework = "jest"
                else:
                    test_framework = "npm"
            except:
                test_framework = "npm"
        else:
            test_framework = "unittest"
    
    if test_framework == "pytest":
        commands.append(
            ShellCommand(
                command="python -m pytest -v",
                command_type=CommandType.TEST,
                description="Run pytest tests",
                timeout=600
            )
        )
    
    elif test_framework == "unittest":
        commands.append(
            ShellCommand(
                command="python -m unittest discover -v",
                command_type=CommandType.TEST,
                description="Run unittest tests",
                timeout=600
            )
        )
    
    elif test_framework == "jest":
        commands.append(
            ShellCommand(
                command="npm test",
                command_type=CommandType.TEST,
                description="Run Jest tests",
                timeout=600
            )
        )
    
    elif test_framework == "npm":
        commands.append(
            ShellCommand(
                command="npm test",
                command_type=CommandType.TEST,
                description="Run npm tests",
                timeout=600
            )
        )
    
    return commands

def get_command_template(command_type: CommandType, language: str = "python") -> str:
    """Get a command template for common operations."""
    templates = {
        CommandType.BUILD: {
            "python": "python setup.py build",
            "javascript": "npm run build",
            "java": "mvn compile",
            "generic": "make build"
        },
        CommandType.TEST: {
            "python": "python -m pytest",
            "javascript": "npm test",
            "java": "mvn test",
            "generic": "make test"
        },
        CommandType.INSTALL: {
            "python": "pip install -r requirements.txt",
            "javascript": "npm install",
            "java": "mvn install",
            "generic": "make install"
        },
        CommandType.LINT: {
            "python": "flake8 .",
            "javascript": "eslint .",
            "java": "checkstyle",
            "generic": "lint"
        },
        CommandType.FORMAT: {
            "python": "black .",
            "javascript": "prettier --write .",
            "java": "google-java-format",
            "generic": "format"
        }
    }
    
    return templates.get(command_type, {}).get(language, f"# {command_type.value} command for {language}")

def parse_command_output(result: CommandResult) -> Dict[str, Any]:
    """Parse command output for useful information."""
    parsed = {
        "success": result.success,
        "exit_code": result.exit_code,
        "execution_time": result.execution_time,
        "has_output": bool(result.stdout.strip()),
        "has_errors": bool(result.stderr.strip()),
        "output_lines": len(result.stdout.split('\n')) if result.stdout else 0,
        "error_lines": len(result.stderr.split('\n')) if result.stderr else 0
    }
    
    # Extract common patterns
    if result.stdout:
        # Look for success indicators
        if any(indicator in result.stdout.lower() for indicator in ["success", "passed", "ok", "done", "completed"]):
            parsed["success_indicators"] = True
        
        # Look for warnings
        if any(warning in result.stdout.lower() for warning in ["warning", "warn", "deprecated"]):
            parsed["warnings_found"] = True
    
    if result.stderr:
        # Look for error patterns
        if any(error in result.stderr.lower() for error in ["error", "failed", "exception", "traceback"]):
            parsed["errors_found"] = True
    
    return parsed