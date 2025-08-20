# tools/execution/dependency_tools.py
import logging
import subprocess
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)


class PackageManager(Enum):
    """Package manager types."""
    PIP = "pip"
    CONDA = "conda"


class DependencyOperation(Enum):
    """Dependency operations."""
    INSTALL = "install"
    UNINSTALL = "uninstall"
    LIST = "list"
    FREEZE = "freeze"
    UPDATE = "update"
    CHECK = "check"


@dataclass
class DependencyContext:
    """Configuration for dependency operations."""
    operation: DependencyOperation
    packages: List[str] = field(default_factory=list)
    requirements_file: Optional[str] = None
    package_manager: PackageManager = PackageManager.PIP
    working_directory: str = "."
    python_executable: Optional[str] = None
    upgrade_packages: bool = False
    force_reinstall: bool = False
    index_url: Optional[str] = None
    extra_index_urls: List[str] = field(default_factory=list)
    timeout_seconds: int = 300


@dataclass
class DependencyResult:
    """Result of dependency operations."""
    success: bool
    message: str
    operation: DependencyOperation
    package_manager: PackageManager
    packages_affected: List[str] = field(default_factory=list)
    commands_executed: List[str] = field(default_factory=list)
    output: Optional[str] = None
    error_output: Optional[str] = None
    conflicts_found: bool = False
    requirements_content: Optional[str] = None
    error_details: Optional[str] = None


def install_packages(context: DependencyContext) -> DependencyResult:
    """Install packages using the specified package manager."""
    
    if not context.packages and not context.requirements_file:
        return DependencyResult(
            success=False,
            message="No packages or requirements file specified",
            operation=context.operation,
            package_manager=context.package_manager,
            error_details="Either packages or requirements_file must be provided"
        )

    python_exe = context.python_executable or sys.executable
    commands_executed = []
    results = []

    try:
        if context.package_manager == PackageManager.PIP:
            # Build pip install command
            cmd = [python_exe, "-m", "pip", "install"]
            
            if context.upgrade_packages:
                cmd.append("--upgrade")
            if context.force_reinstall:
                cmd.extend(["--force-reinstall", "--no-deps"])
            if context.index_url:
                cmd.extend(["--index-url", context.index_url])
            for extra_url in context.extra_index_urls:
                cmd.extend(["--extra-index-url", extra_url])

            # Install from requirements file
            if context.requirements_file:
                req_path = Path(context.working_directory) / context.requirements_file
                if req_path.exists():
                    install_cmd = cmd + ["-r", str(req_path)]
                    result = subprocess.run(
                        install_cmd, 
                        capture_output=True, 
                        text=True, 
                        cwd=context.working_directory,
                        timeout=context.timeout_seconds
                    )
                    commands_executed.append(" ".join(install_cmd))
                    results.append(result)

            # Install individual packages
            if context.packages:
                install_cmd = cmd + context.packages
                result = subprocess.run(
                    install_cmd, 
                    capture_output=True, 
                    text=True, 
                    cwd=context.working_directory,
                    timeout=context.timeout_seconds
                )
                commands_executed.append(" ".join(install_cmd))
                results.append(result)

        elif context.package_manager == PackageManager.CONDA:
            # Build conda install command
            cmd = ["conda", "install", "-y"]
            
            if context.packages:
                install_cmd = cmd + context.packages
                result = subprocess.run(
                    install_cmd, 
                    capture_output=True, 
                    text=True, 
                    cwd=context.working_directory,
                    timeout=context.timeout_seconds
                )
                commands_executed.append(" ".join(install_cmd))
                results.append(result)

        # Check if all installations succeeded
        all_success = all(r.returncode == 0 for r in results)
        
        if all_success:
            output = "\n".join(r.stdout for r in results if r.stdout)
            return DependencyResult(
                success=True,
                message=f"Successfully installed packages using {context.package_manager.value}",
                operation=context.operation,
                package_manager=context.package_manager,
                packages_affected=context.packages,
                commands_executed=commands_executed,
                output=output
            )
        else:
            failed_outputs = [r.stderr for r in results if r.returncode != 0]
            return DependencyResult(
                success=False,
                message=f"Package installation failed with {context.package_manager.value}",
                operation=context.operation,
                package_manager=context.package_manager,
                commands_executed=commands_executed,
                error_output="\n".join(failed_outputs),
                error_details="Some package installations failed"
            )

    except subprocess.TimeoutExpired:
        return DependencyResult(
            success=False,
            message="Package installation timed out",
            operation=context.operation,
            package_manager=context.package_manager,
            commands_executed=commands_executed,
            error_details=f"Operation timed out after {context.timeout_seconds} seconds"
        )
    except Exception as e:
        return DependencyResult(
            success=False,
            message=f"Package installation error: {e}",
            operation=context.operation,
            package_manager=context.package_manager,
            commands_executed=commands_executed,
            error_details=str(e)
        )


def uninstall_packages(context: DependencyContext) -> DependencyResult:
    """Uninstall packages using the specified package manager."""
    
    if not context.packages:
        return DependencyResult(
            success=False,
            message="No packages specified for uninstallation",
            operation=context.operation,
            package_manager=context.package_manager,
            error_details="Packages list cannot be empty for uninstall operation"
        )

    python_exe = context.python_executable or sys.executable

    try:
        if context.package_manager == PackageManager.PIP:
            cmd = [python_exe, "-m", "pip", "uninstall", "-y"] + context.packages
        elif context.package_manager == PackageManager.CONDA:
            cmd = ["conda", "remove", "-y"] + context.packages
        else:
            return DependencyResult(
                success=False,
                message=f"Unsupported package manager: {context.package_manager}",
                operation=context.operation,
                package_manager=context.package_manager,
                error_details="Only pip and conda are supported"
            )

        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            cwd=context.working_directory,
            timeout=120
        )

        if result.returncode == 0:
            return DependencyResult(
                success=True,
                message=f"Successfully uninstalled packages: {', '.join(context.packages)}",
                operation=context.operation,
                package_manager=context.package_manager,
                packages_affected=context.packages,
                commands_executed=[" ".join(cmd)],
                output=result.stdout
            )
        else:
            return DependencyResult(
                success=False,
                message=f"Failed to uninstall packages",
                operation=context.operation,
                package_manager=context.package_manager,
                commands_executed=[" ".join(cmd)],
                error_output=result.stderr,
                error_details=f"Command failed with return code {result.returncode}"
            )

    except Exception as e:
        return DependencyResult(
            success=False,
            message=f"Uninstallation error: {e}",
            operation=context.operation,
            package_manager=context.package_manager,
            error_details=str(e)
        )


def list_packages(context: DependencyContext) -> DependencyResult:
    """List installed packages."""
    
    python_exe = context.python_executable or sys.executable

    try:
        if context.package_manager == PackageManager.PIP:
            cmd = [python_exe, "-m", "pip", "list"]
        elif context.package_manager == PackageManager.CONDA:
            cmd = ["conda", "list"]
        else:
            return DependencyResult(
                success=False,
                message=f"Unsupported package manager: {context.package_manager}",
                operation=context.operation,
                package_manager=context.package_manager,
                error_details="Only pip and conda are supported"
            )

        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            cwd=context.working_directory,
            timeout=60
        )

        if result.returncode == 0:
            return DependencyResult(
                success=True,
                message=f"Successfully listed packages with {context.package_manager.value}",
                operation=context.operation,
                package_manager=context.package_manager,
                commands_executed=[" ".join(cmd)],
                output=result.stdout
            )
        else:
            return DependencyResult(
                success=False,
                message=f"Failed to list packages",
                operation=context.operation,
                package_manager=context.package_manager,
                commands_executed=[" ".join(cmd)],
                error_output=result.stderr,
                error_details=f"Command failed with return code {result.returncode}"
            )

    except Exception as e:
        return DependencyResult(
            success=False,
            message=f"Package listing error: {e}",
            operation=context.operation,
            package_manager=context.package_manager,
            error_details=str(e)
        )


def freeze_requirements(context: DependencyContext) -> DependencyResult:
    """Generate requirements file from installed packages."""
    
    python_exe = context.python_executable or sys.executable

    try:
        if context.package_manager == PackageManager.PIP:
            cmd = [python_exe, "-m", "pip", "freeze"]
        else:
            return DependencyResult(
                success=False,
                message=f"Freeze operation not supported for {context.package_manager.value}",
                operation=context.operation,
                package_manager=context.package_manager,
                error_details="Freeze is only supported for pip"
            )

        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            cwd=context.working_directory,
            timeout=60
        )

        if result.returncode == 0:
            # Write to file if specified
            if context.requirements_file:
                req_path = Path(context.working_directory) / context.requirements_file
                req_path.write_text(result.stdout)
                
            return DependencyResult(
                success=True,
                message=f"Successfully generated requirements {'to ' + context.requirements_file if context.requirements_file else ''}",
                operation=context.operation,
                package_manager=context.package_manager,
                commands_executed=[" ".join(cmd)],
                output=result.stdout,
                requirements_content=result.stdout
            )
        else:
            return DependencyResult(
                success=False,
                message=f"Failed to freeze requirements",
                operation=context.operation,
                package_manager=context.package_manager,
                commands_executed=[" ".join(cmd)],
                error_output=result.stderr,
                error_details=f"Command failed with return code {result.returncode}"
            )

    except Exception as e:
        return DependencyResult(
            success=False,
            message=f"Requirements freeze error: {e}",
            operation=context.operation,
            package_manager=context.package_manager,
            error_details=str(e)
        )


def update_packages(context: DependencyContext) -> DependencyResult:
    """Update packages to latest versions."""
    
    if not context.packages:
        return DependencyResult(
            success=False,
            message="No packages specified for update",
            operation=context.operation,
            package_manager=context.package_manager,
            error_details="Packages list cannot be empty for update operation"
        )

    python_exe = context.python_executable or sys.executable

    try:
        if context.package_manager == PackageManager.PIP:
            cmd = [python_exe, "-m", "pip", "install", "--upgrade"] + context.packages
        elif context.package_manager == PackageManager.CONDA:
            cmd = ["conda", "update", "-y"] + context.packages
        else:
            return DependencyResult(
                success=False,
                message=f"Unsupported package manager: {context.package_manager}",
                operation=context.operation,
                package_manager=context.package_manager,
                error_details="Only pip and conda are supported"
            )

        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            cwd=context.working_directory,
            timeout=context.timeout_seconds
        )

        if result.returncode == 0:
            return DependencyResult(
                success=True,
                message=f"Successfully updated packages: {', '.join(context.packages)}",
                operation=context.operation,
                package_manager=context.package_manager,
                packages_affected=context.packages,
                commands_executed=[" ".join(cmd)],
                output=result.stdout
            )
        else:
            return DependencyResult(
                success=False,
                message=f"Failed to update packages",
                operation=context.operation,
                package_manager=context.package_manager,
                commands_executed=[" ".join(cmd)],
                error_output=result.stderr,
                error_details=f"Command failed with return code {result.returncode}"
            )

    except Exception as e:
        return DependencyResult(
            success=False,
            message=f"Package update error: {e}",
            operation=context.operation,
            package_manager=context.package_manager,
            error_details=str(e)
        )


def check_dependencies(context: DependencyContext) -> DependencyResult:
    """Check for dependency conflicts and issues."""
    
    python_exe = context.python_executable or sys.executable

    try:
        if context.package_manager == PackageManager.PIP:
            cmd = [python_exe, "-m", "pip", "check"]
        else:
            return DependencyResult(
                success=False,
                message=f"Dependency check not supported for {context.package_manager.value}",
                operation=context.operation,
                package_manager=context.package_manager,
                error_details="Check is only supported for pip"
            )

        result = subprocess.run(
            cmd, 
            capture_output=True, 
            text=True, 
            cwd=context.working_directory,
            timeout=60
        )

        # pip check returns 0 if no issues, non-zero if conflicts found
        if result.returncode == 0:
            return DependencyResult(
                success=True,
                message="No dependency conflicts found",
                operation=context.operation,
                package_manager=context.package_manager,
                commands_executed=[" ".join(cmd)],
                output=result.stdout,
                conflicts_found=False
            )
        else:
            return DependencyResult(
                success=False,
                message=f"Dependency conflicts detected",
                operation=context.operation,
                package_manager=context.package_manager,
                commands_executed=[" ".join(cmd)],
                output=result.stdout,
                conflicts_found=True,
                error_details="Conflicts found in dependency tree"
            )

    except Exception as e:
        return DependencyResult(
            success=False,
            message=f"Dependency check error: {e}",
            operation=context.operation,
            package_manager=context.package_manager,
            error_details=str(e)
        )


def execute_dependency_operation(context: DependencyContext) -> DependencyResult:
    """Execute a dependency operation based on the context."""
    
    logger.info(f"Executing {context.operation.value} operation with {context.package_manager.value}")
    
    if context.operation == DependencyOperation.INSTALL:
        return install_packages(context)
    elif context.operation == DependencyOperation.UNINSTALL:
        return uninstall_packages(context)
    elif context.operation == DependencyOperation.LIST:
        return list_packages(context)
    elif context.operation == DependencyOperation.FREEZE:
        return freeze_requirements(context)
    elif context.operation == DependencyOperation.UPDATE:
        return update_packages(context)
    elif context.operation == DependencyOperation.CHECK:
        return check_dependencies(context)
    else:
        return DependencyResult(
            success=False,
            message=f"Unknown operation: {context.operation}",
            operation=context.operation,
            package_manager=context.package_manager,
            error_details=f"Supported operations: {[op.value for op in DependencyOperation]}"
        )