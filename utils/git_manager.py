# utils/git_manager.py
import logging
import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional

# This import establishes the "contract" this class must fulfill.
from core.context import WorkspaceManager

# Get a logger instance for this module.
logger = logging.getLogger(__name__)

class GitWorkspaceManager(WorkspaceManager):
    """
    An implementation of the WorkspaceManager that uses a local Git repository
    for version control. This provides a robust way to manage file changes, create
    atomic save points per task, and revert to previous states, which is perfect
for the AdaptiveEngine's needs.
    """

    def __init__(self, repo_path: str):
        """
        Initializes the workspace manager. If the specified directory does not
        exist or is not a Git repository, it will be initialized as one.

        Args:
            repo_path: The local file system path to the workspace directory.
        """
        self.repo_path = Path(repo_path).resolve()
        logger.info(f"Initializing Git workspace at: {self.repo_path}")

        try:
            # Create the directory if it doesn't exist.
            self.repo_path.mkdir(parents=True, exist_ok=True)

            # Check if it's already a git repository.
            is_git_repo = (self.repo_path / ".git").is_dir()

            if not is_git_repo:
                self._run_git_command(["init"])
                self._run_git_command(["config", "user.name", "AgentCollective"])
                self._run_git_command(["config", "user.email", "agent@localhost"])
                # Create an initial commit so that reverting is always possible.
                (self.repo_path / ".placeholder").touch()
                self._run_git_command(["add", ".placeholder"])
                self._run_git_command(["commit", "-m", "Initial commit"])
                logger.info(f"Initialized new Git repository at {self.repo_path}")

        except Exception:
            logger.critical(f"Failed to initialize Git workspace at {self.repo_path}", exc_info=True)
            raise

    def _run_git_command(self, command: List[str]) -> subprocess.CompletedProcess:
        """A helper function to run a git command and handle errors."""
        git_command = ["git", "-C", str(self.repo_path)] + command
        try:
            result = subprocess.run(
                git_command,
                capture_output=True,
                text=True,
                check=True,  # This will raise a CalledProcessError if the command returns a non-zero exit code.
                encoding="utf-8"
            )
            logger.debug(f"Git command {' '.join(command)} successful. STDOUT: {result.stdout.strip()}")
            return result
        except subprocess.CalledProcessError as e:
            error_message = f"Git command failed: {' '.join(git_command)}\n"
            error_message += f"Exit Code: {e.returncode}\n"
            error_message += f"STDOUT: {e.stdout.strip()}\n"
            error_message += f"STDERR: {e.stderr.strip()}"
            logger.error(error_message)
            raise

    def write_file_content(self, relative_path: str, content: str, task_id: str):
        """
        Writes content to a file and commits it as a single, atomic change
        tagged with the task ID.

        Args:
            relative_path: The path of the file relative to the workspace root.
            content: The string content to write to the file.
            task_id: The ID of the task performing the write, used for the commit message.
        """
        full_path = self.repo_path / relative_path
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # The commit creates an atomic, reversible record of the change.
        self._run_git_command(["add", str(full_path)])
        commit_message = f"Task {task_id}: Write file {relative_path}"
        self._run_git_command(["commit", "-m", commit_message])
        logger.info(f"Wrote and committed file '{relative_path}' for task '{task_id}'.")

    def get_file_content(self, relative_path: str) -> Optional[str]:
        """Reads the content of a file from the workspace."""
        full_path = self.repo_path / relative_path
        if not full_path.is_file():
            logger.warning(f"File not found: {full_path}")
            return None
        with open(full_path, 'r', encoding='utf-8') as f:
            return f.read()

    def list_files(self, relative_path: str = ".") -> List[str]:
        """Lists relevant project files, filtering out build artifacts and dependencies."""
        base_path = self.repo_path / relative_path
        if not base_path.is_dir():
            return []
        
        # Directories to skip entirely
        skip_dirs = {
            '.git', '__pycache__', '.pytest_cache', 'node_modules', 
            '.venv', 'venv', 'env', '.env', 'dist', 'build', 
            'target', '.idea', '.vscode', '.DS_Store', 'site-packages'
        }
        
        # File extensions to skip
        skip_extensions = {
            '.pyc', '.pyo', '.pyd', '.so', '.dll', '.dylib', '.exe',
            '.log', '.tmp', '.cache', '.lock', '.pid', '.swp', '.bak',
            '.class', '.jar', '.war', '.ear'
        }
        
        # Files to skip by name
        skip_files = {
            '.DS_Store', 'Thumbs.db', 'desktop.ini', '.coverage',
            'coverage.xml', 'pytest.ini', 'tox.ini'
        }
        
        relevant_files = []
        for root, dirs, files in os.walk(base_path):
            # Filter out directories we want to skip
            dirs[:] = [d for d in dirs if d not in skip_dirs]
            
            for file in files:
                # Skip files by name or extension
                if file in skip_files:
                    continue
                    
                file_ext = Path(file).suffix.lower()
                if file_ext in skip_extensions:
                    continue
                
                # Get the path relative to the repo root
                full_path = Path(root) / file
                relative_file_path = str(full_path.relative_to(self.repo_path))
                
                # Additional filtering: skip files in certain patterns
                if any(pattern in relative_file_path for pattern in [
                    '__pycache__', '.git/', 'site-packages/', '.dist-info/'
                ]):
                    continue
                
                relevant_files.append(relative_file_path)
        
        return sorted(relevant_files)  # Sort for consistency

    def revert_changes(self, task_id: str):
        """
        Finds the commit(s) associated with a task ID and reverts them.
        This is the core of the backtracking mechanism.

        Args:
            task_id: The ID of the task whose changes should be reverted.
        """
        try:
            # Use `git log` to find the commit hash based on the task_id in the message.
            log_output = self._run_git_command(
                ["log", "--grep", f"Task {task_id}", "--format=%H"]
            ).stdout.strip()
            
            commit_hashes = log_output.split()
            if not commit_hashes:
                logger.warning(f"No commits found for task_id '{task_id}'. Nothing to revert.")
                return

            for commit_hash in commit_hashes:
                logger.info(f"Reverting commit '{commit_hash}' for task '{task_id}'.")
                self._run_git_command(["revert", "--no-edit", commit_hash])
            
            logger.info(f"Successfully reverted all changes for task '{task_id}'.")

        except subprocess.CalledProcessError:
            logger.error(f"Failed to revert changes for task '{task_id}'. The repository may be in a conflicting state.", exc_info=True)


# --- Self-Testing Block ---
# To run this test: `python utils/git_manager.py`
if __name__ == "__main__":
    from utils.logger import setup_logger
    setup_logger(default_level=logging.DEBUG)

    # Use a temporary directory for a clean, sandboxed test environment.
    TEST_REPO_PATH = Path("./temp_test_workspace")
    
    # Clean up any previous test runs.
    if TEST_REPO_PATH.exists():
        shutil.rmtree(TEST_REPO_PATH)

    print("\n--- Testing GitWorkspaceManager ---")

    # 1. Test Initialization
    print("\n[1] Testing workspace initialization...")
    try:
        workspace = GitWorkspaceManager(str(TEST_REPO_PATH))
        assert (TEST_REPO_PATH / ".git").is_dir()
        logger.info("Initialization test passed.")
    except Exception as e:
        logger.error("Initialization test failed.", exc_info=True)
        exit()

    # 2. Test Write and Read Operations
    print("\n[2] Testing write, read, and list operations...")
    task1_id = "task_001"
    file1_path = "src/app.py"
    file1_content = "print('Hello, World!')"
    workspace.write_file_content(file1_path, file1_content, task1_id)
    
    read_content = workspace.get_file_content(file1_path)
    assert read_content == file1_content
    logger.info("File write and read successful.")
    
    file_list = workspace.list_files()
    assert file1_path in file_list
    assert ".placeholder" in file_list
    logger.info(f"File list successful: {file_list}")
    logger.info("Write, read, and list tests passed.")

    # 3. Test a second write operation
    task2_id = "task_002"
    file2_path = "config/settings.py"
    file2_content = "DEBUG = True"
    workspace.write_file_content(file2_path, file2_content, task2_id)
    assert file2_path in workspace.list_files()
    logger.info("Second write operation successful.")
    
    # 4. Test Revert Operation
    print("\n[3] Testing revert operation...")
    # Revert the first task's changes.
    workspace.revert_changes(task1_id)
    
    # The first file should now be gone, but the second should remain.
    reverted_file_list = workspace.list_files()
    assert file1_path not in reverted_file_list
    assert file2_path in reverted_file_list
    logger.info(f"File list after revert: {reverted_file_list}")
    logger.info("Revert operation test passed.")

    # 5. Test reading a non-existent file
    print("\n[4] Testing read on non-existent file...")
    non_existent_content = workspace.get_file_content("non_existent.txt")
    assert non_existent_content is None
    logger.info("Reading non-existent file returned None as expected.")

    # Clean up the test directory.
    shutil.rmtree(TEST_REPO_PATH)
    logger.info(f"Cleaned up test directory: {TEST_REPO_PATH}")

    print("\n--- All GitWorkspaceManager Tests Passed Successfully ---")