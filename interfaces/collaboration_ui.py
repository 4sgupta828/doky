# interfaces/collaboration_ui.py
from core.context import GlobalContext

class CollaborationUI:
    """
    Provides a transparent window into the agent's operations, allowing for
    human-in-the-loop collaboration. This could be implemented as a command-line
    interface, a web dashboard, or an IDE extension.
    """

    def display_status(self, context: GlobalContext):
        """Displays the current state of the TaskGraph and recent logs."""
        pass

    def prompt_for_input(self, question: str) -> str:
        """Pauses execution and asks the user for input or confirmation."""
        pass

    def allow_artifact_edit(self, artifact_key: str, context: GlobalContext):
        """Allows the user to manually inspect and edit a generated artifact."""
        pass