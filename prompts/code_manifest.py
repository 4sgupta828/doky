# prompts/code_manifest.py
import json
from typing import Dict, Any

def build_manifest_generation_prompt(spec: str, context_summary: Dict[str, Any]) -> str:
    """Constructs a high-quality prompt for generating a file manifest."""
    return f"""
    **Persona**: You are an expert tech lead responsible for planning a project's file structure. Your task is to convert a technical specification into a JSON manifest of all the file paths that need to be created.

    **Overall Goal**: Analyze the technical specification and the existing project context to produce a logical and complete list of files for the development team.

    **Context**:
    - **Technical Specification**: 
      ---
      {spec}
      ---
    - **Current Project Files**: 
      {json.dumps(context_summary, indent=2)}

    **Your Task**:
    1.  **Analyze Components**: Identify all the distinct components described in the spec (e.g., data models, API routes, utility functions, tests).
    2.  **Design a File Structure**: Create a logical file and directory structure that separates concerns. Use standard Python project layouts (e.g., a `src` directory for source code, a `tests` directory for tests).
    3.  **List All Files**: Generate a complete list of all the file paths that need to be created to implement the specification.

    **Constraints**:
    - **Be Comprehensive**: Include all necessary files, such as `__init__.py` files for packages, configuration files (`config.py`), and test files (`tests/test_...`).
    - **Follow Conventions**: Adhere to standard Python project structure best practices.

    **Output Format**: You MUST return a single, valid JSON object with one key, "files_to_create", which holds a list of strings representing the file paths.
    
    **Example Output**:
    {{
        "files_to_create": [
            "src/main.py",
            "src/models/__init__.py",
            "src/models/user.py",
            "src/routes/__init__.py",
            "src/routes/auth.py",
            "tests/test_auth.py",
            "config.py",
            "requirements.txt"
        ]
    }}
    """
