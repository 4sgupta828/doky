# tests/test_creator_agent.py
import pytest
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fagents.creator import CreatorAgent, CreationType
from core.context import GlobalContext
from core.workspace import FileSystemWorkspace

class MockLLMClient:
    """Mock LLM client for testing."""
    
    def __init__(self):
        self.responses = {
            "code": """
def hello_world():
    '''Simple hello world function.'''
    return "Hello, World!"

def add_numbers(a: int, b: int) -> int:
    '''Add two numbers and return the result.'''
    return a + b
""",
            "test": """
import pytest

def test_hello_world():
    from main import hello_world
    assert hello_world() == "Hello, World!"

def test_add_numbers():
    from main import add_numbers
    assert add_numbers(2, 3) == 5
    assert add_numbers(-1, 1) == 0
""",
            "spec": """{
    "architecture_summary": "Simple Python application with basic functions",
    "data_models": [
        {
            "name": "User",
            "description": "Basic user model",
            "fields": {
                "id": "integer",
                "name": "string",
                "email": "string"
            }
        }
    ],
    "api_endpoints": [
        {
            "method": "GET",
            "path": "/users",
            "description": "Get all users",
            "request_schema": "No body required",
            "response_schema": "List of user objects"
        }
    ],
    "core_business_logic": [
        "Users can be created, read, updated, and deleted",
        "Email addresses must be unique"
    ],
    "technical_decisions": {
        "database": "SQLite for simplicity",
        "framework": "FastAPI for modern API development"
    }
}""",
            "manifest": """{
    "files_to_create": [
        "src/main.py",
        "src/__init__.py",
        "src/models/__init__.py",
        "src/models/user.py",
        "src/routes/__init__.py",
        "src/routes/users.py",
        "tests/test_users.py",
        "requirements.txt",
        "README.md"
    ]
}""",
            "doc": """# My Project

This is a simple Python project that demonstrates basic functionality.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```python
from main import hello_world, add_numbers

print(hello_world())
print(add_numbers(2, 3))
```

## Testing

```bash
pytest tests/
```
"""
        }
        
    def invoke(self, prompt: str) -> str:
        """Mock invoke method that returns appropriate responses."""
        prompt_lower = prompt.lower()
        
        if "technical specification" in prompt_lower or "json" in prompt_lower and "architecture_summary" in prompt_lower:
            return self.responses["spec"]
        elif "files_to_create" in prompt_lower or "manifest" in prompt_lower:
            return self.responses["manifest"]
        elif "test" in prompt_lower:
            return self.responses["test"]
        elif "documentation" in prompt_lower or "readme" in prompt_lower:
            return self.responses["doc"]
        else:
            return self.responses["code"]

def test_creator_agent_initialization():
    """Test CreatorAgent initialization."""
    agent = CreatorAgent()
    
    assert agent.name == "CreatorAgent"
    assert "code_generation" in agent.get_capabilities()
    assert "test_generation" in agent.get_capabilities()
    assert "documentation_generation" in agent.get_capabilities()

def test_determine_creation_type():
    """Test creation type determination."""
    agent = CreatorAgent()
    
    # Test code generation
    assert agent._determine_creation_type("implement a function", {}) == CreationType.CODE
    assert agent._determine_creation_type("create a class", {}) == CreationType.CODE
    
    # Test test generation
    assert agent._determine_creation_type("write unit tests", {}) == CreationType.TESTS
    assert agent._determine_creation_type("generate test cases", {}) == CreationType.TESTS
    
    # Test documentation generation
    assert agent._determine_creation_type("create documentation", {}) == CreationType.DOCUMENTATION
    assert agent._determine_creation_type("write a README", {}) == CreationType.DOCUMENTATION
    
    # Test specification creation
    assert agent._determine_creation_type("create technical specification", {}) == CreationType.SPECIFICATION
    assert agent._determine_creation_type("write a spec", {}) == CreationType.SPECIFICATION
    
    # Test manifest planning
    assert agent._determine_creation_type("plan file structure", {}) == CreationType.MANIFEST
    assert agent._determine_creation_type("create project manifest", {}) == CreationType.MANIFEST

def test_code_generation():
    """Test code generation functionality."""
    mock_llm = MockLLMClient()
    agent = CreatorAgent(llm_client=mock_llm)
    
    workspace = FileSystemWorkspace("/tmp/test")
    context = GlobalContext(workspace=workspace)
    
    goal = "Create a hello world function"
    inputs = {
        "language": "Python",
        "quality": "decent",
        "code_requirements": "Create a simple function that returns 'Hello, World!'"
    }
    
    result = agent.execute(goal, inputs, context)
    
    assert result.success
    assert "generated_code" in result.outputs
    assert "def hello_world" in result.outputs["generated_code"]

def test_test_generation():
    """Test test generation functionality."""
    mock_llm = MockLLMClient()
    agent = CreatorAgent(llm_client=mock_llm)
    
    workspace = FileSystemWorkspace("/tmp/test")
    context = GlobalContext(workspace=workspace)
    
    goal = "Generate unit tests"
    inputs = {
        "test_type": "unit",
        "test_framework": "pytest",
        "code_to_test": "def hello_world(): return 'Hello, World!'"
    }
    
    result = agent.execute(goal, inputs, context)
    
    assert result.success
    assert "generated_tests" in result.outputs
    assert "test_hello_world" in result.outputs["generated_tests"]

def test_specification_creation():
    """Test specification creation functionality."""
    mock_llm = MockLLMClient()
    agent = CreatorAgent(llm_client=mock_llm)
    
    workspace = FileSystemWorkspace("/tmp/test")
    context = GlobalContext(workspace=workspace)
    
    goal = "Create a technical specification"
    inputs = {
        "clarified_requirements": "Build a simple user management API"
    }
    
    result = agent.execute(goal, inputs, context)
    
    assert result.success
    assert "technical_spec" in result.outputs
    assert "Architecture Summary" in result.outputs["technical_spec"]
    assert "data_models" in result.outputs
    assert len(result.outputs["data_models"]) > 0

def test_manifest_planning():
    """Test manifest planning functionality."""
    mock_llm = MockLLMClient()
    agent = CreatorAgent(llm_client=mock_llm)
    
    workspace = FileSystemWorkspace("/tmp/test")
    context = GlobalContext(workspace=workspace)
    
    goal = "Plan project structure"
    inputs = {
        "technical_spec": "Simple Python API with user management",
        "project_type": "web_api"
    }
    
    result = agent.execute(goal, inputs, context)
    
    assert result.success
    assert "file_manifest" in result.outputs
    assert "files_to_create" in result.outputs["file_manifest"]
    assert len(result.outputs["file_manifest"]["files_to_create"]) > 0

def test_supports_goal():
    """Test goal support checking."""
    agent = CreatorAgent()
    
    # Should support creation-related goals
    assert agent.supports_goal("create a function")
    assert agent.supports_goal("generate tests")
    assert agent.supports_goal("write documentation")
    assert agent.supports_goal("build a project")
    
    # Should not support non-creation goals
    assert not agent.supports_goal("analyze the code")
    assert not agent.supports_goal("debug the issue")

if __name__ == "__main__":
    # Run basic tests
    test_creator_agent_initialization()
    test_determine_creation_type()
    print("All CreatorAgent tests passed!")