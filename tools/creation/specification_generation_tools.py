# tools/creation/specification_generation_tools.py
"""
Specification generation tools for creating technical specifications from requirements.
Extracted from SpecGeneratorAgent to provide atomic, reusable specification generation capabilities.
"""

import json
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

class SpecificationType(Enum):
    """Types of specifications that can be generated."""
    TECHNICAL_SPEC = "technical_spec"
    API_SPEC = "api_spec"
    DATA_MODEL_SPEC = "data_model_spec"
    ARCHITECTURE_SPEC = "architecture_spec"

class SpecificationStyle(Enum):
    """Different styles of specification output."""
    DETAILED = "detailed"
    MINIMAL = "minimal"
    STRUCTURED = "structured"
    COMPREHENSIVE = "comprehensive"

@dataclass
class SpecificationContext:
    """Context for specification generation."""
    requirements: str
    spec_type: SpecificationType = SpecificationType.TECHNICAL_SPEC
    style: SpecificationStyle = SpecificationStyle.DETAILED
    target_stack: str = "Python/FastAPI"
    existing_files: List[str] = None
    domain: str = ""
    constraints: List[str] = None
    include_examples: bool = True
    
    def __post_init__(self):
        if self.existing_files is None:
            self.existing_files = []
        if self.constraints is None:
            self.constraints = []

@dataclass
class DataModel:
    """Represents a data model in the specification."""
    name: str
    description: str
    fields: Dict[str, str]

@dataclass
class APIEndpoint:
    """Represents an API endpoint in the specification."""
    method: str
    path: str
    description: str
    request_schema: str = ""
    response_schema: str = ""

@dataclass
class SpecificationResult:
    """Result of specification generation."""
    success: bool
    specification_content: str
    structured_data: Dict[str, Any]
    message: str
    data_models: List[DataModel] = None
    api_endpoints: List[APIEndpoint] = None
    
    def __post_init__(self):
        if self.data_models is None:
            self.data_models = []
        if self.api_endpoints is None:
            self.api_endpoints = []

def build_specification_prompt(context: SpecificationContext) -> str:
    """Build a comprehensive prompt for specification generation."""
    
    # Base prompt based on specification type
    if context.spec_type == SpecificationType.TECHNICAL_SPEC:
        base_prompt = _build_technical_spec_prompt(context)
    elif context.spec_type == SpecificationType.API_SPEC:
        base_prompt = _build_api_spec_prompt(context)
    elif context.spec_type == SpecificationType.DATA_MODEL_SPEC:
        base_prompt = _build_data_model_spec_prompt(context)
    elif context.spec_type == SpecificationType.ARCHITECTURE_SPEC:
        base_prompt = _build_architecture_spec_prompt(context)
    else:
        base_prompt = _build_technical_spec_prompt(context)
    
    return base_prompt

def _build_technical_spec_prompt(context: SpecificationContext) -> str:
    """Build prompt for technical specification generation."""
    context_summary = {"files_in_workspace": context.existing_files}
    
    return f"""
    **Persona**: You are an expert software architect and tech lead. Your task is to convert a set of user requirements into a detailed, actionable technical specification that another AI agent can use to write code.

    **Overall Goal**: Analyze the user's requirements and the project context to produce a comprehensive technical specification in a structured JSON format.

    **Context**:
    - **User Requirements / Goal**: 
      ---
      {context.requirements}
      ---
    - **Target Stack**: {context.target_stack}
    - **Domain**: {context.domain}
    - **Current Project State**: 
      {json.dumps(context_summary, indent=2)}
    - **Constraints**: {', '.join(context.constraints) if context.constraints else 'None specified'}

    **Your Task**:
    1. **Deconstruct Requirements**: Break down the user's request into concrete technical components.
    2. **Define Data Structures**: Specify the necessary data models, including fields and types.
    3. **Outline API Endpoints**: If it's a web application, define the API endpoints, methods, and expected request/response schemas.
    4. **Specify Core Logic**: Describe the key business logic, algorithms, or processes that need to be implemented.
    5. **Make Technical Decisions**: Propose a technology stack and justify your choices.

    **Constraints**:
    - **Be Specific and Unambiguous**: The output must be clear enough for another AI agent to write code from it directly. Avoid vague language.
    - **Style**: Use {context.style.value} approach
    - **Stack**: Assume {context.target_stack} unless otherwise specified

    **Output Format**: You MUST return a single, valid JSON object with the following structure:
    {{
        "architecture_summary": "A high-level description of the proposed system architecture.",
        "data_models": [
            {{
                "name": "ModelName",
                "description": "What this data model represents.",
                "fields": {{
                    "field_name": "data_type (e.g., string, integer, datetime)"
                }}
            }}
        ],
        "api_endpoints": [
            {{
                "method": "GET|POST|PUT|DELETE",
                "path": "/api/v1/resource",
                "description": "The purpose of this endpoint.",
                "request_schema": "Description of the expected request body or parameters.",
                "response_schema": "Description of the expected success response body."
            }}
        ],
        "core_business_logic": [
            "A list of key business rules or logic to be implemented."
        ],
        "technical_decisions": {{
            "database": "Your recommended database and a brief rationale.",
            "framework": "Your recommended web framework and rationale."
        }}
    }}
    """

def _build_api_spec_prompt(context: SpecificationContext) -> str:
    """Build prompt for API specification generation."""
    return f"""
    **Persona**: You are an expert API architect. Create a detailed API specification from the requirements.

    **Requirements**: {context.requirements}
    **Target Stack**: {context.target_stack}
    **Style**: {context.style.value}

    **Output Format**: Return a JSON object focused on API endpoints:
    {{
        "api_version": "v1",
        "base_url": "/api/v1",
        "endpoints": [
            {{
                "method": "HTTP_METHOD",
                "path": "/endpoint/path",
                "description": "What this endpoint does",
                "parameters": {{}},
                "request_body": {{}},
                "responses": {{}}
            }}
        ],
        "authentication": "Authentication method",
        "error_handling": "Error response format"
    }}
    """

def _build_data_model_spec_prompt(context: SpecificationContext) -> str:
    """Build prompt for data model specification generation."""
    return f"""
    **Persona**: You are a database architect. Create detailed data model specifications.

    **Requirements**: {context.requirements}
    **Target Stack**: {context.target_stack}
    **Style**: {context.style.value}

    **Output Format**: Return a JSON object focused on data models:
    {{
        "models": [
            {{
                "name": "ModelName",
                "description": "Purpose of this model",
                "fields": {{
                    "field_name": {{
                        "type": "data_type",
                        "required": true,
                        "description": "Field purpose"
                    }}
                }},
                "relationships": [],
                "indexes": [],
                "constraints": []
            }}
        ]
    }}
    """

def _build_architecture_spec_prompt(context: SpecificationContext) -> str:
    """Build prompt for architecture specification generation."""
    return f"""
    **Persona**: You are a system architect. Create a high-level architecture specification.

    **Requirements**: {context.requirements}
    **Target Stack**: {context.target_stack}
    **Style**: {context.style.value}

    **Output Format**: Return a JSON object focused on architecture:
    {{
        "architecture_overview": "High-level system description",
        "components": [
            {{
                "name": "ComponentName",
                "responsibility": "What it does",
                "dependencies": []
            }}
        ],
        "data_flow": "How data moves through the system",
        "deployment": "Deployment considerations",
        "scalability": "Scalability considerations"
    }}
    """

def compile_specification_document(spec_data: Dict[str, Any], style: SpecificationStyle = SpecificationStyle.DETAILED) -> str:
    """Compile specification data into a markdown document."""
    content = "# Technical Specification\n\n"
    
    # Architecture summary
    content += f"## Architecture Summary\n\n{spec_data.get('architecture_summary', 'N/A')}\n\n"
    
    # Data models
    if spec_data.get('data_models'):
        content += "## Data Models\n\n"
        for model in spec_data['data_models']:
            content += f"### {model.get('name', 'Unknown Model')}\n\n"
            content += f"{model.get('description', '')}\n\n"
            if model.get('fields'):
                content += "**Fields:**\n"
                for field, f_type in model['fields'].items():
                    content += f"- `{field}`: {f_type}\n"
                content += "\n"
    
    # API endpoints
    if spec_data.get('api_endpoints'):
        content += "## API Endpoints\n\n"
        for endpoint in spec_data['api_endpoints']:
            content += f"### `{endpoint.get('method', 'GET')} {endpoint.get('path', '/unknown')}`\n\n"
            content += f"**Description**: {endpoint.get('description', '')}\n\n"
            if endpoint.get('request_schema'):
                content += f"**Request**: {endpoint['request_schema']}\n\n"
            if endpoint.get('response_schema'):
                content += f"**Response**: {endpoint['response_schema']}\n\n"
    
    # Core business logic
    if spec_data.get('core_business_logic'):
        content += "## Core Business Logic\n\n"
        for rule in spec_data['core_business_logic']:
            content += f"- {rule}\n"
        content += "\n"
    
    # Technical decisions
    if spec_data.get('technical_decisions'):
        content += "## Technical Decisions\n\n"
        for key, value in spec_data['technical_decisions'].items():
            content += f"- **{key.title()}:** {value}\n"
        content += "\n"
    
    return content

def parse_specification_data(spec_data: Dict[str, Any]) -> tuple[List[DataModel], List[APIEndpoint]]:
    """Parse specification data into structured objects."""
    data_models = []
    api_endpoints = []
    
    # Parse data models
    if spec_data.get('data_models'):
        for model_data in spec_data['data_models']:
            data_model = DataModel(
                name=model_data.get('name', ''),
                description=model_data.get('description', ''),
                fields=model_data.get('fields', {})
            )
            data_models.append(data_model)
    
    # Parse API endpoints
    if spec_data.get('api_endpoints'):
        for endpoint_data in spec_data['api_endpoints']:
            api_endpoint = APIEndpoint(
                method=endpoint_data.get('method', 'GET'),
                path=endpoint_data.get('path', '/'),
                description=endpoint_data.get('description', ''),
                request_schema=endpoint_data.get('request_schema', ''),
                response_schema=endpoint_data.get('response_schema', '')
            )
            api_endpoints.append(api_endpoint)
    
    return data_models, api_endpoints

def generate_specification(context: SpecificationContext, llm_client=None) -> SpecificationResult:
    """Generate a technical specification from requirements."""
    if llm_client is None:
        return SpecificationResult(
            success=False,
            specification_content="",
            structured_data={},
            message="No LLM client provided"
        )
    
    try:
        # Build prompt
        prompt = build_specification_prompt(context)
        
        # Get LLM response
        logger.info(f"Generating {context.spec_type.value} specification")
        response_str = llm_client.invoke(prompt)
        
        # Parse JSON response
        spec_data = json.loads(response_str)
        
        # Validate response structure
        if not isinstance(spec_data, dict):
            raise ValueError("LLM response is not a valid JSON object")
        
        if context.spec_type == SpecificationType.TECHNICAL_SPEC:
            if "architecture_summary" not in spec_data:
                raise ValueError("Technical spec missing required 'architecture_summary' field")
        
        # Compile markdown document
        specification_content = compile_specification_document(spec_data, context.style)
        
        # Parse structured data
        data_models, api_endpoints = parse_specification_data(spec_data)
        
        logger.info(f"Successfully generated specification with {len(data_models)} models and {len(api_endpoints)} endpoints")
        
        return SpecificationResult(
            success=True,
            specification_content=specification_content,
            structured_data=spec_data,
            message=f"Successfully generated {context.spec_type.value}",
            data_models=data_models,
            api_endpoints=api_endpoints
        )
        
    except json.JSONDecodeError as e:
        error_msg = f"Failed to parse LLM response as JSON: {e}"
        logger.error(error_msg)
        return SpecificationResult(
            success=False,
            specification_content="",
            structured_data={},
            message=error_msg
        )
        
    except ValueError as e:
        error_msg = f"Invalid specification format: {e}"
        logger.error(error_msg)
        return SpecificationResult(
            success=False,
            specification_content="",
            structured_data={},
            message=error_msg
        )
        
    except Exception as e:
        error_msg = f"Unexpected error generating specification: {e}"
        logger.error(error_msg)
        return SpecificationResult(
            success=False,
            specification_content="",
            structured_data={},
            message=error_msg
        )

def validate_specification(spec_data: Dict[str, Any], spec_type: SpecificationType) -> tuple[bool, str]:
    """Validate a generated specification for completeness and correctness."""
    errors = []
    
    if spec_type == SpecificationType.TECHNICAL_SPEC:
        required_fields = ["architecture_summary", "data_models", "api_endpoints", "core_business_logic", "technical_decisions"]
        for field in required_fields:
            if field not in spec_data:
                errors.append(f"Missing required field: {field}")
        
        # Validate data models structure
        if "data_models" in spec_data and isinstance(spec_data["data_models"], list):
            for i, model in enumerate(spec_data["data_models"]):
                if not isinstance(model, dict):
                    errors.append(f"Data model {i} is not a dictionary")
                elif "name" not in model or "fields" not in model:
                    errors.append(f"Data model {i} missing name or fields")
        
        # Validate API endpoints structure
        if "api_endpoints" in spec_data and isinstance(spec_data["api_endpoints"], list):
            for i, endpoint in enumerate(spec_data["api_endpoints"]):
                if not isinstance(endpoint, dict):
                    errors.append(f"API endpoint {i} is not a dictionary")
                elif "method" not in endpoint or "path" not in endpoint:
                    errors.append(f"API endpoint {i} missing method or path")
    
    is_valid = len(errors) == 0
    message = "Specification is valid" if is_valid else f"Validation errors: {'; '.join(errors)}"
    
    return is_valid, message