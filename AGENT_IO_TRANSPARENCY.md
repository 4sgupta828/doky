# Agent I/O Transparency System

## Overview

The Agent I/O Transparency system provides clean, user-friendly visibility into agent operations. Users can now see exactly what inputs agents receive and what outputs they produce, with smart content trimming and clear visual boundaries that separate agent communications from system logging.

## Features

### 🎯 Smart Content Trimming
- **Intelligent truncation** of different content types (text, JSON, code, dicts, lists)
- **Preserves structure** while showing essential information
- **Clear truncation indicators** show exactly how much content is hidden
- **Customizable limits** for different content types

### 🎨 Clean Visual Boundaries
```
════════════════════════════════════════════════════════════════════════════════
🤖 [AGENT INPUT] CreatorAgent
🧠 Routing reason: User needs code generation based on requirements
────────────────────────────────────────────────────────────────────────────────
🎯 Goal: Create a Python function to calculate fibonacci numbers
📥 Inputs:
   file_path: fibonacci.py
   algorithm: recursive
   requirements: ["handle edge cases", "optimize for small numbers"]
────────────────────────────────────────────────────────────────────────────────
```

### 🔄 Inter-Agent Routing Visibility
- **Routing decisions** show which agent was selected and why
- **Confidence scores** indicate how certain the routing decision was
- **Clear reasoning** explains the logic behind each routing choice

### 🧠 LLM Communication Transparency
- **Prompt previews** show what questions were asked to the LLM
- **Response previews** show the LLM's decision making
- **Smart trimming** keeps the display manageable while preserving key information

## Usage

### Basic Integration

To enable I/O transparency in your agent system:

```python
from interfaces.collaboration_ui import CollaborationUI
from fagents.inter_agent_router import InterAgentRouter

# Create UI interface
ui = CollaborationUI()

# Create router with UI transparency
router = InterAgentRouter(
    llm_client=your_llm_client,
    ui_interface=ui  # Enable I/O transparency
)

# Execute workflow - users will see all I/O automatically
workflow_result = router.execute_workflow(
    user_goal="Your goal here",
    initial_inputs={"param": "value"},
    global_context=context,
    ui_interface=ui
)
```

### Individual Agent Transparency

For individual agents:

```python
from fagents.base import FoundationalAgent
from interfaces.collaboration_ui import CollaborationUI

class YourAgent(FoundationalAgent):
    def __init__(self, ui_interface=None):
        super().__init__("YourAgent", "Description", ui_interface=ui_interface)
    
    def execute(self, goal, inputs, global_context):
        # Report LLM communication if you make LLM calls
        if self.llm_client:
            response = self.llm_client.invoke(prompt)
            self.report_llm_communication(prompt, response)
        
        # Normal agent processing...
        result = self.create_result(True, "Success", outputs)
        
        # I/O reporting is automatic when used with InterAgentRouter
        return result
```

## Content Trimming Examples

The system handles various content types intelligently:

### Text Content
```
Long text gets trimmed at word boundaries...
(+1247 more chars)
```

### JSON/Dict Data
```
{
  users: list(15 items)
  metadata: dict(3 items)
  config: "production_settings"
}
... (+5 more items)
```

### Code Content
```python
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# More code here...
(+23 more lines)
```

### Lists
```
[
  [0] "Item 0 with description"
  [1] "Item 1 with description"
  [2] "Item 2 with description"
]
... (+47 more items)
```

## Configuration

### Custom Trimming Limits

```python
from utils.content_trimmer import ContentTrimmer

# Create trimmer with custom limits
custom_trimmer = ContentTrimmer({
    'text': 600,      # More text content
    'json': 800,      # More JSON content
    'code': 1000,     # More code content
    'dict_summary': 400
})

result = custom_trimmer.trim_content(your_content, "auto")
```

### UI Display Customization

The CollaborationUI provides methods for different types of displays:

```python
ui = CollaborationUI()

# Individual displays
ui.display_agent_input(agent_name, goal, inputs, reasoning)
ui.display_agent_output(agent_name, success, message, outputs)
ui.display_routing_decision(from_agent, to_agent, confidence, reasoning)
ui.display_llm_communication(agent_name, prompt_preview, response_preview)
```

## Integration Points

### 1. InterAgentRouter
- Automatically shows routing decisions
- Displays input/output for each agent execution
- Shows LLM communication for routing decisions

### 2. FoundationalAgent Base Class  
- Provides `report_llm_communication()` for LLM transparency
- Provides `report_agent_io()` for manual I/O reporting
- UI interface is passed through constructor

### 3. CollaborationUI
- New display methods for clean formatting
- Smart content trimming integration
- Consistent visual styling with existing UI

## Benefits

### For Users
- **Complete visibility** into what agents are doing
- **Clear understanding** of routing decisions and reasoning
- **Easy debugging** when things go wrong
- **Confidence** in agent operations through transparency

### For Developers
- **Better debugging** with full I/O visibility
- **Easy integration** with minimal code changes
- **Customizable display** for different content types
- **Backward compatibility** with existing agents

## Testing

Run the test suite to verify the implementation:

```bash
python test_agent_io_transparency.py
```

This will test:
- Content trimming for various data types
- UI display methods with different content
- Agent execution with I/O transparency
- Inter-agent routing with full visibility

## Future Enhancements

Potential improvements for the system:
- **Filtering options** to show/hide certain types of I/O
- **Export capabilities** to save agent conversations
- **Search functionality** to find specific agent interactions
- **Performance metrics** display (execution times, token usage)
- **Interactive exploration** of truncated content