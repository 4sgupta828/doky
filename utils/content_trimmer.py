"""
Smart content trimming utility for agent I/O display.

Provides intelligent truncation of various content types while preserving
structure and readability for user transparency in agent communications.
"""

import json
from typing import Any, Dict, Union
from dataclasses import dataclass


@dataclass
class TrimResult:
    """Result of content trimming operation."""
    content: str
    was_truncated: bool
    original_length: int
    display_length: int
    content_type: str
    truncation_info: str = ""


class ContentTrimmer:
    """Smart content trimming for various data types."""
    
    # Default limits for different content types - increased for better transparency
    DEFAULT_LIMITS = {
        'text': 1200,        # Increased from 400
        'json': 1500,        # Increased from 600  
        'code': 2000,        # Increased from 800
        'dict_summary': 800, # Increased from 300
        'list_summary': 600  # Increased from 200
    }
    
    DEFAULT_LINE_LIMITS = {
        'text': 40,          # Increased from 15
        'json': 50,          # Increased from 20
        'code': 80,          # Increased from 30
        'dict_keys': 25,     # Increased from 10
        'list_items': 20     # Increased from 10
    }
    
    def __init__(self, custom_limits: Dict[str, int] = None):
        """Initialize with optional custom limits."""
        self.limits = {**self.DEFAULT_LIMITS, **(custom_limits or {})}
    
    def trim_content(self, content: Any, content_type: str = "auto") -> TrimResult:
        """
        Smart trim any content type.
        
        Args:
            content: Content to trim
            content_type: Type hint ("auto", "text", "json", "code", "dict", "list")
            
        Returns:
            TrimResult with trimmed content and metadata
        """
        if content is None:
            return TrimResult("None", False, 4, 4, "none")
        
        # Auto-detect content type if needed
        if content_type == "auto":
            content_type = self._detect_content_type(content)
        
        # Route to appropriate trimming method
        if content_type == "dict":
            return self._trim_dict(content)
        elif content_type == "list":
            return self._trim_list(content)
        elif content_type == "json":
            return self._trim_json(content)
        elif content_type == "code":
            return self._trim_code(content)
        else:
            return self._trim_text(str(content))
    
    def _detect_content_type(self, content: Any) -> str:
        """Auto-detect the most appropriate content type."""
        if isinstance(content, dict):
            return "dict"
        elif isinstance(content, list):
            return "list"
        elif isinstance(content, str):
            # Check if it looks like JSON
            if content.strip().startswith(('{', '[')):
                try:
                    json.loads(content)
                    return "json"
                except:
                    pass
            
            # Check if it looks like code (has typical code patterns)
            if self._looks_like_code(content):
                return "code"
                
            return "text"
        else:
            return "text"
    
    def _looks_like_code(self, text: str) -> bool:
        """Heuristic to detect if text looks like code."""
        code_indicators = [
            'def ', 'class ', 'import ', 'from ',  # Python
            'function ', 'const ', 'let ', 'var ',  # JavaScript  
            'public class', 'private ', 'package ',  # Java
            '#include', 'int main', 'void ',  # C/C++
            '#!/bin/', '#!/usr/bin/'  # Scripts
        ]
        
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in code_indicators)
    
    def _trim_text(self, text: str) -> TrimResult:
        """Trim plain text content."""
        lines = text.split('\n')
        char_limit = self.limits['text']
        line_limit = self.DEFAULT_LINE_LIMITS['text']
        
        original_length = len(text)
        
        # If short enough, return as-is
        if len(text) <= char_limit and len(lines) <= line_limit:
            return TrimResult(text, False, original_length, len(text), "text")
        
        # Trim by lines first, then by chars
        if len(lines) > line_limit:
            display_lines = lines[:line_limit]
            remaining_lines = len(lines) - line_limit
            
            trimmed_text = '\n'.join(display_lines)
            truncation_info = f"(+{remaining_lines} more lines)"
        else:
            trimmed_text = text
            truncation_info = ""
        
        # Then trim by characters if still too long
        if len(trimmed_text) > char_limit:
            # Find a good break point (end of word/line)
            break_point = char_limit
            while break_point > char_limit - 50 and break_point < len(trimmed_text):
                if trimmed_text[break_point] in [' ', '\n', '.', ',', ';']:
                    break
                break_point += 1
            
            if break_point >= len(trimmed_text):
                break_point = char_limit
            
            trimmed_text = trimmed_text[:break_point]
            remaining_chars = original_length - len(trimmed_text)
            
            if truncation_info:
                truncation_info = f"(+{remaining_chars} more chars, {truncation_info.strip('()')}"
            else:
                truncation_info = f"(+{remaining_chars} more chars)"
        
        return TrimResult(
            trimmed_text,
            True,
            original_length,
            len(trimmed_text),
            "text",
            truncation_info
        )
    
    def _trim_json(self, content: Union[str, dict, list]) -> TrimResult:
        """Trim JSON content with pretty formatting."""
        if isinstance(content, str):
            try:
                parsed = json.loads(content)
            except:
                return self._trim_text(content)  # Fallback to text
        else:
            parsed = content
        
        try:
            formatted = json.dumps(parsed, indent=2, ensure_ascii=False)
        except:
            return self._trim_text(str(content))  # Fallback
        
        lines = formatted.split('\n')
        line_limit = self.DEFAULT_LINE_LIMITS['json']
        char_limit = self.limits['json']
        
        original_length = len(formatted)
        
        if len(formatted) <= char_limit and len(lines) <= line_limit:
            return TrimResult(formatted, False, original_length, len(formatted), "json")
        
        # Trim by lines first
        if len(lines) > line_limit:
            display_lines = lines[:line_limit]
            remaining_lines = len(lines) - line_limit
            
            # Add proper JSON closing if we cut off in the middle
            trimmed_text = '\n'.join(display_lines)
            if not trimmed_text.rstrip().endswith(('}', ']')):
                # Find the appropriate closing bracket
                open_braces = trimmed_text.count('{') - trimmed_text.count('}')
                open_brackets = trimmed_text.count('[') - trimmed_text.count(']')
                
                closers = '}' * open_braces + ']' * open_brackets
                if closers:
                    trimmed_text += '\n' + closers
            
            truncation_info = f"(+{remaining_lines} more lines)"
        else:
            trimmed_text = formatted
            truncation_info = ""
        
        return TrimResult(
            trimmed_text,
            len(lines) > line_limit or len(formatted) > char_limit,
            original_length,
            len(trimmed_text),
            "json",
            truncation_info
        )
    
    def _trim_code(self, code: str) -> TrimResult:
        """Trim code content while preserving structure."""
        lines = code.split('\n')
        line_limit = self.DEFAULT_LINE_LIMITS['code']
        char_limit = self.limits['code']
        
        original_length = len(code)
        
        if len(code) <= char_limit and len(lines) <= line_limit:
            return TrimResult(code, False, original_length, len(code), "code")
        
        # For code, prioritize line-based trimming
        if len(lines) > line_limit:
            display_lines = lines[:line_limit]
            remaining_lines = len(lines) - line_limit
            
            trimmed_text = '\n'.join(display_lines)
            truncation_info = f"(+{remaining_lines} more lines)"
        else:
            trimmed_text = code
            truncation_info = ""
        
        return TrimResult(
            trimmed_text,
            len(lines) > line_limit,
            original_length,
            len(trimmed_text),
            "code", 
            truncation_info
        )
    
    def _trim_dict(self, data: dict) -> TrimResult:
        """Trim dictionary by showing key-value summary."""
        if not data:
            return TrimResult("{}", False, 2, 2, "dict")
        
        key_limit = self.DEFAULT_LINE_LIMITS['dict_keys']
        char_limit = self.limits['dict_summary']
        
        # Create key-value summary
        lines = []
        total_items = len(data)
        
        for i, (key, value) in enumerate(data.items()):
            if i >= key_limit:
                break
                
            # Summarize the value
            value_summary = self._summarize_value(value)
            line = f"  {key}: {value_summary}"
            
            # Check if adding this line would exceed char limit
            current_length = sum(len(l) for l in lines) + len(line)
            if current_length > char_limit and lines:
                break
                
            lines.append(line)
        
        # Format the result
        if lines:
            content = "{\n" + "\n".join(lines) + "\n}"
        else:
            content = "{...}"
        
        # Add truncation info
        shown_items = len(lines)
        if shown_items < total_items:
            remaining = total_items - shown_items
            content += f"\n... (+{remaining} more items)"
        
        return TrimResult(
            content,
            shown_items < total_items,
            len(str(data)),
            len(content),
            "dict",
            f"(showing {shown_items}/{total_items} items)" if shown_items < total_items else ""
        )
    
    def _trim_list(self, data: list) -> TrimResult:
        """Trim list by showing item summary."""
        if not data:
            return TrimResult("[]", False, 2, 2, "list")
        
        item_limit = self.DEFAULT_LINE_LIMITS['list_items']
        char_limit = self.limits['list_summary']
        
        lines = []
        total_items = len(data)
        
        for i, item in enumerate(data):
            if i >= item_limit:
                break
                
            item_summary = self._summarize_value(item)
            line = f"  [{i}] {item_summary}"
            
            # Check char limit
            current_length = sum(len(l) for l in lines) + len(line)
            if current_length > char_limit and lines:
                break
                
            lines.append(line)
        
        # Format result
        if lines:
            content = "[\n" + "\n".join(lines) + "\n]"
        else:
            content = "[...]"
        
        # Add truncation info
        shown_items = len(lines)
        if shown_items < total_items:
            remaining = total_items - shown_items
            content += f"\n... (+{remaining} more items)"
        
        return TrimResult(
            content,
            shown_items < total_items,
            len(str(data)),
            len(content),
            "list",
            f"(showing {shown_items}/{total_items} items)" if shown_items < total_items else ""
        )
    
    def _summarize_value(self, value: Any, max_length: int = 60) -> str:
        """Create a brief summary of a value."""
        if value is None:
            return "None"
        elif isinstance(value, (int, float, bool)):
            return str(value)
        elif isinstance(value, str):
            if len(value) <= max_length:
                return f'"{value}"'
            else:
                return f'"{value[:max_length-3]}..."'
        elif isinstance(value, dict):
            return f"dict({len(value)} items)"
        elif isinstance(value, list):
            return f"list({len(value)} items)"
        else:
            str_val = str(value)
            if len(str_val) <= max_length:
                return str_val
            else:
                return f"{str_val[:max_length-3]}..."


# Global instance for convenient access
default_trimmer = ContentTrimmer()


def trim_content(content: Any, content_type: str = "auto", 
                custom_limits: Dict[str, int] = None) -> TrimResult:
    """
    Convenience function for trimming content.
    
    Args:
        content: Content to trim
        content_type: Type hint or "auto" for auto-detection
        custom_limits: Optional custom limits for this operation
        
    Returns:
        TrimResult with trimmed content and metadata
    """
    if custom_limits:
        trimmer = ContentTrimmer(custom_limits)
        return trimmer.trim_content(content, content_type)
    else:
        return default_trimmer.trim_content(content, content_type)


# Self-testing
if __name__ == "__main__":
    # Test cases
    trimmer = ContentTrimmer()
    
    # Test long text
    long_text = "This is a very long text. " * 50
    result = trimmer.trim_content(long_text, "text")
    print(f"Long text: {result.was_truncated}, {result.truncation_info}")
    
    # Test dict
    big_dict = {f"key_{i}": f"value_{i} with some content" for i in range(20)}
    result = trimmer.trim_content(big_dict, "dict") 
    print(f"Dict: {result.was_truncated}, {result.truncation_info}")
    
    # Test JSON
    json_data = {"users": [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}] * 10}
    result = trimmer.trim_content(json_data, "json")
    print(f"JSON: {result.was_truncated}, {result.truncation_info}")
    
    print("Content trimmer tests completed!")