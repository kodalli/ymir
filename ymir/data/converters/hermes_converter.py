"""Converter for NousResearch Hermes Function Calling dataset format."""

import json
import re
from typing import Iterator

from loguru import logger

from ymir.core import (
    Message,
    MessageRole,
    ToolCall,
    ToolResult,
    Trajectory,
    TOOL_CALL_START,
    TOOL_CALL_END,
)
from .base_converter import BaseConverter


class HermesFCConverter(BaseConverter):
    """
    Convert Hermes Function Calling format to standard trajectory format.

    Hermes FC format:
    {
        "conversations": [
            {"from": "system", "value": "..."},
            {"from": "human", "value": "..."},
            {"from": "gpt", "value": "...<tool_call>...</tool_call>..."},
            {"from": "tool", "value": "..."},
            {"from": "gpt", "value": "..."}
        ]
    }

    Also handles:
    - interstellarninja/tool-calls-multiturn (same format)
    - <functioncall> tags (normalized to <tool_call>)
    """

    @property
    def format_name(self) -> str:
        return "hermes-fc"

    def supports_format(self, format_name: str) -> bool:
        return format_name.lower() in [
            "hermes",
            "hermes-fc",
            "hermes_fc",
            "multiturn",
            "multiturn_tools",
            "tool-calls-multiturn",
        ]

    def convert_file(self, input_path: str) -> Iterator[Trajectory]:
        """Convert a JSONL file to trajectories."""
        with open(input_path, "r") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    trajectory = self.convert_record(data)
                    if trajectory:
                        yield trajectory
                except json.JSONDecodeError as e:
                    logger.warning(f"Line {line_num}: Failed to parse JSON: {e}")
                except Exception as e:
                    logger.warning(f"Line {line_num}: Conversion error: {e}")

    def _normalize_tool_calls(self, text: str) -> str:
        """Normalize <functioncall> tags to <tool_call> format."""
        # Pattern for <functioncall> ... </functioncall>
        functioncall_pattern = r"<functioncall>\s*(\{.*?\})\s*</functioncall>"

        def convert_functioncall(match):
            try:
                json_str = match.group(1).strip()
                data = json.loads(json_str)
                name = data.get("name", "")
                arguments = data.get("arguments", {})
                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        pass
                normalized = {"name": name, "arguments": arguments}
                return f"{TOOL_CALL_START}{json.dumps(normalized)}{TOOL_CALL_END}"
            except json.JSONDecodeError:
                return f"{TOOL_CALL_START}{match.group(1)}{TOOL_CALL_END}"

        result = re.sub(
            functioncall_pattern, convert_functioncall, text, flags=re.DOTALL
        )
        return result

    def _extract_tool_calls(self, text: str) -> list[ToolCall]:
        """Extract tool calls from text."""
        # First normalize any <functioncall> tags
        normalized = self._normalize_tool_calls(text)
        return ToolCall.parse(normalized)

    def convert_record(self, data: dict) -> Trajectory | None:
        """Convert a single Hermes FC record to a trajectory."""
        conversations = data.get("conversations", [])
        if not conversations:
            return None

        system_prompt = ""
        messages = []
        tools = []

        for turn in conversations:
            role = turn.get("from", turn.get("role", ""))
            content = turn.get("value", turn.get("content", ""))

            if role == "system":
                # Extract tools from system prompt if present
                system_prompt = content
                # Try to parse tools from system message
                tools = self._extract_tools_from_system(content)

            elif role == "human":
                messages.append(Message(role=MessageRole.USER, content=content))

            elif role == "gpt":
                # Normalize and extract tool calls
                normalized_content = self._normalize_tool_calls(content)
                tool_calls = self._extract_tool_calls(content)

                messages.append(
                    Message(
                        role=MessageRole.ASSISTANT,
                        content=normalized_content,
                        tool_calls=tool_calls if tool_calls else None,
                    )
                )

            elif role == "tool":
                # Tool result
                try:
                    result_data = json.loads(content) if content.strip().startswith("{") else {"result": content}
                except (json.JSONDecodeError, TypeError):
                    result_data = {"result": content}

                # Get the previous tool call name if available
                tool_name = "unknown"
                for msg in reversed(messages):
                    if msg.tool_calls:
                        tool_name = msg.tool_calls[0].name
                        break

                messages.append(
                    Message(
                        role=MessageRole.TOOL_RESULT,
                        content=content,
                        tool_result=ToolResult(
                            tool_name=tool_name,
                            result=result_data,
                            success=True,
                        ),
                    )
                )

        if not messages:
            return None

        return Trajectory(
            scenario_description="Imported from Hermes FC",
            tools=tools,
            system_prompt=system_prompt,
            messages=messages,
            source="imported",
            original_source="hermes-fc",
        )

    def _extract_tools_from_system(self, system_prompt: str) -> list[dict]:
        """Try to extract tool definitions from system prompt."""
        tools = []

        # Look for JSON tool definitions
        try:
            # Try to find JSON array of tools
            match = re.search(r"\[[\s\S]*?\{[\s\S]*?\"name\"[\s\S]*?\}[\s\S]*?\]", system_prompt)
            if match:
                tools_json = match.group()
                parsed = json.loads(tools_json)
                if isinstance(parsed, list):
                    tools = parsed
        except (json.JSONDecodeError, TypeError):
            pass

        return tools
