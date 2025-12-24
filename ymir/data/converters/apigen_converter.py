"""Converter for Salesforce APIGen-MT dataset format."""

import json
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


class APIGenMTConverter(BaseConverter):
    """
    Convert Salesforce APIGen-MT format to standard trajectory format.

    APIGen-MT format:
    {
        "conversations": [
            {"from": "human", "value": "..."},
            {"from": "function_call", "value": '{"name": "...", "arguments": {...}}'},
            {"from": "observation", "value": "..."},
            {"from": "gpt", "value": "..."}
        ],
        "system": "...",
        "tools": "..."
    }
    """

    @property
    def format_name(self) -> str:
        return "apigen-mt"

    def supports_format(self, format_name: str) -> bool:
        return format_name.lower() in ["apigen", "apigen-mt", "apigen_mt", "salesforce"]

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

    def convert_record(self, data: dict) -> Trajectory | None:
        """Convert a single APIGen-MT record to a trajectory."""
        conversations = data.get("conversations", [])
        if not conversations:
            return None

        system_prompt = data.get("system", "")
        tools_str = data.get("tools", "")

        # Parse tools if present
        tools = []
        if tools_str:
            try:
                tools = json.loads(tools_str) if isinstance(tools_str, str) else tools_str
            except json.JSONDecodeError:
                # Tools might be in a different format, keep as-is
                pass

        messages = []
        for turn in conversations:
            role = turn.get("from", "")
            value = turn.get("value", "")

            if role == "human":
                messages.append(Message(role=MessageRole.USER, content=value))

            elif role == "function_call":
                # Parse the function call and wrap in <tool_call> tags
                try:
                    fc_data = json.loads(value) if isinstance(value, str) else value
                    tool_call = ToolCall(
                        name=fc_data.get("name", "unknown"),
                        arguments=fc_data.get("arguments", {}),
                    )
                    formatted_content = tool_call.format()
                    messages.append(
                        Message(
                            role=MessageRole.ASSISTANT,
                            content=formatted_content,
                            tool_calls=[tool_call],
                        )
                    )
                except (json.JSONDecodeError, TypeError):
                    # If we can't parse, wrap raw value
                    messages.append(
                        Message(
                            role=MessageRole.ASSISTANT,
                            content=f"{TOOL_CALL_START}{value}{TOOL_CALL_END}",
                        )
                    )

            elif role == "observation":
                # Tool result
                try:
                    result_data = json.loads(value) if isinstance(value, str) else value
                except (json.JSONDecodeError, TypeError):
                    result_data = {"raw": value}

                # Get the previous tool call name if available
                tool_name = "unknown"
                for msg in reversed(messages):
                    if msg.tool_calls:
                        tool_name = msg.tool_calls[0].name
                        break

                messages.append(
                    Message(
                        role=MessageRole.TOOL_RESULT,
                        content=json.dumps(result_data, indent=2) if isinstance(result_data, dict) else str(result_data),
                        tool_result=ToolResult(
                            tool_name=tool_name,
                            result=result_data if isinstance(result_data, dict) else {"result": result_data},
                            success=True,
                        ),
                    )
                )

            elif role == "gpt":
                # Final assistant response (or intermediate without tool call)
                messages.append(Message(role=MessageRole.ASSISTANT, content=value))

        if not messages:
            return None

        return Trajectory(
            scenario_description="Imported from APIGen-MT",
            tools=tools if isinstance(tools, list) else [],
            system_prompt=system_prompt,
            messages=messages,
            source="imported",
            original_source="apigen-mt",
        )
