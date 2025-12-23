"""Core data models for trajectories, messages, and tool calls."""

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4
import json

from pydantic import BaseModel, Field

from .constants import TOOL_CALL_START, TOOL_CALL_END


class MessageRole(str, Enum):
    """Role of a message in a trajectory."""

    USER = "user"
    ASSISTANT = "assistant"
    TOOL_RESULT = "tool_result"


class TrajectoryStatus(str, Enum):
    """Status of a trajectory in the annotation workflow."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_EDIT = "needs_edit"


class ToolCall(BaseModel):
    """Represents a single tool call in the trajectory."""

    name: str
    arguments: dict[str, Any]

    def format(self) -> str:
        """Format as <tool_call>...</tool_call> for training."""
        payload = json.dumps({"name": self.name, "arguments": self.arguments})
        return f"{TOOL_CALL_START}{payload}{TOOL_CALL_END}"

    @classmethod
    def parse(cls, text: str) -> list["ToolCall"]:
        """Parse tool calls from text containing <tool_call> tags."""
        tool_calls = []
        start_tag = TOOL_CALL_START
        end_tag = TOOL_CALL_END

        pos = 0
        while True:
            start = text.find(start_tag, pos)
            if start == -1:
                break
            end = text.find(end_tag, start)
            if end == -1:
                break

            json_str = text[start + len(start_tag) : end]
            try:
                data = json.loads(json_str)
                tool_calls.append(
                    cls(name=data["name"], arguments=data.get("arguments", {}))
                )
            except (json.JSONDecodeError, KeyError):
                pass  # Skip malformed tool calls

            pos = end + len(end_tag)

        return tool_calls


class ToolResult(BaseModel):
    """Result/observation from a tool execution."""

    tool_name: str
    result: dict[str, Any]
    success: bool = True


class Message(BaseModel):
    """A single message in the trajectory."""

    role: MessageRole
    content: str
    tool_calls: list[ToolCall] | None = None
    tool_result: ToolResult | None = None


class Trajectory(BaseModel):
    """A complete multi-turn tool-calling trajectory."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # Scenario info
    scenario_id: str | None = None
    scenario_description: str

    # Tools available in this trajectory (OpenAI format)
    tools: list[dict[str, Any]]

    # The conversation
    system_prompt: str
    messages: list[Message]

    # Metadata
    source: str = "generated"  # "generated", "imported", "edited"
    original_source: str | None = None  # For imported: "apigen-mt", "hermes-fc", etc.

    # Annotation status
    status: TrajectoryStatus = TrajectoryStatus.PENDING
    quality_score: float | None = None
    annotator_notes: str | None = None
    reviewed_at: datetime | None = None

    def to_chatml(self) -> dict[str, Any]:
        """Convert trajectory to ChatML format for training export."""
        messages = [{"role": "system", "content": self.system_prompt}]

        for msg in self.messages:
            if msg.role == MessageRole.TOOL_RESULT:
                messages.append(
                    {
                        "role": "tool",
                        "content": msg.content,
                        "name": msg.tool_result.tool_name if msg.tool_result else "unknown",
                    }
                )
            else:
                messages.append({"role": msg.role.value, "content": msg.content})

        return {"messages": messages, "tools": self.tools}

    def turn_count(self) -> int:
        """Count the number of assistant turns."""
        return sum(1 for m in self.messages if m.role == MessageRole.ASSISTANT)

    def tool_call_count(self) -> int:
        """Count the total number of tool calls."""
        count = 0
        for m in self.messages:
            if m.tool_calls:
                count += len(m.tool_calls)
        return count
