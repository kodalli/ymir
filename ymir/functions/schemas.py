"""Schemas for function/tool definitions and scenarios."""

from typing import Any
from pydantic import BaseModel, Field


class FunctionDefinition(BaseModel):
    """Definition of a function/tool that can be called by the agent."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema format
    category: str = "general"

    def to_openai_format(self) -> dict[str, Any]:
        """Convert to OpenAI function calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    @classmethod
    def from_openai_format(cls, data: dict[str, Any], category: str = "general") -> "FunctionDefinition":
        """Create from OpenAI function calling format."""
        func = data.get("function", data)
        return cls(
            name=func["name"],
            description=func.get("description", ""),
            parameters=func.get("parameters", {"type": "object", "properties": {}}),
            category=category,
        )


class ScenarioTemplate(BaseModel):
    """Template for generating trajectories in a specific domain."""

    id: str
    name: str
    description: str
    category: str

    # Tools/functions available in this scenario
    functions: list[FunctionDefinition]

    # System prompt template (can include {variables})
    system_prompt: str

    # Example user queries for this scenario
    example_queries: list[str] = Field(default_factory=list)

    # Mock data templates for simulating tool responses
    mock_responses: dict[str, Any] = Field(default_factory=dict)

    def get_tools(self) -> list[dict[str, Any]]:
        """Get tools in OpenAI format."""
        return [f.to_openai_format() for f in self.functions]
