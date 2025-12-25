"""Pydantic models for scenario management system."""

from datetime import datetime
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


# ============================================================================
# Tool Models
# ============================================================================


class Tool(BaseModel):
    """Represents a tool/function that can be called in a scenario."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    category: str | None = None
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

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


class ToolCreate(BaseModel):
    """Schema for creating a new tool."""

    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    category: str | None = None


class ToolUpdate(BaseModel):
    """Schema for updating a tool."""

    name: str | None = None
    description: str | None = None
    parameters: dict[str, Any] | None = None
    category: str | None = None
    is_active: bool | None = None


# ============================================================================
# Scenario Models
# ============================================================================


class Scenario(BaseModel):
    """Represents a conversation scenario/domain."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: str
    category: str | None = None
    system_prompt: str
    example_queries: list[str] = Field(default_factory=list)
    mock_responses: dict[str, Any] = Field(default_factory=dict)
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # Computed field (not stored)
    tool_count: int = 0


class ScenarioCreate(BaseModel):
    """Schema for creating a new scenario."""

    id: str | None = None  # Optional override
    name: str
    description: str
    category: str | None = None
    system_prompt: str
    example_queries: list[str] = Field(default_factory=list)
    mock_responses: dict[str, Any] = Field(default_factory=dict)
    tool_ids: list[str] = Field(default_factory=list)


class ScenarioUpdate(BaseModel):
    """Schema for updating a scenario."""

    name: str | None = None
    description: str | None = None
    category: str | None = None
    system_prompt: str | None = None
    example_queries: list[str] | None = None
    mock_responses: dict[str, Any] | None = None
    is_active: bool | None = None


class ScenarioWithTools(Scenario):
    """Scenario with populated tools for joined queries."""

    tools: list[Tool] = Field(default_factory=list)


# ============================================================================
# Actor Models
# ============================================================================


class Actor(BaseModel):
    """Represents a user persona/actor in a scenario."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    icon: str | None = None
    background: str
    goal: str
    tags: list[str] = Field(default_factory=list)
    category: str | None = None
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class ActorCreate(BaseModel):
    """Schema for creating a new actor."""

    id: str | None = None  # Optional override
    name: str
    icon: str | None = None
    background: str
    goal: str
    tags: list[str] = Field(default_factory=list)
    category: str | None = None


class ActorUpdate(BaseModel):
    """Schema for updating an actor."""

    name: str | None = None
    icon: str | None = None
    background: str | None = None
    goal: str | None = None
    tags: list[str] | None = None
    category: str | None = None
    is_active: bool | None = None


# ============================================================================
# ToolPreset Models
# ============================================================================


class ToolPreset(BaseModel):
    """Represents a predefined set of tools for a scenario."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    scenario_id: str
    name: str
    description: str
    tool_ids: list[str] = Field(default_factory=list)
    is_default: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # Optional joined data (not stored)
    tools: list[Tool] | None = None


class ToolPresetCreate(BaseModel):
    """Schema for creating a new tool preset."""

    scenario_id: str
    name: str
    description: str
    tool_ids: list[str] = Field(default_factory=list)
    is_default: bool = False


class ToolPresetUpdate(BaseModel):
    """Schema for updating a tool preset."""

    name: str | None = None
    description: str | None = None
    tool_ids: list[str] | None = None
    is_default: bool | None = None


# ============================================================================
# GenerationTemplate Models
# ============================================================================


class GenerationTemplate(BaseModel):
    """Represents a saved generation configuration template."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: str
    scenario_id: str
    actor_id: str | None = None
    tool_preset_id: str | None = None
    model: str
    temperature: float = 0.7
    is_favorite: bool = False
    use_count: int = 0
    last_used_at: datetime | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # Optional joined data (not stored)
    scenario: Scenario | None = None
    actor: Actor | None = None
    tool_preset: ToolPreset | None = None


class GenerationTemplateCreate(BaseModel):
    """Schema for creating a new generation template."""

    name: str
    description: str
    scenario_id: str
    actor_id: str | None = None
    tool_preset_id: str | None = None
    model: str
    temperature: float = 0.7


class GenerationTemplateUpdate(BaseModel):
    """Schema for updating a generation template."""

    name: str | None = None
    description: str | None = None
    scenario_id: str | None = None
    actor_id: str | None = None
    tool_preset_id: str | None = None
    model: str | None = None
    temperature: float | None = None
    is_favorite: bool | None = None
