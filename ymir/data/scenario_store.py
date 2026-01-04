"""Data access layer for managing scenarios, tools, actors, presets, and templates."""

import json
from datetime import datetime
from typing import Any
from uuid import uuid4

from loguru import logger

from ymir.core.scenario_schemas import (
    Actor,
    ActorCreate,
    ActorUpdate,
    GenerationTemplate,
    GenerationTemplateCreate,
    GenerationTemplateUpdate,
    Scenario,
    ScenarioCreate,
    ScenarioUpdate,
    ScenarioWithTools,
    Tool,
    ToolCreate,
    ToolPreset,
    ToolPresetCreate,
    ToolPresetUpdate,
    ToolUpdate,
)

from .database import Database


class ScenarioStore:
    """Data access layer for scenario management system."""

    def __init__(self, db: Database):
        self.db = db

    # ========================================================================
    # Tool Methods
    # ========================================================================

    async def create_tool(self, data: ToolCreate) -> Tool:
        """Create a new tool."""
        try:
            tool = Tool(
                id=str(uuid4()),
                name=data.name,
                description=data.description,
                parameters=data.parameters,
                category=data.category,
                is_active=True,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )

            await self.db.execute(
                """
                INSERT INTO tools (
                    id, name, description, parameters, category,
                    is_active, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    tool.id,
                    tool.name,
                    tool.description,
                    json.dumps(tool.parameters),
                    tool.category,
                    1 if tool.is_active else 0,
                    tool.created_at.isoformat(),
                    tool.updated_at.isoformat(),
                ),
            )
            logger.debug(f"Created tool {tool.id}: {tool.name}")
            return tool
        except Exception as e:
            logger.error(f"Error creating tool: {e}")
            raise

    async def get_tool(self, id: str) -> Tool | None:
        """Get a tool by ID."""
        try:
            row = await self.db.fetchone("SELECT * FROM tools WHERE id = ?", (id,))
            if row is None:
                return None
            return self._parse_tool(row)
        except Exception as e:
            logger.error(f"Error getting tool {id}: {e}")
            return None

    async def get_tool_by_name(self, name: str, category: str) -> Tool | None:
        """Get a tool by name and category."""
        try:
            row = await self.db.fetchone(
                "SELECT * FROM tools WHERE name = ? AND category = ?",
                (name, category),
            )
            if row is None:
                return None
            return self._parse_tool(row)
        except Exception as e:
            logger.error(f"Error getting tool by name {name}/{category}: {e}")
            return None

    async def update_tool(self, id: str, updates: ToolUpdate) -> Tool | None:
        """Update a tool."""
        try:
            # Get existing tool
            tool = await self.get_tool(id)
            if tool is None:
                return None

            # Build update query dynamically
            fields = []
            params = []

            if updates.name is not None:
                fields.append("name = ?")
                params.append(updates.name)
            if updates.description is not None:
                fields.append("description = ?")
                params.append(updates.description)
            if updates.parameters is not None:
                fields.append("parameters = ?")
                params.append(json.dumps(updates.parameters))
            if updates.category is not None:
                fields.append("category = ?")
                params.append(updates.category)
            if updates.is_active is not None:
                fields.append("is_active = ?")
                params.append(1 if updates.is_active else 0)

            # Always update updated_at
            fields.append("updated_at = ?")
            params.append(datetime.utcnow().isoformat())

            # Add ID to params
            params.append(id)

            await self.db.execute(
                f"UPDATE tools SET {', '.join(fields)} WHERE id = ?",
                tuple(params),
            )

            logger.debug(f"Updated tool {id}")
            return await self.get_tool(id)
        except Exception as e:
            logger.error(f"Error updating tool {id}: {e}")
            return None

    async def delete_tool(self, id: str) -> bool:
        """Delete a tool."""
        try:
            await self.db.execute("DELETE FROM tools WHERE id = ?", (id,))
            logger.debug(f"Deleted tool {id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting tool {id}: {e}")
            return False

    async def list_tools(self, category: str | None = None) -> list[Tool]:
        """List all tools, optionally filtered by category."""
        try:
            if category is None:
                rows = await self.db.fetchall(
                    "SELECT * FROM tools ORDER BY category, name"
                )
            else:
                rows = await self.db.fetchall(
                    "SELECT * FROM tools WHERE category = ? ORDER BY name",
                    (category,),
                )
            return [self._parse_tool(row) for row in rows]
        except Exception as e:
            logger.error(f"Error listing tools: {e}")
            return []

    async def search_tools(self, text: str) -> list[Tool]:
        """Search tools using FTS5 full-text search."""
        try:
            rows = await self.db.fetchall(
                """
                SELECT t.* FROM tools t
                INNER JOIN tools_fts fts ON t.id = fts.id
                WHERE tools_fts MATCH ?
                ORDER BY rank
                """,
                (text,),
            )
            return [self._parse_tool(row) for row in rows]
        except Exception as e:
            logger.error(f"Error searching tools: {e}")
            return []

    # ========================================================================
    # Scenario Methods
    # ========================================================================

    async def create_scenario(self, data: ScenarioCreate) -> Scenario:
        """Create a new scenario."""
        try:
            scenario = Scenario(
                id=data.id if data.id else str(uuid4()),
                name=data.name,
                description=data.description,
                category=data.category,
                system_prompt=data.system_prompt,
                example_queries=data.example_queries,
                mock_responses=data.mock_responses,
                is_active=True,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )

            await self.db.execute(
                """
                INSERT INTO scenarios (
                    id, name, description, category, system_prompt,
                    example_queries, mock_responses, is_active,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    scenario.id,
                    scenario.name,
                    scenario.description,
                    scenario.category,
                    scenario.system_prompt,
                    json.dumps(scenario.example_queries),
                    json.dumps(scenario.mock_responses),
                    1 if scenario.is_active else 0,
                    scenario.created_at.isoformat(),
                    scenario.updated_at.isoformat(),
                ),
            )

            # Attach tools if provided
            if data.tool_ids:
                await self.attach_tools_to_scenario(scenario.id, data.tool_ids)

            logger.debug(f"Created scenario {scenario.id}: {scenario.name}")
            return scenario
        except Exception as e:
            logger.error(f"Error creating scenario: {e}")
            raise

    async def get_scenario(self, id: str) -> Scenario | None:
        """Get a scenario by ID."""
        try:
            row = await self.db.fetchone(
                "SELECT * FROM scenarios WHERE id = ?", (id,)
            )
            if row is None:
                return None
            return self._parse_scenario(row)
        except Exception as e:
            logger.error(f"Error getting scenario {id}: {e}")
            return None

    async def get_scenario_with_tools(self, id: str) -> ScenarioWithTools | None:
        """Get a scenario with its associated tools."""
        try:
            scenario = await self.get_scenario(id)
            if scenario is None:
                return None

            tools = await self.get_scenario_tools(id)

            return ScenarioWithTools(
                **scenario.model_dump(exclude={"tool_count"}),
                tools=tools,
                tool_count=len(tools),
            )
        except Exception as e:
            logger.error(f"Error getting scenario with tools {id}: {e}")
            return None

    async def update_scenario(
        self, id: str, updates: ScenarioUpdate
    ) -> Scenario | None:
        """Update a scenario."""
        try:
            # Get existing scenario
            scenario = await self.get_scenario(id)
            if scenario is None:
                return None

            # Build update query dynamically
            fields = []
            params = []

            if updates.name is not None:
                fields.append("name = ?")
                params.append(updates.name)
            if updates.description is not None:
                fields.append("description = ?")
                params.append(updates.description)
            if updates.category is not None:
                fields.append("category = ?")
                params.append(updates.category)
            if updates.system_prompt is not None:
                fields.append("system_prompt = ?")
                params.append(updates.system_prompt)
            if updates.example_queries is not None:
                fields.append("example_queries = ?")
                params.append(json.dumps(updates.example_queries))
            if updates.mock_responses is not None:
                fields.append("mock_responses = ?")
                params.append(json.dumps(updates.mock_responses))
            if updates.is_active is not None:
                fields.append("is_active = ?")
                params.append(1 if updates.is_active else 0)

            # Always update updated_at
            fields.append("updated_at = ?")
            params.append(datetime.utcnow().isoformat())

            # Add ID to params
            params.append(id)

            await self.db.execute(
                f"UPDATE scenarios SET {', '.join(fields)} WHERE id = ?",
                tuple(params),
            )

            logger.debug(f"Updated scenario {id}")
            return await self.get_scenario(id)
        except Exception as e:
            logger.error(f"Error updating scenario {id}: {e}")
            return None

    async def delete_scenario(self, id: str) -> bool:
        """Delete a scenario."""
        try:
            await self.db.execute("DELETE FROM scenarios WHERE id = ?", (id,))
            logger.debug(f"Deleted scenario {id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting scenario {id}: {e}")
            return False

    async def list_scenarios(self, category: str | None = None) -> list[Scenario]:
        """List all scenarios, optionally filtered by category."""
        try:
            if category is None:
                rows = await self.db.fetchall(
                    "SELECT * FROM scenarios ORDER BY category, name"
                )
            else:
                rows = await self.db.fetchall(
                    "SELECT * FROM scenarios WHERE category = ? ORDER BY name",
                    (category,),
                )
            return [self._parse_scenario(row) for row in rows]
        except Exception as e:
            logger.error(f"Error listing scenarios: {e}")
            return []

    async def search_scenarios(self, text: str) -> list[Scenario]:
        """Search scenarios using FTS5 full-text search."""
        try:
            rows = await self.db.fetchall(
                """
                SELECT s.* FROM scenarios s
                INNER JOIN scenarios_fts fts ON s.id = fts.id
                WHERE scenarios_fts MATCH ?
                ORDER BY rank
                """,
                (text,),
            )
            return [self._parse_scenario(row) for row in rows]
        except Exception as e:
            logger.error(f"Error searching scenarios: {e}")
            return []

    # ========================================================================
    # Scenario-Tool Relationship Methods
    # ========================================================================

    async def attach_tools_to_scenario(
        self, scenario_id: str, tool_ids: list[str], replace: bool = False
    ) -> int:
        """Attach tools to a scenario."""
        try:
            if replace:
                # Remove existing tools
                await self.db.execute(
                    "DELETE FROM scenario_tools WHERE scenario_id = ?",
                    (scenario_id,),
                )

            # Insert new tools
            for idx, tool_id in enumerate(tool_ids):
                await self.db.execute(
                    """
                    INSERT OR IGNORE INTO scenario_tools
                    (scenario_id, tool_id, display_order)
                    VALUES (?, ?, ?)
                    """,
                    (scenario_id, tool_id, idx),
                )

            logger.debug(
                f"Attached {len(tool_ids)} tools to scenario {scenario_id}"
            )
            return len(tool_ids)
        except Exception as e:
            logger.error(f"Error attaching tools to scenario {scenario_id}: {e}")
            return 0

    async def detach_tools_from_scenario(
        self, scenario_id: str, tool_ids: list[str]
    ) -> int:
        """Detach tools from a scenario."""
        try:
            if not tool_ids:
                return 0

            placeholders = ",".join(["?" for _ in tool_ids])
            query = f"""
                DELETE FROM scenario_tools
                WHERE scenario_id = ? AND tool_id IN ({placeholders})
            """
            params = [scenario_id] + tool_ids

            await self.db.execute(query, tuple(params))
            logger.debug(
                f"Detached {len(tool_ids)} tools from scenario {scenario_id}"
            )
            return len(tool_ids)
        except Exception as e:
            logger.error(
                f"Error detaching tools from scenario {scenario_id}: {e}"
            )
            return 0

    async def get_scenario_tools(self, scenario_id: str) -> list[Tool]:
        """Get all tools associated with a scenario."""
        try:
            rows = await self.db.fetchall(
                """
                SELECT t.* FROM tools t
                INNER JOIN scenario_tools st ON t.id = st.tool_id
                WHERE st.scenario_id = ?
                ORDER BY st.display_order, t.name
                """,
                (scenario_id,),
            )
            return [self._parse_tool(row) for row in rows]
        except Exception as e:
            logger.error(f"Error getting scenario tools {scenario_id}: {e}")
            return []

    # ========================================================================
    # Actor Methods
    # ========================================================================

    async def create_actor(self, data: ActorCreate) -> Actor:
        """Create a new actor."""
        try:
            actor = Actor(
                id=data.id if data.id else str(uuid4()),
                name=data.name,
                icon=data.icon,
                background=data.background,
                goal=data.goal,
                tags=data.tags,
                category=data.category,
                is_active=True,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )

            await self.db.execute(
                """
                INSERT INTO actors (
                    id, name, icon, background, goal, tags, category,
                    is_active, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    actor.id,
                    actor.name,
                    actor.icon,
                    actor.background,
                    actor.goal,
                    json.dumps(actor.tags),
                    actor.category,
                    1 if actor.is_active else 0,
                    actor.created_at.isoformat(),
                    actor.updated_at.isoformat(),
                ),
            )

            logger.debug(f"Created actor {actor.id}: {actor.name}")
            return actor
        except Exception as e:
            logger.error(f"Error creating actor: {e}")
            raise

    async def get_actor(self, id: str) -> Actor | None:
        """Get an actor by ID."""
        try:
            row = await self.db.fetchone("SELECT * FROM actors WHERE id = ?", (id,))
            if row is None:
                return None
            return self._parse_actor(row)
        except Exception as e:
            logger.error(f"Error getting actor {id}: {e}")
            return None

    async def update_actor(self, id: str, updates: ActorUpdate) -> Actor | None:
        """Update an actor."""
        try:
            # Get existing actor
            actor = await self.get_actor(id)
            if actor is None:
                return None

            # Build update query dynamically
            fields = []
            params = []

            if updates.name is not None:
                fields.append("name = ?")
                params.append(updates.name)
            if updates.icon is not None:
                fields.append("icon = ?")
                params.append(updates.icon)
            if updates.background is not None:
                fields.append("background = ?")
                params.append(updates.background)
            if updates.goal is not None:
                fields.append("goal = ?")
                params.append(updates.goal)
            if updates.tags is not None:
                fields.append("tags = ?")
                params.append(json.dumps(updates.tags))
            if updates.category is not None:
                fields.append("category = ?")
                params.append(updates.category)
            if updates.is_active is not None:
                fields.append("is_active = ?")
                params.append(1 if updates.is_active else 0)

            # Always update updated_at
            fields.append("updated_at = ?")
            params.append(datetime.utcnow().isoformat())

            # Add ID to params
            params.append(id)

            await self.db.execute(
                f"UPDATE actors SET {', '.join(fields)} WHERE id = ?",
                tuple(params),
            )

            logger.debug(f"Updated actor {id}")
            return await self.get_actor(id)
        except Exception as e:
            logger.error(f"Error updating actor {id}: {e}")
            return None

    async def delete_actor(self, id: str) -> bool:
        """Delete an actor."""
        try:
            await self.db.execute("DELETE FROM actors WHERE id = ?", (id,))
            logger.debug(f"Deleted actor {id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting actor {id}: {e}")
            return False

    async def list_actors(self, category: str | None = None) -> list[Actor]:
        """List all actors, optionally filtered by category."""
        try:
            if category is None:
                rows = await self.db.fetchall(
                    "SELECT * FROM actors ORDER BY category, name"
                )
            else:
                rows = await self.db.fetchall(
                    "SELECT * FROM actors WHERE category = ? ORDER BY name",
                    (category,),
                )
            return [self._parse_actor(row) for row in rows]
        except Exception as e:
            logger.error(f"Error listing actors: {e}")
            return []

    async def search_actors(self, text: str) -> list[Actor]:
        """Search actors using FTS5 full-text search."""
        try:
            rows = await self.db.fetchall(
                """
                SELECT a.* FROM actors a
                INNER JOIN actors_fts fts ON a.id = fts.id
                WHERE actors_fts MATCH ?
                ORDER BY rank
                """,
                (text,),
            )
            return [self._parse_actor(row) for row in rows]
        except Exception as e:
            logger.error(f"Error searching actors: {e}")
            return []

    # ========================================================================
    # Scenario-Actor Relationship Methods
    # ========================================================================

    async def link_actor_to_scenario(
        self, scenario_id: str, actor_id: str
    ) -> bool:
        """Link an actor to a scenario."""
        try:
            await self.db.execute(
                """
                INSERT OR IGNORE INTO scenario_actors
                (scenario_id, actor_id)
                VALUES (?, ?)
                """,
                (scenario_id, actor_id),
            )
            logger.debug(f"Linked actor {actor_id} to scenario {scenario_id}")
            return True
        except Exception as e:
            logger.error(
                f"Error linking actor {actor_id} to scenario {scenario_id}: {e}"
            )
            return False

    async def unlink_actor_from_scenario(
        self, scenario_id: str, actor_id: str
    ) -> bool:
        """Unlink an actor from a scenario."""
        try:
            await self.db.execute(
                """
                DELETE FROM scenario_actors
                WHERE scenario_id = ? AND actor_id = ?
                """,
                (scenario_id, actor_id),
            )
            logger.debug(
                f"Unlinked actor {actor_id} from scenario {scenario_id}"
            )
            return True
        except Exception as e:
            logger.error(
                f"Error unlinking actor {actor_id} from scenario {scenario_id}: {e}"
            )
            return False

    async def get_scenario_actors(self, scenario_id: str) -> list[Actor]:
        """Get all actors linked to a scenario."""
        try:
            rows = await self.db.fetchall(
                """
                SELECT a.* FROM actors a
                INNER JOIN scenario_actors sa ON a.id = sa.actor_id
                WHERE sa.scenario_id = ?
                ORDER BY a.name
                """,
                (scenario_id,),
            )
            return [self._parse_actor(row) for row in rows]
        except Exception as e:
            logger.error(f"Error getting scenario actors {scenario_id}: {e}")
            return []

    async def get_actors_for_category(self, category: str) -> list[Actor]:
        """Get all actors for a specific category."""
        try:
            rows = await self.db.fetchall(
                "SELECT * FROM actors WHERE category = ? ORDER BY name",
                (category,),
            )
            return [self._parse_actor(row) for row in rows]
        except Exception as e:
            logger.error(f"Error getting actors for category {category}: {e}")
            return []

    # ========================================================================
    # ToolPreset Methods
    # ========================================================================

    async def create_tool_preset(self, data: ToolPresetCreate) -> ToolPreset:
        """Create a new tool preset."""
        try:
            preset = ToolPreset(
                id=str(uuid4()),
                scenario_id=data.scenario_id,
                name=data.name,
                description=data.description,
                tool_ids=data.tool_ids,
                is_default=data.is_default,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )

            # If this is a default preset, unset other defaults for this scenario
            if preset.is_default:
                await self.db.execute(
                    """
                    UPDATE tool_presets
                    SET is_default = 0
                    WHERE scenario_id = ?
                    """,
                    (preset.scenario_id,),
                )

            await self.db.execute(
                """
                INSERT INTO tool_presets (
                    id, scenario_id, name, description, tool_ids,
                    is_default, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    preset.id,
                    preset.scenario_id,
                    preset.name,
                    preset.description,
                    json.dumps(preset.tool_ids),
                    1 if preset.is_default else 0,
                    preset.created_at.isoformat(),
                ),
            )

            logger.debug(f"Created tool preset {preset.id}: {preset.name}")
            return preset
        except Exception as e:
            logger.error(f"Error creating tool preset: {e}")
            raise

    async def get_tool_preset(self, id: str) -> ToolPreset | None:
        """Get a tool preset by ID."""
        try:
            row = await self.db.fetchone(
                "SELECT * FROM tool_presets WHERE id = ?", (id,)
            )
            if row is None:
                return None
            return self._parse_tool_preset(row)
        except Exception as e:
            logger.error(f"Error getting tool preset {id}: {e}")
            return None

    async def get_tool_preset_with_tools(self, id: str) -> ToolPreset | None:
        """Get a tool preset with joined tool data."""
        try:
            preset = await self.get_tool_preset(id)
            if preset is None:
                return None

            # Fetch the actual tool objects
            tools = []
            for tool_id in preset.tool_ids:
                tool = await self.get_tool(tool_id)
                if tool:
                    tools.append(tool)

            preset.tools = tools
            return preset
        except Exception as e:
            logger.error(f"Error getting tool preset with tools {id}: {e}")
            return None

    async def update_tool_preset(
        self, id: str, updates: ToolPresetUpdate
    ) -> ToolPreset | None:
        """Update a tool preset."""
        try:
            # Get existing preset
            preset = await self.get_tool_preset(id)
            if preset is None:
                return None

            # Build update query dynamically
            fields = []
            params = []

            if updates.name is not None:
                fields.append("name = ?")
                params.append(updates.name)
            if updates.description is not None:
                fields.append("description = ?")
                params.append(updates.description)
            if updates.tool_ids is not None:
                fields.append("tool_ids = ?")
                params.append(json.dumps(updates.tool_ids))
            if updates.is_default is not None:
                # If setting as default, unset other defaults for this scenario
                if updates.is_default:
                    await self.db.execute(
                        """
                        UPDATE tool_presets
                        SET is_default = 0
                        WHERE scenario_id = ?
                        """,
                        (preset.scenario_id,),
                    )
                fields.append("is_default = ?")
                params.append(1 if updates.is_default else 0)

            if not fields:
                return preset

            # Add ID to params
            params.append(id)

            await self.db.execute(
                f"UPDATE tool_presets SET {', '.join(fields)} WHERE id = ?",
                tuple(params),
            )

            logger.debug(f"Updated tool preset {id}")
            return await self.get_tool_preset(id)
        except Exception as e:
            logger.error(f"Error updating tool preset {id}: {e}")
            return None

    async def delete_tool_preset(self, id: str) -> bool:
        """Delete a tool preset."""
        try:
            await self.db.execute("DELETE FROM tool_presets WHERE id = ?", (id,))
            logger.debug(f"Deleted tool preset {id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting tool preset {id}: {e}")
            return False

    async def list_scenario_presets(self, scenario_id: str) -> list[ToolPreset]:
        """List all tool presets for a scenario."""
        try:
            rows = await self.db.fetchall(
                """
                SELECT * FROM tool_presets
                WHERE scenario_id = ?
                ORDER BY is_default DESC, name
                """,
                (scenario_id,),
            )
            return [self._parse_tool_preset(row) for row in rows]
        except Exception as e:
            logger.error(f"Error listing scenario presets {scenario_id}: {e}")
            return []

    async def get_default_preset(self, scenario_id: str) -> ToolPreset | None:
        """Get the default tool preset for a scenario."""
        try:
            row = await self.db.fetchone(
                """
                SELECT * FROM tool_presets
                WHERE scenario_id = ? AND is_default = 1
                LIMIT 1
                """,
                (scenario_id,),
            )
            if row is None:
                return None
            return self._parse_tool_preset(row)
        except Exception as e:
            logger.error(f"Error getting default preset {scenario_id}: {e}")
            return None

    # ========================================================================
    # GenerationTemplate Methods
    # ========================================================================

    async def create_generation_template(
        self, data: GenerationTemplateCreate
    ) -> GenerationTemplate:
        """Create a new generation template."""
        try:
            template = GenerationTemplate(
                id=str(uuid4()),
                name=data.name,
                description=data.description,
                scenario_id=data.scenario_id,
                actor_id=data.actor_id,
                tool_preset_id=data.tool_preset_id,
                model=data.model,
                temperature=data.temperature,
                is_favorite=False,
                use_count=0,
                last_used_at=None,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
            )

            await self.db.execute(
                """
                INSERT INTO generation_templates (
                    id, name, description, scenario_id, actor_id,
                    tool_preset_id, model, temperature, is_favorite,
                    use_count, last_used_at, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    template.id,
                    template.name,
                    template.description,
                    template.scenario_id,
                    template.actor_id,
                    template.tool_preset_id,
                    template.model,
                    template.temperature,
                    1 if template.is_favorite else 0,
                    template.use_count,
                    None,
                    template.created_at.isoformat(),
                ),
            )

            logger.debug(
                f"Created generation template {template.id}: {template.name}"
            )
            return template
        except Exception as e:
            logger.error(f"Error creating generation template: {e}")
            raise

    async def get_generation_template(self, id: str) -> GenerationTemplate | None:
        """Get a generation template by ID."""
        try:
            row = await self.db.fetchone(
                "SELECT * FROM generation_templates WHERE id = ?", (id,)
            )
            if row is None:
                return None
            return self._parse_generation_template(row)
        except Exception as e:
            logger.error(f"Error getting generation template {id}: {e}")
            return None

    async def get_generation_template_full(
        self, id: str
    ) -> GenerationTemplate | None:
        """Get a generation template with joined scenario, actor, and preset data."""
        try:
            template = await self.get_generation_template(id)
            if template is None:
                return None

            # Load related data
            if template.scenario_id:
                template.scenario = await self.get_scenario(template.scenario_id)
            if template.actor_id:
                template.actor = await self.get_actor(template.actor_id)
            if template.tool_preset_id:
                template.tool_preset = await self.get_tool_preset(
                    template.tool_preset_id
                )

            return template
        except Exception as e:
            logger.error(
                f"Error getting generation template with joins {id}: {e}"
            )
            return None

    async def update_generation_template(
        self, id: str, updates: GenerationTemplateUpdate
    ) -> GenerationTemplate | None:
        """Update a generation template."""
        try:
            # Get existing template
            template = await self.get_generation_template(id)
            if template is None:
                return None

            # Build update query dynamically
            fields = []
            params = []

            if updates.name is not None:
                fields.append("name = ?")
                params.append(updates.name)
            if updates.description is not None:
                fields.append("description = ?")
                params.append(updates.description)
            if updates.scenario_id is not None:
                fields.append("scenario_id = ?")
                params.append(updates.scenario_id)
            if updates.actor_id is not None:
                fields.append("actor_id = ?")
                params.append(updates.actor_id)
            if updates.tool_preset_id is not None:
                fields.append("tool_preset_id = ?")
                params.append(updates.tool_preset_id)
            if updates.model is not None:
                fields.append("model = ?")
                params.append(updates.model)
            if updates.temperature is not None:
                fields.append("temperature = ?")
                params.append(updates.temperature)
            if updates.is_favorite is not None:
                fields.append("is_favorite = ?")
                params.append(1 if updates.is_favorite else 0)

            if not fields:
                return template

            # Add ID to params
            params.append(id)

            await self.db.execute(
                f"UPDATE generation_templates SET {', '.join(fields)} WHERE id = ?",
                tuple(params),
            )

            logger.debug(f"Updated generation template {id}")
            return await self.get_generation_template(id)
        except Exception as e:
            logger.error(f"Error updating generation template {id}: {e}")
            return None

    async def delete_generation_template(self, id: str) -> bool:
        """Delete a generation template."""
        try:
            await self.db.execute(
                "DELETE FROM generation_templates WHERE id = ?", (id,)
            )
            logger.debug(f"Deleted generation template {id}")
            return True
        except Exception as e:
            logger.error(f"Error deleting generation template {id}: {e}")
            return False

    async def list_generation_templates(
        self, scenario_id: str | None = None, favorites_only: bool = False
    ) -> list[GenerationTemplate]:
        """List generation templates, optionally filtered."""
        try:
            conditions = []
            params = []

            if scenario_id is not None:
                conditions.append("scenario_id = ?")
                params.append(scenario_id)
            if favorites_only:
                conditions.append("is_favorite = 1")

            where_clause = (
                f"WHERE {' AND '.join(conditions)}" if conditions else ""
            )

            query = f"""
                SELECT * FROM generation_templates
                {where_clause}
                ORDER BY is_favorite DESC, use_count DESC, name
            """

            rows = await self.db.fetchall(query, tuple(params) if params else None)
            return [self._parse_generation_template(row) for row in rows]
        except Exception as e:
            logger.error(f"Error listing generation templates: {e}")
            return []

    async def record_template_use(self, id: str) -> None:
        """Increment use count and update last_used_at for a template."""
        try:
            await self.db.execute(
                """
                UPDATE generation_templates
                SET use_count = use_count + 1,
                    last_used_at = ?
                WHERE id = ?
                """,
                (datetime.utcnow().isoformat(), id),
            )
            logger.debug(f"Recorded use of template {id}")
        except Exception as e:
            logger.error(f"Error recording template use {id}: {e}")

    async def get_recent_templates(
        self, limit: int = 5
    ) -> list[GenerationTemplate]:
        """Get recently used generation templates."""
        try:
            rows = await self.db.fetchall(
                """
                SELECT * FROM generation_templates
                WHERE last_used_at IS NOT NULL
                ORDER BY last_used_at DESC
                LIMIT ?
                """,
                (limit,),
            )
            return [self._parse_generation_template(row) for row in rows]
        except Exception as e:
            logger.error(f"Error getting recent templates: {e}")
            return []

    # ========================================================================
    # Migration Helper
    # ========================================================================

    async def migrate_from_hardcoded(
        self, scenario_template: Any, personas: list[Any]
    ) -> tuple[str, list[str], list[str]]:
        """
        Migrate from hardcoded ScenarioTemplate and PersonaPreset lists.

        Args:
            scenario_template: ScenarioTemplate from ymir.functions.schemas
            personas: List of PersonaPreset from ymir.pipeline.personas

        Returns:
            Tuple of (scenario_id, tool_ids, actor_ids)
        """
        try:
            # Create tools from scenario functions
            tool_ids = []
            for func_def in scenario_template.functions:
                # Check if tool already exists
                existing = await self.get_tool_by_name(
                    func_def.name, func_def.category
                )
                if existing:
                    tool_ids.append(existing.id)
                    logger.debug(
                        f"Using existing tool {func_def.name} ({existing.id})"
                    )
                else:
                    tool = await self.create_tool(
                        ToolCreate(
                            name=func_def.name,
                            description=func_def.description,
                            parameters=func_def.parameters,
                            category=func_def.category,
                        )
                    )
                    tool_ids.append(tool.id)
                    logger.debug(f"Created tool {func_def.name} ({tool.id})")

            # Create scenario
            scenario = await self.create_scenario(
                ScenarioCreate(
                    id=scenario_template.id,
                    name=scenario_template.name,
                    description=scenario_template.description,
                    category=scenario_template.category,
                    system_prompt=scenario_template.system_prompt,
                    example_queries=scenario_template.example_queries,
                    mock_responses=scenario_template.mock_responses,
                    tool_ids=tool_ids,
                )
            )
            logger.debug(f"Created scenario {scenario.id}: {scenario.name}")

            # Create actors from personas
            actor_ids = []
            for persona in personas:
                # Check if actor already exists
                existing_actors = await self.list_actors(category=persona.category)
                existing = next(
                    (a for a in existing_actors if a.name == persona.name), None
                )

                if existing:
                    actor_ids.append(existing.id)
                    logger.debug(
                        f"Using existing actor {persona.name} ({existing.id})"
                    )
                else:
                    actor = await self.create_actor(
                        ActorCreate(
                            id=persona.id,
                            name=persona.name,
                            icon=persona.icon,
                            background=persona.background,
                            goal=persona.goal,
                            tags=persona.tags,
                            category=persona.category,
                        )
                    )
                    actor_ids.append(actor.id)
                    logger.debug(f"Created actor {persona.name} ({actor.id})")

                # Link actor to scenario
                await self.link_actor_to_scenario(scenario.id, actor_ids[-1])

            logger.info(
                f"Migrated scenario {scenario.name}: "
                f"{len(tool_ids)} tools, {len(actor_ids)} actors"
            )

            return scenario.id, tool_ids, actor_ids

        except Exception as e:
            logger.error(f"Error migrating from hardcoded data: {e}")
            raise

    # ========================================================================
    # Helper Methods for Parsing Rows
    # ========================================================================

    def _parse_tool(self, row: dict) -> Tool:
        """Parse a tool from a database row."""
        return Tool(
            id=row["id"],
            name=row["name"],
            description=row["description"] or "",
            parameters=json.loads(row["parameters"]) if row["parameters"] else {},
            category=row["category"],
            is_active=bool(row["is_active"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    def _parse_scenario(self, row: dict) -> Scenario:
        """Parse a scenario from a database row."""
        return Scenario(
            id=row["id"],
            name=row["name"],
            description=row["description"] or "",
            category=row["category"],
            system_prompt=row["system_prompt"] or "",
            example_queries=json.loads(row["example_queries"])
            if row["example_queries"]
            else [],
            mock_responses=json.loads(row["mock_responses"])
            if row["mock_responses"]
            else {},
            is_active=bool(row["is_active"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    def _parse_actor(self, row: dict) -> Actor:
        """Parse an actor from a database row."""
        return Actor(
            id=row["id"],
            name=row["name"],
            icon=row["icon"],
            background=row["background"] or "",
            goal=row["goal"] or "",
            tags=json.loads(row["tags"]) if row["tags"] else [],
            category=row["category"],
            is_active=bool(row["is_active"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    def _parse_tool_preset(self, row: dict) -> ToolPreset:
        """Parse a tool preset from a database row."""
        return ToolPreset(
            id=row["id"],
            scenario_id=row["scenario_id"],
            name=row["name"],
            description=row["description"] or "",
            tool_ids=json.loads(row["tool_ids"]) if row["tool_ids"] else [],
            is_default=bool(row["is_default"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["created_at"]),  # Use created_at as fallback
        )

    def _parse_generation_template(self, row: dict) -> GenerationTemplate:
        """Parse a generation template from a database row."""
        return GenerationTemplate(
            id=row["id"],
            name=row["name"],
            description=row["description"] or "",
            scenario_id=row["scenario_id"],
            actor_id=row["actor_id"],
            tool_preset_id=row["tool_preset_id"],
            model=row["model"],
            temperature=row["temperature"],
            is_favorite=bool(row["is_favorite"]),
            use_count=row["use_count"],
            last_used_at=datetime.fromisoformat(row["last_used_at"])
            if row["last_used_at"]
            else None,
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["created_at"]),  # Use created_at as fallback
        )


# Global store instance
_scenario_store: ScenarioStore | None = None


def get_scenario_store(db: Database | None = None) -> ScenarioStore:
    """Get the global scenario store instance."""
    global _scenario_store
    if _scenario_store is None:
        if db is None:
            from .database import get_database

            db = get_database()
        _scenario_store = ScenarioStore(db)
    return _scenario_store
