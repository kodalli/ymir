"""SQLite database module for async operations."""

import aiosqlite
import json
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from loguru import logger


class Database:
    """Async SQLite database with full-text search support."""

    def __init__(self, db_path: str = "ymir/data/runtime/ymir.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: aiosqlite.Connection | None = None

    async def _get_connection(self) -> aiosqlite.Connection:
        """Get or create database connection."""
        if self._conn is None:
            self._conn = await aiosqlite.connect(self.db_path)
            self._conn.row_factory = aiosqlite.Row
        return self._conn

    async def initialize(self) -> None:
        """Create tables and indexes."""
        conn = await self._get_connection()

        # Enable pragmas for better performance and reliability
        await conn.execute("PRAGMA foreign_keys = ON")
        await conn.execute("PRAGMA journal_mode = WAL")
        await conn.execute("PRAGMA cache_size = -64000")

        # Sessions table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                scenario_id TEXT,
                scenario_description TEXT NOT NULL,
                system_prompt TEXT NOT NULL,
                messages TEXT NOT NULL,
                tools TEXT NOT NULL,
                source TEXT NOT NULL DEFAULT 'generated',
                original_source TEXT,
                status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'approved', 'rejected', 'needs_edit')),
                quality_score REAL,
                annotator_notes TEXT,
                reviewed_at TEXT
            )
        """)

        # Indexes for sessions
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_sessions_status ON sessions(status)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_sessions_scenario_id ON sessions(scenario_id)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_sessions_created_at ON sessions(created_at DESC)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_sessions_status_scenario ON sessions(status, scenario_id)"
        )

        # Datasets table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS datasets (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                description TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now')),
                tags TEXT,
                metadata TEXT
            )
        """)

        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_datasets_name ON datasets(name)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_datasets_created_at ON datasets(created_at DESC)"
        )

        # Junction table for many-to-many
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS dataset_sessions (
                dataset_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                added_at TEXT NOT NULL DEFAULT (datetime('now')),
                PRIMARY KEY (dataset_id, session_id),
                FOREIGN KEY (dataset_id) REFERENCES datasets(id) ON DELETE CASCADE,
                FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
            )
        """)

        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_dataset_sessions_dataset_id ON dataset_sessions(dataset_id)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_dataset_sessions_session_id ON dataset_sessions(session_id)"
        )

        # Scenarios table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS scenarios (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                description TEXT,
                category TEXT,
                system_prompt TEXT,
                example_queries TEXT,
                mock_responses TEXT,
                is_active INTEGER DEFAULT 1,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
        """)

        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_scenarios_name ON scenarios(name)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_scenarios_category ON scenarios(category)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_scenarios_is_active ON scenarios(is_active)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_scenarios_category_active ON scenarios(category, is_active)"
        )

        # Tools table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS tools (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                parameters TEXT,
                category TEXT,
                is_active INTEGER DEFAULT 1,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now')),
                UNIQUE(name, category)
            )
        """)

        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_tools_name ON tools(name)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_tools_category ON tools(category)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_tools_is_active ON tools(is_active)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_tools_category_active ON tools(category, is_active)"
        )

        # Scenario-Tools junction table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS scenario_tools (
                scenario_id TEXT NOT NULL,
                tool_id TEXT NOT NULL,
                display_order INTEGER DEFAULT 0,
                PRIMARY KEY (scenario_id, tool_id),
                FOREIGN KEY (scenario_id) REFERENCES scenarios(id) ON DELETE CASCADE,
                FOREIGN KEY (tool_id) REFERENCES tools(id) ON DELETE CASCADE
            )
        """)

        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_scenario_tools_scenario_id ON scenario_tools(scenario_id)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_scenario_tools_tool_id ON scenario_tools(tool_id)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_scenario_tools_order ON scenario_tools(scenario_id, display_order)"
        )

        # Actor Groups table (for organizing batch-generated actors)
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS actor_groups (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                template_id TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY (template_id) REFERENCES actor_templates(id) ON DELETE SET NULL
            )
        """)

        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_actor_groups_name ON actor_groups(name)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_actor_groups_template_id ON actor_groups(template_id)"
        )

        # Actors table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS actors (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                icon TEXT,
                background TEXT,
                goal TEXT,
                tags TEXT,
                category TEXT,
                group_id TEXT,
                is_active INTEGER DEFAULT 1,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY (group_id) REFERENCES actor_groups(id) ON DELETE SET NULL
            )
        """)

        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_actors_name ON actors(name)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_actors_category ON actors(category)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_actors_is_active ON actors(is_active)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_actors_category_active ON actors(category, is_active)"
        )

        # Migration: Add group_id column to actors table if it doesn't exist
        cursor = await conn.execute("PRAGMA table_info(actors)")
        columns = await cursor.fetchall()
        column_names = [col[1] for col in columns]
        if "group_id" not in column_names:
            logger.info("Migrating actors table: adding group_id column")
            await conn.execute("ALTER TABLE actors ADD COLUMN group_id TEXT")

        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_actors_group_id ON actors(group_id)"
        )

        # Migration: Add situation and template_id columns to actors table
        if "situation" not in column_names:
            logger.info("Migrating actors table: adding situation column")
            await conn.execute("ALTER TABLE actors ADD COLUMN situation TEXT")

        if "template_id" not in column_names:
            logger.info("Migrating actors table: adding template_id column")
            await conn.execute("ALTER TABLE actors ADD COLUMN template_id TEXT")

        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_actors_template_id ON actors(template_id)"
        )

        # Scenario-Actors junction table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS scenario_actors (
                scenario_id TEXT NOT NULL,
                actor_id TEXT NOT NULL,
                PRIMARY KEY (scenario_id, actor_id),
                FOREIGN KEY (scenario_id) REFERENCES scenarios(id) ON DELETE CASCADE,
                FOREIGN KEY (actor_id) REFERENCES actors(id) ON DELETE CASCADE
            )
        """)

        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_scenario_actors_scenario_id ON scenario_actors(scenario_id)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_scenario_actors_actor_id ON scenario_actors(actor_id)"
        )

        # Tool Presets table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS tool_presets (
                id TEXT PRIMARY KEY,
                scenario_id TEXT NOT NULL,
                name TEXT NOT NULL,
                description TEXT,
                tool_ids TEXT,
                is_default INTEGER DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY (scenario_id) REFERENCES scenarios(id) ON DELETE CASCADE,
                UNIQUE(scenario_id, name)
            )
        """)

        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_tool_presets_scenario_id ON tool_presets(scenario_id)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_tool_presets_is_default ON tool_presets(scenario_id, is_default)"
        )

        # Generation Templates table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS generation_templates (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                description TEXT,
                scenario_id TEXT,
                actor_id TEXT,
                tool_preset_id TEXT,
                model TEXT,
                temperature REAL,
                is_favorite INTEGER DEFAULT 0,
                use_count INTEGER DEFAULT 0,
                last_used_at TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY (scenario_id) REFERENCES scenarios(id) ON DELETE CASCADE,
                FOREIGN KEY (actor_id) REFERENCES actors(id) ON DELETE SET NULL,
                FOREIGN KEY (tool_preset_id) REFERENCES tool_presets(id) ON DELETE SET NULL
            )
        """)

        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_generation_templates_name ON generation_templates(name)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_generation_templates_scenario_id ON generation_templates(scenario_id)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_generation_templates_is_favorite ON generation_templates(is_favorite)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_generation_templates_use_count ON generation_templates(use_count DESC)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_generation_templates_last_used ON generation_templates(last_used_at DESC)"
        )

        # FTS5 for full-text search on sessions
        await conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS sessions_fts USING fts5(
                id UNINDEXED,
                content,
                system_prompt,
                scenario_description
            )
        """)

        # Triggers to keep FTS in sync
        await conn.execute("""
            CREATE TRIGGER IF NOT EXISTS sessions_ai AFTER INSERT ON sessions BEGIN
                INSERT INTO sessions_fts(id, content, system_prompt, scenario_description)
                VALUES (new.id, new.messages, new.system_prompt, new.scenario_description);
            END
        """)

        await conn.execute("""
            CREATE TRIGGER IF NOT EXISTS sessions_ad AFTER DELETE ON sessions BEGIN
                DELETE FROM sessions_fts WHERE id = old.id;
            END
        """)

        await conn.execute("""
            CREATE TRIGGER IF NOT EXISTS sessions_au AFTER UPDATE ON sessions BEGIN
                DELETE FROM sessions_fts WHERE id = old.id;
                INSERT INTO sessions_fts(id, content, system_prompt, scenario_description)
                VALUES (new.id, new.messages, new.system_prompt, new.scenario_description);
            END
        """)

        # FTS5 for scenarios
        await conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS scenarios_fts USING fts5(
                id UNINDEXED,
                name,
                description,
                category
            )
        """)

        await conn.execute("""
            CREATE TRIGGER IF NOT EXISTS scenarios_ai AFTER INSERT ON scenarios BEGIN
                INSERT INTO scenarios_fts(id, name, description, category)
                VALUES (new.id, new.name, new.description, new.category);
            END
        """)

        await conn.execute("""
            CREATE TRIGGER IF NOT EXISTS scenarios_ad AFTER DELETE ON scenarios BEGIN
                DELETE FROM scenarios_fts WHERE id = old.id;
            END
        """)

        await conn.execute("""
            CREATE TRIGGER IF NOT EXISTS scenarios_au AFTER UPDATE ON scenarios BEGIN
                DELETE FROM scenarios_fts WHERE id = old.id;
                INSERT INTO scenarios_fts(id, name, description, category)
                VALUES (new.id, new.name, new.description, new.category);
            END
        """)

        # FTS5 for tools
        await conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS tools_fts USING fts5(
                id UNINDEXED,
                name,
                description,
                category
            )
        """)

        await conn.execute("""
            CREATE TRIGGER IF NOT EXISTS tools_ai AFTER INSERT ON tools BEGIN
                INSERT INTO tools_fts(id, name, description, category)
                VALUES (new.id, new.name, new.description, new.category);
            END
        """)

        await conn.execute("""
            CREATE TRIGGER IF NOT EXISTS tools_ad AFTER DELETE ON tools BEGIN
                DELETE FROM tools_fts WHERE id = old.id;
            END
        """)

        await conn.execute("""
            CREATE TRIGGER IF NOT EXISTS tools_au AFTER UPDATE ON tools BEGIN
                DELETE FROM tools_fts WHERE id = old.id;
                INSERT INTO tools_fts(id, name, description, category)
                VALUES (new.id, new.name, new.description, new.category);
            END
        """)

        # FTS5 for actors
        await conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS actors_fts USING fts5(
                id UNINDEXED,
                name,
                background,
                goal,
                category
            )
        """)

        await conn.execute("""
            CREATE TRIGGER IF NOT EXISTS actors_ai AFTER INSERT ON actors BEGIN
                INSERT INTO actors_fts(id, name, background, goal, category)
                VALUES (new.id, new.name, new.background, new.goal, new.category);
            END
        """)

        await conn.execute("""
            CREATE TRIGGER IF NOT EXISTS actors_ad AFTER DELETE ON actors BEGIN
                DELETE FROM actors_fts WHERE id = old.id;
            END
        """)

        await conn.execute("""
            CREATE TRIGGER IF NOT EXISTS actors_au AFTER UPDATE ON actors BEGIN
                DELETE FROM actors_fts WHERE id = old.id;
                INSERT INTO actors_fts(id, name, background, goal, category)
                VALUES (new.id, new.name, new.background, new.goal, new.category);
            END
        """)

        # Actor Templates table
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS actor_templates (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                template_text TEXT NOT NULL,
                background_template TEXT,
                goal_template TEXT,
                category TEXT,
                scenario_id TEXT,
                is_active INTEGER DEFAULT 1,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY (scenario_id) REFERENCES scenarios(id) ON DELETE SET NULL
            )
        """)

        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_actor_templates_name ON actor_templates(name)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_actor_templates_category ON actor_templates(category)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_actor_templates_scenario_id ON actor_templates(scenario_id)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_actor_templates_is_active ON actor_templates(is_active)"
        )

        # FTS5 for actor templates
        await conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS actor_templates_fts USING fts5(
                id UNINDEXED,
                name,
                description,
                template_text
            )
        """)

        await conn.execute("""
            CREATE TRIGGER IF NOT EXISTS actor_templates_ai AFTER INSERT ON actor_templates BEGIN
                INSERT INTO actor_templates_fts(id, name, description, template_text)
                VALUES (new.id, new.name, new.description, new.template_text);
            END
        """)

        await conn.execute("""
            CREATE TRIGGER IF NOT EXISTS actor_templates_ad AFTER DELETE ON actor_templates BEGIN
                DELETE FROM actor_templates_fts WHERE id = old.id;
            END
        """)

        await conn.execute("""
            CREATE TRIGGER IF NOT EXISTS actor_templates_au AFTER UPDATE ON actor_templates BEGIN
                DELETE FROM actor_templates_fts WHERE id = old.id;
                INSERT INTO actor_templates_fts(id, name, description, template_text)
                VALUES (new.id, new.name, new.description, new.template_text);
            END
        """)

        await conn.commit()
        logger.info(f"Database initialized at {self.db_path}")

        # Seed initial scenario and persona data if database is empty
        await self._seed_initial_data()

        # Seed demo data if database is empty
        await self._seed_demo_data_if_empty()

    async def _seed_initial_data(self) -> None:
        """Seed the database with initial scenario and persona data if scenarios table is empty."""
        count = await self.fetchone("SELECT COUNT(*) as count FROM scenarios")
        if count and count["count"] > 0:
            return  # Already has scenarios

        logger.info("Seeding initial scenario and persona data...")

        try:
            # Import the hardcoded data
            from ymir.functions.templates.scheduling import SCHEDULING_SCENARIO
            from ymir.pipeline.personas import SCHEDULING_PERSONAS
            from ymir.data.scenario_store import ScenarioStore

            # Get ScenarioStore instance
            store = ScenarioStore(self)

            # Migrate the hardcoded data to the database
            scenario_id, tool_ids, actor_ids = await store.migrate_from_hardcoded(
                SCHEDULING_SCENARIO, SCHEDULING_PERSONAS
            )

            # Create a default "Full Access" tool preset with all tools
            from ymir.core.scenario_schemas import ToolPresetCreate

            await store.create_tool_preset(
                ToolPresetCreate(
                    scenario_id=scenario_id,
                    name="Full Access",
                    description="All available tools for this scenario",
                    tool_ids=tool_ids,
                    is_default=True,
                )
            )

            logger.info(
                f"Initial data seeded: 1 scenario ({scenario_id}), "
                f"{len(tool_ids)} tools, {len(actor_ids)} actors, 1 tool preset"
            )
        except Exception as e:
            logger.error(f"Error seeding initial data: {e}")
            # Don't raise - allow app to continue even if seeding fails

    async def _seed_demo_data_if_empty(self) -> None:
        """Seed the database with demo trajectories if empty."""
        count = await self.fetchone("SELECT COUNT(*) as count FROM sessions")
        if count and count["count"] > 0:
            return  # Already has data

        logger.info("Seeding demo data...")

        # Appointment scheduling demo
        appointment_traj = {
            "id": f"demo-appt-{uuid4().hex[:8]}",
            "created_at": datetime.utcnow().isoformat(),
            "scenario_id": "appointment-scheduling",
            "scenario_description": "Healthcare front desk appointment scheduling",
            "tools": json.dumps([
                {
                    "type": "function",
                    "function": {
                        "name": "check_provider_availability",
                        "description": "Check available appointment slots for a provider",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "provider_id": {"type": "string", "description": "Provider/doctor ID"},
                                "date": {"type": "string", "description": "Date in YYYY-MM-DD format"},
                                "appointment_type": {"type": "string", "enum": ["new_patient", "follow_up", "annual_physical", "sick_visit"]}
                            },
                            "required": ["provider_id", "date"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "book_appointment",
                        "description": "Book an appointment for a patient",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "patient_id": {"type": "string"},
                                "provider_id": {"type": "string"},
                                "date": {"type": "string"},
                                "time": {"type": "string"},
                                "appointment_type": {"type": "string"},
                                "reason": {"type": "string"}
                            },
                            "required": ["patient_id", "provider_id", "date", "time", "appointment_type"]
                        }
                    }
                }
            ]),
            "system_prompt": "You are a healthcare front desk assistant. Help patients schedule appointments by checking provider availability and booking slots. Always verify patient information and confirm appointment details.",
            "messages": json.dumps([
                {"role": "user", "content": "Hi, I need to schedule an appointment with Dr. Smith for a follow-up visit. I'm available next Tuesday.", "tool_calls": None, "tool_result": None},
                {"role": "assistant", "content": "I'd be happy to help you schedule a follow-up appointment with Dr. Smith. Let me check her availability for next Tuesday.\n\n<tool_call>{\"name\": \"check_provider_availability\", \"arguments\": {\"provider_id\": \"dr-smith-001\", \"date\": \"2025-12-30\", \"appointment_type\": \"follow_up\"}}</tool_call>", "tool_calls": [{"name": "check_provider_availability", "arguments": {"provider_id": "dr-smith-001", "date": "2025-12-30", "appointment_type": "follow_up"}}], "tool_result": None},
                {"role": "tool_result", "content": "{\"available_slots\": [{\"time\": \"09:00\", \"duration\": 30}, {\"time\": \"10:30\", \"duration\": 30}, {\"time\": \"14:00\", \"duration\": 30}, {\"time\": \"15:30\", \"duration\": 30}], \"provider_name\": \"Dr. Sarah Smith\", \"location\": \"Main Clinic - Room 204\"}", "tool_calls": None, "tool_result": {"tool_name": "check_provider_availability", "result": {"available_slots": [{"time": "09:00"}, {"time": "10:30"}, {"time": "14:00"}, {"time": "15:30"}]}, "success": True}},
                {"role": "assistant", "content": "Dr. Smith has four openings on Tuesday, December 30th. She's available at 9 AM, 10:30 AM, 2 PM, or 3:30 PM. Which time works best for you? Also, may I have your name or patient ID to complete the booking?", "tool_calls": None, "tool_result": None},
                {"role": "user", "content": "The 2pm slot works. My name is John Davis, patient ID is PT-78432.", "tool_calls": None, "tool_result": None},
                {"role": "assistant", "content": "Perfect, let me book the 2:00 PM appointment for you, Mr. Davis.\n\n<tool_call>{\"name\": \"book_appointment\", \"arguments\": {\"patient_id\": \"PT-78432\", \"provider_id\": \"dr-smith-001\", \"date\": \"2025-12-30\", \"time\": \"14:00\", \"appointment_type\": \"follow_up\", \"reason\": \"Follow-up visit\"}}</tool_call>", "tool_calls": [{"name": "book_appointment", "arguments": {"patient_id": "PT-78432", "provider_id": "dr-smith-001", "date": "2025-12-30", "time": "14:00", "appointment_type": "follow_up", "reason": "Follow-up visit"}}], "tool_result": None},
                {"role": "tool_result", "content": "{\"confirmation_number\": \"APT-2025-89234\", \"status\": \"confirmed\", \"appointment\": {\"date\": \"2025-12-30\", \"time\": \"14:00\", \"provider\": \"Dr. Sarah Smith\", \"location\": \"Main Clinic - Room 204\", \"type\": \"follow_up\"}}", "tool_calls": None, "tool_result": {"tool_name": "book_appointment", "result": {"confirmation_number": "APT-2025-89234", "status": "confirmed"}, "success": True}},
                {"role": "assistant", "content": "Your appointment is confirmed. Your confirmation number is APT-2025-89234. That's Tuesday, December 30th, 2025 at 2 PM with Dr. Sarah Smith at the Main Clinic, Room 204 for a follow-up visit. Please arrive 15 minutes early and bring your insurance card and photo ID. Is there anything else I can help you with?", "tool_calls": None, "tool_result": None}
            ]),
            "source": "generated",
            "original_source": None,
            "status": "approved",
            "quality_score": 0.92,
            "annotator_notes": "Good multi-step appointment booking flow",
            "reviewed_at": datetime.utcnow().isoformat(),
        }

        # Insurance verification demo
        insurance_traj = {
            "id": f"demo-ins-{uuid4().hex[:8]}",
            "created_at": datetime.utcnow().isoformat(),
            "scenario_id": "insurance-verification",
            "scenario_description": "Patient insurance eligibility and benefits check",
            "tools": json.dumps([
                {
                    "type": "function",
                    "function": {
                        "name": "verify_insurance",
                        "description": "Verify patient insurance eligibility and coverage",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "patient_id": {"type": "string"},
                                "insurance_id": {"type": "string"},
                                "service_type": {"type": "string", "description": "Type of service to check coverage for"}
                            },
                            "required": ["patient_id"]
                        }
                    }
                },
                {
                    "type": "function",
                    "function": {
                        "name": "get_copay_info",
                        "description": "Get copay and deductible information",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "insurance_id": {"type": "string"},
                                "visit_type": {"type": "string", "enum": ["office_visit", "specialist", "urgent_care", "emergency", "lab_work"]}
                            },
                            "required": ["insurance_id", "visit_type"]
                        }
                    }
                }
            ]),
            "system_prompt": "You are a healthcare front desk assistant specializing in insurance verification. Help patients understand their coverage, copays, and benefits. Always protect patient privacy and verify identity before sharing information.",
            "messages": json.dumps([
                {"role": "user", "content": "I have an appointment tomorrow and want to know what my copay will be. My patient ID is PT-45621.", "tool_calls": None, "tool_result": None},
                {"role": "assistant", "content": "I'll look up your insurance information to check your copay. Let me verify your coverage first.\n\n<tool_call>{\"name\": \"verify_insurance\", \"arguments\": {\"patient_id\": \"PT-45621\"}}</tool_call>", "tool_calls": [{"name": "verify_insurance", "arguments": {"patient_id": "PT-45621"}}], "tool_result": None},
                {"role": "tool_result", "content": "{\"patient_name\": \"Maria Garcia\", \"insurance_carrier\": \"BlueCross BlueShield\", \"plan_name\": \"PPO Gold\", \"member_id\": \"BCB-998877665\", \"group_number\": \"GRP-12345\", \"status\": \"active\", \"effective_date\": \"2025-01-01\", \"primary_care_provider\": \"Dr. Johnson\"}", "tool_calls": None, "tool_result": {"tool_name": "verify_insurance", "result": {"insurance_carrier": "BlueCross BlueShield", "status": "active"}, "success": True}},
                {"role": "assistant", "content": "I found your insurance information, Ms. Garcia. You have active coverage with BlueCross BlueShield PPO Gold. Let me check your copay for tomorrow's visit.\n\n<tool_call>{\"name\": \"get_copay_info\", \"arguments\": {\"insurance_id\": \"BCB-998877665\", \"visit_type\": \"office_visit\"}}</tool_call>", "tool_calls": [{"name": "get_copay_info", "arguments": {"insurance_id": "BCB-998877665", "visit_type": "office_visit"}}], "tool_result": None},
                {"role": "tool_result", "content": "{\"copay\": 25.00, \"deductible\": 500.00, \"deductible_met\": 350.00, \"out_of_pocket_max\": 3000.00, \"out_of_pocket_spent\": 425.00}", "tool_calls": None, "tool_result": {"tool_name": "get_copay_info", "result": {"copay": 25.00, "deductible_met": 350.00}, "success": True}},
                {"role": "assistant", "content": "For tomorrow's office visit, your copay is $25, which is due at the time of your visit. Regarding your deductible, your annual deductible is $500 and you've met $350 of that so far, leaving $150 remaining. Your out-of-pocket maximum for the year is $3,000, and you've spent $425 so far. Please bring your $25 copay tomorrow. We accept cash, check, and all major credit cards. Do you have any other questions about your coverage?", "tool_calls": None, "tool_result": None}
            ]),
            "source": "generated",
            "original_source": None,
            "status": "pending",
            "quality_score": 0.88,
            "annotator_notes": None,
            "reviewed_at": None,
        }

        # Billing inquiry demo (with hallucination issue)
        billing_traj = {
            "id": f"demo-bill-{uuid4().hex[:8]}",
            "created_at": datetime.utcnow().isoformat(),
            "scenario_id": "billing-inquiry",
            "scenario_description": "Patient billing questions and payment assistance",
            "tools": json.dumps([{
                "type": "function",
                "function": {
                    "name": "get_patient_balance",
                    "description": "Get patient's current account balance and recent charges",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "patient_id": {"type": "string"},
                            "include_pending": {"type": "boolean", "description": "Include pending insurance claims"}
                        },
                        "required": ["patient_id"]
                    }
                }
            }]),
            "system_prompt": "You are a healthcare billing assistant. Help patients understand their bills, payment options, and account balances. Be empathetic and offer payment plan options when appropriate.",
            "messages": json.dumps([
                {"role": "user", "content": "I received a bill for $450 and I don't understand why it's so high. Patient ID: PT-33892", "tool_calls": None, "tool_result": None},
                {"role": "assistant", "content": "I understand receiving an unexpected bill can be concerning. Let me pull up your account to review the charges.\n\n<tool_call>{\"name\": \"get_patient_balance\", \"arguments\": {\"patient_id\": \"PT-33892\", \"include_pending\": true}}</tool_call>", "tool_calls": [{"name": "get_patient_balance", "arguments": {"patient_id": "PT-33892", "include_pending": True}}], "tool_result": None},
                {"role": "tool_result", "content": "{\"patient_name\": \"Robert Chen\", \"total_balance\": 450.00, \"charges\": [{\"date\": \"2025-12-15\", \"description\": \"Office Visit - Level 4\", \"amount\": 275.00, \"insurance_paid\": 220.00, \"patient_responsibility\": 55.00}, {\"date\": \"2025-12-15\", \"description\": \"Lab Work - Comprehensive Panel\", \"amount\": 850.00, \"insurance_paid\": 455.00, \"patient_responsibility\": 395.00}], \"pending_claims\": []}", "tool_calls": None, "tool_result": {"tool_name": "get_patient_balance", "result": {"total_balance": 450.00, "charges": [{"description": "Office Visit", "patient_responsibility": 55.00}, {"description": "Lab Work", "patient_responsibility": 395.00}]}, "success": True}},
                {"role": "assistant", "content": "I found your account, Mr. Chen. Let me break down that $450 balance for you. From your December 15th visit, you had two charges. The first was an office visit for $275, and after insurance paid $220, your portion is $55. The second charge was for lab work totaling $850, and after insurance covered $455, you're responsible for $395. That brings your total to $450. The lab work is the larger portion because comprehensive panels often apply toward your deductible. Would you like to set up a payment plan? We offer interest-free options for up to 12 months.", "tool_calls": None, "tool_result": None}
            ]),
            "source": "generated",
            "original_source": None,
            "status": "needs_edit",
            "quality_score": 0.62,
            "annotator_notes": "Hallucination: Assistant added details not in tool results (Level 4 coding explanation, specific lab tests included). Only balance and basic charge info was returned.",
            "reviewed_at": datetime.utcnow().isoformat(),
        }

        # Insert demo data
        for traj in [appointment_traj, insurance_traj, billing_traj]:
            await self.execute(
                """
                INSERT INTO sessions (
                    id, created_at, scenario_id, scenario_description,
                    tools, system_prompt, messages, source, original_source,
                    status, quality_score, annotator_notes, reviewed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    traj["id"],
                    traj["created_at"],
                    traj["scenario_id"],
                    traj["scenario_description"],
                    traj["tools"],
                    traj["system_prompt"],
                    traj["messages"],
                    traj["source"],
                    traj["original_source"],
                    traj["status"],
                    traj["quality_score"],
                    traj["annotator_notes"],
                    traj["reviewed_at"],
                ),
            )

        logger.info("Demo data seeded: 3 sample trajectories")

    async def close(self) -> None:
        """Close database connection."""
        if self._conn is not None:
            await self._conn.close()
            self._conn = None
            logger.debug("Database connection closed")

    async def execute(
        self, query: str, parameters: tuple[Any, ...] | dict[str, Any] | None = None
    ) -> aiosqlite.Cursor:
        """Execute a query and return the cursor."""
        conn = await self._get_connection()
        if parameters is None:
            cursor = await conn.execute(query)
        else:
            cursor = await conn.execute(query, parameters)
        await conn.commit()
        return cursor

    async def executemany(
        self, query: str, parameters: list[tuple[Any, ...]] | list[dict[str, Any]]
    ) -> aiosqlite.Cursor:
        """Execute a query with multiple parameter sets."""
        conn = await self._get_connection()
        cursor = await conn.executemany(query, parameters)
        await conn.commit()
        return cursor

    async def fetchone(
        self, query: str, parameters: tuple[Any, ...] | dict[str, Any] | None = None
    ) -> aiosqlite.Row | None:
        """Execute a query and fetch one result."""
        conn = await self._get_connection()
        if parameters is None:
            cursor = await conn.execute(query)
        else:
            cursor = await conn.execute(query, parameters)
        return await cursor.fetchone()

    async def fetchall(
        self, query: str, parameters: tuple[Any, ...] | dict[str, Any] | None = None
    ) -> list[aiosqlite.Row]:
        """Execute a query and fetch all results."""
        conn = await self._get_connection()
        if parameters is None:
            cursor = await conn.execute(query)
        else:
            cursor = await conn.execute(query, parameters)
        return await cursor.fetchall()


# Global database instance
_database: Database | None = None


def get_database() -> Database:
    """Get the global database instance."""
    global _database
    if _database is None:
        _database = Database()
    return _database
