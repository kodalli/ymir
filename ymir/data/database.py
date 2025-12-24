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

        # FTS5 for full-text search
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

        await conn.commit()
        logger.info(f"Database initialized at {self.db_path}")

        # Seed demo data if database is empty
        await self._seed_demo_data_if_empty()

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
                {"role": "assistant", "content": "Dr. Smith has the following openings on Tuesday, December 30th:\n\n- 9:00 AM\n- 10:30 AM\n- 2:00 PM\n- 3:30 PM\n\nWhich time works best for you? Also, may I have your name or patient ID to complete the booking?", "tool_calls": None, "tool_result": None},
                {"role": "user", "content": "The 2pm slot works. My name is John Davis, patient ID is PT-78432.", "tool_calls": None, "tool_result": None},
                {"role": "assistant", "content": "Perfect, let me book the 2:00 PM appointment for you, Mr. Davis.\n\n<tool_call>{\"name\": \"book_appointment\", \"arguments\": {\"patient_id\": \"PT-78432\", \"provider_id\": \"dr-smith-001\", \"date\": \"2025-12-30\", \"time\": \"14:00\", \"appointment_type\": \"follow_up\", \"reason\": \"Follow-up visit\"}}</tool_call>", "tool_calls": [{"name": "book_appointment", "arguments": {"patient_id": "PT-78432", "provider_id": "dr-smith-001", "date": "2025-12-30", "time": "14:00", "appointment_type": "follow_up", "reason": "Follow-up visit"}}], "tool_result": None},
                {"role": "tool_result", "content": "{\"confirmation_number\": \"APT-2025-89234\", \"status\": \"confirmed\", \"appointment\": {\"date\": \"2025-12-30\", \"time\": \"14:00\", \"provider\": \"Dr. Sarah Smith\", \"location\": \"Main Clinic - Room 204\", \"type\": \"follow_up\"}}", "tool_calls": None, "tool_result": {"tool_name": "book_appointment", "result": {"confirmation_number": "APT-2025-89234", "status": "confirmed"}, "success": True}},
                {"role": "assistant", "content": "Your appointment has been confirmed. Here are the details:\n\n**Confirmation #**: APT-2025-89234\n**Date**: Tuesday, December 30, 2025\n**Time**: 2:00 PM\n**Provider**: Dr. Sarah Smith\n**Location**: Main Clinic - Room 204\n**Visit Type**: Follow-up\n\nPlease arrive 15 minutes early and bring your insurance card and photo ID. Is there anything else I can help you with?", "tool_calls": None, "tool_result": None}
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
                {"role": "assistant", "content": "Here's your copay information for tomorrow's office visit:\n\n**Copay**: $25.00 (due at time of visit)\n\n**Deductible Status**:\n- Annual Deductible: $500.00\n- Amount Met: $350.00\n- Remaining: $150.00\n\n**Out-of-Pocket Maximum**: $3,000.00\n- Spent YTD: $425.00\n\nPlease bring your $25 copay tomorrow. We accept cash, check, and all major credit cards. Do you have any other questions about your coverage?", "tool_calls": None, "tool_result": None}
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
                {"role": "assistant", "content": "I found your account, Mr. Chen. Let me break down the $450 balance:\n\n**December 15, 2025 Visit:**\n\n1. **Office Visit (Level 4)**: $275.00\n   - Insurance Paid: $220.00\n   - Your Responsibility: $55.00\n   - *This was coded as a complex visit due to the comprehensive examination*\n\n2. **Lab Work (Comprehensive Panel)**: $850.00\n   - Insurance Paid: $455.00\n   - Your Responsibility: $395.00\n   - *This included CBC, metabolic panel, lipid panel, and thyroid function tests*\n\n**Total Patient Responsibility: $450.00**\n\nThe lab work is the larger portion because comprehensive panels often apply to your deductible. Would you like to set up a payment plan? We offer interest-free plans up to 12 months.", "tool_calls": None, "tool_result": None}
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
