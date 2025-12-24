"""SQLite database module for async operations."""

import aiosqlite
from pathlib import Path
from typing import Any

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
