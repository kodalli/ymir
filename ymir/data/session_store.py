"""SQLite-backed trajectory storage."""

import json
from datetime import datetime
from typing import AsyncIterator

from loguru import logger

from ymir.core import Trajectory, TrajectoryStatus

from .database import Database


class SessionStore:
    """SQLite-backed storage for trajectories."""

    def __init__(self, db: Database):
        self.db = db

    async def save(self, trajectory: Trajectory) -> None:
        """Save a trajectory to storage."""
        try:
            data = self._serialize_trajectory(trajectory)
            await self.db.execute(
                """
                INSERT INTO sessions (
                    id, created_at, scenario_id, scenario_description,
                    tools, system_prompt, messages, source, original_source,
                    status, quality_score, annotator_notes, reviewed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    data["id"],
                    data["created_at"],
                    data["scenario_id"],
                    data["scenario_description"],
                    data["tools"],
                    data["system_prompt"],
                    data["messages"],
                    data["source"],
                    data["original_source"],
                    data["status"],
                    data["quality_score"],
                    data["annotator_notes"],
                    data["reviewed_at"],
                ),
            )
            logger.debug(f"Saved trajectory {trajectory.id}")
        except Exception as e:
            logger.error(f"Error saving trajectory {trajectory.id}: {e}")
            raise

    async def get(self, trajectory_id: str) -> Trajectory | None:
        """Get a trajectory by ID."""
        try:
            row = await self.db.fetchone(
                "SELECT * FROM sessions WHERE id = ?", (trajectory_id,)
            )
            if row is None:
                return None
            return self._deserialize_trajectory(row)
        except Exception as e:
            logger.error(f"Error getting trajectory {trajectory_id}: {e}")
            return None

    async def update(self, trajectory: Trajectory) -> None:
        """Update an existing trajectory."""
        try:
            data = self._serialize_trajectory(trajectory)
            await self.db.execute(
                """
                UPDATE sessions SET
                    created_at = ?,
                    scenario_id = ?,
                    scenario_description = ?,
                    tools = ?,
                    system_prompt = ?,
                    messages = ?,
                    source = ?,
                    original_source = ?,
                    status = ?,
                    quality_score = ?,
                    annotator_notes = ?,
                    reviewed_at = ?
                WHERE id = ?
                """,
                (
                    data["created_at"],
                    data["scenario_id"],
                    data["scenario_description"],
                    data["tools"],
                    data["system_prompt"],
                    data["messages"],
                    data["source"],
                    data["original_source"],
                    data["status"],
                    data["quality_score"],
                    data["annotator_notes"],
                    data["reviewed_at"],
                    data["id"],
                ),
            )
            logger.debug(f"Updated trajectory {trajectory.id}")
        except Exception as e:
            logger.error(f"Error updating trajectory {trajectory.id}: {e}")
            raise

    async def delete(self, trajectory_id: str) -> bool:
        """Delete a trajectory."""
        try:
            await self.db.execute(
                "DELETE FROM sessions WHERE id = ?", (trajectory_id,)
            )
            deleted = True
            if deleted:
                logger.debug(f"Deleted trajectory {trajectory_id}")
            return deleted
        except Exception as e:
            logger.error(f"Error deleting trajectory {trajectory_id}: {e}")
            return False

    async def get_by_status(self, status: TrajectoryStatus) -> list[Trajectory]:
        """Get all trajectories with a given status."""
        try:
            rows = await self.db.fetchall(
                "SELECT * FROM sessions WHERE status = ? ORDER BY created_at DESC",
                (status.value,),
            )
            return [self._deserialize_trajectory(row) for row in rows]
        except Exception as e:
            logger.error(f"Error getting trajectories by status {status}: {e}")
            return []

    async def count_by_status(self, status: TrajectoryStatus) -> int:
        """Count trajectories by status."""
        try:
            row = await self.db.fetchone(
                "SELECT COUNT(*) as count FROM sessions WHERE status = ?",
                (status.value,),
            )
            return row["count"] if row else 0
        except Exception as e:
            logger.error(f"Error counting trajectories by status {status}: {e}")
            return 0

    async def get_all(self) -> AsyncIterator[Trajectory]:
        """Get all trajectories."""
        try:
            rows = await self.db.fetchall(
                "SELECT * FROM sessions ORDER BY created_at DESC"
            )
            for row in rows:
                yield self._deserialize_trajectory(row)
        except Exception as e:
            logger.error(f"Error getting all trajectories: {e}")

    async def count_all(self) -> int:
        """Count all trajectories."""
        try:
            row = await self.db.fetchone("SELECT COUNT(*) as count FROM sessions")
            return row["count"] if row else 0
        except Exception as e:
            logger.error(f"Error counting all trajectories: {e}")
            return 0

    async def get_by_scenario(self, scenario_id: str) -> list[Trajectory]:
        """Get all trajectories for a specific scenario."""
        try:
            rows = await self.db.fetchall(
                "SELECT * FROM sessions WHERE scenario_id = ? ORDER BY created_at DESC",
                (scenario_id,),
            )
            return [self._deserialize_trajectory(row) for row in rows]
        except Exception as e:
            logger.error(f"Error getting trajectories by scenario {scenario_id}: {e}")
            return []

    async def query(
        self,
        status: TrajectoryStatus | None = None,
        scenario_id: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Trajectory]:
        """Query trajectories with filters."""
        try:
            conditions = []
            params = []

            if status is not None:
                conditions.append("status = ?")
                params.append(status.value)

            if scenario_id is not None:
                conditions.append("scenario_id = ?")
                params.append(scenario_id)

            where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
            query = f"""
                SELECT * FROM sessions
                {where_clause}
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
            """
            params.extend([limit, offset])

            rows = await self.db.fetchall(query, tuple(params))
            return [self._deserialize_trajectory(row) for row in rows]
        except Exception as e:
            logger.error(f"Error querying trajectories: {e}")
            return []

    async def search(self, text: str, limit: int = 100) -> list[Trajectory]:
        """Search trajectories using FTS5 full-text search."""
        try:
            # Search in FTS5 table and join with main table
            rows = await self.db.fetchall(
                """
                SELECT s.* FROM sessions s
                INNER JOIN sessions_fts fts ON s.id = fts.id
                WHERE sessions_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (text, limit),
            )
            return [self._deserialize_trajectory(row) for row in rows]
        except Exception as e:
            logger.error(f"Error searching trajectories: {e}")
            return []

    async def bulk_update_status(
        self, ids: list[str], status: TrajectoryStatus
    ) -> int:
        """Update status for multiple trajectories."""
        try:
            if not ids:
                return 0

            # Create placeholders for the IN clause
            placeholders = ",".join(["?" for _ in ids])
            query = f"""
                UPDATE sessions
                SET status = ?, reviewed_at = ?
                WHERE id IN ({placeholders})
            """
            params = [status.value, datetime.utcnow().isoformat()] + ids

            await self.db.execute(query, tuple(params))
            logger.debug(f"Updated status for {len(ids)} sessions")
            return len(ids)
        except Exception as e:
            logger.error(f"Error bulk updating status: {e}")
            return 0

    def _serialize_trajectory(self, trajectory: Trajectory) -> dict:
        """Convert trajectory to database-storable format."""
        return {
            "id": trajectory.id,
            "created_at": trajectory.created_at.isoformat(),
            "scenario_id": trajectory.scenario_id,
            "scenario_description": trajectory.scenario_description,
            "tools": json.dumps(trajectory.tools),
            "system_prompt": trajectory.system_prompt,
            "messages": json.dumps([msg.model_dump(mode="json") for msg in trajectory.messages]),
            "source": trajectory.source,
            "original_source": trajectory.original_source,
            "status": trajectory.status.value,
            "quality_score": trajectory.quality_score,
            "annotator_notes": trajectory.annotator_notes,
            "reviewed_at": trajectory.reviewed_at.isoformat()
            if trajectory.reviewed_at
            else None,
        }

    def _deserialize_trajectory(self, row: dict) -> Trajectory:
        """Parse trajectory from database row."""
        # Parse JSON fields
        tools = json.loads(row["tools"])
        messages_data = json.loads(row["messages"])

        # Parse datetime fields
        created_at = datetime.fromisoformat(row["created_at"])
        reviewed_at = (
            datetime.fromisoformat(row["reviewed_at"])
            if row["reviewed_at"]
            else None
        )

        # Parse status enum
        status = TrajectoryStatus(row["status"])

        return Trajectory(
            id=row["id"],
            created_at=created_at,
            scenario_id=row["scenario_id"],
            scenario_description=row["scenario_description"],
            tools=tools,
            system_prompt=row["system_prompt"],
            messages=messages_data,
            source=row["source"],
            original_source=row["original_source"],
            status=status,
            quality_score=row["quality_score"],
            annotator_notes=row["annotator_notes"],
            reviewed_at=reviewed_at,
        )


# Global store instance
_session_store: SessionStore | None = None


def get_session_store(db: Database) -> SessionStore:
    """Get the global session store instance."""
    global _session_store
    if _session_store is None:
        _session_store = SessionStore(db)
    return _session_store
