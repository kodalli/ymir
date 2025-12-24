"""SQLite-based dataset storage with many-to-many session relationships."""

import json
from datetime import datetime

from loguru import logger

from ymir.core import Dataset, DatasetCreate, DatasetUpdate, Trajectory, TrajectoryStatus
from .database import Database, get_database


class DatasetStore:
    """SQLite-based storage for datasets and their session relationships."""

    def __init__(self, db: Database):
        self.db = db

    def _parse_dataset(self, row: dict) -> Dataset | None:
        """Parse a dataset from database row."""
        try:
            # Parse datetime fields
            created_at = datetime.fromisoformat(row["created_at"])
            updated_at = datetime.fromisoformat(row["updated_at"])

            # Parse JSON fields that were stored as strings
            tags = json.loads(row["tags"]) if row["tags"] else []
            metadata = json.loads(row["metadata"]) if row["metadata"] else {}

            return Dataset(
                id=row["id"],
                name=row["name"],
                description=row["description"],
                created_at=created_at,
                updated_at=updated_at,
                tags=tags,
                metadata=metadata,
                session_count=0,  # Will be computed separately
            )
        except Exception as e:
            logger.warning(f"Error parsing dataset: {e}")
            return None

    # Dataset CRUD
    async def create(self, data: DatasetCreate) -> Dataset:
        """Create a new dataset."""
        dataset = Dataset(
            name=data.name,
            description=data.description,
            tags=data.tags,
            metadata=data.metadata,
        )

        # Convert to database format
        tags_json = json.dumps(dataset.tags)
        metadata_json = json.dumps(dataset.metadata)

        await self.db.execute(
            """
            INSERT INTO datasets (id, name, description, created_at, updated_at, tags, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                dataset.id,
                dataset.name,
                dataset.description,
                dataset.created_at.isoformat(),
                dataset.updated_at.isoformat(),
                tags_json,
                metadata_json,
            ),
        )

        logger.info(f"Created dataset: {dataset.name} ({dataset.id})")
        return dataset

    async def get(self, dataset_id: str) -> Dataset | None:
        """Get a dataset by ID."""
        row = await self.db.fetchone(
            "SELECT * FROM datasets WHERE id = ?", (dataset_id,)
        )

        if not row:
            return None

        dataset = self._parse_dataset(dict(row))
        if dataset:
            # Get session count
            count_row = await self.db.fetchone(
                "SELECT COUNT(*) as count FROM dataset_sessions WHERE dataset_id = ?",
                (dataset_id,),
            )
            dataset.session_count = count_row["count"] if count_row else 0

        return dataset

    async def get_by_name(self, name: str) -> Dataset | None:
        """Get a dataset by name."""
        row = await self.db.fetchone(
            "SELECT * FROM datasets WHERE name = ?", (name,)
        )

        if not row:
            return None

        dataset = self._parse_dataset(dict(row))
        if dataset:
            # Get session count
            count_row = await self.db.fetchone(
                "SELECT COUNT(*) as count FROM dataset_sessions WHERE dataset_id = ?",
                (dataset.id,),
            )
            dataset.session_count = count_row["count"] if count_row else 0

        return dataset

    async def update(self, dataset_id: str, updates: DatasetUpdate) -> Dataset | None:
        """Update an existing dataset."""
        # First check if dataset exists
        existing = await self.get(dataset_id)
        if not existing:
            return None

        # Build update query dynamically based on provided fields
        update_fields = []
        params = []

        if updates.name is not None:
            update_fields.append("name = ?")
            params.append(updates.name)

        if updates.description is not None:
            update_fields.append("description = ?")
            params.append(updates.description)

        if updates.tags is not None:
            update_fields.append("tags = ?")
            params.append(json.dumps(updates.tags))

        if updates.metadata is not None:
            update_fields.append("metadata = ?")
            params.append(json.dumps(updates.metadata))

        # Always update updated_at
        update_fields.append("updated_at = ?")
        params.append(datetime.utcnow().isoformat())

        # Add dataset_id to params
        params.append(dataset_id)

        if update_fields:
            query = f"UPDATE datasets SET {', '.join(update_fields)} WHERE id = ?"
            await self.db.execute(query, tuple(params))

        logger.info(f"Updated dataset: {dataset_id}")

        # Return updated dataset
        return await self.get(dataset_id)

    async def delete(self, dataset_id: str) -> bool:
        """Delete a dataset and all its session relationships."""
        # Check if dataset exists
        existing = await self.get(dataset_id)
        if not existing:
            return False

        # Delete dataset (CASCADE will handle relationships)
        await self.db.execute("DELETE FROM datasets WHERE id = ?", (dataset_id,))

        logger.info(f"Deleted dataset: {existing.name} ({dataset_id})")
        return True

    async def list_all(self) -> list[Dataset]:
        """Get all datasets."""
        rows = await self.db.fetchall(
            "SELECT * FROM datasets ORDER BY created_at DESC"
        )

        datasets = []
        for row in rows:
            dataset = self._parse_dataset(dict(row))
            if dataset:
                # Get session count for each dataset
                count_row = await self.db.fetchone(
                    "SELECT COUNT(*) as count FROM dataset_sessions WHERE dataset_id = ?",
                    (dataset.id,),
                )
                dataset.session_count = count_row["count"] if count_row else 0
                datasets.append(dataset)

        return datasets

    # Session management
    async def add_sessions(self, dataset_id: str, session_ids: list[str]) -> int:
        """Add sessions to a dataset. Returns number of sessions added."""
        # Check if dataset exists
        existing = await self.get(dataset_id)
        if not existing:
            logger.error(f"Dataset not found: {dataset_id}")
            return 0

        added_count = 0
        for session_id in session_ids:
            try:
                # Insert only if not already exists (ignore duplicates)
                await self.db.execute(
                    """
                    INSERT OR IGNORE INTO dataset_sessions (dataset_id, session_id, added_at)
                    VALUES (?, ?, ?)
                    """,
                    (dataset_id, session_id, datetime.utcnow().isoformat()),
                )
                # Check if it was actually inserted
                count_row = await self.db.fetchone(
                    "SELECT changes() as changes"
                )
                if count_row and count_row["changes"] > 0:
                    added_count += 1
            except Exception as e:
                logger.warning(f"Error adding session {session_id} to dataset {dataset_id}: {e}")

        if added_count > 0:
            logger.info(f"Added {added_count} sessions to dataset {dataset_id}")

        return added_count

    async def remove_sessions(self, dataset_id: str, session_ids: list[str]) -> int:
        """Remove sessions from a dataset. Returns number of sessions removed."""
        if not session_ids:
            return 0

        # Create placeholders for IN clause
        placeholders = ",".join("?" * len(session_ids))
        query = f"DELETE FROM dataset_sessions WHERE dataset_id = ? AND session_id IN ({placeholders})"

        params = [dataset_id] + session_ids
        cursor = await self.db.execute(query, tuple(params))

        # Get number of rows affected
        count_row = await self.db.fetchone("SELECT changes() as changes")
        removed_count = count_row["changes"] if count_row else 0

        if removed_count > 0:
            logger.info(f"Removed {removed_count} sessions from dataset {dataset_id}")

        return removed_count

    async def get_sessions(
        self,
        dataset_id: str,
        status: TrajectoryStatus | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Trajectory]:
        """Get sessions in a dataset, optionally filtered by status."""
        # Build query with optional status filter
        if status:
            query = """
                SELECT s.* FROM sessions s
                JOIN dataset_sessions ds ON s.id = ds.session_id
                WHERE ds.dataset_id = ? AND s.status = ?
                ORDER BY s.created_at DESC
                LIMIT ? OFFSET ?
            """
            params = (dataset_id, status.value, limit, offset)
        else:
            query = """
                SELECT s.* FROM sessions s
                JOIN dataset_sessions ds ON s.id = ds.session_id
                WHERE ds.dataset_id = ?
                ORDER BY s.created_at DESC
                LIMIT ? OFFSET ?
            """
            params = (dataset_id, limit, offset)

        rows = await self.db.fetchall(query, params)

        sessions = []
        for row in rows:
            try:
                # Parse session data
                row_dict = dict(row)

                # Parse datetime fields
                created_at = datetime.fromisoformat(row_dict["created_at"])
                reviewed_at = (
                    datetime.fromisoformat(row_dict["reviewed_at"])
                    if row_dict.get("reviewed_at")
                    else None
                )

                # Parse JSON fields
                messages = json.loads(row_dict["messages"])
                tools = json.loads(row_dict["tools"])

                session = Trajectory(
                    id=row_dict["id"],
                    created_at=created_at,
                    scenario_id=row_dict["scenario_id"],
                    scenario_description=row_dict["scenario_description"],
                    tools=tools,
                    system_prompt=row_dict["system_prompt"],
                    messages=messages,
                    source=row_dict["source"],
                    original_source=row_dict.get("original_source"),
                    status=TrajectoryStatus(row_dict["status"]),
                    quality_score=row_dict.get("quality_score"),
                    annotator_notes=row_dict.get("annotator_notes"),
                    reviewed_at=reviewed_at,
                )
                sessions.append(session)
            except Exception as e:
                logger.warning(f"Error parsing session from database: {e}")

        return sessions

    async def get_session_count(self, dataset_id: str) -> int:
        """Get the number of sessions in a dataset."""
        row = await self.db.fetchone(
            "SELECT COUNT(*) as count FROM dataset_sessions WHERE dataset_id = ?",
            (dataset_id,),
        )
        return row["count"] if row else 0

    # Reverse lookup
    async def get_datasets_for_session(self, session_id: str) -> list[Dataset]:
        """Get all datasets that contain a given session."""
        rows = await self.db.fetchall(
            """
            SELECT d.* FROM datasets d
            JOIN dataset_sessions ds ON d.id = ds.dataset_id
            WHERE ds.session_id = ?
            ORDER BY d.name
            """,
            (session_id,),
        )

        datasets = []
        for row in rows:
            dataset = self._parse_dataset(dict(row))
            if dataset:
                # Get session count for each dataset
                count_row = await self.db.fetchone(
                    "SELECT COUNT(*) as count FROM dataset_sessions WHERE dataset_id = ?",
                    (dataset.id,),
                )
                dataset.session_count = count_row["count"] if count_row else 0
                datasets.append(dataset)

        return datasets


# Global store instance
_dataset_store: DatasetStore | None = None


def get_dataset_store() -> DatasetStore:
    """Get the global dataset store instance."""
    global _dataset_store
    if _dataset_store is None:
        db = get_database()
        _dataset_store = DatasetStore(db)
    return _dataset_store
