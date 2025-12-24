"""File-based trajectory storage."""

import json
from datetime import datetime
from pathlib import Path
from typing import Iterator

from loguru import logger

from ymir.core import Trajectory, TrajectoryStatus


class TrajectoryStore:
    """Simple file-based storage for trajectories."""

    def __init__(self, data_dir: str = "ymir/data/runtime/trajectories"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._index: dict[str, Path] = {}
        self._load_index()

    def _load_index(self) -> None:
        """Load index of all trajectories."""
        for jsonl_file in self.data_dir.glob("*.jsonl"):
            try:
                with open(jsonl_file, "r") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        data = json.loads(line)
                        traj_id = data.get("id")
                        if traj_id:
                            self._index[traj_id] = jsonl_file
            except Exception as e:
                logger.warning(f"Error loading index from {jsonl_file}: {e}")

    def _get_file_for_date(self, date: datetime | None = None) -> Path:
        """Get the file path for a given date."""
        if date is None:
            date = datetime.utcnow()
        return self.data_dir / f"trajectories_{date.strftime('%Y%m%d')}.jsonl"

    async def save(self, trajectory: Trajectory) -> None:
        """Save a trajectory to storage."""
        file_path = self._get_file_for_date(trajectory.created_at)

        # Write to file
        with open(file_path, "a") as f:
            data = trajectory.model_dump(mode="json")
            # Convert datetime objects to ISO format strings
            data["created_at"] = trajectory.created_at.isoformat()
            if trajectory.reviewed_at:
                data["reviewed_at"] = trajectory.reviewed_at.isoformat()
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

        # Update index
        self._index[trajectory.id] = file_path

    async def get(self, trajectory_id: str) -> Trajectory | None:
        """Get a trajectory by ID."""
        file_path = self._index.get(trajectory_id)
        if not file_path or not file_path.exists():
            return None

        try:
            with open(file_path, "r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    if data.get("id") == trajectory_id:
                        return self._parse_trajectory(data)
        except Exception as e:
            logger.error(f"Error reading trajectory {trajectory_id}: {e}")

        return None

    async def update(self, trajectory: Trajectory) -> None:
        """Update an existing trajectory."""
        file_path = self._index.get(trajectory.id)
        if not file_path or not file_path.exists():
            # New trajectory, just save it
            await self.save(trajectory)
            return

        # Read all lines, update the matching one, rewrite file
        lines = []
        updated = False

        try:
            with open(file_path, "r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    if data.get("id") == trajectory.id:
                        # Update this entry
                        data = trajectory.model_dump(mode="json")
                        data["created_at"] = trajectory.created_at.isoformat()
                        if trajectory.reviewed_at:
                            data["reviewed_at"] = trajectory.reviewed_at.isoformat()
                        updated = True
                    lines.append(json.dumps(data, ensure_ascii=False))

            if updated:
                with open(file_path, "w") as f:
                    f.write("\n".join(lines) + "\n")
            else:
                # Not found in file, append
                await self.save(trajectory)

        except Exception as e:
            logger.error(f"Error updating trajectory {trajectory.id}: {e}")

    async def delete(self, trajectory_id: str) -> bool:
        """Delete a trajectory."""
        file_path = self._index.get(trajectory_id)
        if not file_path or not file_path.exists():
            return False

        # Read all lines except the one to delete
        lines = []
        deleted = False

        try:
            with open(file_path, "r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    if data.get("id") == trajectory_id:
                        deleted = True
                        continue
                    lines.append(line.strip())

            if deleted:
                with open(file_path, "w") as f:
                    f.write("\n".join(lines) + "\n" if lines else "")
                del self._index[trajectory_id]
                return True

        except Exception as e:
            logger.error(f"Error deleting trajectory {trajectory_id}: {e}")

        return False

    async def get_by_status(self, status: TrajectoryStatus) -> list[Trajectory]:
        """Get all trajectories with a given status."""
        trajectories = []

        for jsonl_file in self.data_dir.glob("*.jsonl"):
            try:
                with open(jsonl_file, "r") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        data = json.loads(line)
                        if data.get("status") == status.value:
                            traj = self._parse_trajectory(data)
                            if traj:
                                trajectories.append(traj)
            except Exception as e:
                logger.warning(f"Error reading {jsonl_file}: {e}")

        return trajectories

    async def count_by_status(self, status: TrajectoryStatus) -> int:
        """Count trajectories by status."""
        count = 0

        for jsonl_file in self.data_dir.glob("*.jsonl"):
            try:
                with open(jsonl_file, "r") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        data = json.loads(line)
                        if data.get("status") == status.value:
                            count += 1
            except Exception as e:
                logger.warning(f"Error reading {jsonl_file}: {e}")

        return count

    async def get_all(self) -> Iterator[Trajectory]:
        """Get all trajectories."""
        for jsonl_file in sorted(self.data_dir.glob("*.jsonl")):
            try:
                with open(jsonl_file, "r") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        data = json.loads(line)
                        traj = self._parse_trajectory(data)
                        if traj:
                            yield traj
            except Exception as e:
                logger.warning(f"Error reading {jsonl_file}: {e}")

    async def count_all(self) -> int:
        """Count all trajectories."""
        count = 0
        for jsonl_file in self.data_dir.glob("*.jsonl"):
            try:
                with open(jsonl_file, "r") as f:
                    for line in f:
                        if line.strip():
                            count += 1
            except Exception as e:
                logger.warning(f"Error counting {jsonl_file}: {e}")
        return count

    def _parse_trajectory(self, data: dict) -> Trajectory | None:
        """Parse a trajectory from JSON data."""
        try:
            # Handle datetime parsing
            if isinstance(data.get("created_at"), str):
                data["created_at"] = datetime.fromisoformat(data["created_at"])
            if data.get("reviewed_at") and isinstance(data["reviewed_at"], str):
                data["reviewed_at"] = datetime.fromisoformat(data["reviewed_at"])

            # Handle status enum
            if isinstance(data.get("status"), str):
                data["status"] = TrajectoryStatus(data["status"])

            return Trajectory(**data)
        except Exception as e:
            logger.warning(f"Error parsing trajectory: {e}")
            return None


# Global store instance
_store: TrajectoryStore | None = None


def get_store() -> TrajectoryStore:
    """Get the global trajectory store instance."""
    global _store
    if _store is None:
        _store = TrajectoryStore()
    return _store
