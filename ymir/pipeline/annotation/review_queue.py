"""Review queue for trajectory annotation."""

from datetime import datetime
from typing import TYPE_CHECKING

from ymir.core import Message, Trajectory, TrajectoryStatus

from .quality_scorer import QualityScorer

if TYPE_CHECKING:
    from ymir.data import TrajectoryStore


class ReviewQueue:
    """Manages the annotation review queue."""

    def __init__(self, store: "TrajectoryStore"):
        self.store = store
        self.scorer = QualityScorer()

    async def get_next_for_review(
        self,
        status: TrajectoryStatus = TrajectoryStatus.PENDING,
        min_quality: float | None = None,
    ) -> Trajectory | None:
        """Get next trajectory needing review."""
        trajectories = await self.store.get_by_status(status)

        for traj in trajectories:
            # Apply quality filter if specified
            if min_quality is not None:
                if traj.quality_score is None:
                    traj.quality_score = self.scorer.score(traj)
                if traj.quality_score < min_quality:
                    continue
            return traj

        return None

    async def submit_review(
        self,
        trajectory_id: str,
        status: TrajectoryStatus,
        notes: str | None = None,
        edited_messages: list[Message] | None = None,
    ) -> Trajectory | None:
        """Submit a review for a trajectory."""
        trajectory = await self.store.get(trajectory_id)
        if not trajectory:
            return None

        trajectory.status = status
        trajectory.annotator_notes = notes
        trajectory.reviewed_at = datetime.utcnow()

        if edited_messages:
            trajectory.messages = edited_messages
            trajectory.source = "edited"

        # Recalculate quality score after edits
        trajectory.quality_score = self.scorer.score(trajectory)

        await self.store.update(trajectory)
        return trajectory

    async def bulk_approve(self, trajectory_ids: list[str]) -> int:
        """Bulk approve multiple trajectories."""
        count = 0
        for tid in trajectory_ids:
            result = await self.submit_review(tid, TrajectoryStatus.APPROVED)
            if result:
                count += 1
        return count

    async def bulk_reject(self, trajectory_ids: list[str]) -> int:
        """Bulk reject multiple trajectories."""
        count = 0
        for tid in trajectory_ids:
            result = await self.submit_review(tid, TrajectoryStatus.REJECTED)
            if result:
                count += 1
        return count

    async def get_queue_stats(self) -> dict[str, int]:
        """Get queue statistics."""
        return {
            "pending": await self.store.count_by_status(TrajectoryStatus.PENDING),
            "approved": await self.store.count_by_status(TrajectoryStatus.APPROVED),
            "rejected": await self.store.count_by_status(TrajectoryStatus.REJECTED),
            "needs_edit": await self.store.count_by_status(TrajectoryStatus.NEEDS_EDIT),
        }

    async def auto_score_pending(self) -> int:
        """Score all pending trajectories that don't have a quality score."""
        trajectories = await self.store.get_by_status(TrajectoryStatus.PENDING)
        count = 0

        for traj in trajectories:
            if traj.quality_score is None:
                traj.quality_score = self.scorer.score(traj)
                await self.store.update(traj)
                count += 1

        return count

    def get_quality_issues(self, trajectory: Trajectory) -> list[str]:
        """Get quality issues for a trajectory."""
        return self.scorer.get_issues(trajectory)
