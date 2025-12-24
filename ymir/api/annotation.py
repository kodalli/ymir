"""Routes for trajectory annotation and review."""

import json
from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse

from ymir.pipeline.annotation import ReviewQueue
from ymir.core import TrajectoryStatus
from ymir.data import get_store
from ymir.api.shared import templates

router = APIRouter(prefix="/annotation", tags=["annotation"])


def get_queue() -> ReviewQueue:
    """Get a review queue instance."""
    return ReviewQueue(get_store())


@router.get("/", response_class=HTMLResponse)
async def annotation_page(request: Request):
    """Render the annotation review page."""
    queue = get_queue()
    stats = await queue.get_queue_stats()
    next_item = await queue.get_next_for_review()

    issues = []
    if next_item:
        issues = queue.get_quality_issues(next_item)

    return templates.TemplateResponse(
        "annotation/index.html",
        {
            "request": request,
            "stats": stats,
            "trajectory": next_item,
            "issues": issues,
        },
    )


@router.get("/next", response_class=HTMLResponse)
async def get_next_trajectory(request: Request, status: str = "pending"):
    """Get next trajectory for review."""
    queue = get_queue()

    try:
        traj_status = TrajectoryStatus(status)
    except ValueError:
        traj_status = TrajectoryStatus.PENDING

    trajectory = await queue.get_next_for_review(status=traj_status)

    if not trajectory:
        return templates.TemplateResponse(
            "annotation/queue_empty.html",
            {"request": request, "status": status},
        )

    issues = queue.get_quality_issues(trajectory)

    return templates.TemplateResponse(
        "annotation/trajectory_card.html",
        {"request": request, "trajectory": trajectory, "issues": issues},
    )


@router.post("/review/{trajectory_id}", response_class=HTMLResponse)
async def submit_review(
    request: Request,
    trajectory_id: str,
    status: str = Form(...),
    notes: str = Form(""),
):
    """Submit a review for a trajectory."""
    queue = get_queue()

    try:
        traj_status = TrajectoryStatus(status)
    except ValueError:
        return templates.TemplateResponse(
            "components/error.html",
            {"request": request, "error": f"Invalid status: {status}"},
        )

    trajectory = await queue.submit_review(
        trajectory_id=trajectory_id,
        status=traj_status,
        notes=notes if notes else None,
    )

    if not trajectory:
        return templates.TemplateResponse(
            "components/error.html",
            {"request": request, "error": "Trajectory not found"},
        )

    # Get next trajectory
    next_traj = await queue.get_next_for_review()
    stats = await queue.get_queue_stats()
    issues = queue.get_quality_issues(next_traj) if next_traj else []

    return templates.TemplateResponse(
        "annotation/review_result.html",
        {
            "request": request,
            "reviewed": trajectory,
            "next": next_traj,
            "stats": stats,
            "issues": issues,
        },
    )


@router.post("/bulk-approve", response_class=JSONResponse)
async def bulk_approve(
    request: Request,
    trajectory_ids: str = Form(...),
):
    """Bulk approve multiple trajectories."""
    queue = get_queue()

    try:
        ids = json.loads(trajectory_ids)
        if not isinstance(ids, list):
            return JSONResponse({"error": "trajectory_ids must be a JSON array"}, status_code=400)

        count = await queue.bulk_approve(ids)
        return JSONResponse({"success": True, "approved_count": count})
    except json.JSONDecodeError:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)


@router.post("/bulk-reject", response_class=JSONResponse)
async def bulk_reject(
    request: Request,
    trajectory_ids: str = Form(...),
):
    """Bulk reject multiple trajectories."""
    queue = get_queue()

    try:
        ids = json.loads(trajectory_ids)
        if not isinstance(ids, list):
            return JSONResponse({"error": "trajectory_ids must be a JSON array"}, status_code=400)

        count = await queue.bulk_reject(ids)
        return JSONResponse({"success": True, "rejected_count": count})
    except json.JSONDecodeError:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)


@router.get("/stats", response_class=JSONResponse)
async def get_stats():
    """Get annotation queue statistics."""
    queue = get_queue()
    stats = await queue.get_queue_stats()
    return JSONResponse(stats)


@router.post("/auto-score", response_class=JSONResponse)
async def auto_score_pending():
    """Score all pending trajectories."""
    queue = get_queue()
    count = await queue.auto_score_pending()
    return JSONResponse({"success": True, "scored_count": count})
