"""Routes for data management and trajectory annotation."""

import json
import math
from datetime import datetime

from fastapi import APIRouter, Form, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse

from ymir.core import TrajectoryStatus
from ymir.data import get_database, get_session_store
from ymir.pipeline.annotation import QualityScorer
from ymir.api.shared import render_page, templates

router = APIRouter(prefix="/annotation", tags=["annotation"])

# Shared scorer instance
_scorer = QualityScorer()


def get_store():
    """Get the session store."""
    db = get_database()
    return get_session_store(db)


@router.get("/", response_class=HTMLResponse)
async def annotation_page(request: Request):
    """Render the data management page."""
    store = get_store()

    # Get stats
    stats = await store.get_stats()

    # Get scenarios for filter dropdown
    scenarios = await store.get_unique_scenarios()

    return render_page(
        request,
        "annotation/index.html",
        {
            "stats": stats,
            "scenarios": scenarios,
        },
        page_title="Data",
    )


@router.get("/trajectories", response_class=HTMLResponse)
async def list_trajectories(
    request: Request,
    status: str | None = Query(None),
    scenario: str | None = Query(None),
    min_quality: float | None = Query(None),
    max_quality: float | None = Query(None),
    issue_hallucination: str | None = Query(None),
    issue_goal_drift: str | None = Query(None),
    issue_tool_skip: str | None = Query(None),
    issue_suspicious: str | None = Query(None),
    sort_by: str = Query("created_at"),
    sort_order: str = Query("desc"),
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=10, le=100),
):
    """List trajectories with filtering, sorting, and pagination."""
    store = get_store()

    # Parse status filter
    traj_status = None
    if status and status != "all":
        try:
            traj_status = TrajectoryStatus(status)
        except ValueError:
            pass

    # Calculate offset
    offset = (page - 1) * page_size

    # Query trajectories
    trajectories, total_count = await store.query(
        status=traj_status,
        scenario_id=scenario if scenario else None,
        min_quality=min_quality,
        max_quality=max_quality,
        sort_by=sort_by,
        sort_order=sort_order,
        limit=page_size,
        offset=offset,
    )

    # Add issues to each trajectory (computed on-the-fly)
    items = []
    for traj in trajectories:
        issues = _scorer.get_all_issues(traj)
        items.append({
            "trajectory": traj,
            "issues": issues,
        })

    # Filter by issue types (post-query filtering)
    issue_filters = {
        "hallucination": issue_hallucination == "1",
        "goal_drift": issue_goal_drift == "1",
        "tool_skipping": issue_tool_skip == "1",
        "suspicious_pattern": issue_suspicious == "1",
    }
    active_filters = [k for k, v in issue_filters.items() if v]

    if active_filters:
        filtered_items = []
        for item in items:
            issue_types = {i.type.value for i in item["issues"]}
            # Keep item if it has ANY of the selected issue types
            if any(f in issue_types for f in active_filters):
                filtered_items.append(item)
        items = filtered_items

    # Pagination info
    total_pages = math.ceil(total_count / page_size) if total_count > 0 else 1

    # Get scenarios for filter (only on initial load)
    scenarios = await store.get_unique_scenarios()

    return templates.TemplateResponse(
        "annotation/trajectory_table.html",
        {
            "request": request,
            "items": items,
            "total_count": total_count,
            "page": page,
            "page_size": page_size,
            "total_pages": total_pages,
            "sort_by": sort_by,
            "sort_order": sort_order,
            "status": status or "all",
            "scenario": scenario,
            "min_quality": min_quality,
            "max_quality": max_quality,
            "issue_hallucination": issue_hallucination,
            "issue_goal_drift": issue_goal_drift,
            "issue_tool_skip": issue_tool_skip,
            "issue_suspicious": issue_suspicious,
            "scenarios": scenarios,
        },
    )


@router.get("/trajectory/{trajectory_id}/expand", response_class=HTMLResponse)
async def expand_trajectory(request: Request, trajectory_id: str):
    """Get expanded detail view for a trajectory."""
    store = get_store()
    trajectory = await store.get(trajectory_id)

    if not trajectory:
        return HTMLResponse(content="Trajectory not found", status_code=404)

    # Get detailed issues
    issues = _scorer.get_all_issues(trajectory)

    return templates.TemplateResponse(
        "annotation/trajectory_detail.html",
        {
            "request": request,
            "trajectory": trajectory,
            "issues": issues,
        },
    )


@router.post("/trajectory/{trajectory_id}/review", response_class=HTMLResponse)
async def quick_review(
    request: Request,
    trajectory_id: str,
    status: str = Form(...),
    notes: str = Form(""),
):
    """Quick review action from expanded row."""
    store = get_store()

    # Get trajectory
    trajectory = await store.get(trajectory_id)
    if not trajectory:
        return HTMLResponse(content="Trajectory not found", status_code=404)

    # Update status
    try:
        traj_status = TrajectoryStatus(status)
    except ValueError:
        return HTMLResponse(content=f"Invalid status: {status}", status_code=400)

    trajectory.status = traj_status
    trajectory.annotator_notes = notes if notes else None
    trajectory.reviewed_at = datetime.utcnow()

    await store.update(trajectory)

    # Return updated row
    issues = _scorer.get_all_issues(trajectory)
    return templates.TemplateResponse(
        "annotation/trajectory_row.html",
        {
            "request": request,
            "item": {"trajectory": trajectory, "issues": issues},
        },
    )


@router.post("/bulk-action", response_class=HTMLResponse)
async def bulk_action(
    request: Request,
    action: str = Form(...),
    trajectory_ids: str = Form(...),
):
    """Perform bulk action on multiple trajectories."""
    store = get_store()

    try:
        ids = json.loads(trajectory_ids)
        if not isinstance(ids, list):
            return JSONResponse({"error": "trajectory_ids must be a JSON array"}, status_code=400)
    except json.JSONDecodeError:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    if action == "approve":
        count = await store.bulk_update_status(ids, TrajectoryStatus.APPROVED)
    elif action == "reject":
        count = await store.bulk_update_status(ids, TrajectoryStatus.REJECTED)
    else:
        return JSONResponse({"error": f"Unknown action: {action}"}, status_code=400)

    # Return updated stats
    stats = await store.get_stats()

    return templates.TemplateResponse(
        "annotation/stats_bar.html",
        {
            "request": request,
            "stats": stats,
            "action_result": {
                "action": action,
                "count": count,
            },
        },
    )


@router.get("/stats", response_class=JSONResponse)
async def get_stats():
    """Get annotation statistics."""
    store = get_store()
    stats = await store.get_stats()
    return JSONResponse(stats)


@router.get("/stats-bar", response_class=HTMLResponse)
async def stats_bar(request: Request):
    """Get stats bar HTML for dynamic refresh."""
    store = get_store()
    stats = await store.get_stats()
    return templates.TemplateResponse(
        "annotation/stats_bar.html",
        {"request": request, "stats": stats},
    )


@router.get("/scenarios", response_class=JSONResponse)
async def list_scenarios():
    """Get list of unique scenarios for filter dropdown."""
    store = get_store()
    scenarios = await store.get_unique_scenarios()
    return JSONResponse({"scenarios": scenarios})
