"""Routes for dataset management."""

from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from ymir.core import (
    DatasetCreate,
    DatasetUpdate,
    TrajectoryStatus,
)
from ymir.data import get_database, get_dataset_store, get_session_store
from ymir.data.converters import TrainingDataExporter

router = APIRouter(prefix="/datasets", tags=["datasets"])

EXPORT_DIR = Path("ymir/data/runtime/exports")
EXPORT_DIR.mkdir(parents=True, exist_ok=True)


# Request/Response models
class AddSessionsRequest(BaseModel):
    session_ids: list[str]


class RemoveSessionsRequest(BaseModel):
    session_ids: list[str]


class ExportRequest(BaseModel):
    format: str = "chatml"
    status_filter: str | None = None


# Dataset CRUD endpoints
@router.get("/", response_class=JSONResponse)
async def list_datasets():
    """List all datasets."""
    store = get_dataset_store()
    datasets = await store.list_all()
    return JSONResponse({
        "datasets": [d.model_dump(mode="json") for d in datasets]
    })


@router.post("/", response_class=JSONResponse)
async def create_dataset(data: DatasetCreate):
    """Create a new dataset."""
    store = get_dataset_store()

    # Check if name already exists
    existing = await store.get_by_name(data.name)
    if existing:
        raise HTTPException(status_code=400, detail=f"Dataset '{data.name}' already exists")

    dataset = await store.create(data)
    return JSONResponse(dataset.model_dump(mode="json"), status_code=201)


@router.get("/{dataset_id}", response_class=JSONResponse)
async def get_dataset(dataset_id: str):
    """Get a dataset by ID."""
    store = get_dataset_store()

    dataset = await store.get(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    return JSONResponse(dataset.model_dump(mode="json"))


@router.put("/{dataset_id}", response_class=JSONResponse)
async def update_dataset(dataset_id: str, updates: DatasetUpdate):
    """Update a dataset."""
    store = get_dataset_store()

    dataset = await store.update(dataset_id, updates)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    return JSONResponse(dataset.model_dump(mode="json"))


@router.delete("/{dataset_id}", response_class=JSONResponse)
async def delete_dataset(dataset_id: str):
    """Delete a dataset."""
    store = get_dataset_store()

    deleted = await store.delete(dataset_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Dataset not found")

    return JSONResponse({"success": True})


# Session management endpoints
@router.get("/{dataset_id}/sessions", response_class=JSONResponse)
async def get_dataset_sessions(
    dataset_id: str,
    status: str | None = None,
    limit: int = 100,
    offset: int = 0,
):
    """Get sessions in a dataset."""
    store = get_dataset_store()

    # Check dataset exists
    dataset = await store.get(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Parse status filter
    status_filter = None
    if status:
        try:
            status_filter = TrajectoryStatus(status)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid status: {status}")

    sessions = await store.get_sessions(dataset_id, status_filter, limit, offset)
    total = await store.get_session_count(dataset_id)

    return JSONResponse({
        "sessions": [s.model_dump(mode="json") for s in sessions],
        "total": total,
        "limit": limit,
        "offset": offset,
    })


@router.post("/{dataset_id}/sessions", response_class=JSONResponse)
async def add_sessions_to_dataset(dataset_id: str, data: AddSessionsRequest):
    """Add sessions to a dataset."""
    store = get_dataset_store()

    # Check dataset exists
    dataset = await store.get(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    added = await store.add_sessions(dataset_id, data.session_ids)
    return JSONResponse({"added": added})


@router.delete("/{dataset_id}/sessions", response_class=JSONResponse)
async def remove_sessions_from_dataset(dataset_id: str, data: RemoveSessionsRequest):
    """Remove sessions from a dataset."""
    store = get_dataset_store()

    # Check dataset exists
    dataset = await store.get(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    removed = await store.remove_sessions(dataset_id, data.session_ids)
    return JSONResponse({"removed": removed})


# Export endpoints
@router.post("/{dataset_id}/export", response_class=JSONResponse)
async def export_dataset(dataset_id: str, options: ExportRequest):
    """Export a dataset to JSONL."""
    store = get_dataset_store()

    # Check dataset exists
    dataset = await store.get(dataset_id)
    if not dataset:
        raise HTTPException(status_code=404, detail="Dataset not found")

    # Parse status filter
    status_filter = None
    if options.status_filter:
        try:
            status_filter = TrajectoryStatus(options.status_filter)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid status: {options.status_filter}")

    # Get sessions
    sessions = await store.get_sessions(dataset_id, status_filter, limit=100000, offset=0)

    if not sessions:
        raise HTTPException(status_code=400, detail="No sessions to export")

    # Generate filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = dataset.name.replace(" ", "_").replace("/", "-")[:50]
    filename = f"{safe_name}_{options.format}_{timestamp}.jsonl"
    export_path = EXPORT_DIR / filename

    # Export
    exporter = TrainingDataExporter()

    if options.format == "training":
        count = exporter.export_for_training(iter(sessions), str(export_path))
    else:
        count = exporter.export_jsonl(iter(sessions), str(export_path), options.format)

    return JSONResponse({
        "filename": filename,
        "count": count,
        "format": options.format,
    })


@router.get("/{dataset_id}/export/{filename}")
async def download_dataset_export(dataset_id: str, filename: str):
    """Download an exported dataset file."""
    file_path = EXPORT_DIR / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Export file not found")

    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type="application/jsonl",
    )


# Session search endpoint (convenience)
@router.get("/search/sessions", response_class=JSONResponse)
async def search_sessions(
    q: str,
    limit: int = 100,
):
    """Search sessions using full-text search."""
    db = get_database()
    session_store = get_session_store(db)

    sessions = await session_store.search(q, limit)
    return JSONResponse({
        "sessions": [s.model_dump(mode="json") for s in sessions],
        "total": len(sessions),
    })
