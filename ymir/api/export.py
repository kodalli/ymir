"""Routes for exporting trajectories."""

import os
from pathlib import Path

from fastapi import APIRouter, Form, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse

from ymir.data.converters import TrainingDataExporter
from ymir.core import TrajectoryStatus
from ymir.data import get_store
from ymir.api.shared import templates

router = APIRouter(prefix="/export", tags=["export"])

EXPORT_DIR = Path("ymir/data/runtime/exports")
EXPORT_DIR.mkdir(parents=True, exist_ok=True)


@router.get("/", response_class=HTMLResponse)
async def export_page(request: Request):
    """Render the export page."""
    store = get_store()

    # Get counts by status
    stats = {
        "approved": await store.count_by_status(TrajectoryStatus.APPROVED),
        "pending": await store.count_by_status(TrajectoryStatus.PENDING),
        "all": await store.count_all(),
    }

    # List existing exports
    exports = []
    for f in EXPORT_DIR.glob("*.jsonl"):
        exports.append({
            "name": f.name,
            "size": f.stat().st_size,
            "modified": f.stat().st_mtime,
        })

    return templates.TemplateResponse(
        "export/index.html",
        {
            "request": request,
            "stats": stats,
            "exports": sorted(exports, key=lambda x: x["modified"], reverse=True),
        },
    )


@router.post("/create", response_class=HTMLResponse)
async def create_export(
    request: Request,
    status_filter: str = Form("approved"),
    export_format: str = Form("chatml"),
    filename: str = Form(None),
):
    """Create a new export file."""
    store = get_store()
    exporter = TrainingDataExporter()

    # Determine which trajectories to export
    if status_filter == "all":
        trajectories = store.get_all()
    else:
        try:
            traj_status = TrajectoryStatus(status_filter)
            trajectories = iter(await store.get_by_status(traj_status))
        except ValueError:
            return templates.TemplateResponse(
                "components/error.html",
                {"request": request, "error": f"Invalid status: {status_filter}"},
            )

    # Generate filename if not provided
    if not filename:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"export_{status_filter}_{export_format}_{timestamp}.jsonl"

    export_path = EXPORT_DIR / filename

    try:
        if export_format == "training":
            count = exporter.export_for_training(trajectories, str(export_path))
        else:
            count = exporter.export_jsonl(trajectories, str(export_path), export_format)

        return templates.TemplateResponse(
            "export/export_result.html",
            {
                "request": request,
                "success": True,
                "filename": filename,
                "count": count,
                "format": export_format,
            },
        )
    except Exception as e:
        return templates.TemplateResponse(
            "components/error.html",
            {"request": request, "error": str(e)},
        )


@router.get("/download/{filename}")
async def download_export(filename: str):
    """Download an export file."""
    file_path = EXPORT_DIR / filename

    if not file_path.exists():
        return JSONResponse({"error": "File not found"}, status_code=404)

    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type="application/jsonl",
    )


@router.delete("/{filename}")
async def delete_export(filename: str):
    """Delete an export file."""
    file_path = EXPORT_DIR / filename

    if not file_path.exists():
        return JSONResponse({"error": "File not found"}, status_code=404)

    try:
        os.unlink(file_path)
        return JSONResponse({"success": True})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@router.get("/list", response_class=JSONResponse)
async def list_exports():
    """List all export files."""
    exports = []
    for f in EXPORT_DIR.glob("*.jsonl"):
        exports.append({
            "name": f.name,
            "size": f.stat().st_size,
            "modified": f.stat().st_mtime,
        })

    return JSONResponse({
        "exports": sorted(exports, key=lambda x: x["modified"], reverse=True)
    })
