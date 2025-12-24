"""Routes for dataset import and conversion."""

import tempfile
import os
from pathlib import Path

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from ymir.data.converters import APIGenMTConverter, HermesFCConverter
from ymir.data import get_store
from ymir.api.shared import templates

router = APIRouter(prefix="/conversion", tags=["conversion"])

CONVERTERS = {
    "apigen-mt": APIGenMTConverter(),
    "hermes-fc": HermesFCConverter(),
}


@router.get("/", response_class=HTMLResponse)
async def conversion_page(request: Request):
    """Render the dataset conversion page."""
    return templates.TemplateResponse(
        "conversion/index.html",
        {
            "request": request,
            "formats": list(CONVERTERS.keys()),
        },
    )


@router.post("/import", response_class=HTMLResponse)
async def import_dataset(
    request: Request,
    file: UploadFile = File(...),
    format: str = Form(...),
    auto_score: bool = Form(True),
):
    """Import and convert a dataset file."""
    if format not in CONVERTERS:
        return templates.TemplateResponse(
            "components/error.html",
            {"request": request, "error": f"Unknown format: {format}"},
        )

    converter = CONVERTERS[format]
    store = get_store()

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jsonl") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Import with quality scoring
        from ymir.pipeline.annotation import QualityScorer
        scorer = QualityScorer() if auto_score else None

        count = 0
        errors = 0

        for trajectory in converter.convert_file(tmp_path):
            if scorer:
                trajectory.quality_score = scorer.score(trajectory)
            await store.save(trajectory)
            count += 1

        return templates.TemplateResponse(
            "conversion/import_result.html",
            {
                "request": request,
                "success": True,
                "count": count,
                "errors": errors,
                "format": format,
                "filename": file.filename,
            },
        )
    except Exception as e:
        return templates.TemplateResponse(
            "components/error.html",
            {"request": request, "error": str(e)},
        )
    finally:
        os.unlink(tmp_path)


@router.get("/stats", response_class=HTMLResponse)
async def get_stats(request: Request):
    """Get storage statistics as HTML."""
    store = get_store()
    total = await store.count_all()

    from ymir.core import TrajectoryStatus
    stats = {
        "total": total,
        "pending": await store.count_by_status(TrajectoryStatus.PENDING),
        "approved": await store.count_by_status(TrajectoryStatus.APPROVED),
        "rejected": await store.count_by_status(TrajectoryStatus.REJECTED),
        "needs_edit": await store.count_by_status(TrajectoryStatus.NEEDS_EDIT),
    }

    return templates.TemplateResponse(
        "conversion/stats.html",
        {"request": request, "stats": stats},
    )
