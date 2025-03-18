from fastapi import APIRouter
from ymir.routes.shared import rlhf_builder, triplet_dataset, templates
from fastapi.responses import HTMLResponse
from fastapi import Request

router = APIRouter()


@router.get("/dataset_counts")
async def dataset_counts():
    """Get the number of samples in each dataset"""
    return {"rlhf": len(rlhf_builder.get_samples()), "triplet": len(triplet_dataset)}


@router.post("/import_dataset")
async def import_dataset():
    """Import a dataset from a file"""
    # This would parse and import the uploaded file
    return {
        "status": "success",
        "message": "Dataset import functionality will be implemented soon",
    }


@router.get("/datasets", response_class=HTMLResponse)
async def datasets_page(request: Request):
    """Render the Datasets Management page"""
    return templates.TemplateResponse(
        "datasets_processing/datasets.html",
        {
            "request": request,
        },
    )
