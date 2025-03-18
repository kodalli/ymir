from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse
from ymir.routes.shared import get_supported_providers, templates, triplet_dataset
from ymir.triplets.text_to_triplets import extract_triplets
from loguru import logger
from pydantic import BaseModel
from typing import Optional

router = APIRouter()


class TripletRequest(BaseModel):
    subject: str
    predicate: str
    object: str
    description: Optional[str] = ""


@router.get("/triplet", response_class=HTMLResponse)
async def triplet_page(request: Request):
    """Render the Triplet Generation tool page"""
    return templates.TemplateResponse(
        "triplet_processing/triplet.html",
        {
            "request": request,
            "providers": get_supported_providers(),
        },
    )


@router.post("/extract_triplets")
async def process_triplets(
    text: str = Form(...),
    provider: str = Form(...),
    model: str = Form(...),
    entity_types: str = Form("organization,person,geo,event,category"),
):
    """Extract knowledge graph triplets from text using the selected LLM"""
    try:
        # Split entity types string into a list
        entity_types_list = [t.strip() for t in entity_types.split(",")]

        # Use our extract_triplets function with the specified provider and model
        result = extract_triplets(
            text=text,
            provider=provider,
            model_name=model,
            entity_types=entity_types_list,
        )

        # Add the extracted triplets to our dataset
        for triplet in result["triplets"]:
            triplet_dataset.append(
                {
                    "id": len(triplet_dataset) + 1,
                    "subject": triplet.subject,
                    "predicate": triplet.predicate,
                    "object": triplet.object,
                    "description": triplet.description,
                    "confidence": triplet.confidence,
                    "created_at": "2023-06-01T12:00:00Z",  # This would be the current timestamp in production
                }
            )

        # Format the triplets for display
        formatted_triplets = []
        for t in result["triplets"]:
            formatted_triplets.append(
                {
                    "subject": t.subject,
                    "predicate": t.predicate,
                    "object": t.object,
                    "description": t.description,
                    "confidence": t.confidence,
                }
            )

        # Return HTML for displaying the results
        return templates.TemplateResponse(
            "triplet_processing/triplet_extraction_results.html",
            {
                "triplets": formatted_triplets,
                "count": len(formatted_triplets),
                "entities": len(result["entities"]),
            },
        )
    except Exception as e:
        logger.error(f"Error extracting triplets: {e}")
        return f"""
        <div class="p-4 bg-red-100 text-red-700 rounded-md">
            <h3 class="font-bold">Error</h3>
            <p>{str(e)}</p>
        </div>
        """


@router.post("/add_manual_triplet")
async def add_manual_triplet(triplet: TripletRequest):
    """Add a manually created SPO triplet to the dataset"""
    # Add the triplet to our in-memory storage
    triplet_dataset.append(
        {
            "id": len(triplet_dataset) + 1,
            "subject": triplet.subject,
            "predicate": triplet.predicate,
            "object": triplet.object,
            "description": triplet.description,
            "confidence": 1.0,  # Manual triplets have full confidence
            "created_at": "2023-06-01T12:00:00Z",  # This would be the current timestamp in production
        }
    )

    return {
        "status": "success",
        "message": "SPO triplet added successfully",
        "count": len(triplet_dataset),
    }


@router.get("/view_triplets")
async def view_triplets(request: Request, format: str = "table", query: str = ""):
    """Get the triplet dataset"""
    # Filter the dataset if there's a search query
    data = triplet_dataset
    if query:
        filtered_data = []
        for triplet in data:
            if (
                query.lower() in triplet.get("subject", "").lower()
                or query.lower() in triplet.get("predicate", "").lower()
                or query.lower() in triplet.get("object", "").lower()
                or query.lower() in triplet.get("description", "").lower()
            ):
                filtered_data.append(triplet)
        data = filtered_data

    # Return as JSON if requested
    if format == "json":
        return JSONResponse(content={"data": data})

    # Otherwise return as HTML table
    return templates.TemplateResponse(
        "triplet_processing/triplet_table.html",
        {
            "request": request,
            "data": data,
        },
    )


@router.post("/download_triplets")
async def download_triplets():
    """Download the triplet dataset"""
    # In a real app, this would save to disk or provide a download link
    count = len(triplet_dataset)
    if count == 0:
        return {"status": "error", "message": "No triplets to download"}

    return {
        "status": "success",
        "message": f"Dataset with {count} triplets would be downloaded",
        "count": count,
    }


@router.delete("/delete_triplet/{triplet_id}")
async def delete_triplet(request: Request, triplet_id: int):
    """Delete a triplet from the dataset"""
    global triplet_dataset

    # Find the triplet by ID
    for i, triplet in enumerate(triplet_dataset):
        if triplet["id"] == triplet_id:
            # Remove the triplet
            del triplet_dataset[i]

            # Return the updated table
            return await view_triplets(request)

    # If triplet not found, return an error
    return JSONResponse(
        content={
            "status": "error",
            "message": f"Triplet with ID {triplet_id} not found",
        }
    )
