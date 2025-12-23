"""Routes for trajectory generation."""

import json
from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse

from ymir.functions import get_registry
from ymir.generators import TrajectoryGenerator
from ymir.llm import get_available_models
from ymir.storage import get_store
from ymir.routes import templates

router = APIRouter(prefix="/generation", tags=["generation"])


@router.get("/", response_class=HTMLResponse)
async def generation_page(request: Request):
    """Render the trajectory generation page."""
    registry = get_registry()
    scenarios = registry.list_scenarios()
    models = get_available_models()

    return templates.TemplateResponse(
        "generation/index.html",
        {
            "request": request,
            "scenarios": scenarios,
            "models": models,
        },
    )


@router.post("/generate", response_class=HTMLResponse)
async def generate_trajectory(
    request: Request,
    scenario_id: str = Form(...),
    user_query: str = Form(...),
    model: str = Form("llama3.2"),
    temperature: float = Form(0.7),
    save: bool = Form(True),
):
    """Generate a single trajectory."""
    registry = get_registry()
    scenario = registry.get_scenario(scenario_id)

    if not scenario:
        return templates.TemplateResponse(
            "components/error.html",
            {"request": request, "error": f"Scenario not found: {scenario_id}"},
        )

    try:
        generator = TrajectoryGenerator(
            model=model,
            temperature=temperature,
        )
        trajectory = await generator.generate(scenario, user_query)

        if save:
            store = get_store()
            await store.save(trajectory)

        return templates.TemplateResponse(
            "generation/trajectory_preview.html",
            {"request": request, "trajectory": trajectory, "saved": save},
        )
    except Exception as e:
        return templates.TemplateResponse(
            "components/error.html",
            {"request": request, "error": str(e)},
        )


@router.post("/generate-batch", response_class=JSONResponse)
async def generate_batch(
    request: Request,
    scenario_id: str = Form(...),
    queries_json: str = Form(...),  # JSON array of queries
    model: str = Form("llama3.2"),
    temperature: float = Form(0.7),
):
    """Generate multiple trajectories."""
    registry = get_registry()
    scenario = registry.get_scenario(scenario_id)

    if not scenario:
        return JSONResponse({"error": f"Scenario not found: {scenario_id}"}, status_code=404)

    try:
        queries = json.loads(queries_json)
        if not isinstance(queries, list):
            return JSONResponse({"error": "queries_json must be a JSON array"}, status_code=400)

        generator = TrajectoryGenerator(
            model=model,
            temperature=temperature,
        )
        trajectories = await generator.generate_batch(scenario, queries)

        store = get_store()
        for traj in trajectories:
            await store.save(traj)

        return JSONResponse({
            "success": True,
            "count": len(trajectories),
            "ids": [t.id for t in trajectories],
        })
    except json.JSONDecodeError:
        return JSONResponse({"error": "Invalid JSON for queries"}, status_code=400)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@router.get("/models", response_class=JSONResponse)
async def list_models():
    """List available Ollama models."""
    models = get_available_models()
    return JSONResponse({"models": models})
