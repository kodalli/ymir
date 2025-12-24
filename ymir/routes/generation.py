"""Routes for trajectory generation."""

import json
from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse

from ymir.functions import get_registry
from ymir.generators import TrajectoryGenerator
from ymir.llm import get_available_models
from ymir.personas import get_personas_for_category
from ymir.storage import get_store
from ymir.routes.shared import templates

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
            "default_scenario": scenarios[0] if scenarios else None,
        },
    )


@router.get("/scenario-info/{scenario_id}", response_class=HTMLResponse)
async def get_scenario_info(request: Request, scenario_id: str):
    """Get detailed info about a scenario for the generation UI."""
    registry = get_registry()
    scenario = registry.get_scenario(scenario_id)
    if not scenario:
        return HTMLResponse(content="Scenario not found", status_code=404)
    
    return templates.TemplateResponse(
        "generation/scenario_info.html",
        {"request": request, "scenario": scenario},
    )


@router.get("/scenario-info", response_class=HTMLResponse)
async def get_scenario_info_from_form(request: Request, scenario_id: str = None):
    """Get scenario info from form parameter (for HTMX)."""
    if not scenario_id:
        # Try to get from query params
        scenario_id = request.query_params.get("scenario_id")
        if not scenario_id:
            return HTMLResponse(content="No scenario_id provided", status_code=400)
    
    registry = get_registry()
    scenario = registry.get_scenario(scenario_id)
    if not scenario:
        return HTMLResponse(content="Scenario not found", status_code=404)
    
    return templates.TemplateResponse(
        "generation/scenario_info.html",
        {"request": request, "scenario": scenario},
    )


@router.post("/generate", response_class=HTMLResponse)
async def generate_trajectory(
    request: Request,
    scenario_id: str = Form(...),
    user_query: str = Form(""),
    user_situation: str = Form(None),
    user_background: str = Form(None),
    user_goal: str = Form(None),
    enabled_tools: str = Form(None),  # JSON array of tool names
    model: str = Form("qwen3:4b"),
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

    # Parse enabled_tools if provided
    enabled_tools_list = None
    if enabled_tools:
        try:
            enabled_tools_list = json.loads(enabled_tools)
            if not isinstance(enabled_tools_list, list):
                enabled_tools_list = None
        except json.JSONDecodeError:
            enabled_tools_list = None

    try:
        generator = TrajectoryGenerator(
            model=model,
            temperature=temperature,
        )
        trajectory = await generator.generate(
            scenario,
            user_query,
            user_situation=user_situation if user_situation else None,
            user_background=user_background if user_background else None,
            user_goal=user_goal if user_goal else None,
            enabled_tools=enabled_tools_list,
        )

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
    model: str = Form("qwen3:4b"),
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


# Wizard endpoints


@router.get("/step/{step_num}", response_class=HTMLResponse)
async def wizard_step(request: Request, step_num: int, scenario_id: str = None):
    """Get content for a specific wizard step."""
    registry = get_registry()
    scenarios = registry.list_scenarios()
    models = get_available_models()

    # Get scenario if specified
    scenario = None
    if scenario_id:
        scenario = registry.get_scenario(scenario_id)

    # Get personas for scenario category
    personas = []
    if scenario:
        personas = get_personas_for_category(scenario.category)

    template_map = {
        1: "generation/wizard/step_scenario.html",
        2: "generation/wizard/step_tools.html",
        3: "generation/wizard/step_actor.html",
        4: "generation/wizard/step_generate.html",
    }

    template = template_map.get(step_num, "generation/wizard/step_scenario.html")

    return templates.TemplateResponse(
        template,
        {
            "request": request,
            "step": step_num,
            "scenarios": scenarios,
            "scenario": scenario,
            "models": models,
            "personas": personas,
        },
    )


@router.get("/tools/{scenario_id}", response_class=HTMLResponse)
async def get_scenario_tools(request: Request, scenario_id: str):
    """Get tools for a scenario with toggle UI."""
    registry = get_registry()
    scenario = registry.get_scenario(scenario_id)

    if not scenario:
        return HTMLResponse(content="Scenario not found", status_code=404)

    return templates.TemplateResponse(
        "generation/wizard/step_tools.html",
        {"request": request, "scenario": scenario, "step": 2},
    )


@router.get("/personas/{scenario_id}", response_class=HTMLResponse)
async def get_persona_presets(request: Request, scenario_id: str):
    """Get persona presets for a scenario category."""
    registry = get_registry()
    scenario = registry.get_scenario(scenario_id)

    if not scenario:
        return HTMLResponse(content="Scenario not found", status_code=404)

    personas = get_personas_for_category(scenario.category)

    return templates.TemplateResponse(
        "generation/wizard/step_actor.html",
        {"request": request, "scenario": scenario, "personas": personas, "step": 3},
    )


@router.get("/stepper/{step_num}", response_class=HTMLResponse)
async def get_stepper(request: Request, step_num: int):
    """Get just the stepper component for a given step."""
    return templates.TemplateResponse(
        "generation/wizard/stepper.html",
        {"request": request, "step": step_num},
    )
