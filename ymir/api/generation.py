"""Routes for trajectory generation."""

import json
from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse

from ymir.functions import get_registry
from ymir.functions.schemas import FunctionDefinition, ScenarioTemplate
from ymir.pipeline import TrajectoryGenerator
from ymir.pipeline.llm import get_available_models
from ymir.pipeline.personas import get_personas_for_category
from ymir.pipeline.actor_generator import ActorGenerator
from ymir.data import get_store, get_database
from ymir.data.scenario_store import get_scenario_store
from ymir.core.scenario_schemas import ScenarioWithTools
from ymir.api.shared import render_page, templates

router = APIRouter(prefix="/generation", tags=["generation"])


def scenario_with_tools_to_template(scenario: ScenarioWithTools) -> ScenarioTemplate:
    """Convert a ScenarioWithTools from DB to a ScenarioTemplate for generation."""
    # Convert tools to function definitions
    functions = [
        FunctionDefinition(
            name=tool.name,
            description=tool.description,
            parameters=tool.parameters,
            category=tool.category or scenario.category or "general",
        )
        for tool in scenario.tools
    ]

    return ScenarioTemplate(
        id=scenario.id,
        name=scenario.name,
        description=scenario.description,
        category=scenario.category or "general",
        functions=functions,
        system_prompt=scenario.system_prompt,
        example_queries=scenario.example_queries,
        mock_responses=scenario.mock_responses,
    )


@router.get("/", response_class=HTMLResponse)
async def generation_page(request: Request):
    """Render the trajectory generation page."""
    store = get_scenario_store()
    scenarios = await store.list_scenarios()
    models = get_available_models()

    return render_page(
        request,
        "generation/index.html",
        {
            "scenarios": scenarios,
            "models": models,
            "default_scenario": scenarios[0] if scenarios else None,
        },
        page_title="Generate",
    )


@router.get("/scenario-info/{scenario_id}", response_class=HTMLResponse)
async def get_scenario_info(request: Request, scenario_id: str):
    """Get detailed info about a scenario for the generation UI."""
    store = get_scenario_store()
    scenario = await store.get_scenario_with_tools(scenario_id)
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

    store = get_scenario_store()
    scenario = await store.get_scenario_with_tools(scenario_id)
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
    save_as_template: bool = Form(False),
    template_name: str = Form(""),
):
    """Generate a single trajectory."""
    scenario_store = get_scenario_store()
    scenario_db = await scenario_store.get_scenario_with_tools(scenario_id)

    if not scenario_db:
        return templates.TemplateResponse(
            "components/error.html",
            {"request": request, "error": f"Scenario not found: {scenario_id}"},
        )

    # Convert to ScenarioTemplate for generation
    scenario = scenario_with_tools_to_template(scenario_db)

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

        # Save as template if requested
        if save_as_template and template_name:
            from ymir.core.scenario_schemas import GenerationTemplateCreate

            # Create a tool preset if custom tools are selected
            tool_preset_id = None
            if enabled_tools_list:
                from ymir.core.scenario_schemas import ToolPresetCreate

                # Get tool IDs from names
                tool_ids = []
                for tool_name in enabled_tools_list:
                    for tool in scenario_db.tools:
                        if tool.name == tool_name:
                            tool_ids.append(tool.id)
                            break

                if tool_ids:
                    preset = await scenario_store.create_tool_preset(
                        ToolPresetCreate(
                            scenario_id=scenario_id,
                            name=f"{template_name} Tools",
                            description=f"Tool preset for {template_name}",
                            tool_ids=tool_ids,
                        )
                    )
                    tool_preset_id = preset.id

            template = await scenario_store.create_generation_template(
                GenerationTemplateCreate(
                    name=template_name,
                    description=f"Template for {scenario_db.name}",
                    scenario_id=scenario_id,
                    tool_preset_id=tool_preset_id,
                    model=model,
                    temperature=temperature,
                )
            )

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
    scenario_store = get_scenario_store()
    scenario_db = await scenario_store.get_scenario_with_tools(scenario_id)

    if not scenario_db:
        return JSONResponse({"error": f"Scenario not found: {scenario_id}"}, status_code=404)

    # Convert to ScenarioTemplate for generation
    scenario = scenario_with_tools_to_template(scenario_db)

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
    scenario_store = get_scenario_store()
    scenarios = await scenario_store.list_scenarios()
    models = get_available_models()

    # Get scenario if specified
    scenario = None
    actors = []
    tool_presets = []
    actor_templates = []
    if scenario_id:
        scenario = await scenario_store.get_scenario_with_tools(scenario_id)
        if scenario:
            # Get actors for this scenario's category
            actors = await scenario_store.get_actors_for_category(scenario.category or "general")
            # Get tool presets for this scenario
            tool_presets = await scenario_store.list_scenario_presets(scenario_id)
            # Get actor templates for this scenario
            actor_templates = await scenario_store.list_actor_templates(scenario_id=scenario_id)
            # Also include templates for the scenario's category
            if scenario.category:
                category_templates = await scenario_store.list_actor_templates(category=scenario.category)
                existing_ids = {t.id for t in actor_templates}
                for t in category_templates:
                    if t.id not in existing_ids:
                        actor_templates.append(t)

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
            "actors": actors,
            "tool_presets": tool_presets,
            "actor_templates": actor_templates,
        },
    )


@router.get("/tools/{scenario_id}", response_class=HTMLResponse)
async def get_scenario_tools(request: Request, scenario_id: str):
    """Get tools for a scenario with toggle UI."""
    scenario_store = get_scenario_store()
    scenario = await scenario_store.get_scenario_with_tools(scenario_id)

    if not scenario:
        return HTMLResponse(content="Scenario not found", status_code=404)

    # Get tool presets for this scenario
    tool_presets = await scenario_store.list_scenario_presets(scenario_id)

    return templates.TemplateResponse(
        "generation/wizard/step_tools.html",
        {"request": request, "scenario": scenario, "step": 2, "tool_presets": tool_presets},
    )


@router.get("/personas/{scenario_id}", response_class=HTMLResponse)
async def get_persona_presets(request: Request, scenario_id: str):
    """Get persona presets for a scenario category."""
    scenario_store = get_scenario_store()
    scenario = await scenario_store.get_scenario_with_tools(scenario_id)

    if not scenario:
        return HTMLResponse(content="Scenario not found", status_code=404)

    # Get actors from database for this scenario's category
    actors = await scenario_store.get_actors_for_category(scenario.category or "general")

    # Get actor templates for this scenario
    actor_templates = await scenario_store.list_actor_templates(scenario_id=scenario_id)
    # Also include templates for the scenario's category
    if scenario.category:
        category_templates = await scenario_store.list_actor_templates(category=scenario.category)
        existing_ids = {t.id for t in actor_templates}
        for t in category_templates:
            if t.id not in existing_ids:
                actor_templates.append(t)

    return templates.TemplateResponse(
        "generation/wizard/step_actor.html",
        {"request": request, "scenario": scenario, "actors": actors, "actor_templates": actor_templates, "step": 3},
    )


@router.get("/stepper/{step_num}", response_class=HTMLResponse)
async def get_stepper(request: Request, step_num: int):
    """Get just the stepper component for a given step."""
    return templates.TemplateResponse(
        "generation/wizard/stepper.html",
        {"request": request, "step": step_num},
    )


@router.post("/generate-actor-from-template", response_class=JSONResponse)
async def generate_actor_from_template(
    request: Request,
    template_id: str = Form(...),
):
    """Generate an actor from a template and return the data."""
    scenario_store = get_scenario_store()
    template = await scenario_store.get_actor_template(template_id)

    if not template:
        return JSONResponse({"error": "Template not found"}, status_code=404)

    try:
        generator = ActorGenerator()
        actor_data = await generator.agenerate_one(template)

        return JSONResponse({
            "success": True,
            "situation": actor_data.situation,
            "background": actor_data.background,
            "goal": actor_data.goal,
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
