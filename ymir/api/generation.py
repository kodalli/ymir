"""Routes for trajectory generation."""

import json
from fastapi import APIRouter, Form, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse

from ymir.functions import get_registry
from ymir.functions.schemas import FunctionDefinition, ScenarioTemplate
from ymir.pipeline import TrajectoryGenerator
from ymir.pipeline.llm import get_available_models
from ymir.pipeline.personas import get_personas_for_category
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
    actor_ids: str = Form("[]"),  # JSON array of selected actor IDs
    enabled_tools: str = Form(None),  # JSON array of tool names
    model: str = Form("qwen3:4b"),
    temperature: float = Form(0.7),
    save: bool = Form(True),
    save_as_template: bool = Form(False),
    template_name: str = Form(""),
):
    """Generate trajectories for selected actors."""
    scenario_store = get_scenario_store()
    scenario_db = await scenario_store.get_scenario_with_tools(scenario_id)

    if not scenario_db:
        return templates.TemplateResponse(
            "components/error.html",
            {"request": request, "error": f"Scenario not found: {scenario_id}"},
        )

    # Parse actor_ids
    actor_id_list = []
    try:
        actor_id_list = json.loads(actor_ids)
        if not isinstance(actor_id_list, list):
            actor_id_list = []
    except json.JSONDecodeError:
        actor_id_list = []

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

        trajectories = []
        store = get_store()

        # Generate for each actor
        for actor_id in actor_id_list:
            actor = await scenario_store.get_actor(actor_id)
            if not actor:
                continue

            trajectory = await generator.generate(
                scenario,
                user_query,
                user_situation=actor.situation if actor.situation else None,
                user_background=actor.background if actor.background else None,
                user_goal=actor.goal if actor.goal else None,
                enabled_tools=enabled_tools_list,
            )

            if save:
                await store.save(trajectory)

            trajectories.append(trajectory)

        # Save as template if requested (only once, not per actor)
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

            await scenario_store.create_generation_template(
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
            {"request": request, "trajectories": trajectories, "saved": save, "count": len(trajectories)},
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
    categories = []
    actor_groups = []

    if scenario_id:
        scenario = await scenario_store.get_scenario_with_tools(scenario_id)
        if scenario:
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

    # For step 3 (actor selection), get all actors with filters
    if step_num == 3:
        actors = await scenario_store.list_actors()
        # Get unique categories
        categories = sorted(set(a.category for a in actors if a.category))
        # Get actor groups
        actor_groups = await scenario_store.list_actor_groups()

    template_map = {
        1: "generation/wizard/step_scenario.html",
        2: "generation/wizard/step_tools.html",
        3: "generation/wizard/step_actor.html",
        4: "generation/wizard/step_generate.html",
    }

    template = template_map.get(step_num, "generation/wizard/step_scenario.html")

    # Build actor_groups dict for template
    actor_groups_dict = {g.id: g for g in actor_groups}

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
            "categories": categories,
            "actor_groups": actor_groups,
            "actor_groups_dict": actor_groups_dict,
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


@router.get("/actors-list", response_class=HTMLResponse)
async def list_actors_for_wizard(
    request: Request,
    search: str | None = Query(None),
    category: str | None = Query(None),
    group_id: str | None = Query(None),
):
    """Get filtered actor cards for wizard selection."""
    scenario_store = get_scenario_store()

    # Get actors with optional filters
    if group_id:
        actors = await scenario_store.get_actors_in_group(group_id)
    elif category:
        actors = await scenario_store.list_actors(category=category)
    else:
        actors = await scenario_store.list_actors()

    # Apply search filter
    if search:
        search_lower = search.lower()
        actors = [
            a for a in actors
            if search_lower in a.name.lower()
            or search_lower in a.background.lower()
            or search_lower in a.goal.lower()
            or (a.situation and search_lower in a.situation.lower())
        ]

    # Get actor groups for display
    actor_groups = await scenario_store.list_actor_groups()
    actor_groups_dict = {g.id: g for g in actor_groups}

    return templates.TemplateResponse(
        "generation/wizard/actor_cards.html",
        {
            "request": request,
            "actors": actors,
            "actor_groups": actor_groups_dict,
            "search": search,
            "category": category,
            "group_id": group_id,
        },
    )
