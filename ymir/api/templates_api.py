"""Routes for managing generation templates."""

from datetime import datetime

from fastapi import APIRouter, Form, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response

from ymir.api.shared import render_page, templates
from ymir.core.scenario_schemas import GenerationTemplateCreate, GenerationTemplateUpdate
from ymir.data import get_database
from ymir.data.scenario_store import get_scenario_store
from ymir.functions import get_registry
from ymir.pipeline import TrajectoryGenerator
from ymir.pipeline.llm import get_available_models

router = APIRouter(prefix="/templates", tags=["templates"])


def get_store():
    """Get the scenario store instance."""
    db = get_database()
    return get_scenario_store(db)


@router.get("/", response_class=HTMLResponse)
async def templates_page(request: Request):
    """Render the templates management page."""
    store = get_store()

    # Get all scenarios for filter dropdown
    scenarios = await store.list_scenarios()

    # Get all templates
    all_templates = await store.list_generation_templates()

    # Get available models
    models = get_available_models()

    return render_page(
        request,
        "templates/index.html",
        {
            "scenarios": scenarios,
            "templates": all_templates,
            "models": models,
        },
        page_title="Templates",
    )


@router.get("/list", response_class=HTMLResponse)
async def list_templates(
    request: Request,
    scenario_id: str | None = Query(None),
    favorites_only: bool = Query(False),
    page: int = Query(1, ge=1),
):
    """HTMX table body refresh for templates."""
    store = get_store()

    # Query templates with filters
    all_templates = await store.list_generation_templates(
        scenario_id=scenario_id if scenario_id and scenario_id != "all" else None,
        favorites_only=favorites_only,
    )

    # For each template, load full details (scenario, actor, preset)
    enriched_templates = []
    for template in all_templates:
        full_template = await store.get_generation_template_full(template.id)
        if full_template:
            enriched_templates.append(full_template)

    return templates.TemplateResponse(
        "templates/table.html",
        {
            "request": request,
            "templates": enriched_templates,
            "page": page,
        },
    )


@router.post("/", response_class=HTMLResponse)
async def create_template(
    request: Request,
    name: str = Form(...),
    description: str = Form(""),
    scenario_id: str = Form(...),
    actor_id: str | None = Form(None),
    tool_preset_id: str | None = Form(None),
    model: str = Form(...),
    temperature: float = Form(0.7),
):
    """Create a new generation template."""
    store = get_store()

    try:
        # Create the template
        template_data = GenerationTemplateCreate(
            name=name,
            description=description,
            scenario_id=scenario_id,
            actor_id=actor_id if actor_id else None,
            tool_preset_id=tool_preset_id if tool_preset_id else None,
            model=model,
            temperature=temperature,
        )

        template = await store.create_generation_template(template_data)

        # Load full details for rendering
        full_template = await store.get_generation_template_full(template.id)

        # Return the new row with HX-Trigger header
        response = templates.TemplateResponse(
            "templates/row.html",
            {
                "request": request,
                "template": full_template,
            },
        )
        response.headers["HX-Trigger"] = "templateCreated"
        return response

    except Exception as e:
        return templates.TemplateResponse(
            "components/error.html",
            {"request": request, "error": str(e)},
        )


@router.get("/{id}", response_class=JSONResponse)
async def get_template(id: str):
    """Get template JSON with full details."""
    store = get_store()

    template = await store.get_generation_template_full(id)
    if not template:
        return JSONResponse({"error": "Template not found"}, status_code=404)

    # Convert to dict with all related data
    result = template.model_dump()

    # Include related objects if they exist
    if template.scenario:
        result["scenario"] = template.scenario.model_dump()
    if template.actor:
        result["actor"] = template.actor.model_dump()
    if template.tool_preset:
        result["tool_preset"] = template.tool_preset.model_dump()

    return JSONResponse(result)


@router.put("/{id}", response_class=HTMLResponse)
async def update_template(
    request: Request,
    id: str,
    name: str = Form(None),
    description: str = Form(None),
    scenario_id: str = Form(None),
    actor_id: str = Form(None),
    tool_preset_id: str = Form(None),
    model: str = Form(None),
    temperature: float = Form(None),
    is_favorite: bool = Form(None),
):
    """Update a generation template."""
    store = get_store()

    try:
        # Build update object
        updates = GenerationTemplateUpdate(
            name=name,
            description=description,
            scenario_id=scenario_id if scenario_id else None,
            actor_id=actor_id if actor_id else None,
            tool_preset_id=tool_preset_id if tool_preset_id else None,
            model=model,
            temperature=temperature,
            is_favorite=is_favorite,
        )

        updated_template = await store.update_generation_template(id, updates)
        if not updated_template:
            return HTMLResponse(content="Template not found", status_code=404)

        # Load full details for rendering
        full_template = await store.get_generation_template_full(id)

        return templates.TemplateResponse(
            "templates/row.html",
            {
                "request": request,
                "template": full_template,
            },
        )

    except Exception as e:
        return templates.TemplateResponse(
            "components/error.html",
            {"request": request, "error": str(e)},
        )


@router.delete("/{id}")
async def delete_template(id: str):
    """Delete a generation template."""
    store = get_store()

    success = await store.delete_generation_template(id)
    if not success:
        return Response(status_code=404)

    return Response(status_code=200)


@router.post("/{id}/clone", response_class=HTMLResponse)
async def clone_template(
    request: Request,
    id: str,
    new_name: str = Form(...),
):
    """Clone an existing template with a new name."""
    store = get_store()

    try:
        # Get the original template
        original = await store.get_generation_template(id)
        if not original:
            return HTMLResponse(content="Template not found", status_code=404)

        # Create a new template with the same settings but new name
        template_data = GenerationTemplateCreate(
            name=new_name,
            description=f"Cloned from {original.name}",
            scenario_id=original.scenario_id,
            actor_id=original.actor_id,
            tool_preset_id=original.tool_preset_id,
            model=original.model,
            temperature=original.temperature,
        )

        cloned_template = await store.create_generation_template(template_data)

        # Load full details for rendering
        full_template = await store.get_generation_template_full(cloned_template.id)

        return templates.TemplateResponse(
            "templates/row.html",
            {
                "request": request,
                "template": full_template,
            },
        )

    except Exception as e:
        return templates.TemplateResponse(
            "components/error.html",
            {"request": request, "error": str(e)},
        )


@router.post("/{id}/generate", response_class=HTMLResponse)
async def generate_from_template(
    request: Request,
    id: str,
    user_query: str = Form(...),
):
    """Quick generate from template."""
    store = get_store()

    try:
        # Get the template with full details
        template = await store.get_generation_template_full(id)
        if not template:
            return HTMLResponse(content="Template not found", status_code=404)

        if not template.scenario:
            return templates.TemplateResponse(
                "components/error.html",
                {"request": request, "error": "Template scenario not found"},
            )

        # Record template usage
        await store.record_template_use(id)

        # Get the scenario from registry
        registry = get_registry()
        scenario = registry.get_scenario(template.scenario_id)
        if not scenario:
            return templates.TemplateResponse(
                "components/error.html",
                {"request": request, "error": f"Scenario not found: {template.scenario_id}"},
            )

        # Prepare generation parameters
        user_situation = None
        user_background = None
        user_goal = None
        enabled_tools = None

        # If actor is specified, use its data
        if template.actor:
            user_background = template.actor.background
            user_goal = template.actor.goal

        # If tool preset is specified, get tool IDs
        if template.tool_preset_id:
            preset = await store.get_tool_preset_with_tools(template.tool_preset_id)
            if preset and preset.tool_ids:
                # Get tool names from IDs
                tool_names = []
                for tool_id in preset.tool_ids:
                    tool = await store.get_tool(tool_id)
                    if tool:
                        tool_names.append(tool.name)
                enabled_tools = tool_names

        # Generate the trajectory
        generator = TrajectoryGenerator(
            model=template.model,
            temperature=template.temperature,
        )
        trajectory = await generator.generate(
            scenario,
            user_query,
            user_situation=user_situation,
            user_background=user_background,
            user_goal=user_goal,
            enabled_tools=enabled_tools,
        )

        # Save the trajectory
        from ymir.data import get_store as get_trajectory_store
        traj_store = get_trajectory_store()
        await traj_store.save(trajectory)

        return templates.TemplateResponse(
            "generation/trajectory_preview.html",
            {"request": request, "trajectory": trajectory, "saved": True},
        )

    except Exception as e:
        return templates.TemplateResponse(
            "components/error.html",
            {"request": request, "error": str(e)},
        )


@router.post("/{id}/favorite", response_class=HTMLResponse)
async def toggle_favorite(
    request: Request,
    id: str,
):
    """Toggle favorite status of a template."""
    store = get_store()

    try:
        # Get current template
        template = await store.get_generation_template(id)
        if not template:
            return HTMLResponse(content="Template not found", status_code=404)

        # Toggle favorite status
        updates = GenerationTemplateUpdate(
            is_favorite=not template.is_favorite,
        )

        await store.update_generation_template(id, updates)

        # Load full details for rendering
        full_template = await store.get_generation_template_full(id)

        return templates.TemplateResponse(
            "templates/row.html",
            {
                "request": request,
                "template": full_template,
            },
        )

    except Exception as e:
        return templates.TemplateResponse(
            "components/error.html",
            {"request": request, "error": str(e)},
        )


@router.get("/recent", response_class=JSONResponse)
async def get_recent_templates(
    limit: int = Query(5, ge=1, le=20),
):
    """Get recently used templates."""
    store = get_store()

    templates_list = await store.get_recent_templates(limit=limit)

    # Convert to dicts
    result = []
    for template in templates_list:
        template_dict = template.model_dump()
        # Load related data
        full_template = await store.get_generation_template_full(template.id)
        if full_template:
            if full_template.scenario:
                template_dict["scenario_name"] = full_template.scenario.name
            if full_template.actor:
                template_dict["actor_name"] = full_template.actor.name
        result.append(template_dict)

    return JSONResponse({"templates": result})
