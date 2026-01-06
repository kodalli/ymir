"""Routes for managing actor templates."""

from fastapi import APIRouter, Form, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse

from ymir.core.scenario_schemas import ActorTemplateCreate, ActorTemplateUpdate
from ymir.data import get_database, get_scenario_store
from ymir.pipeline.actor_generator import ActorGenerator
from ymir.pipeline.llm.ollama import get_available_models
from ymir.api.shared import render_page, templates

router = APIRouter(prefix="/actor-templates", tags=["actor-templates"])


def get_store():
    """Get the scenario store."""
    db = get_database()
    return get_scenario_store(db)


@router.get("/", response_class=HTMLResponse)
async def actor_templates_page(request: Request):
    """Render the actor templates page."""
    store = get_store()

    # Get all actor templates
    actor_templates = await store.list_actor_templates()

    # Get unique categories
    categories = sorted(
        set(t.category for t in actor_templates if t.category)
    )

    # Get scenarios for linking
    scenarios = await store.list_scenarios()

    return render_page(
        request,
        "actor_templates/index.html",
        {
            "actor_templates": actor_templates,
            "categories": categories,
            "scenarios": scenarios,
            "total_count": len(actor_templates),
        },
        page_title="Actor Templates",
    )


@router.get("/list", response_class=HTMLResponse)
async def list_actor_templates(
    request: Request,
    search: str | None = Query(None),
    category: str | None = Query(None),
    scenario_id: str | None = Query(None),
):
    """HTMX table refresh with filtering."""
    store = get_store()

    # Get templates with optional filters
    if category and category != "all":
        actor_templates = await store.list_actor_templates(category=category)
    elif scenario_id and scenario_id != "all":
        actor_templates = await store.list_actor_templates(scenario_id=scenario_id)
    else:
        actor_templates = await store.list_actor_templates()

    # Apply search filter
    if search:
        search_lower = search.lower()
        actor_templates = [
            t
            for t in actor_templates
            if search_lower in t.name.lower()
            or search_lower in t.description.lower()
            or search_lower in t.template_text.lower()
        ]

    return templates.TemplateResponse(
        "actor_templates/table.html",
        {
            "request": request,
            "actor_templates": actor_templates,
            "total_count": len(actor_templates),
        },
    )


@router.get("/new", response_class=HTMLResponse)
async def new_template_form(request: Request):
    """Render create form modal."""
    store = get_store()
    scenarios = await store.list_scenarios()
    models = get_available_models()

    return templates.TemplateResponse(
        "actor_templates/form_modal.html",
        {
            "request": request,
            "template": None,
            "scenarios": scenarios,
            "mode": "create",
            "ollama_models": models,
        },
    )


@router.post("/", response_class=HTMLResponse)
async def create_actor_template(
    request: Request,
    name: str = Form(...),
    description: str = Form(""),
    template_text: str = Form(...),
    background_template: str = Form(None),
    goal_template: str = Form(None),
    category: str = Form(None),
    scenario_id: str = Form(None),
):
    """Create a new actor template."""
    store = get_store()

    # Validate template syntax
    generator = ActorGenerator()
    is_valid, errors = generator.validate_template(template_text)

    if not is_valid:
        return templates.TemplateResponse(
            "actor_templates/form_modal.html",
            {
                "request": request,
                "template": None,
                "scenarios": await store.list_scenarios(),
                "mode": "create",
                "errors": errors,
                "ollama_models": get_available_models(),
            },
        )

    # Create the template
    template = await store.create_actor_template(
        ActorTemplateCreate(
            name=name,
            description=description,
            template_text=template_text,
            background_template=background_template if background_template else None,
            goal_template=goal_template if goal_template else None,
            category=category if category else None,
            scenario_id=scenario_id if scenario_id else None,
        )
    )

    # Return updated table
    actor_templates = await store.list_actor_templates()

    response = templates.TemplateResponse(
        "actor_templates/table.html",
        {
            "request": request,
            "actor_templates": actor_templates,
            "total_count": len(actor_templates),
        },
    )
    response.headers["HX-Trigger"] = "closeModal"
    return response


@router.get("/{id}", response_class=JSONResponse)
async def get_actor_template(id: str):
    """Get a template by ID (JSON)."""
    store = get_store()
    template = await store.get_actor_template(id)

    if not template:
        return JSONResponse({"error": "Template not found"}, status_code=404)

    return JSONResponse(template.model_dump(mode="json"))


@router.get("/{id}/edit", response_class=HTMLResponse)
async def edit_template_form(request: Request, id: str):
    """Render edit form modal."""
    store = get_store()
    template = await store.get_actor_template(id)

    if not template:
        return HTMLResponse(content="Template not found", status_code=404)

    scenarios = await store.list_scenarios()
    models = get_available_models()

    return templates.TemplateResponse(
        "actor_templates/form_modal.html",
        {
            "request": request,
            "template": template,
            "scenarios": scenarios,
            "mode": "edit",
            "ollama_models": models,
        },
    )


@router.put("/{id}", response_class=HTMLResponse)
async def update_actor_template(
    request: Request,
    id: str,
    name: str = Form(...),
    description: str = Form(""),
    template_text: str = Form(...),
    background_template: str = Form(None),
    goal_template: str = Form(None),
    category: str = Form(None),
    scenario_id: str = Form(None),
):
    """Update an actor template."""
    store = get_store()

    # Validate template syntax
    generator = ActorGenerator()
    is_valid, errors = generator.validate_template(template_text)

    if not is_valid:
        template = await store.get_actor_template(id)
        return templates.TemplateResponse(
            "actor_templates/form_modal.html",
            {
                "request": request,
                "template": template,
                "scenarios": await store.list_scenarios(),
                "mode": "edit",
                "errors": errors,
                "ollama_models": get_available_models(),
            },
        )

    # Update the template
    await store.update_actor_template(
        id,
        ActorTemplateUpdate(
            name=name,
            description=description,
            template_text=template_text,
            background_template=background_template if background_template else None,
            goal_template=goal_template if goal_template else None,
            category=category if category else None,
            scenario_id=scenario_id if scenario_id else None,
        ),
    )

    # Return updated table
    actor_templates = await store.list_actor_templates()

    response = templates.TemplateResponse(
        "actor_templates/table.html",
        {
            "request": request,
            "actor_templates": actor_templates,
            "total_count": len(actor_templates),
        },
    )
    response.headers["HX-Trigger"] = "closeModal"
    return response


@router.delete("/{id}", response_class=HTMLResponse)
async def delete_actor_template(request: Request, id: str):
    """Delete an actor template."""
    store = get_store()

    await store.delete_actor_template(id)

    # Return updated table
    actor_templates = await store.list_actor_templates()

    return templates.TemplateResponse(
        "actor_templates/table.html",
        {
            "request": request,
            "actor_templates": actor_templates,
            "total_count": len(actor_templates),
        },
    )


@router.post("/{id}/preview", response_class=HTMLResponse)
async def preview_actor_template(
    request: Request,
    id: str,
    count: int = Form(3),
):
    """Generate preview actors from template."""
    store = get_store()
    template = await store.get_actor_template(id)

    if not template:
        return HTMLResponse(content="Template not found", status_code=404)

    # Generate preview actors
    generator = ActorGenerator()
    actors = await generator.agenerate_batch(template, count)

    return templates.TemplateResponse(
        "actor_templates/preview.html",
        {
            "request": request,
            "template": template,
            "actors": actors,
        },
    )


@router.post("/preview-text", response_class=HTMLResponse)
async def preview_template_text(
    request: Request,
    template_text: str = Form(""),
    background_template: str = Form(""),
    goal_template: str = Form(""),
    count: int = Form(3),
    model: str = Form("mistral-small:latest"),
    temperature: float = Form(0.7),
    num_predict: int = Form(128),
):
    """Preview all template fields without saving."""
    generator = ActorGenerator(
        model=model,
        temperature=temperature,
        num_predict=num_predict,
    )

    all_errors = []
    results = {
        "situation": [],
        "background": [],
        "goal": [],
    }

    # Generate situation previews
    if template_text.strip():
        is_valid, errors = generator.validate_template(template_text)
        if not is_valid:
            all_errors.extend([f"Situation: {e}" for e in errors])
        else:
            results["situation"] = await generator.apreview(template_text, count)

    # Generate background previews
    if background_template.strip():
        is_valid, errors = generator.validate_template(background_template)
        if not is_valid:
            all_errors.extend([f"Background: {e}" for e in errors])
        else:
            results["background"] = await generator.apreview(background_template, count)

    # Generate goal previews
    if goal_template.strip():
        is_valid, errors = generator.validate_template(goal_template)
        if not is_valid:
            all_errors.extend([f"Goal: {e}" for e in errors])
        else:
            results["goal"] = await generator.apreview(goal_template, count)

    return templates.TemplateResponse(
        "actor_templates/preview.html",
        {
            "request": request,
            "results": results,
            "errors": all_errors if all_errors else None,
        },
    )


@router.get("/models", response_class=JSONResponse)
async def list_ollama_models():
    """Get list of available Ollama models."""
    models = get_available_models()
    return JSONResponse({"models": models})


@router.post("/validate", response_class=JSONResponse)
async def validate_template_text(template_text: str = Form(...)):
    """Validate template syntax."""
    generator = ActorGenerator()
    is_valid, errors = generator.validate_template(template_text)

    return JSONResponse(
        {
            "valid": is_valid,
            "errors": errors,
        }
    )


@router.get("/for-scenario/{scenario_id}", response_class=HTMLResponse)
async def get_templates_for_scenario(request: Request, scenario_id: str):
    """Get actor templates for a specific scenario (for wizard integration)."""
    store = get_store()

    # Get scenario to check category
    scenario = await store.get_scenario(scenario_id)
    if not scenario:
        return HTMLResponse(content="Scenario not found", status_code=404)

    # Get templates for this scenario or its category
    templates_list = await store.list_actor_templates(scenario_id=scenario_id)

    # Also get templates for the scenario's category if different
    if scenario.category:
        category_templates = await store.list_actor_templates(category=scenario.category)
        # Merge without duplicates
        existing_ids = {t.id for t in templates_list}
        for t in category_templates:
            if t.id not in existing_ids:
                templates_list.append(t)

    return templates.TemplateResponse(
        "actor_templates/select_list.html",
        {
            "request": request,
            "actor_templates": templates_list,
        },
    )
