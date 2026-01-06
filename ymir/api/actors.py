"""Routes for managing actors/personas."""

import json
from uuid import uuid4

from fastapi import APIRouter, Form, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse
from loguru import logger

from ymir.core.scenario_schemas import ActorCreate, ActorGroupCreate, ActorUpdate
from ymir.data import get_database, get_scenario_store
from ymir.api.shared import render_page, templates
from ymir.pipeline.actor_generator import ActorGenerator

router = APIRouter(prefix="/actors", tags=["actors"])


def get_store():
    """Get the scenario store."""
    db = get_database()
    return get_scenario_store(db)


@router.get("/", response_class=HTMLResponse)
async def actors_page(request: Request):
    """Render the actors table page."""
    store = get_store()

    # Get all actors for initial load
    actors = await store.list_actors()

    # Get unique categories for filter
    categories = sorted(set(actor.category for actor in actors if actor.category))

    # Get actor groups for filter
    actor_groups = await store.list_actor_groups()

    return render_page(
        request,
        "actors/index.html",
        {
            "actors": actors,
            "categories": categories,
            "actor_groups": actor_groups,
            "total_count": len(actors),
        },
        page_title="Actors",
    )


@router.get("/list", response_class=HTMLResponse)
async def list_actors(
    request: Request,
    search: str | None = Query(None),
    category: str | None = Query(None),
    group_id: str | None = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    sort_by: str = Query("name"),
    sort_order: str = Query("asc"),
):
    """HTMX table body refresh with filtering."""
    store = get_store()

    # Get actors with optional category filter
    if group_id and group_id != "all":
        actors = await store.get_actors_in_group(group_id)
    elif category and category != "all":
        actors = await store.list_actors(category=category)
    else:
        actors = await store.list_actors()

    # Get all groups for display
    groups = await store.list_actor_groups()
    actor_groups = {g.id: g for g in groups}

    # Apply search filter if provided
    if search:
        search_lower = search.lower()
        actors = [
            actor
            for actor in actors
            if search_lower in actor.name.lower()
            or search_lower in actor.background.lower()
            or search_lower in actor.goal.lower()
            or any(search_lower in tag.lower() for tag in actor.tags)
        ]

    # Apply sorting
    if sort_by == "name":
        actors = sorted(actors, key=lambda a: a.name.lower(), reverse=(sort_order == "desc"))
    elif sort_by == "category":
        actors = sorted(actors, key=lambda a: (a.category or "").lower(), reverse=(sort_order == "desc"))
    elif sort_by == "created_at":
        actors = sorted(actors, key=lambda a: a.created_at, reverse=(sort_order == "desc"))

    # Compute pagination
    total_count = len(actors)
    total_pages = (total_count + page_size - 1) // page_size if total_count > 0 else 1

    # Apply pagination
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    paginated_actors = actors[start_idx:end_idx]

    return templates.TemplateResponse(
        "actors/table.html",
        {
            "request": request,
            "actors": paginated_actors,
            "actor_groups": actor_groups,
            "total_count": total_count,
            "total_pages": total_pages,
            "page_size": page_size,
            "search": search or "",
            "category": category or "",
            "page": page,
            "sort_by": sort_by,
            "sort_order": sort_order,
        },
    )


@router.post("/", response_class=HTMLResponse)
async def create_actor(
    request: Request,
    name: str = Form(...),
    icon: str = Form(""),
    background: str = Form(...),
    goal: str = Form(...),
    tags: str = Form("[]"),
    category: str = Form(...),
):
    """Create a new actor."""
    store = get_store()

    # Parse tags JSON
    try:
        tags_list = json.loads(tags) if tags else []
    except json.JSONDecodeError:
        tags_list = []

    # Create actor
    actor = await store.create_actor(
        ActorCreate(
            name=name,
            icon=icon if icon else None,
            background=background,
            goal=goal,
            tags=tags_list,
            category=category,
        )
    )

    # Return row template with HX-Trigger for success notification
    response = templates.TemplateResponse(
        "actors/row.html",
        {
            "request": request,
            "actor": actor,
        },
    )
    response.headers["HX-Trigger"] = "actorCreated"
    return response


@router.post("/generate-batch", response_class=HTMLResponse)
async def generate_actors_from_template(
    request: Request,
    template_id: str = Form(...),
    count: int = Form(5),
    group_name: str = Form(""),
    category: str = Form(""),
):
    """Generate multiple actors from a template and save them."""
    store = get_store()

    # Get the template
    template = await store.get_actor_template(template_id)
    if not template:
        return HTMLResponse(content="Template not found", status_code=404)

    # Create actor group
    group_name = group_name.strip() or f"{template.name} Batch"
    group = await store.create_actor_group(
        ActorGroupCreate(
            name=group_name,
            description=f"Generated from template: {template.name}",
            template_id=template_id,
        )
    )

    # Generate actors using ActorGenerator
    generator = ActorGenerator()
    try:
        generated_data = await generator.agenerate_batch(template, count)
    except Exception as e:
        logger.error(f"Error generating actors: {e}")
        return HTMLResponse(content=f"Error generating actors: {e}", status_code=500)

    # Save each generated actor
    actors = []
    for i, data in enumerate(generated_data, 1):
        actor = await store.create_actor(
            ActorCreate(
                name=f"{template.name} #{i}",
                icon="user",
                background=data.background,
                goal=data.goal,
                tags=[template.category] if template.category else [],
                category=category or template.category,
                group_id=group.id,
            )
        )
        actors.append(actor)

    logger.info(f"Generated {len(actors)} actors in group '{group_name}'")

    # Get updated list of actors for table refresh
    all_actors = await store.list_actors()
    groups = await store.list_actor_groups()
    actor_groups = {g.id: g for g in groups}

    # Return updated table
    response = templates.TemplateResponse(
        "actors/table.html",
        {
            "request": request,
            "actors": all_actors,
            "actor_groups": actor_groups,
            "total_count": len(all_actors),
            "total_pages": 1,
            "page_size": 20,
            "search": "",
            "category": "",
            "page": 1,
            "sort_by": "created_at",
            "sort_order": "desc",
        },
    )
    response.headers["HX-Trigger"] = "actorsGenerated"
    return response


@router.get("/preview-batch", response_class=HTMLResponse)
async def preview_batch_generation(
    request: Request,
    template_id: str = Query(None, alias="template-select"),
    count: int = Query(1, alias="batch-count"),
):
    """Preview a sample actor from a template."""
    if not template_id:
        return HTMLResponse(content="<p class='text-slate-500 text-sm'>Select a template to preview</p>")

    store = get_store()
    template = await store.get_actor_template(template_id)

    if not template:
        return HTMLResponse(content="<p class='text-red-400 text-sm'>Template not found</p>")

    # Generate a single preview
    generator = ActorGenerator()
    try:
        data = await generator.agenerate_one(template)
    except Exception as e:
        logger.error(f"Error previewing actor: {e}")
        return HTMLResponse(content=f"<p class='text-red-400 text-sm'>Error: {e}</p>")

    return templates.TemplateResponse(
        "actors/preview.html",
        {
            "request": request,
            "preview": data,
            "template": template,
        },
    )


@router.get("/modal/new", response_class=HTMLResponse)
async def new_actor_modal(request: Request):
    """Render new actor modal."""
    store = get_store()

    # Get all categories for dropdown
    all_actors = await store.list_actors()
    categories = sorted(set(a.category for a in all_actors if a.category))

    # Get actor templates for generation
    actor_templates = await store.list_actor_templates()

    return templates.TemplateResponse(
        "actors/form_modal.html",
        {
            "request": request,
            "actor": None,
            "categories": categories,
            "actor_templates": actor_templates,
        },
    )


@router.get("/{id}", response_class=JSONResponse)
async def get_actor(id: str):
    """Get actor JSON by ID."""
    store = get_store()
    actor = await store.get_actor(id)

    if not actor:
        return JSONResponse({"error": "Actor not found"}, status_code=404)

    return JSONResponse(actor.model_dump(mode="json"))


@router.get("/modal/edit/{id}", response_class=HTMLResponse)
async def edit_actor_modal(request: Request, id: str):
    """Render edit modal for an actor (modal path)."""
    return await edit_actor_form(request, id)


@router.get("/{id}/edit", response_class=HTMLResponse)
async def edit_actor_form(request: Request, id: str):
    """Render edit modal for an actor."""
    store = get_store()
    actor = await store.get_actor(id)

    if not actor:
        return HTMLResponse(content="Actor not found", status_code=404)

    # Get all categories for dropdown
    all_actors = await store.list_actors()
    categories = sorted(
        set(a.category for a in all_actors if a.category)
    )

    return templates.TemplateResponse(
        "actors/form_modal.html",
        {
            "request": request,
            "actor": actor,
            "categories": categories,
        },
    )


@router.put("/{id}", response_class=HTMLResponse)
async def update_actor(
    request: Request,
    id: str,
    name: str = Form(...),
    icon: str = Form(""),
    background: str = Form(...),
    goal: str = Form(...),
    tags: str = Form("[]"),
    category: str = Form(...),
):
    """Update an actor."""
    store = get_store()

    # Parse tags JSON
    try:
        tags_list = json.loads(tags) if tags else []
    except json.JSONDecodeError:
        tags_list = []

    # Update actor
    actor = await store.update_actor(
        id,
        ActorUpdate(
            name=name,
            icon=icon if icon else None,
            background=background,
            goal=goal,
            tags=tags_list,
            category=category,
        ),
    )

    if not actor:
        return HTMLResponse(content="Actor not found", status_code=404)

    # Return updated row
    return templates.TemplateResponse(
        "actors/row.html",
        {
            "request": request,
            "actor": actor,
        },
    )


@router.delete("/{id}", response_class=HTMLResponse)
async def delete_actor(id: str):
    """Delete an actor."""
    store = get_store()
    success = await store.delete_actor(id)

    if not success:
        return HTMLResponse(content="Failed to delete actor", status_code=500)

    # Return empty response (row will be removed by HTMX)
    return HTMLResponse(content="", status_code=200)


@router.post("/{id}/link/{scenario_id}", response_class=JSONResponse)
async def link_actor_to_scenario(id: str, scenario_id: str):
    """Link an actor to a scenario."""
    store = get_store()
    success = await store.link_actor_to_scenario(scenario_id, id)

    if not success:
        return JSONResponse(
            {"error": "Failed to link actor to scenario"}, status_code=500
        )

    return JSONResponse({"success": True})


@router.delete("/{id}/link/{scenario_id}", response_class=JSONResponse)
async def unlink_actor_from_scenario(id: str, scenario_id: str):
    """Unlink an actor from a scenario."""
    store = get_store()
    success = await store.unlink_actor_from_scenario(scenario_id, id)

    if not success:
        return JSONResponse(
            {"error": "Failed to unlink actor from scenario"}, status_code=500
        )

    return JSONResponse({"success": True})


@router.get("/by-scenario/{scenario_id}", response_class=JSONResponse)
async def get_actors_by_scenario(scenario_id: str):
    """Get all actors linked to a scenario."""
    store = get_store()
    actors = await store.get_scenario_actors(scenario_id)

    return JSONResponse(
        {
            "actors": [actor.model_dump(mode="json") for actor in actors],
            "count": len(actors),
        }
    )


@router.get("/by-category/{category}", response_class=JSONResponse)
async def get_actors_by_category(category: str):
    """Get all actors in a specific category."""
    store = get_store()
    actors = await store.list_actors(category=category)

    return JSONResponse(
        {
            "actors": [actor.model_dump(mode="json") for actor in actors],
            "count": len(actors),
        }
    )
