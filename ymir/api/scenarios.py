"""Routes for scenario management - scenarios, tools, and presets."""

import json
from typing import Any

from fastapi import APIRouter, Form, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse

from ymir.core.scenario_schemas import (
    ScenarioCreate,
    ScenarioUpdate,
    ToolPresetCreate,
)
from ymir.data.scenario_store import get_scenario_store
from ymir.api.shared import render_page, templates

router = APIRouter(prefix="/scenarios", tags=["scenarios"])


def get_store():
    """Get the scenario store."""
    return get_scenario_store()


# ============================================================================
# Scenario Management Endpoints
# ============================================================================


@router.get("/", response_class=HTMLResponse)
async def scenarios_page(request: Request):
    """Render the scenarios management page."""
    store = get_store()

    # Get initial data
    scenarios = await store.list_scenarios()

    # Add tool count to each scenario
    for scenario in scenarios:
        tools = await store.get_scenario_tools(scenario.id)
        scenario.tool_count = len(tools)

    # Get unique categories for filter
    categories = sorted(set(s.category for s in scenarios if s.category))

    return render_page(
        request,
        "scenarios/index.html",
        {
            "scenarios": scenarios,
            "categories": categories,
            "total_count": len(scenarios),
        },
        page_title="Scenarios",
    )


@router.get("/list", response_class=HTMLResponse)
async def list_scenarios(
    request: Request,
    search: str | None = Query(None),
    category: str | None = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(25, ge=10, le=100),
):
    """HTMX table body refresh with filtering and pagination."""
    store = get_store()

    # Query scenarios with filters
    if search:
        scenarios = await store.search_scenarios(search)
    elif category and category != "all":
        scenarios = await store.list_scenarios(category=category)
    else:
        scenarios = await store.list_scenarios()

    # Add tool count to each scenario
    for scenario in scenarios:
        tools = await store.get_scenario_tools(scenario.id)
        scenario.tool_count = len(tools)

    # Pagination
    total_count = len(scenarios)
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    paginated_scenarios = scenarios[start_idx:end_idx]

    total_pages = (total_count + page_size - 1) // page_size if total_count > 0 else 1

    return templates.TemplateResponse(
        "scenarios/table.html",
        {
            "request": request,
            "scenarios": paginated_scenarios,
            "total_count": total_count,
            "page": page,
            "page_size": page_size,
            "total_pages": total_pages,
            "search": search,
            "category": category,
        },
    )


@router.post("/", response_class=HTMLResponse)
async def create_scenario(
    request: Request,
    name: str = Form(...),
    description: str = Form(""),
    category: str = Form(...),
    system_prompt: str = Form(""),
    example_queries: str = Form(""),
    mock_responses: str = Form("{}"),
):
    """Create a new scenario."""
    store = get_store()

    try:
        # Parse example_queries - can be JSON array or newline-separated text
        if example_queries:
            try:
                example_queries_list = json.loads(example_queries)
            except json.JSONDecodeError:
                # Treat as newline-separated text
                example_queries_list = [q.strip() for q in example_queries.split("\n") if q.strip()]
        else:
            example_queries_list = []
        mock_responses_dict = json.loads(mock_responses) if mock_responses else {}

        # Create scenario
        scenario_data = ScenarioCreate(
            name=name,
            description=description,
            category=category,
            system_prompt=system_prompt,
            example_queries=example_queries_list,
            mock_responses=mock_responses_dict,
        )

        await store.create_scenario(scenario_data)

        # Return the full table with all scenarios
        scenarios = await store.list_scenarios()
        for s in scenarios:
            tools = await store.get_scenario_tools(s.id)
            s.tool_count = len(tools)

        return templates.TemplateResponse(
            "scenarios/table.html",
            {
                "request": request,
                "scenarios": scenarios,
                "page": 1,
                "page_size": 25,
                "total_pages": 1,
                "total_count": len(scenarios),
            },
            headers={"HX-Trigger": "scenarioCreated"},
        )

    except json.JSONDecodeError as e:
        return HTMLResponse(
            content=f"Invalid JSON in fields: {e}",
            status_code=400,
        )
    except Exception as e:
        return HTMLResponse(
            content=f"Error creating scenario: {e}",
            status_code=500,
        )


@router.get("/new", response_class=HTMLResponse)
async def new_scenario_modal(request: Request):
    """Render new scenario modal."""
    store = get_store()

    # Get categories for dropdown
    scenarios = await store.list_scenarios()
    categories = sorted(set(s.category for s in scenarios if s.category))
    if not categories:
        categories = ["scheduling", "healthcare", "finance", "retail", "education", "general"]

    return templates.TemplateResponse(
        "scenarios/form_modal.html",
        {
            "request": request,
            "scenario": None,
            "categories": categories,
            "mode": "create",
        },
    )


@router.get("/{id}", response_class=JSONResponse)
async def get_scenario(id: str):
    """Get scenario JSON data."""
    store = get_store()

    scenario_with_tools = await store.get_scenario_with_tools(id)
    if scenario_with_tools is None:
        return JSONResponse(
            {"error": "Scenario not found"},
            status_code=404,
        )

    return JSONResponse(scenario_with_tools.model_dump(mode="json"))


@router.get("/{id}/edit", response_class=HTMLResponse)
async def edit_scenario_modal(request: Request, id: str):
    """Render edit scenario modal."""
    store = get_store()

    scenario = await store.get_scenario(id)
    if scenario is None:
        return HTMLResponse(content="Scenario not found", status_code=404)

    # Get categories for dropdown
    scenarios = await store.list_scenarios()
    categories = sorted(set(s.category for s in scenarios if s.category))

    return templates.TemplateResponse(
        "scenarios/form_modal.html",
        {
            "request": request,
            "scenario": scenario,
            "categories": categories,
            "mode": "edit",
        },
    )


@router.put("/{id}", response_class=HTMLResponse)
async def update_scenario(
    request: Request,
    id: str,
    name: str = Form(...),
    description: str = Form(""),
    category: str = Form(...),
    system_prompt: str = Form(""),
    example_queries: str = Form(""),
    mock_responses: str = Form("{}"),
):
    """Update an existing scenario."""
    store = get_store()

    try:
        # Parse example_queries - can be JSON array or newline-separated text
        if example_queries:
            try:
                example_queries_list = json.loads(example_queries)
            except json.JSONDecodeError:
                # Treat as newline-separated text
                example_queries_list = [q.strip() for q in example_queries.split("\n") if q.strip()]
        else:
            example_queries_list = []
        mock_responses_dict = json.loads(mock_responses) if mock_responses else {}

        # Update scenario
        updates = ScenarioUpdate(
            name=name,
            description=description,
            category=category,
            system_prompt=system_prompt,
            example_queries=example_queries_list,
            mock_responses=mock_responses_dict,
        )

        scenario = await store.update_scenario(id, updates)
        if scenario is None:
            return HTMLResponse(content="Scenario not found", status_code=404)

        # Return the full table with all scenarios
        scenarios = await store.list_scenarios()
        for s in scenarios:
            tools = await store.get_scenario_tools(s.id)
            s.tool_count = len(tools)

        return templates.TemplateResponse(
            "scenarios/table.html",
            {
                "request": request,
                "scenarios": scenarios,
                "page": 1,
                "page_size": 25,
                "total_pages": 1,
                "total_count": len(scenarios),
            },
        )

    except json.JSONDecodeError as e:
        return HTMLResponse(
            content=f"Invalid JSON in fields: {e}",
            status_code=400,
        )
    except Exception as e:
        return HTMLResponse(
            content=f"Error updating scenario: {e}",
            status_code=500,
        )


@router.delete("/{id}", response_class=HTMLResponse)
async def delete_scenario(id: str):
    """Delete a scenario."""
    store = get_store()

    success = await store.delete_scenario(id)
    if not success:
        return HTMLResponse(content="Failed to delete scenario", status_code=500)

    # Return empty response - row will be removed via HX-Swap
    return HTMLResponse(content="", status_code=200)


# ============================================================================
# Scenario Tools Management Endpoints
# ============================================================================


@router.get("/{id}/tools", response_class=HTMLResponse)
async def get_scenario_tools_panel(request: Request, id: str):
    """Render tools management panel for a scenario."""
    store = get_store()

    scenario = await store.get_scenario(id)
    if scenario is None:
        return HTMLResponse(content="Scenario not found", status_code=404)

    # Get current tools attached to this scenario
    current_tools = await store.get_scenario_tools(id)
    selected_tool_ids = [t.id for t in current_tools]

    # Get all available tools
    all_tools = await store.list_tools()

    # Get presets for this scenario
    presets = await store.list_scenario_presets(id)

    return templates.TemplateResponse(
        "scenarios/tools_panel.html",
        {
            "request": request,
            "scenario": scenario,
            "all_tools": all_tools,
            "selected_tools": selected_tool_ids,
            "presets": presets,
        },
    )


@router.post("/{id}/tools", response_class=HTMLResponse)
async def update_scenario_tools(
    request: Request,
    id: str,
    tool_ids: list[str] = Form(default=[]),
):
    """Update tools for a scenario (replaces all current tools)."""
    store = get_store()

    # Check scenario exists
    scenario = await store.get_scenario(id)
    if scenario is None:
        return HTMLResponse(content="Scenario not found", status_code=404)

    # Replace all tools with the new selection
    await store.attach_tools_to_scenario(id, tool_ids, replace=True)

    # Return updated scenario list
    scenarios = await store.list_scenarios()

    return templates.TemplateResponse(
        "scenarios/table.html",
        {
            "request": request,
            "scenarios": scenarios,
            "page": 1,
            "total_pages": 1,
            "total_count": len(scenarios),
        },
    )


@router.delete("/{id}/tools/{tool_id}", response_class=HTMLResponse)
async def remove_tool_from_scenario(id: str, tool_id: str):
    """Remove a tool from a scenario."""
    store = get_store()

    await store.detach_tools_from_scenario(id, [tool_id])

    # Return empty response - row will be removed via HX-Swap
    return HTMLResponse(content="", status_code=200)


@router.put("/{id}/tools/reorder", response_class=JSONResponse)
async def reorder_scenario_tools(
    id: str,
    tool_ids: list[str],
):
    """Reorder tools for a scenario."""
    store = get_store()

    # Replace tools with new order
    await store.attach_tools_to_scenario(id, tool_ids, replace=True)

    return JSONResponse({"success": True})


# ============================================================================
# Tool Presets Management Endpoints
# ============================================================================


@router.get("/{id}/presets", response_class=HTMLResponse)
async def list_scenario_presets(request: Request, id: str):
    """List presets for a scenario."""
    store = get_store()

    scenario = await store.get_scenario(id)
    if scenario is None:
        return HTMLResponse(content="Scenario not found", status_code=404)

    presets = await store.list_scenario_presets(id)

    # Enrich with tool data
    for preset in presets:
        preset.tools = []
        for tool_id in preset.tool_ids:
            tool = await store.get_tool(tool_id)
            if tool:
                preset.tools.append(tool)

    return templates.TemplateResponse(
        "scenarios/preset_list.html",
        {
            "request": request,
            "scenario": scenario,
            "presets": presets,
        },
    )


@router.post("/{id}/presets", response_class=HTMLResponse)
async def create_tool_preset(
    request: Request,
    id: str,
    name: str = Form(...),
    description: str = Form(""),
    tool_ids: str = Form(...),
    is_default: bool = Form(False),
):
    """Create a tool preset for a scenario."""
    store = get_store()

    try:
        # Parse tool_ids JSON
        tool_ids_list = json.loads(tool_ids) if tool_ids else []

        preset_data = ToolPresetCreate(
            scenario_id=id,
            name=name,
            description=description,
            tool_ids=tool_ids_list,
            is_default=is_default,
        )

        await store.create_tool_preset(preset_data)

        # Return the full tools panel to refresh the modal
        scenario = await store.get_scenario(id)
        current_tools = await store.get_scenario_tools(id)
        selected_tool_ids = [t.id for t in current_tools]
        all_tools = await store.list_tools()
        presets = await store.list_scenario_presets(id)

        return templates.TemplateResponse(
            "scenarios/tools_panel.html",
            {
                "request": request,
                "scenario": scenario,
                "all_tools": all_tools,
                "selected_tools": selected_tool_ids,
                "presets": presets,
            },
        )

    except json.JSONDecodeError as e:
        return HTMLResponse(
            content=f"Invalid JSON for tool_ids: {e}",
            status_code=400,
        )
    except Exception as e:
        return HTMLResponse(
            content=f"Error creating preset: {e}",
            status_code=500,
        )


@router.delete("/{id}/presets/{preset_id}", response_class=HTMLResponse)
async def delete_tool_preset(id: str, preset_id: str):
    """Delete a tool preset."""
    store = get_store()

    success = await store.delete_tool_preset(preset_id)
    if not success:
        return HTMLResponse(content="Failed to delete preset", status_code=500)

    # Return empty response - row will be removed via HX-Swap
    return HTMLResponse(content="", status_code=200)
