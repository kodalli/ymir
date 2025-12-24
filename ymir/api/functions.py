"""Routes for function/tool definition management."""

import json
from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse

from ymir.functions import FunctionDefinition, ScenarioTemplate, get_registry
from ymir.api.shared import render_page

router = APIRouter(prefix="/functions", tags=["functions"])


@router.get("/", response_class=HTMLResponse)
async def functions_page(request: Request):
    """Render the function management page."""
    registry = get_registry()
    functions = registry.list_functions()
    scenarios = registry.list_scenarios()
    categories = registry.get_categories()

    return render_page(
        request,
        "functions/index.html",
        {
            "functions": functions,
            "scenarios": scenarios,
            "categories": categories,
        },
        page_title="Scenarios",
    )


@router.get("/list", response_class=HTMLResponse)
async def list_functions(request: Request, category: str = None):
    """List functions, optionally filtered by category."""
    registry = get_registry()
    functions = registry.list_functions(category)

    return templates.TemplateResponse(
        "functions/function_list.html",
        {"request": request, "functions": functions},
    )


@router.post("/add", response_class=HTMLResponse)
async def add_function(
    request: Request,
    name: str = Form(...),
    description: str = Form(...),
    parameters_json: str = Form(...),
    category: str = Form("custom"),
):
    """Add a new function definition."""
    registry = get_registry()

    try:
        parameters = json.loads(parameters_json)
    except json.JSONDecodeError:
        return templates.TemplateResponse(
            "components/error.html",
            {"request": request, "error": "Invalid JSON for parameters"},
        )

    func = FunctionDefinition(
        name=name,
        description=description,
        parameters=parameters,
        category=category,
    )
    registry.add_function(func)

    # Return updated function list
    functions = registry.list_functions()
    return templates.TemplateResponse(
        "functions/function_list.html",
        {"request": request, "functions": functions, "message": f"Added function: {name}"},
    )


@router.delete("/{name}")
async def delete_function(name: str):
    """Delete a function definition."""
    registry = get_registry()
    success = registry.delete_function(name)
    return JSONResponse({"success": success})


@router.get("/scenario-details", response_class=HTMLResponse)
async def scenario_details(request: Request, scenario_id: str):
    """Get HTML details for a scenario (tools, description)."""
    registry = get_registry()
    scenario = registry.get_scenario(scenario_id)
    return templates.TemplateResponse(
        "generation/scenario_info.html",
        {"request": request, "scenario": scenario},
    )


@router.get("/scenario-details", response_class=HTMLResponse)
async def get_scenario_details(request: Request, scenario_id: str):
    """Get details of a scenario for the generation UI."""
    registry = get_registry()
    scenario = registry.get_scenario(scenario_id)
    
    if not scenario:
        return HTMLResponse("Scenario not found", status_code=404)
        
    return templates.TemplateResponse(
        "functions/scenario_tools_compact.html",
        {"request": request, "scenario": scenario},
    )


@router.post("/scenario", response_class=HTMLResponse)
async def add_scenario(
    request: Request,
    id: str = Form(...),
    name: str = Form(...),
    description: str = Form(...),
    category: str = Form("custom"),
    system_prompt: str = Form(...),
    function_names: str = Form(...),  # Comma-separated function names
):
    """Add a new scenario template."""
    registry = get_registry()

    # Get functions by name
    func_names = [n.strip() for n in function_names.split(",") if n.strip()]
    functions = []
    for fname in func_names:
        func = registry.get_function(fname)
        if func:
            functions.append(func)

    if not functions:
        return templates.TemplateResponse(
            "components/error.html",
            {"request": request, "error": "No valid functions specified"},
        )

    scenario = ScenarioTemplate(
        id=id,
        name=name,
        description=description,
        category=category,
        functions=functions,
        system_prompt=system_prompt,
    )
    registry.add_scenario(scenario)

    scenarios = registry.list_scenarios()
    return templates.TemplateResponse(
        "functions/scenario_list.html",
        {"request": request, "scenarios": scenarios, "message": f"Added scenario: {name}"},
    )
