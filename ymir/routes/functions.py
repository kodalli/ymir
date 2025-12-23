"""Routes for function/tool definition management."""

import json
from fastapi import APIRouter, Form, Request
from fastapi.responses import HTMLResponse, JSONResponse

from ymir.functions import FunctionDefinition, ScenarioTemplate, get_registry
from ymir.routes import templates

router = APIRouter(prefix="/functions", tags=["functions"])


@router.get("/", response_class=HTMLResponse)
async def functions_page(request: Request):
    """Render the function management page."""
    registry = get_registry()
    functions = registry.list_functions()
    scenarios = registry.list_scenarios()
    categories = registry.get_categories()

    return templates.TemplateResponse(
        "functions/index.html",
        {
            "request": request,
            "functions": functions,
            "scenarios": scenarios,
            "categories": categories,
        },
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


@router.get("/scenarios", response_class=HTMLResponse)
async def list_scenarios(request: Request, category: str = None):
    """List scenario templates."""
    registry = get_registry()
    scenarios = registry.list_scenarios(category)

    return templates.TemplateResponse(
        "functions/scenario_list.html",
        {"request": request, "scenarios": scenarios},
    )


@router.get("/scenario/{scenario_id}", response_class=JSONResponse)
async def get_scenario(scenario_id: str):
    """Get a scenario by ID."""
    registry = get_registry()
    scenario = registry.get_scenario(scenario_id)

    if not scenario:
        return JSONResponse({"error": "Scenario not found"}, status_code=404)

    return JSONResponse(scenario.model_dump())


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
