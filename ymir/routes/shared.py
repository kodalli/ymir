from ymir.rlhf import RLHFDatasetBuilder
from fastapi.templating import Jinja2Templates
from ymir.llm import (
    OPENAI_CHAT_MODELS,
    DEEPSEEK_CHAT_MODELS,
    get_supported_configurations,
)
from fastapi import APIRouter
from loguru import logger
import os
from fastapi import HTTPException
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi import Form, Request
import json


router = APIRouter()

rlhf_builder = RLHFDatasetBuilder()

templates = Jinja2Templates(directory="ymir/templates")


# Initialize an empty triplet dataset (this would be replaced by a proper database in production)
triplet_dataset = []

# Chat history
chat_history = {
    "llm_1": [],
    "llm_2": [],
}

# Configuration variables - the current provider and model for each LLM slot
llm_config = {
    "llm_1": {"provider": "OpenAI", "model": OPENAI_CHAT_MODELS[0]},
    "llm_2": {"provider": "DeepSeek", "model": DEEPSEEK_CHAT_MODELS[0]},
}


def get_supported_providers():
    providers = []
    configurations = get_supported_configurations()
    for provider_info in configurations:
        providers.append(provider_info["provider"])
    return providers


def get_provider_models(provider):
    """Get all models for a given provider."""
    configurations = get_supported_configurations()
    for provider_info in configurations:
        if provider_info["provider"] == provider:
            return provider_info["models"]
    return []


@router.post("/update_provider")
async def update_provider(llm_key: str = Form(...), provider: str = Form(...)):
    """Update the provider for an LLM slot and return available models."""
    llm_config[llm_key]["provider"] = provider
    models = get_provider_models(provider)
    llm_config[llm_key]["model"] = models[0] if models else ""

    # Return HTML options for the model dropdown
    options_html = ""
    for model in models:
        options_html += f'<option value="{model}">{model}</option>'

    return HTMLResponse(content=options_html)


@router.post("/update_model")
async def update_model(llm_key: str = Form(...), model: str = Form(...)):
    """Update the model for an LLM slot."""
    provider = llm_config[llm_key]["provider"]
    models = get_provider_models(provider)

    response = None
    if model in models:
        llm_config[llm_key]["model"] = model
        response = JSONResponse(content={"status": "success"})

        # Add toast notification header
        response.headers["HX-Trigger-After-Swap"] = json.dumps(
            {
                "showToast": {
                    "message": f"Model updated to {model} for {llm_key}",
                    "type": "success",
                    "duration": 2000,
                }
            }
        )
    else:
        response = JSONResponse(
            content={
                "status": "error",
                "message": f"Model {model} not available for provider {provider}",
            }
        )

        # Add error toast notification header
        response.headers["HX-Trigger-After-Swap"] = json.dumps(
            {
                "showToast": {
                    "message": f"Error: Model {model} not available for provider {provider}",
                    "type": "error",
                    "duration": 3000,
                }
            }
        )

    return response


@router.get("/provider_models")
async def provider_models(provider: str):
    """Get the list of models available for a specific provider"""
    try:
        models = get_provider_models(provider)
        return {"models": models}
    except Exception as e:
        logger.error(f"Error getting models for provider {provider}: {e}")
        return {"error": str(e), "models": []}


@router.get("/download")
async def download_file(request: Request, file: str):
    """Download a file from the server"""
    try:
        # Validate the file path (ensure it's in the data directory to prevent directory traversal attacks)
        if not os.path.exists(file) or not file.startswith("ymir/data/"):
            raise HTTPException(status_code=404, detail="File not found")

        # Create a FileResponse to serve the file
        return FileResponse(
            path=file,
            filename=os.path.basename(file),
            media_type="application/octet-stream",
        )
    except Exception as e:
        logger.error(f"Error downloading file: {e}")
        raise HTTPException(status_code=500, detail=str(e))
