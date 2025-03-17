from typing import Optional
import uvicorn
from fastapi import FastAPI, Request, Form, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import re
from loguru import logger
import json
import os
import tempfile
import time
from openai import OpenAI
import csv
from pathlib import Path
from contextlib import asynccontextmanager

from ymir.llm import (
    get_llm,
    get_supported_configurations,
    OPENAI_CHAT_MODELS,
    DEEPSEEK_CHAT_MODELS,
)
from ymir.llm.openai_llm import OpenAIBatchProcessor
from ymir.rlhf import RLHFDatasetBuilder
from ymir.triplets.text_to_triplets import extract_triplets
from ymir.prompt.pdf import extract_chapter_starts, split_pdf_by_chapters
import pandas as pd


# Define lifespan context manager for startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    rlhf_builder.load_dataset_jsonl()
    logger.info("Ymir Dataset Tools application started")
    yield
    # Shutdown logic (if any)
    logger.info("Shutting down Ymir Dataset Tools application")


# Initialize global variables
app = FastAPI(title="Ymir AI Dataset Tools", lifespan=lifespan)
rlhf_builder = RLHFDatasetBuilder()
templates = Jinja2Templates(directory="ymir/templates")

# Initialize an empty triplet dataset (this would be replaced by a proper database in production)
triplet_dataset = []

# Progress tracking for PDF processing
pdf_processing_progress = {}

# Create static files directory for CSS and JS
try:
    app.mount("/static", StaticFiles(directory="ymir/static"), name="static")
except RuntimeError:
    # This happens when reloading the app (files already mounted)
    pass

# Configuration variables - the current provider and model for each LLM slot
llm_config = {
    "llm_1": {"provider": "OpenAI", "model": OPENAI_CHAT_MODELS[0]},
    "llm_2": {"provider": "DeepSeek", "model": DEEPSEEK_CHAT_MODELS[0]},
}

# Chat history
chat_history = {
    "llm_1": [],
    "llm_2": [],
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


def to_langchain_messages(message, history):
    messages = []
    for m in history:
        messages.append((m["role"], m["content"]))
    messages.append(("user", message))
    return messages


def convert_reasoning_to_markdown(message):
    match = re.search(r"<think>(.*?)</think>", message, re.DOTALL)
    if match:
        think_content = match.group(1).strip()
        message = message.replace(
            f"<think>{think_content}</think>", f"```think\n{think_content}\n```"
        )
    return message


def convert_markdown_to_reasoning(message):
    return message


def generate_response(llm, message, history):
    llm_client = get_llm(llm["provider"], llm["model"])
    return llm_client.invoke(message, history)


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    llm_key: str  # "llm_1" or "llm_2"


class LLMConfig(BaseModel):
    provider: str
    model: str


class RatingRequest(BaseModel):
    chosen: str  # "llm_1" or "llm_2"
    notes: Optional[str] = ""


class TripletRequest(BaseModel):
    subject: str
    predicate: str
    object: str
    description: Optional[str] = ""


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Render the main layout template only (content will be loaded via HTMX)"""
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
        },
    )


@app.get("/rlhf", response_class=HTMLResponse)
async def rlhf_page(request: Request):
    """Render the RLHF tool page content only (not the entire index.html)"""
    return templates.TemplateResponse(
        "rlhf_processing/rlhf_content.html",
        {
            "request": request,
            "providers": get_supported_providers(),
            "models": {
                "llm_1": get_provider_models(llm_config["llm_1"]["provider"]),
                "llm_2": get_provider_models(llm_config["llm_2"]["provider"]),
            },
            "chat_history_1": chat_history["llm_1"],
            "chat_history_2": chat_history["llm_2"],
        },
    )


@app.get("/triplet", response_class=HTMLResponse)
async def triplet_page(request: Request):
    """Render the Triplet Generation tool page"""
    return templates.TemplateResponse(
        "triplet_processing/triplet.html",
        {
            "request": request,
            "providers": get_supported_providers(),
        },
    )


@app.get("/datasets", response_class=HTMLResponse)
async def datasets_page(request: Request):
    """Render the Datasets Management page"""
    return templates.TemplateResponse(
        "datasets_processing/datasets.html",
        {
            "request": request,
        },
    )


@app.post("/update_provider")
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


@app.post("/update_model")
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


@app.post("/chat")
async def chat(request: ChatRequest):
    """Generate a response from an LLM."""
    logger.info(f"Chat request received: {request}")
    if request.llm_key not in llm_config:
        raise HTTPException(
            status_code=400, detail=f"Invalid LLM key: {request.llm_key}"
        )

    # Add the user message to the chat history
    chat_history[request.llm_key].append({"role": "user", "content": request.message})

    # Generate a response
    messages = to_langchain_messages(
        request.message, chat_history[request.llm_key][:-1]
    )
    llm = llm_config[request.llm_key]
    response = generate_response(llm, request.message, messages)
    response_content = convert_reasoning_to_markdown(response)

    # Add the assistant response to the chat history
    chat_history[request.llm_key].append(
        {"role": "assistant", "content": response_content}
    )

    # Return the updated chat history for this LLM
    return templates.TemplateResponse(
        "rlhf_processing/rlhf_chat.html",
        {
            "chat_history": chat_history[request.llm_key],
        },
    )


@app.post("/rate")
async def rate(rating: RatingRequest):
    # Similar to the submit_rating function in the original app
    # Save the rating to the RLHF dataset
    if not chat_history["llm_1"] or not chat_history["llm_2"]:
        response = JSONResponse(
            content={
                "status": "error",
                "message": "Both LLMs must have chat history before rating",
            }
        )

        # Add error toast notification
        response.headers["HX-Trigger-After-Swap"] = json.dumps(
            {
                "showToast": {
                    "message": "Error: Both LLMs must have chat history before rating",
                    "type": "error",
                    "duration": 3000,
                }
            }
        )
        return response

    # Get the most recent query (should be the same for both)
    latest_user_message_index_1 = max(
        [i for i, m in enumerate(chat_history["llm_1"]) if m["role"] == "user"],
        default=-1,
    )
    latest_user_message_index_2 = max(
        [i for i, m in enumerate(chat_history["llm_2"]) if m["role"] == "user"],
        default=-1,
    )

    if latest_user_message_index_1 == -1 or latest_user_message_index_2 == -1:
        response = JSONResponse(
            content={
                "status": "error",
                "message": "Both LLMs must have at least one user message before rating",
            }
        )

        # Add error toast notification
        response.headers["HX-Trigger-After-Swap"] = json.dumps(
            {
                "showToast": {
                    "message": "Error: Both LLMs must have at least one user message before rating",
                    "type": "error",
                    "duration": 3000,
                }
            }
        )
        return response

    # Save the rating
    user_prompt_1 = chat_history["llm_1"][latest_user_message_index_1]["content"]
    user_prompt_2 = chat_history["llm_2"][latest_user_message_index_2]["content"]

    if user_prompt_1 != user_prompt_2:
        response = JSONResponse(
            content={
                "status": "error",
                "message": "The most recent user messages for both LLMs must match",
            }
        )

        # Add error toast notification
        response.headers["HX-Trigger-After-Swap"] = json.dumps(
            {
                "showToast": {
                    "message": "Error: The most recent user messages for both LLMs must match",
                    "type": "error",
                    "duration": 3000,
                }
            }
        )
        return response

    # Get the most recent assistant response
    latest_assistant_message_index_1 = max(
        [
            i
            for i, m in enumerate(chat_history["llm_1"])
            if m["role"] == "assistant" and i > latest_user_message_index_1
        ],
        default=-1,
    )
    latest_assistant_message_index_2 = max(
        [
            i
            for i, m in enumerate(chat_history["llm_2"])
            if m["role"] == "assistant" and i > latest_user_message_index_2
        ],
        default=-1,
    )

    if latest_assistant_message_index_1 == -1 or latest_assistant_message_index_2 == -1:
        response = JSONResponse(
            content={
                "status": "error",
                "message": "Both LLMs must have responded to the latest user message before rating",
            }
        )

        # Add error toast notification
        response.headers["HX-Trigger-After-Swap"] = json.dumps(
            {
                "showToast": {
                    "message": "Error: Both LLMs must have responded to the latest user message before rating",
                    "type": "error",
                    "duration": 3000,
                }
            }
        )
        return response

    # Now save the RLHF data
    system_prompt = ""  # Default empty system prompt
    user_prompt = user_prompt_1  # Verified to be the same as user_prompt_2
    llm1_name = llm_config["llm_1"]["provider"] + " " + llm_config["llm_1"]["model"]
    llm2_name = llm_config["llm_2"]["provider"] + " " + llm_config["llm_2"]["model"]
    response1 = chat_history["llm_1"][latest_assistant_message_index_1]["content"]
    response2 = chat_history["llm_2"][latest_assistant_message_index_2]["content"]
    chosen_rating = rating.chosen  # "llm_1" or "llm_2"
    notes = rating.notes

    # Save the entry
    rlhf_builder.save_rlhf_entry(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        llm1_name=llm1_name,
        llm2_name=llm2_name,
        response1=response1,
        response2=response2,
        rating=chosen_rating.upper(),  # Convert to uppercase
        notes=notes,
    )

    # Clear the chat history for both LLMs
    chat_history["llm_1"] = []
    chat_history["llm_2"] = []

    response = JSONResponse(content={"status": "success", "message": "Rating saved"})

    # Add success toast notification
    response.headers["HX-Trigger-After-Swap"] = json.dumps(
        {
            "showToast": {
                "message": f"Rating saved: {chosen_rating.upper()} preferred",
                "type": "success",
                "duration": 2000,
            }
        }
    )

    return response


@app.get("/rlhf_data")
async def get_rlhf_data(request: Request, format: str = "table", query: str = ""):
    # Get RLHF dataset samples
    data = rlhf_builder.get_samples()

    # Filter if there's a search query
    if query:
        filtered_data = []
        for sample in data:
            if (
                query.lower() in sample.prompt.lower()
                or query.lower() in sample.chosen.lower()
                or query.lower() in sample.rejected.lower()
                or query.lower() in (sample.notes or "").lower()
            ):
                filtered_data.append(sample)
        data = filtered_data

    # Format the data for display
    formatted_data = []
    for i, sample in enumerate(data):
        formatted_data.append(
            {
                "id": str(i),  # Add ID for each entry
                "user_prompt": sample.prompt,
                "llm1": sample.chosen_model,
                "llm2": sample.rejected_model,
                "response1": sample.chosen,
                "response2": sample.rejected,
                "rating": "LLM1 better than LLM2",
                "notes": sample.notes,
            }
        )

    # Return as JSON if requested
    if format == "json":
        return JSONResponse(content={"data": formatted_data})

    # Otherwise return as HTML table
    return templates.TemplateResponse(
        "rlhf_processing/rlhf_table.html",
        {
            "request": request,
            "data": formatted_data,
        },
    )


@app.post("/download_rlhf")
async def download_rlhf():
    try:
        filepath = rlhf_builder.save_dataset_jsonl()
        return {
            "status": "success",
            "message": f"Dataset saved to {filepath}",
            "file": filepath,
        }
    except Exception as e:
        logger.error(f"Error saving dataset: {e}")
        return {"status": "error", "message": f"Error saving dataset: {str(e)}"}


@app.post("/generate_queries")
async def generate_queries(
    domain: str = Form("general"),
    count: int = Form(10),
    custom_domain: str = Form(None),
):
    """Generate queries for triplet dataset"""
    # This would be implemented to use an LLM to generate diverse queries
    # For now, return some sample queries
    sample_queries = [
        "What are the main features of Python 3.10?",
        "How do I deploy a FastAPI application to production?",
        "Explain the difference between supervised and unsupervised learning",
        "What are the best practices for REST API design?",
        "How does BERT handle natural language processing tasks?",
    ]

    # Limit to the requested count
    return {"queries": sample_queries[:count]}


@app.post("/generate_responses")
async def generate_responses(
    query: str = Form(...), provider: str = Form(...), model: str = Form(...)
):
    """Generate positive and negative responses for a query using the selected LLM"""
    try:
        # Use the LLM to generate responses
        llm_client = get_llm(provider, model)

        # Generate a positive response
        positive_prompt = f"Please provide a high-quality, detailed, and informative response to this query: {query}"
        positive_response = llm_client.invoke(positive_prompt, [])

        # Generate a negative response
        negative_prompt = f"Please provide a brief, low-quality, uninformative response to this query: {query}"
        negative_response = llm_client.invoke(negative_prompt, [])

        return {
            "positive": positive_response,
            "negative": negative_response,
        }
    except Exception as e:
        logger.error(f"Error generating responses: {e}")
        return {
            "error": f"Failed to generate responses: {str(e)}",
            "positive": "",
            "negative": "",
        }


@app.post("/extract_triplets")
async def process_triplets(
    text: str = Form(...),
    provider: str = Form(...),
    model: str = Form(...),
    entity_types: str = Form("organization,person,geo,event,category"),
):
    """Extract knowledge graph triplets from text using the selected LLM"""
    try:
        # Split entity types string into a list
        entity_types_list = [t.strip() for t in entity_types.split(",")]

        # Use our extract_triplets function with the specified provider and model
        result = extract_triplets(
            text=text,
            provider=provider,
            model_name=model,
            entity_types=entity_types_list,
        )

        # Add the extracted triplets to our dataset
        for triplet in result["triplets"]:
            triplet_dataset.append(
                {
                    "id": len(triplet_dataset) + 1,
                    "subject": triplet.subject,
                    "predicate": triplet.predicate,
                    "object": triplet.object,
                    "description": triplet.description,
                    "confidence": triplet.confidence,
                    "created_at": "2023-06-01T12:00:00Z",  # This would be the current timestamp in production
                }
            )

        # Format the triplets for display
        formatted_triplets = []
        for t in result["triplets"]:
            formatted_triplets.append(
                {
                    "subject": t.subject,
                    "predicate": t.predicate,
                    "object": t.object,
                    "description": t.description,
                    "confidence": t.confidence,
                }
            )

        # Return HTML for displaying the results
        return templates.TemplateResponse(
            "triplet_processing/triplet_extraction_results.html",
            {
                "triplets": formatted_triplets,
                "count": len(formatted_triplets),
                "entities": len(result["entities"]),
            },
        )
    except Exception as e:
        logger.error(f"Error extracting triplets: {e}")
        return f"""
        <div class="p-4 bg-red-100 text-red-700 rounded-md">
            <h3 class="font-bold">Error</h3>
            <p>{str(e)}</p>
        </div>
        """


@app.post("/add_manual_triplet")
async def add_manual_triplet(triplet: TripletRequest):
    """Add a manually created SPO triplet to the dataset"""
    # Add the triplet to our in-memory storage
    triplet_dataset.append(
        {
            "id": len(triplet_dataset) + 1,
            "subject": triplet.subject,
            "predicate": triplet.predicate,
            "object": triplet.object,
            "description": triplet.description,
            "confidence": 1.0,  # Manual triplets have full confidence
            "created_at": "2023-06-01T12:00:00Z",  # This would be the current timestamp in production
        }
    )

    return {
        "status": "success",
        "message": "SPO triplet added successfully",
        "count": len(triplet_dataset),
    }


@app.get("/view_triplets")
async def view_triplets(request: Request, format: str = "table", query: str = ""):
    """Get the triplet dataset"""
    # Filter the dataset if there's a search query
    data = triplet_dataset
    if query:
        filtered_data = []
        for triplet in data:
            if (
                query.lower() in triplet.get("subject", "").lower()
                or query.lower() in triplet.get("predicate", "").lower()
                or query.lower() in triplet.get("object", "").lower()
                or query.lower() in triplet.get("description", "").lower()
            ):
                filtered_data.append(triplet)
        data = filtered_data

    # Return as JSON if requested
    if format == "json":
        return JSONResponse(content={"data": data})

    # Otherwise return as HTML table
    return templates.TemplateResponse(
        "triplet_processing/triplet_table.html",
        {
            "request": request,
            "data": data,
        },
    )


@app.post("/download_triplets")
async def download_triplets():
    """Download the triplet dataset"""
    # In a real app, this would save to disk or provide a download link
    count = len(triplet_dataset)
    if count == 0:
        return {"status": "error", "message": "No triplets to download"}

    return {
        "status": "success",
        "message": f"Dataset with {count} triplets would be downloaded",
        "count": count,
    }


@app.get("/dataset_counts")
async def dataset_counts():
    """Get the number of samples in each dataset"""
    return {"rlhf": len(rlhf_builder.get_samples()), "triplet": len(triplet_dataset)}


@app.post("/import_dataset")
async def import_dataset():
    """Import a dataset from a file"""
    # This would parse and import the uploaded file
    return {
        "status": "success",
        "message": "Dataset import functionality will be implemented soon",
    }


@app.get("/expand_text")
async def expand_text(request: Request, id: str, field: str):
    """Return the full text of a field in an RLHF dataset entry."""
    entry = rlhf_builder.get_entry_by_id(id)
    if not entry:
        return HTMLResponse(f"Entry with ID {id} not found.")

    if field not in entry:
        return HTMLResponse(f"Field {field} not found in entry.")

    full_text = entry[field]

    # Return both the expanded text and a button to collapse it
    return HTMLResponse(f"""
        <div>{full_text}</div>
        <button class="text-blue-600 text-sm mt-2"
                hx-get="/collapse_text?id={id}&field={field}"
                hx-target="#{field}-{id}"
                hx-swap="innerHTML">
            Show less
        </button>
    """)


@app.get("/collapse_text")
async def collapse_text(request: Request, id: str, field: str):
    """Return the truncated text of a field in an RLHF dataset entry."""
    entry = rlhf_builder.get_entry_by_id(id)
    if not entry:
        return HTMLResponse(f"Entry with ID {id} not found.")

    if field not in entry:
        return HTMLResponse(f"Field {field} not found in entry.")

    full_text = entry[field]
    truncated = full_text[:150] + "..." if len(full_text) > 150 else full_text

    # Return both the truncated text and a button to expand it
    return HTMLResponse(f"""
        <div>{truncated}</div>
        <button class="text-blue-600 text-sm mt-2"
                hx-get="/expand_text?id={id}&field={field}"
                hx-target="#{field}-{id}"
                hx-swap="innerHTML"
                hx-indicator="#indicator-{id}">
            Show more
        </button>
        <div id="indicator-{id}" class="htmx-indicator">
            <div class="spinner"></div>
        </div>
    """)


@app.get("/provider_models")
async def provider_models(provider: str):
    """Get the list of models available for a specific provider"""
    try:
        models = get_provider_models(provider)
        return {"models": models}
    except Exception as e:
        logger.error(f"Error getting models for provider {provider}: {e}")
        return {"error": str(e), "models": []}


@app.get("/batch", response_class=HTMLResponse)
async def batch_page(request: Request):
    """Render the Batch Dataset Builder page"""
    return templates.TemplateResponse(
        "batch_processing/batch.html",
        {
            "request": request,
            "openai_models": OPENAI_CHAT_MODELS,
        },
    )


@app.post("/process_batch")
async def process_batch(
    request: Request,
    csv_file: UploadFile = File(...),
    system_prompt: str = Form(...),
    user_prompt: str = Form(...),
    model: str = Form(...),
    max_tokens: int = Form(1000),
    temperature: float = Form(0.7),
    reasoning_effort: Optional[str] = Form(None),
):
    """Process a batch of data using OpenAI Batch API"""
    try:
        # Create temp file to store uploaded CSV
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
            temp_file_path = temp_file.name
            content = await csv_file.read()
            temp_file.write(content)

        # Read the CSV file with pandas for robust parsing
        df = pd.read_csv(temp_file_path)
        os.unlink(temp_file_path)  # Clean up temp file

        # Validate CSV content
        if df.empty:
            return templates.TemplateResponse(
                "batch_processing/batch_error.html",
                {
                    "request": request,
                    "error_message": "The CSV file is empty.",
                },
            )

        # Create output file for batch results
        timestamp = int(time.time())
        output_path = f"ymir/data/batch_results_{timestamp}.jsonl"

        # Initialize the batch processor
        batch_processor = OpenAIBatchProcessor(
            description=f"Batch processing with model {model}",
            output_path=output_path,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning_effort=reasoning_effort if reasoning_effort else None,
        )

        # Create a PromptConfig for formatting
        from ymir.prompt.config import PromptConfig

        prompt_config = PromptConfig(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
        )

        # Save the prompt configuration for reference
        config_path = prompt_config.save_to_file()
        logger.info(f"Saved prompt configuration to {config_path}")

        # Define the template function that formats prompts using CSV data
        def format_prompts(row_dict):
            return prompt_config.format_prompts(row_dict)

        # Prepare template arguments - convert DataFrame to list of dicts
        template_args = df.to_dict(orient="records")

        # Create a unique name for the input JSONL file
        batch_input_file = f"ymir/data/batch_input_{timestamp}.jsonl"

        # Create the batch
        batch_processor.create_batch(
            template_args=template_args,
            template_func=format_prompts,
            batch_jsonl_input_save_file=batch_input_file,
        )

        # Submit the batch
        batch_processor.submit()

        # Return the batch status and monitoring info
        return templates.TemplateResponse(
            "batch_processing/batch_results.html",
            {
                "request": request,
                "batch_id": batch_processor.batch_id,
                "input_file": batch_input_file,
                "output_file": output_path,
                "num_rows": len(df),
                "model": model,
                "timestamp": timestamp,
                "config_path": config_path,
            },
        )
    except Exception as e:
        logger.error(f"Error processing batch: {e}")
        return templates.TemplateResponse(
            "batch_processing/batch_error.html",
            {
                "request": request,
                "error_message": f"Error processing batch: {str(e)}",
            },
        )


@app.get("/check_batch_status")
async def check_batch_status(request: Request, batch_id: str):
    """Check the status of a batch job"""
    try:
        client = OpenAI()
        status = client.batches.retrieve(batch_id)

        return {
            "status": status.status,
            "created_at": status.created_at,
            "completed_at": status.completed_at,
            "error": status.error,
            "total_requests": status.total_requests,
            "completed_count": status.completed_count,
        }
    except Exception as e:
        logger.error(f"Error checking batch status: {e}")
        return {"error": str(e)}


@app.get("/download")
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


@app.get("/document", response_class=HTMLResponse)
async def document_page(request: Request):
    """Render the Document Processor page"""
    return templates.TemplateResponse(
        "document_processing/document_processor.html",
        {
            "request": request,
        },
    )


@app.post("/upload_pdf")
async def upload_pdf(request: Request, pdf_file: UploadFile = File(...)):
    """Upload a PDF file and return basic information about it"""
    try:
        logger.info(f"PDF upload received: {pdf_file.filename}")

        # Create data directory if it doesn't exist
        data_dir = Path("ymir/data/documents")
        data_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Using data directory: {data_dir.absolute()}")

        # Save the uploaded file
        timestamp = int(time.time())
        filename = f"{timestamp}_{pdf_file.filename}"
        file_path = data_dir / filename
        logger.info(f"Saving file to: {file_path.absolute()}")

        with open(file_path, "wb") as f:
            content = await pdf_file.read()
            f.write(content)
            logger.info(f"Wrote {len(content)} bytes to file")

        # Get basic PDF info
        from pypdf import PdfReader

        reader = PdfReader(file_path)
        num_pages = len(reader.pages)
        logger.info(f"PDF has {num_pages} pages")

        # Return PDF info
        response = templates.TemplateResponse(
            "document_processing/document_pdf_info.html",
            {
                "request": request,
                "filename": pdf_file.filename,
                "file_path": str(file_path),
                "num_pages": num_pages,
            },
        )
        logger.info("Returning PDF info template response")
        return response
    except Exception as e:
        logger.error(f"Error uploading PDF: {e}", exc_info=True)
        return templates.TemplateResponse(
            "document_processing/document_error.html",
            {
                "request": request,
                "error_message": f"Error uploading PDF: {str(e)}",
            },
        )


@app.post("/detect_toc")
async def detect_toc(
    request: Request,
    pdf_path: str = Form(...),
    toc_start_page: Optional[int] = Form(None),
    toc_end_page: Optional[int] = Form(None),
):
    """Detect table of contents in a PDF file"""
    try:
        logger.info(f"Detecting TOC in PDF: {pdf_path}")
        logger.info(f"TOC page range provided: {toc_start_page}-{toc_end_page}")

        # Check if manual TOC range was provided
        toc_page_range = None
        if toc_start_page is not None and toc_end_page is not None:
            toc_page_range = (toc_start_page, toc_end_page)
            logger.info(f"Using manually specified TOC page range: {toc_page_range}")

        # Extract chapter starts using the function from pdf.py
        chapter_starts = extract_chapter_starts(pdf_path, toc_page_range)
        logger.info(f"Detected chapter starts (0-indexed): {chapter_starts}")

        if not chapter_starts:
            logger.warning("No chapter starts detected in the PDF")
            return templates.TemplateResponse(
                "document_processing/document_error.html",
                {
                    "request": request,
                    "error_message": "No chapters detected in the PDF. Try specifying a manual TOC page range where the table of contents is located.",
                },
            )

        # Convert to 1-indexed for display
        chapter_pages = [page + 1 for page in chapter_starts]
        logger.info(f"Chapter pages (1-indexed for display): {chapter_pages}")

        # Create chapter info
        chapters = []
        for i, page in enumerate(chapter_pages):
            next_page = (
                chapter_pages[i + 1] - 1 if i < len(chapter_pages) - 1 else "end"
            )
            chapters.append(
                {
                    "number": i + 1,
                    "start_page": page,
                    "end_page": next_page,
                }
            )

        # Return chapter info
        return templates.TemplateResponse(
            "document_processing/document_toc_results.html",
            {
                "request": request,
                "chapters": chapters,
                "chapter_starts": ",".join(map(str, chapter_starts)),
                "pdf_path": pdf_path,
            },
        )
    except Exception as e:
        logger.error(f"Error detecting TOC: {e}", exc_info=True)
        return templates.TemplateResponse(
            "document_processing/document_error.html",
            {
                "request": request,
                "error_message": f"Error detecting table of contents: {str(e)}",
            },
        )


@app.post("/process_pdf")
async def process_pdf(
    request: Request,
    pdf_path: str = Form(...),
    chapter_starts: str = Form(...),
    split_chapters: bool = Form(False),
    extract_text: bool = Form(True),
    create_csv: bool = Form(True),
    toc_start_page: Optional[int] = Form(None),
    toc_end_page: Optional[int] = Form(None),
):
    """Process a PDF file based on detected chapters"""
    try:
        # Generate a unique job ID for tracking progress
        job_id = f"pdf_job_{int(time.time())}"

        # Initialize progress tracking
        update_progress(job_id, "init", 0, 100, "Starting PDF processing")

        # Parse chapter starts
        chapter_starts_list = [int(page) for page in chapter_starts.split(",")]
        update_progress(
            job_id, "chapters", 1, 3, f"Identified {len(chapter_starts_list)} chapters"
        )

        # Create output directory
        timestamp = int(time.time())
        output_dir = Path("ymir/data/documents") / f"output_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)
        update_progress(job_id, "prepare", 2, 3, "Created output directory")

        # Base filename for outputs
        pdf_filename = os.path.basename(pdf_path)
        base_name = os.path.splitext(pdf_filename)[0]
        output_prefix = str(output_dir / base_name)

        results = {
            "chapters_processed": 0,
            "csv_path": None,
            "chapter_pdfs": [],
            "job_id": job_id,  # Include job ID in results
        }

        update_progress(job_id, "processing", 3, 3, "Starting PDF operations")

        # Split PDF if requested
        if split_chapters:
            # Define a progress callback to track PDF splitting
            def progress_callback(current, total, message=""):
                # progress_percent = 10 + int(
                #     (current / total) * 60
                # )  # Scale to 10-70% range
                update_progress(job_id, "splitting", current, total, message)

            update_progress(
                job_id, "splitting", 0, 100, "Starting PDF chapter splitting"
            )

            # Call split_pdf_by_chapters with the progress callback
            chapter_contents = split_pdf_by_chapters(
                pdf_path,
                output_prefix,
                chapter_starts_list,
                progress_callback=progress_callback,
            )

            results["chapters_processed"] = len(chapter_contents)
            results["chapter_pdfs"] = list(chapter_contents.keys())

            update_progress(
                job_id,
                "splitting",
                100,
                100,
                f"Split PDF into {len(chapter_contents)} chapters",
            )

        # Create CSV if requested
        if create_csv:
            csv_path = f"{output_prefix}_chapters.csv"
            update_progress(job_id, "csv", 0, 100, "Creating CSV dataset")

            # If we already have chapter contents from splitting
            if split_chapters and extract_text:
                with open(csv_path, "w", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(["chapter", "pages", "content"])

                    total_chapters = len(chapter_contents)
                    for i, (pdf_path, content) in enumerate(chapter_contents.items()):
                        chapter_num = i + 1
                        start_page = chapter_starts_list[i] + 1  # Convert to 1-indexed
                        end_page = (
                            chapter_starts_list[i + 1]
                            if i + 1 < len(chapter_starts_list)
                            else "end"
                        )
                        pages = f"{start_page}-{end_page}"

                        # Join all page content
                        full_content = "\n\n".join(content.strip())
                        writer.writerow([chapter_num, pages, full_content.strip()])

                        # Update progress (70-90% range for CSV creation)
                        # progress_percent = 70 + int((i / total_chapters) * 20)
                        update_progress(
                            job_id,
                            "csv",
                            i + 1,
                            total_chapters,
                            f"Processing chapter {chapter_num} for CSV",
                        )

            # If we need to extract text without splitting
            elif extract_text:
                update_progress(
                    job_id, "extracting_text", 0, 100, "Extracting text from PDF"
                )
                from pypdf import PdfReader

                reader = PdfReader(pdf_path)

                with open(csv_path, "w", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(["chapter", "pages", "content"])

                    total_chapters = len(chapter_starts_list) - 1
                    for i in range(total_chapters):
                        chapter_num = i + 1
                        start_page = chapter_starts_list[i]
                        end_page = chapter_starts_list[i + 1]
                        pages = f"{start_page + 1}-{end_page}"

                        # Extract text from all pages in this chapter
                        chapter_content = []
                        total_pages = end_page - start_page
                        for page_idx, page_num in enumerate(
                            range(start_page, end_page)
                        ):
                            if page_num < len(reader.pages):
                                page_text = (
                                    reader.pages[page_num].extract_text().strip()
                                )
                                chapter_content.append(page_text)

                                # Update progress for each page processed
                                # sub_progress = int(
                                #     (page_idx / max(total_pages, 1)) * 100
                                update_progress(
                                    job_id,
                                    "extracting_page",
                                    page_idx + 1,
                                    total_pages,
                                    f"Extracting text from chapter {chapter_num}, page {page_num + 1}",
                                )

                        full_content = "\n\n".join(chapter_content)
                        writer.writerow([chapter_num, pages, full_content])

                        # Update overall progress (70-90% range)
                        # progress_percent = 70 + int((i / total_chapters) * 20)
                        update_progress(
                            job_id,
                            "csv",
                            i + 1,
                            total_chapters,
                            f"Added chapter {chapter_num} to CSV",
                        )

            # Just create a CSV with chapter info but no content
            else:
                with open(csv_path, "w", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(["chapter", "pages", "content"])

                    total_chapters = len(chapter_starts_list) - 1
                    for i in range(total_chapters):
                        chapter_num = i + 1
                        start_page = chapter_starts_list[i] + 1  # Convert to 1-indexed
                        end_page = chapter_starts_list[i + 1]
                        pages = f"{start_page}-{end_page}"

                        writer.writerow([chapter_num, pages, ""])

                        # Update progress
                        # progress_percent = 70 + int((i / total_chapters) * 20)
                        update_progress(
                            job_id,
                            "csv",
                            i + 1,
                            total_chapters,
                            f"Added chapter {chapter_num} info to CSV",
                        )

            results["csv_path"] = csv_path
            update_progress(job_id, "completed", 100, 100, "PDF processing complete")

        # Return processing results
        return templates.TemplateResponse(
            "document_processing/document_processing_results.html",
            {
                "request": request,
                "results": results,
                "pdf_path": pdf_path,
            },
        )
    except Exception as e:
        logger.error(f"Error processing PDF: {e}", exc_info=True)
        return templates.TemplateResponse(
            "document_processing/document_error.html",
            {
                "request": request,
                "error_message": f"Error processing PDF: {str(e)}",
            },
        )


@app.delete("/delete_triplet/{triplet_id}")
async def delete_triplet(request: Request, triplet_id: int):
    """Delete a triplet from the dataset"""
    global triplet_dataset

    # Find the triplet by ID
    for i, triplet in enumerate(triplet_dataset):
        if triplet["id"] == triplet_id:
            # Remove the triplet
            del triplet_dataset[i]

            # Return the updated table
            return await view_triplets(request)

    # If triplet not found, return an error
    return JSONResponse(
        content={
            "status": "error",
            "message": f"Triplet with ID {triplet_id} not found",
        }
    )


def update_progress(
    job_id: str, step: str, current: int, total: int, message: str = ""
):
    """Update the progress for a given job ID"""
    progress = {
        "step": step,
        "current": current,
        "total": total,
        "percent": int((current / max(total, 1)) * 100),
        "message": message,
        "time": time.time(),
    }
    pdf_processing_progress[job_id] = progress
    logger.info(f"Progress update for job {job_id}: {progress['percent']}% - {message}")
    return progress


@app.get("/pdf_progress/{job_id}")
async def get_pdf_progress(job_id: str):
    """Get the progress for a PDF processing job"""
    if job_id in pdf_processing_progress:
        return pdf_processing_progress[job_id]
    else:
        return {"error": "Job not found", "percent": 0, "message": "Unknown job ID"}


@app.post("/parse_csv")
async def parse_csv(request: Request, csv_file: UploadFile = File(...)):
    """Parse a CSV file and return headers and a preview of rows"""
    try:
        # Create temp file to store uploaded CSV
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
            temp_file_path = temp_file.name
            content = await csv_file.read()
            temp_file.write(content)

        # Read the CSV file with pandas for robust parsing
        df = pd.read_csv(temp_file_path)
        os.unlink(temp_file_path)  # Clean up temp file

        # Validate CSV content
        if df.empty:
            return JSONResponse(content={"error": "The CSV file is empty."})

        # Get headers (column names)
        headers = df.columns.tolist()

        # Get a preview of rows (first 5 rows)
        # Convert to list of lists for simpler JSON serialization
        preview_rows = df.head(5).values.tolist()

        return JSONResponse(content={"headers": headers, "rows": preview_rows})

    except Exception as e:
        logger.error(f"Error parsing CSV: {str(e)}")
        return JSONResponse(
            content={"error": f"Error parsing CSV file: {str(e)}"}, status_code=400
        )


@app.post("/save_prompt_config")
async def save_prompt_config(
    system_prompt: str = Form(...),
    user_prompt: str = Form(...),
    model: str = Form(...),
    max_tokens: int = Form(1000),
    temperature: float = Form(0.7),
    reasoning_effort: Optional[str] = Form(None),
):
    """Save prompt configuration to a YAML file and return it for download"""
    try:
        from ymir.prompt.config import PromptConfig

        # Create config object
        config = PromptConfig(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
        )

        # Save the config to a file using the model's method
        file_path = config.save_to_file()

        # Get the YAML content
        yaml_content = config.to_yaml()

        # Get just the filename
        filename = os.path.basename(file_path)

        # Return the file content and path for download
        return JSONResponse(
            {
                "status": "success",
                "message": "Prompt configuration saved",
                "file_path": file_path,
                "file_name": filename,
                "yaml_content": yaml_content,
            }
        )

    except Exception as e:
        logger.error(f"Error saving prompt configuration: {str(e)}")
        return JSONResponse(
            content={"error": f"Error saving prompt configuration: {str(e)}"},
            status_code=400,
        )


@app.post("/load_prompt_config")
async def load_prompt_config(config_file: UploadFile = File(...)):
    """Load prompt configuration from a YAML file"""
    try:
        # Read the uploaded file content
        content = await config_file.read()

        # Parse the YAML using PromptConfig
        from ymir.prompt.config import PromptConfig

        config = PromptConfig.from_yaml(content.decode("utf-8"))

        # Return the configuration dictionary
        return JSONResponse(content=config.dict())

    except Exception as e:
        logger.error(f"Error loading prompt configuration: {str(e)}")
        return JSONResponse(
            content={"error": f"Error loading prompt configuration: {str(e)}"},
            status_code=400,
        )


@app.post("/calculate_token_stats")
async def calculate_token_stats(
    system_prompt: str = Form(...),
    user_prompt: str = Form(...),
    csv_data: str = Form(...),
    headers: str = Form(...),
):
    """Calculate token statistics for formatted prompts using the CSV data"""
    try:
        # Import token counting function
        from ymir.llm.openai_llm import count_tokens

        # Parse CSV data and headers from JSON strings
        import json

        csv_data = json.loads(csv_data)
        headers = json.loads(headers)

        # Create PromptConfig for formatting
        from ymir.prompt.config import PromptConfig

        config = PromptConfig(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model="gpt-4o",  # Default model for token counting
            max_tokens=1000,
            temperature=0.7,
        )

        # Calculate token counts for each row
        token_counts = []
        for row in csv_data:
            # Convert row to dictionary with headers as keys
            row_dict = {
                header: row[i] if i < len(row) else ""
                for i, header in enumerate(headers)
            }

            # Format prompts
            formatted_system, formatted_user = config.format_prompts(row_dict)

            # Count tokens in combined prompt
            combined_prompt = formatted_system + "\n" + formatted_user
            token_count = count_tokens(combined_prompt)
            token_counts.append(token_count)

        # Calculate statistics
        if token_counts:
            min_tokens = min(token_counts)
            max_tokens = max(token_counts)
            avg_tokens = sum(token_counts) / len(token_counts)

            return JSONResponse(
                {
                    "min_tokens": min_tokens,
                    "max_tokens": max_tokens,
                    "avg_tokens": round(avg_tokens),
                    "total_rows": len(token_counts),
                    "sample_counts": token_counts[
                        :5
                    ],  # Include first 5 counts for reference
                }
            )
        else:
            return JSONResponse(
                {"error": "No data to calculate token statistics"}, status_code=400
            )

    except Exception as e:
        logger.error(f"Error calculating token statistics: {str(e)}")
        return JSONResponse(
            {"error": f"Error calculating token statistics: {str(e)}"}, status_code=400
        )


if __name__ == "__main__":
    uvicorn.run("ymir.app:app", host="0.0.0.0", port=8008, reload=True)
