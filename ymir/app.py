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

# Initialize global variables
app = FastAPI(title="Ymir AI Dataset Tools")
rlhf_builder = RLHFDatasetBuilder()
templates = Jinja2Templates(directory="ymir/templates")

# Initialize an empty triplet dataset (this would be replaced by a proper database in production)
triplet_dataset = []

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
    query: str
    positive: str
    negative: str


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
        "rlhf_content.html",
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
        "triplet.html",
        {
            "request": request,
            "providers": get_supported_providers(),
        },
    )


@app.get("/datasets", response_class=HTMLResponse)
async def datasets_page(request: Request):
    """Render the Datasets Management page"""
    return templates.TemplateResponse(
        "datasets.html",
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
        "rlhf_chat.html",
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
        "rlhf_table.html",
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
                    "query": f"What is the relationship between {triplet.subject} and {triplet.object}?",
                    "positive": f"{triplet.subject} {triplet.predicate} {triplet.object}. {triplet.description}",
                    "negative": f"{triplet.subject} and {triplet.object} may be related.",
                    "created_at": "2023-06-01T12:00:00Z",  # This would be the current timestamp in production
                    "confidence": triplet.confidence,
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
            "triplet_extraction_results.html",
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
    """Add a manually created triplet to the dataset"""
    # Add the triplet to our in-memory storage
    triplet_dataset.append(
        {
            "id": len(triplet_dataset) + 1,
            "query": triplet.query,
            "positive": triplet.positive,
            "negative": triplet.negative,
            "created_at": "2023-06-01T12:00:00Z",  # This would be the current timestamp in production
        }
    )

    return {
        "status": "success",
        "message": "Triplet added successfully",
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
                query.lower() in triplet["query"].lower()
                or query.lower() in triplet["positive"].lower()
                or query.lower() in triplet["negative"].lower()
            ):
                filtered_data.append(triplet)
        data = filtered_data

    # Return as JSON if requested
    if format == "json":
        return JSONResponse(content={"data": data})

    # Otherwise return as HTML table
    return templates.TemplateResponse(
        "triplet_table.html",
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


@app.on_event("startup")
async def startup_event():
    rlhf_builder.load_dataset_jsonl()
    logger.info("Ymir RLHF application started")


@app.get("/batch", response_class=HTMLResponse)
async def batch_page(request: Request):
    """Render the Batch Dataset Builder page"""
    return templates.TemplateResponse(
        "batch.html",
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

        # Read the CSV file
        df = pd.read_csv(temp_file_path)
        os.unlink(temp_file_path)  # Clean up temp file

        # Validate CSV content
        if df.empty:
            return templates.TemplateResponse(
                "batch_error.html",
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

        # Define the template function that formats prompts using CSV data
        def format_prompts(row_dict):
            formatted_system = system_prompt
            formatted_user = user_prompt

            # Replace placeholders with values from the row
            for key, value in row_dict.items():
                formatted_system = formatted_system.replace(f"{{{key}}}", str(value))
                formatted_user = formatted_user.replace(f"{{{key}}}", str(value))

            # No need to include system content in user prompt for 'o' models
            # as we now use the developer role

            return formatted_system, formatted_user

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
            "batch_results.html",
            {
                "request": request,
                "batch_id": batch_processor.batch_id,
                "input_file": batch_input_file,
                "output_file": output_path,
                "num_rows": len(df),
                "model": model,
                "timestamp": timestamp,
            },
        )
    except Exception as e:
        logger.error(f"Error processing batch: {e}")
        return templates.TemplateResponse(
            "batch_error.html",
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
        "document_processor.html",
        {
            "request": request,
        },
    )


@app.post("/upload_pdf")
async def upload_pdf(request: Request, pdf_file: UploadFile = File(...)):
    """Upload a PDF file and return basic information about it"""
    try:
        # Create data directory if it doesn't exist
        data_dir = Path("ymir/data/documents")
        data_dir.mkdir(parents=True, exist_ok=True)

        # Save the uploaded file
        timestamp = int(time.time())
        filename = f"{timestamp}_{pdf_file.filename}"
        file_path = data_dir / filename

        with open(file_path, "wb") as f:
            content = await pdf_file.read()
            f.write(content)

        # Get basic PDF info
        from pypdf import PdfReader

        reader = PdfReader(file_path)
        num_pages = len(reader.pages)

        # Return PDF info
        return templates.TemplateResponse(
            "document_pdf_info.html",
            {
                "request": request,
                "filename": pdf_file.filename,
                "file_path": str(file_path),
                "num_pages": num_pages,
            },
        )
    except Exception as e:
        logger.error(f"Error uploading PDF: {e}")
        return templates.TemplateResponse(
            "document_error.html",
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
        # Check if manual TOC range was provided
        toc_page_range = None
        if toc_start_page is not None and toc_end_page is not None:
            toc_page_range = (toc_start_page, toc_end_page)

        # Extract chapter starts
        chapter_starts = extract_chapter_starts(pdf_path, toc_page_range)

        # Convert to 1-indexed for display
        chapter_pages = [page + 1 for page in chapter_starts]

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
            "document_toc_results.html",
            {
                "request": request,
                "chapters": chapters,
                "chapter_starts": ",".join(map(str, chapter_starts)),
                "pdf_path": pdf_path,
            },
        )
    except Exception as e:
        logger.error(f"Error detecting TOC: {e}")
        return templates.TemplateResponse(
            "document_error.html",
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
        # Parse chapter starts
        chapter_starts_list = [int(page) for page in chapter_starts.split(",")]

        # Create output directory
        timestamp = int(time.time())
        output_dir = Path("ymir/data/documents") / f"output_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)

        # Base filename for outputs
        pdf_filename = os.path.basename(pdf_path)
        base_name = os.path.splitext(pdf_filename)[0]
        output_prefix = str(output_dir / base_name)

        results = {
            "chapters_processed": 0,
            "csv_path": None,
            "chapter_pdfs": [],
        }

        # Split PDF if requested
        if split_chapters:
            chapter_contents = split_pdf_by_chapters(
                pdf_path, output_prefix, chapter_starts_list
            )
            results["chapters_processed"] = len(chapter_contents)
            results["chapter_pdfs"] = list(chapter_contents.keys())

        # Create CSV if requested
        if create_csv:
            csv_path = f"{output_prefix}_chapters.csv"

            # If we already have chapter contents from splitting
            if split_chapters and extract_text:
                with open(csv_path, "w", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(["chapter", "pages", "content"])

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
                        full_content = "\n\n".join(content)

                        writer.writerow([chapter_num, pages, full_content])

            # If we need to extract text without splitting
            elif extract_text:
                from pypdf import PdfReader

                reader = PdfReader(pdf_path)

                with open(csv_path, "w", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(["chapter", "pages", "content"])

                    for i in range(len(chapter_starts_list) - 1):
                        chapter_num = i + 1
                        start_page = chapter_starts_list[i]
                        end_page = chapter_starts_list[i + 1]
                        pages = f"{start_page + 1}-{end_page}"

                        # Extract text from all pages in this chapter
                        chapter_content = []
                        for page_num in range(start_page, end_page):
                            if page_num < len(reader.pages):
                                page_text = reader.pages[page_num].extract_text()
                                chapter_content.append(page_text)

                        full_content = "\n\n".join(chapter_content)
                        writer.writerow([chapter_num, pages, full_content])

            # Just create a CSV with chapter info but no content
            else:
                with open(csv_path, "w", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(["chapter", "pages", "content"])

                    for i in range(len(chapter_starts_list) - 1):
                        chapter_num = i + 1
                        start_page = chapter_starts_list[i] + 1  # Convert to 1-indexed
                        end_page = chapter_starts_list[i + 1]
                        pages = f"{start_page}-{end_page}"

                        writer.writerow([chapter_num, pages, ""])

            results["csv_path"] = csv_path

        # Return processing results
        return templates.TemplateResponse(
            "document_processing_results.html",
            {
                "request": request,
                "results": results,
                "pdf_path": pdf_path,
            },
        )
    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        return templates.TemplateResponse(
            "document_error.html",
            {
                "request": request,
                "error_message": f"Error processing PDF: {str(e)}",
            },
        )


if __name__ == "__main__":
    uvicorn.run("ymir.app:app", host="0.0.0.0", port=8008, reload=True)
