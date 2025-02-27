from typing import Optional
import uvicorn
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import re
from loguru import logger

from ymir.llm import (
    get_llm,
    get_supported_configurations,
    OPENAI_CHAT_MODELS,
    DEEPSEEK_CHAT_MODELS,
)
from ymir.rlhf import RLHFDatasetBuilder

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
    return templates.TemplateResponse(
        "index.html",
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


@app.get("/rlhf", response_class=HTMLResponse)
async def rlhf_page(request: Request):
    """Render the RLHF tool page (default content of index.html)"""
    return templates.TemplateResponse(
        "index.html",
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
    if model in models:
        llm_config[llm_key]["model"] = model
        return {"status": "success"}
    else:
        return {
            "status": "error",
            "message": f"Model {model} not available for provider {provider}",
        }


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
        return {
            "status": "error",
            "message": "Both LLMs must have chat history before rating",
        }

    # Get the most recent query (should be the same for both)
    latest_user_message_index_1 = max(
        [i for i, msg in enumerate(chat_history["llm_1"]) if msg["role"] == "user"]
        or [-1]
    )
    latest_user_message_index_2 = max(
        [i for i, msg in enumerate(chat_history["llm_2"]) if msg["role"] == "user"]
        or [-1]
    )

    if latest_user_message_index_1 == -1 or latest_user_message_index_2 == -1:
        return {"status": "error", "message": "No user messages found in chat history"}

    query = chat_history["llm_1"][latest_user_message_index_1]["content"]
    query2 = chat_history["llm_2"][latest_user_message_index_2]["content"]

    if query != query2:
        logger.warning(f"Queries don't match: {query} vs {query2}")

    # Get the responses to the most recent query
    response1_index = latest_user_message_index_1 + 1
    response2_index = latest_user_message_index_2 + 1

    if response1_index >= len(chat_history["llm_1"]) or response2_index >= len(
        chat_history["llm_2"]
    ):
        return {"status": "error", "message": "Missing response from one or both LLMs"}

    response1 = chat_history["llm_1"][response1_index]["content"]
    response2 = chat_history["llm_2"][response2_index]["content"]

    # Convert markdown reasoning back to the original format if needed
    response1 = convert_markdown_to_reasoning(response1)
    response2 = convert_markdown_to_reasoning(response2)

    chosen_index = 0 if rating.chosen == "llm_1" else 1
    rejected_index = 1 if rating.chosen == "llm_1" else 0

    rlhf_builder.add_sample(
        prompt=query,
        chosen=response1 if chosen_index == 0 else response2,
        rejected=response2 if rejected_index == 0 else response1,
        chosen_model=f"{llm_config['llm_1']['provider']}/{llm_config['llm_1']['model']}"
        if chosen_index == 0
        else f"{llm_config['llm_2']['provider']}/{llm_config['llm_2']['model']}",
        rejected_model=f"{llm_config['llm_2']['provider']}/{llm_config['llm_2']['model']}"
        if rejected_index == 0
        else f"{llm_config['llm_1']['provider']}/{llm_config['llm_1']['model']}",
        notes=rating.notes,
    )

    # Clear the chat history after saving the rating
    chat_history["llm_1"] = []
    chat_history["llm_2"] = []

    return {"status": "success", "message": "Rating saved successfully"}


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
    for sample in data:
        formatted_data.append(
            {
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
    """Generate positive and negative responses for a query"""
    # This would use the selected LLM to generate responses
    # For now, return sample responses
    return {
        "positive": "Here's a detailed explanation of the main features in Python 3.10: Structural pattern matching (match/case statements), better error messages with precise line indicators, parenthesized context managers, and typing improvements like Union operator using the pipe symbol (|). These features make Python code more readable and maintainable.",
        "negative": "Python 3.10 has some new stuff. You can look it up online.",
    }


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


@app.on_event("startup")
async def startup_event():
    rlhf_builder.load_dataset_jsonl()
    logger.info("Ymir RLHF application started")


if __name__ == "__main__":
    uvicorn.run("ymir.app:app", host="0.0.0.0", port=8008, reload=True)
