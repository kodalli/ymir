from fastapi import APIRouter, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from ymir.llm import get_llm
from contextlib import asynccontextmanager
from fastapi import FastAPI
from loguru import logger
from pydantic import BaseModel
from typing import Optional
import json
import re
from ymir.routes.shared import (
    rlhf_builder,
    templates,
    chat_history,
    get_supported_providers,
    get_provider_models,
    llm_config,
)

router = APIRouter()


class ChatRequest(BaseModel):
    llm_key: str
    message: str


class RatingRequest(BaseModel):
    chosen: str  # "llm_1" or "llm_2"
    notes: Optional[str] = ""


class Message(BaseModel):
    role: str
    content: str


class LLMConfig(BaseModel):
    provider: str
    model: str


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


# Define lifespan context manager for startup/shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    rlhf_builder.load_dataset_jsonl()
    logger.info("Ymir Dataset Tools application started")
    yield
    # Shutdown logic (if any)
    logger.info("Shutting down Ymir Dataset Tools application")


@router.get("/expand_text")
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


@router.get("/collapse_text")
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


@router.post("/rate")
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


@router.get("/rlhf_data")
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


@router.post("/download_rlhf")
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


@router.post("/chat")
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


@router.get("/rlhf", response_class=HTMLResponse)
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
