from fastapi import APIRouter, Request, Form, UploadFile, File
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
import tempfile
import pandas as pd
import os
from typing import Optional
from loguru import logger
from openai import OpenAI
from ymir.llm.openai_llm import OPENAI_CHAT_MODELS
from ymir.llm.openai_llm import OpenAIBatchProcessor
import time

router = APIRouter()

templates = Jinja2Templates(directory="ymir/templates")


@router.get("/batch", response_class=HTMLResponse)
async def batch_page(request: Request):
    """Render the Batch Dataset Builder page"""
    return templates.TemplateResponse(
        "batch_processing/batch.html",
        {
            "request": request,
            "openai_models": OPENAI_CHAT_MODELS,
        },
    )


@router.post("/process_batch")
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


@router.get("/check_batch_status")
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


@router.post("/parse_csv")
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


@router.post("/save_prompt_config")
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


@router.post("/load_prompt_config")
async def load_prompt_config(config_file: UploadFile = File(...)):
    """Load prompt configuration from a YAML file"""
    try:
        # Read the uploaded file content
        content = await config_file.read()

        # Parse the YAML using PromptConfig
        from ymir.prompt.config import PromptConfig

        config = PromptConfig.from_yaml(content.decode("utf-8"))

        # Return the configuration dictionary
        return JSONResponse(content=config.model_dump())

    except Exception as e:
        logger.error(f"Error loading prompt configuration: {str(e)}")
        return JSONResponse(
            content={"error": f"Error loading prompt configuration: {str(e)}"},
            status_code=400,
        )


@router.post("/calculate_token_stats")
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
    pass
