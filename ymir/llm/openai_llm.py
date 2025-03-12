from langchain_openai import ChatOpenAI
from typing import Dict, Optional, Callable
import json
from tqdm import tqdm
from openai import OpenAI
from loguru import logger
import time

OPENAI_CHAT_MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "o3-mini",
    "o1-mini",
    "o1-preview",
]

REASONING_EFFORT = ["low", "medium", "high"]


def get_openai_config_components(
    model_name: Optional[str] = None,
) -> Dict[str, Dict]:
    """Returns configuration options for OpenAI model configuration."""
    config = {
        "temperature": {
            "type": "slider",
            "min": 0.0,
            "max": 2.0,
            "default": 0.7,
            "step": 0.1,
            "label": "Temperature",
        },
        "max_tokens": {
            "type": "slider",
            "min": 50,
            "max": 64000,
            "default": 1000,
            "step": 50,
            "label": "Max Tokens",
        },
    }
    if model_name and model_name.startswith(("o")):
        config["reasoning_effort"] = {
            "type": "dropdown",
            "choices": REASONING_EFFORT,
            "default": "medium",
            "label": "Reasoning Effort",
        }
    return config


def get_openai_llm(model_name: str, config: Optional[Dict] = None) -> ChatOpenAI:
    if config is None:
        config = {}
    return ChatOpenAI(model=model_name, **config)


class OpenAIBatchProcessor:
    def __init__(
        self,
        description: str,
        output_path: str,
        interval_seconds: int = 60,
        max_attempts: Optional[int] = None,
        model: str = "gpt-4o-mini",
        max_tokens: int = 8192,
        seed: Optional[int] = None,
        temperature: Optional[float] = None,
        reasoning_effort: Optional[str] = None,
    ):
        """
        Initialize BatchProcessor for handling OpenAI batch operations.

        Args:
            description: Description of the batch job
            output_path: Path where to save the results
            interval_seconds: Time between status checks
            max_attempts: Maximum number of status checks
            model: OpenAI model to use
            max_tokens: Maximum tokens for completion
            seed: Random seed for completions
            temperature: Temperature for completions
        """
        self.description = description
        self.output_path = output_path
        self.interval_seconds = interval_seconds
        self.max_attempts = max_attempts
        self.model = model
        self.max_tokens = max_tokens
        self.seed = seed
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort
        self.client = OpenAI()

        self.jsonl_path = None
        self.batch_id = None
        self.input_file_id = None

    @staticmethod
    def apply_batch_template(
        custom_id: str,
        system_message: str,
        user_message: str,
        model: str = "gpt-4o-mini",
        max_tokens: int = 8192,
        seed: Optional[int] = None,
        temperature: Optional[float] = None,
        reasoning_effort: Optional[str] = None,
    ):
        template = {
            "custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": model,
                "messages": [],
                "max_tokens": max_tokens,
            },
        }

        # For 'o' models (reasoning models), only include user message as system messages are not supported
        if model.startswith("o"):
            template["body"]["messages"] = [{"role": "user", "content": user_message}]
        else:
            template["body"]["messages"] = [
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_message},
            ]

        if seed is not None:
            template["body"]["seed"] = seed
        if temperature is not None:
            template["body"]["temperature"] = temperature
        if reasoning_effort is not None and model.startswith("o"):
            template["body"]["reasoning_effort"] = reasoning_effort
        return template

    def create_batch(
        self,
        template_args: list[dict],
        template_func: Callable,
        batch_jsonl_input_save_file: str,
    ) -> "OpenAIBatchProcessor":
        """
        Create a JSONL file for batch processing.

        Args:
            template_args: List of dictionaries containing arguments for template_func
            template_func: Function that returns (system_message, user_message) tuple
            batch_jsonl_input_save_file: Path to save the JSONL file
        """
        with open(batch_jsonl_input_save_file, "wb") as f:
            for i, args in tqdm(enumerate(template_args), total=len(template_args)):
                custom_id = f"request-{i}"
                system_message, user_message = template_func(**args)
                batch = self.apply_batch_template(
                    custom_id=custom_id,
                    system_message=system_message,
                    user_message=user_message,
                    model=self.model,
                    max_tokens=self.max_tokens,
                    seed=self.seed,
                    temperature=self.temperature,
                    reasoning_effort=self.reasoning_effort,
                )

                json_line = json.dumps(batch) + "\n"
                f.write(json_line.encode("utf-8"))

        self.jsonl_path = batch_jsonl_input_save_file
        logger.info(f"Created batch request file: {batch_jsonl_input_save_file}")
        return self

    def submit(self) -> "OpenAIBatchProcessor":
        """Submit batch job to OpenAI and return self for chaining."""
        if not self.jsonl_path:
            raise ValueError("No JSONL file created. Call create_batch first.")

        # Upload input file
        batch_input_file = self.client.files.create(
            file=open(self.jsonl_path, "rb"), purpose="batch"
        )
        self.input_file_id = batch_input_file.id

        # Create batch job
        batch_job = self.client.batches.create(
            input_file_id=self.input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": self.description},
        )
        self.batch_id = batch_job.id
        logger.info(f"Submitted batch job: {self.batch_id}")
        return self

    def monitor(self) -> "OpenAIBatchProcessor":
        """Monitor batch status and download results when complete."""
        if not self.batch_id:
            raise ValueError("No batch job submitted. Call submit first.")

        attempt = 0
        while True:
            status = self.client.batches.retrieve(self.batch_id)
            logger.info(f"Batch {self.batch_id} status: {status.status}")

            if status.status == "completed":
                logger.info("Batch completed successfully. Downloading results...")
                self._save_results(status.output_file_id)
                break
            elif status.status == "failed":
                logger.error("Batch failed during validation process")
                break
            elif status.status == "cancelled":
                logger.info("Batch was cancelled")
                break
            elif status.status == "expired":
                logger.error("Batch expired - not completed within 24-hour window")
                break
            elif status.status in [
                "validating",
                "in_progress",
                "finalizing",
                "cancelling",
            ]:
                # These are all valid intermediate states, continue monitoring
                logger.info(f"Batch is {status.status}...")

            attempt += 1
            if self.max_attempts and attempt >= self.max_attempts:
                logger.warning(f"Reached maximum attempts ({self.max_attempts})")
                break

            time.sleep(self.interval_seconds)
        return self

    def _save_results(self, output_file_id: str) -> None:
        """Download and save batch results."""
        try:
            file_response = self.client.files.content(output_file_id)
            with open(self.output_path, "w") as f:
                f.write(file_response.text)
            logger.info(f"Results saved to: {self.output_path}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")

    def cancel(self) -> "OpenAIBatchProcessor":
        """Cancel the batch job if it's running."""
        if self.batch_id:
            status = self.client.batches.cancel(self.batch_id)
            logger.info(f"Cancelled batch {self.batch_id}: {status}")
        return self
