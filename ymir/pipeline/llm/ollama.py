"""Ollama LLM provider for trajectory generation."""

from typing import Any
from loguru import logger

try:
    from ollama import Client, AsyncClient
except ImportError:
    Client = None
    AsyncClient = None
    logger.warning("ollama package not installed. Run: pip install ollama")


def get_available_models() -> list[str]:
    """Get list of available Ollama models."""
    if Client is None:
        return []
    try:
        client = Client()
        return [model.model for model in client.list().models]
    except Exception as e:
        logger.error(f"Error listing Ollama models: {e}")
        return []


class OllamaLLM:
    """Ollama LLM wrapper for trajectory generation."""

    def __init__(
        self,
        model: str,
        temperature: float = 0.7,
        num_predict: int = 2048,
        num_ctx: int = 8192,
    ):
        if Client is None:
            raise RuntimeError("ollama package not installed")

        self.model = model
        self.temperature = temperature
        self.num_predict = num_predict
        self.num_ctx = num_ctx
        self.client = Client()
        self.async_client = AsyncClient() if AsyncClient else None

    def generate(
        self,
        messages: list[dict[str, str]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> str:
        """Generate a response synchronously."""
        formatted_messages = []

        if system:
            formatted_messages.append({"role": "system", "content": system})

        formatted_messages.extend(messages)

        # Build options
        options = {
            "temperature": self.temperature,
            "num_predict": self.num_predict,
            "num_ctx": self.num_ctx,
        }

        try:
            response = self.client.chat(
                model=self.model,
                messages=formatted_messages,
                options=options,
            )
            return response.message.content
        except Exception as e:
            logger.error(f"Ollama generation error: {e}")
            raise

    async def agenerate(
        self,
        messages: list[dict[str, str]],
        system: str | None = None,
        tools: list[dict[str, Any]] | None = None,
    ) -> str:
        """Generate a response asynchronously."""
        if self.async_client is None:
            # Fall back to sync if async client not available
            return self.generate(messages, system, tools)

        formatted_messages = []

        if system:
            formatted_messages.append({"role": "system", "content": system})

        formatted_messages.extend(messages)

        options = {
            "temperature": self.temperature,
            "num_predict": self.num_predict,
            "num_ctx": self.num_ctx,
        }

        try:
            response = await self.async_client.chat(
                model=self.model,
                messages=formatted_messages,
                options=options,
            )
            return response.message.content
        except Exception as e:
            logger.error(f"Ollama async generation error: {e}")
            raise

    @staticmethod
    def list_models() -> list[str]:
        """List available models."""
        return get_available_models()
