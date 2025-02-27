from langchain_ollama import ChatOllama
from typing import Dict
from loguru import logger

OLLAMA_CHAT_MODELS = []
try:
    from ollama import Client

    client = Client()
    OLLAMA_CHAT_MODELS = [model.model for model in client.list().models]
except Exception as e:
    logger.error(
        f"Error listing Ollama models: {e}. Please ensure Ollama is installed and running."
    )
    pass


def get_ollama_config_components(model_name: str) -> Dict[str, Dict]:
    """Returns configuration options for Ollama model configuration."""
    return {
        "temperature": {
            "type": "slider",
            "min": 0.0,
            "max": 2.0,
            "default": 0.7,
            "step": 0.1,
            "label": "Temperature",
        },
        "context_length": {
            "type": "slider",
            "min": 512,
            "max": 16384,
            "default": 2048,
            "step": 512,
            "label": "Context Length",
        },
        "num_predict": {
            "type": "slider",
            "min": 1,
            "max": 8192,
            "default": 256,
            "step": 32,
            "label": "Max Tokens",
        },
    }


def get_ollama_llm(model_name: str, config: Dict = None) -> ChatOllama:
    if config is None:
        config = {}
    if model_name not in OLLAMA_CHAT_MODELS:
        raise ValueError(
            f"Model {model_name} is not available. Available models: {OLLAMA_CHAT_MODELS}"
        )
    return ChatOllama(model=model_name, **config)
