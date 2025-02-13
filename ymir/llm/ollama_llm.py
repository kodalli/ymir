from langchain_ollama import ChatOllama
import gradio as gr
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


def get_ollama_config_components(model_name: str) -> Dict[str, gr.Component]:
    """Returns Gradio components for Ollama model configuration."""
    return {
        "temperature": gr.Slider(
            minimum=0.0, maximum=2.0, value=0.7, step=0.1, label="Temperature"
        ),
        "context_length": gr.Slider(
            minimum=512, maximum=16384, value=2048, step=512, label="Context Length"
        ),
        "num_predict": gr.Slider(
            minimum=1,
            maximum=8192,
            value=1024,
            step=1,
            label="Number of Tokens to Predict",
        ),
        "top_p": gr.Slider(
            minimum=0.0, maximum=1.0, value=0.95, step=0.05, label="Top P"
        ),
        "top_k": gr.Slider(minimum=1, maximum=100, value=50, step=1, label="Top K"),
    }


def get_ollama_llm(model_name: str, config: Dict = None) -> ChatOllama:
    if config is None:
        config = {}
    if model_name not in OLLAMA_CHAT_MODELS:
        raise ValueError(
            f"Model {model_name} is not available. Available models: {OLLAMA_CHAT_MODELS}"
        )
    return ChatOllama(model=model_name, **config)
