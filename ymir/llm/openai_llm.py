from langchain_openai import ChatOpenAI
from typing import Dict, Optional
import gradio as gr

OPENAI_CHAT_MODELS = [
    "gpt-4o",
    "gpt-4o-mini",
    "o3-mini",
    "o3-mini-high",
    "o1-mini",
    "o1-preview",
]

REASONING_EFFORT = ["low", "medium", "high"]


def get_openai_config_components(
    model_name: Optional[str] = None,
) -> Dict[str, gr.Component]:
    """Returns Gradio components for OpenAI model configuration."""
    config = {
        "temperature": gr.Slider(
            minimum=0.0, maximum=2.0, value=0.7, step=0.1, label="Temperature"
        ),
        "max_tokens": gr.Slider(
            minimum=50, maximum=64000, value=1000, step=50, label="Max Tokens"
        ),
    }
    if model_name.startswith(("o")):
        config["reasoning_effort"] = gr.Dropdown(
            choices=REASONING_EFFORT, value="medium", label="Reasoning Effort"
        )
    return config


def get_openai_llm(model_name: str, config: Optional[Dict] = None) -> ChatOpenAI:
    if config is None:
        config = {}
    return ChatOpenAI(model=model_name, **config)
