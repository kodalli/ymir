from langchain_openai import ChatOpenAI
from typing import Literal, Dict, Optional
import gradio as gr

OPENAI_CHAT_MODELS = Literal[
    "gpt-4o",
    "gpt-4o-mini",
    "o3-mini",
    "o3-mini-high",
    "o1-mini",
    "o1-preview",
]

REASONING_EFFORT = Literal["low", "medium", "high"]


def get_openai_config_components(model_name: str) -> Dict[str, gr.Component]:
    """Returns Gradio components for OpenAI model configuration."""
    components = {
        "temperature": gr.Slider(
            minimum=0.0, maximum=2.0, value=0.7, step=0.1, label="Temperature"
        ),
        "max_tokens": gr.Slider(
            minimum=50, maximum=64000, value=1000, step=50, label="Max Tokens"
        ),
    }

    # Add reasoning_effort only for o-series models
    if model_name.startswith(("o")):
        components["reasoning_effort"] = gr.Dropdown(
            choices=["low", "medium", "high"], value="medium", label="Reasoning Effort"
        )

    return components


def get_openai_llm(
    model_name: OPENAI_CHAT_MODELS, config: Optional[Dict] = None
) -> ChatOpenAI:
    if config is None:
        config = {}
    return ChatOpenAI(model=model_name, **config)
