import json
import os
from typing import List, Optional
import gradio as gr

# ------------------------------
# LLM Wrappers and Utilities
# ------------------------------

from langchain.schema import SystemMessage, HumanMessage, AIMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import ChatHuggingFace
from langchain_ollama import ChatOllama

##############################################
# Utility: Instantiate the appropriate LLM wrapper
##############################################
def get_llm(model_name: str):
    if model_name in ["gpt-4o", "o3-mini"]:
        return ChatOpenAI(model=model_name, temperature=0.7)
    elif model_name == "gemini":
        return ChatGoogleGenerativeAI(model=model_name, temperature=0.7)
    elif model_name.startswith("hf-"):
        repo_id = model_name[3:]  # remove the "hf-" prefix
        return ChatHuggingFace(model=repo_id, temperature= 0.7)
    elif model_name.startswith("ollama-"):
        repo_id = model_name[7:]  # remove the "ollama-" prefix
        return ChatOllama(model=repo_id, temperature=0.7)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

##############################################
# Chat Arena Function (for RLHF)
##############################################
def chat_arena(system_prompt: str, user_prompt: str, llm1_name: str, llm2_name: str):
    """
    Send the same system and user prompt to two different LLMs.
    """
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]
    llm1 = get_llm(llm1_name)
    llm2 = get_llm(llm2_name)
    response1 = llm1(messages).content
    response2 = llm2(messages).content
    return response1, response2

# ------------------------------
# RLHF Dataset Functions
# ------------------------------

# Global list to hold RLHF ranking entries.
rlhf_data = []

def save_rlhf_entry(system_prompt, user_prompt, llm1_name, llm2_name, response1, response2, rating, notes):
    """
    Save an entry containing the prompts, the two model responses, a rating, and any notes.
    """
    entry = {
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "llm1": llm1_name,
        "llm2": llm2_name,
        "response1": response1,
        "response2": response2,
        "rating": rating,
        "notes": notes
    }
    rlhf_data.append(entry)
    return "RLHF entry saved!"

def download_rlhf_dataset():
    """
    Write the RLHF dataset to a JSONL file and return its filename.
    """
    filename = "rlhf_dataset.jsonl"
    with open(filename, "w", encoding="utf-8") as f:
        for entry in rlhf_data:
            f.write(json.dumps(entry) + "\n")
    return filename

# ------------------------------
# Conversation Dataset Functions
# ------------------------------

def initialize_conversation(system_prompt: str):
    """
    Start a conversation using the provided system prompt.
    Returns the conversation (a list of message dicts) and a formatted text representation.
    """
    conversation = []
    text_display = ""
    if system_prompt.strip():
        conversation.append({"role": "system", "content": system_prompt})
        text_display += f"System: {system_prompt}\n"
    return conversation, text_display

def conversation_send(conversation, user_message, conv_model_name):
    """
    Append a user message, get the assistant response using the chosen model,
    and return the updated conversation list and formatted conversation text.
    """
    from langchain.schema import SystemMessage, HumanMessage, AIMessage

    conversation.append({"role": "user", "content": user_message})

    # Convert our conversation (list of dicts) into LangChain message objects.
    messages = []
    for msg in conversation:
        if msg["role"] == "system":
            messages.append(SystemMessage(content=msg["content"]))
        elif msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))

    llm = get_llm(conv_model_name)
    assistant_response = llm(messages).content
    conversation.append({"role": "assistant", "content": assistant_response})

    # Format the conversation for display.
    text_display = ""
    for msg in conversation:
        text_display += f"{msg['role'].capitalize()}: {msg['content']}\n"
    return conversation, text_display

def download_conversation(conversation):
    """
    Write the conversation (in OpenAI chat format) to a JSON file and return its filename.
    """
    filename = "conversation_dataset.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump({"messages": conversation}, f, indent=2)
    return filename

# ------------------------------
# Gradio Interface
# ------------------------------

# Define available model options.

model_options = [
    "gpt-4o",
    "o3-mini",
    "o3-mini-high",
    "gemini-flash-lite-2",
    "gemini-flash-pro-2",
    "ollama-deepseek-r1:1.5b",
    "ollama-deepseek-r1:7b",
    "ollama-deepseek-r1:8b",
    "ollama-deepseek-r1:14b",
    "ollama-deepseek-r1:32b",
]

with gr.Blocks() as demo:
    gr.Markdown("# Chat Arena & Dataset Builder")
    gr.Markdown(
        "This app serves two purposes:\n\n"
        "1. **RLHF Dataset Creation:** Compare two model outputs side by side and rate which one is better.\n\n"
        "2. **Conversation Dataset Builder:** Build a multi-turn conversation (in OpenAI chat format) using a chosen model."
    )

    with gr.Tabs():
        # ==============================
        # Tab 1: RLHF Dataset Creation
        # ==============================
        with gr.TabItem("RLHF Dataset Creation"):
            gr.Markdown("### Compare Model Responses and Rate")
            with gr.Row():
                system_input = gr.Textbox(label="System Prompt", placeholder="Enter system prompt here...", lines=4)
                user_input = gr.Textbox(label="User Prompt", placeholder="Enter user prompt here...", lines=4)
            with gr.Row():
                llm1_dropdown = gr.Dropdown(choices=model_options, label="LLM 1", value="gpt-3.5-turbo")
                llm2_dropdown = gr.Dropdown(choices=model_options, label="LLM 2", value="claude")
            get_response_btn = gr.Button("Get Responses")
            with gr.Row():
                response1_output = gr.Textbox(label="LLM 1 Output", lines=10)
                response2_output = gr.Textbox(label="LLM 2 Output", lines=10)

            # When you click the Get Responses button, call the chat_arena function.
            get_response_btn.click(
                fn=chat_arena,
                inputs=[system_input, user_input, llm1_dropdown, llm2_dropdown],
                outputs=[response1_output, response2_output]
            )

            gr.Markdown("### Rate and Save This Comparison")
            rating_radio = gr.Radio(
                choices=["LLM 1", "LLM 2", "Tie"],
                label="Preferred Response",
                value="LLM 1"
            )
            notes_input = gr.Textbox(label="Optional Notes", placeholder="Enter any comments...", lines=2)
            save_entry_btn = gr.Button("Save RLHF Entry")
            save_status = gr.Textbox(label="Status", interactive=False)

            save_entry_btn.click(
                fn=save_rlhf_entry,
                inputs=[
                    system_input, user_input, llm1_dropdown, llm2_dropdown,
                    response1_output, response2_output, rating_radio, notes_input
                ],
                outputs=save_status
            )

            download_rlhf_btn = gr.Button("Download RLHF Dataset")
            download_file = gr.File(label="RLHF Dataset File")
            download_rlhf_btn.click(
                fn=download_rlhf_dataset,
                inputs=[],
                outputs=download_file
            )

        # ======================================
        # Tab 2: Conversation Dataset Creation
        # ======================================
        with gr.TabItem("Conversation Dataset Creation"):
            gr.Markdown("### Build a Conversation")
            system_conv = gr.Textbox(label="System Prompt", placeholder="Enter a system prompt (optional)...", lines=2)
            start_conv_btn = gr.Button("Start Conversation")
            conv_display = gr.Textbox(label="Conversation Log", lines=10, interactive=False)
            # Use a State component to hold the conversation (a list of messages).
            conv_state = gr.State([])
            conv_model_dropdown = gr.Dropdown(choices=model_options, label="Conversation Model", value="gpt-3.5-turbo")

            with gr.Row():
                user_conv_input = gr.Textbox(label="Your Message", placeholder="Enter your message...", lines=2)
                send_conv_btn = gr.Button("Send")

            with gr.Row():
                clear_conv_btn = gr.Button("Clear Conversation")
                download_conv_btn = gr.Button("Download Conversation")
                conv_download_file = gr.File(label="Conversation Dataset File")

            # Initialize conversation state with the system prompt.
            def start_conversation(system_prompt):
                conv, disp = initialize_conversation(system_prompt)
                return conv, disp
            start_conv_btn.click(
                fn=start_conversation,
                inputs=system_conv,
                outputs=[conv_state, conv_display]
            )

            # Append a new turn: add user message and generate assistant response.
            send_conv_btn.click(
                fn=conversation_send,
                inputs=[conv_state, user_conv_input, conv_model_dropdown],
                outputs=[conv_state, conv_display]
            )

            clear_conv_btn.click(
                fn=lambda: ([], ""),
                inputs=None,
                outputs=[conv_state, conv_display]
            )

            download_conv_btn.click(
                fn=download_conversation,
                inputs=conv_state,
                outputs=conv_download_file
            )

demo.launch()

