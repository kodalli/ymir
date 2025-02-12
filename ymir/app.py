import json
import gradio as gr
from ymir.llm import (
    get_llm,
    ALL_CHAT_MODELS,
    get_openai_config_components,
    get_google_config_components,
    get_deepseek_config_components,
    get_ollama_config_components,
)
from ymir.rlhf import RLHFDatasetBuilder

# Create global RLHF dataset builder
rlhf_builder = RLHFDatasetBuilder()


##############################################
# Chat Arena Function (for RLHF)
##############################################
def chat_arena(system_prompt: str, user_prompt: str, llm1_name: str, llm2_name: str):
    """
    Send the same system and user prompt to two different LLMs.
    """
    return rlhf_builder.chat_arena(system_prompt, user_prompt, llm1_name, llm2_name)


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

    conversation.append({"role": "user", "content": user_message})

    # Convert our conversation (list of dicts) into LangChain message objects.
    messages = []
    for msg in conversation:
        messages.append((msg["role"], msg["content"]))

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

model_options = ALL_CHAT_MODELS


def create_model_config_components(model_name: str):
    """Dynamically create configuration components based on the selected model."""
    if model_name.startswith(("gpt-", "o1-", "o3-")):
        return get_openai_config_components(model_name)
    elif model_name.startswith("gemini-"):
        return get_google_config_components()
    elif model_name.startswith("deepseek-"):
        return get_deepseek_config_components()
    else:  # Ollama models
        return get_ollama_config_components()


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
                system_input = gr.Textbox(
                    label="System Prompt",
                    placeholder="Enter system prompt here...",
                    lines=4,
                )
                user_input = gr.Textbox(
                    label="User Prompt",
                    placeholder="Enter user prompt here...",
                    lines=4,
                )
            with gr.Row():
                llm1_dropdown = gr.Dropdown(
                    choices=model_options, label="LLM 1", value="gpt-3.5-turbo"
                )
                llm2_dropdown = gr.Dropdown(
                    choices=model_options, label="LLM 2", value="claude"
                )
            with gr.Row():
                llm1_config_box = gr.Box(visible=True)
                llm2_config_box = gr.Box(visible=True)
            get_response_btn = gr.Button("Get Responses")
            with gr.Row():
                response1_output = gr.Textbox(label="LLM 1 Output", lines=10)
                response2_output = gr.Textbox(label="LLM 2 Output", lines=10)

            # Update configurations when models are selected
            def update_config_box(model_name, box_number):
                components = create_model_config_components(model_name)
                if box_number == 1:
                    return {llm1_config_box: gr.Group(list(components.values()))}
                else:
                    return {llm2_config_box: gr.Group(list(components.values()))}

            llm1_dropdown.change(
                fn=lambda m: update_config_box(m, 1),
                inputs=[llm1_dropdown],
                outputs=[llm1_config_box],
            )

            llm2_dropdown.change(
                fn=lambda m: update_config_box(m, 2),
                inputs=[llm2_dropdown],
                outputs=[llm2_config_box],
            )

            # Updated to use rlhf_builder directly
            get_response_btn.click(
                fn=rlhf_builder.chat_arena,
                inputs=[
                    system_input,
                    user_input,
                    llm1_dropdown,
                    llm2_dropdown,
                    llm1_config_box,
                    llm2_config_box,
                ],
                outputs=[response1_output, response2_output],
            )

            gr.Markdown("### Rate and Save This Comparison")
            rating_radio = gr.Radio(
                choices=["LLM 1", "LLM 2", "Tie"],
                label="Preferred Response",
                value="LLM 1",
            )
            notes_input = gr.Textbox(
                label="Optional Notes", placeholder="Enter any comments...", lines=2
            )
            save_entry_btn = gr.Button("Save RLHF Entry")
            save_status = gr.Textbox(label="Status", interactive=False)

            save_entry_btn.click(
                fn=rlhf_builder.save_rlhf_entry,
                inputs=[
                    system_input,
                    user_input,
                    llm1_dropdown,
                    llm2_dropdown,
                    response1_output,
                    response2_output,
                    rating_radio,
                    notes_input,
                ],
                outputs=save_status,
            )

            download_rlhf_btn = gr.Button("Download RLHF Dataset")
            filename_input = gr.Textbox(
                label="Filename",
                placeholder="Enter filename (e.g. rlhf_dataset.json)",
                value="rlhf_dataset.json",
            )
            download_file = gr.File(label="RLHF Dataset File")
            download_rlhf_btn.click(
                fn=rlhf_builder.download_rlhf_dataset,
                inputs=[filename_input],
                outputs=download_file,
            )

        # ======================================
        # Tab 2: Conversation Dataset Creation
        # ======================================
        with gr.TabItem("Conversation Dataset Creation"):
            gr.Markdown("### Build a Conversation")
            system_conv = gr.Textbox(
                label="System Prompt",
                placeholder="Enter a system prompt (optional)...",
                lines=2,
            )
            start_conv_btn = gr.Button("Start Conversation")
            conv_display = gr.Textbox(
                label="Conversation Log", lines=10, interactive=False
            )
            # Use a State component to hold the conversation (a list of messages).
            conv_state = gr.State([])
            conv_model_dropdown = gr.Dropdown(
                choices=model_options, label="Conversation Model", value="gpt-4o-mini"
            )

            with gr.Row():
                user_conv_input = gr.Textbox(
                    label="Your Message", placeholder="Enter your message...", lines=2
                )
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
                outputs=[conv_state, conv_display],
            )

            # Append a new turn: add user message and generate assistant response.
            send_conv_btn.click(
                fn=conversation_send,
                inputs=[conv_state, user_conv_input, conv_model_dropdown],
                outputs=[conv_state, conv_display],
            )

            clear_conv_btn.click(
                fn=lambda: ([], ""), inputs=None, outputs=[conv_state, conv_display]
            )

            download_conv_btn.click(
                fn=download_conversation, inputs=conv_state, outputs=conv_download_file
            )

demo.launch()
