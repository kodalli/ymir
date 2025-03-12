# Ymir: RLHF Dataset Builder (WIP)

Ymir is a tool for creating and managing RLHF (Reinforcement Learning from Human Feedback) datasets for language models.

## Features

- Side-by-side comparison of responses from different LLMs
- Support for multiple LLM providers (OpenAI, Google, DeepSeek, Ollama)
- Collection and export of human preferences
- Built with FastAPI and HTMX for a responsive, modern UI

## Development Setup

After cloning the repository:

1. Ensure you have Python 3.12 installed on your system.

2. Install uv:
   ```bash
   curl -sSf https://astral.sh/uv/install.sh | bash
   ```

3. Create a virtual environment and install dependencies:
   ```bash
   uv venv
   source .venv/bin/activate  # On Unix/macOS
   # or
   .venv\Scripts\activate  # On Windows
   uv pip install -e .
   ```

4. Set up pre-commit hooks:
   ```bash
   pre-commit install
   pre-commit install --hook-type post-checkout --hook-type post-merge
   ```

## Running the Application

To start the RLHF Dataset Builder:

```bash
./main.py --reload
```

or

```bash
python main.py --reload
```

This will start the FastAPI server with hot reloading enabled.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Main application interface |
| `/chat` | POST | Send a message to an LLM |
| `/update_provider` | POST | Change the LLM provider |
| `/update_model` | POST | Change the LLM model |
| `/rate` | POST | Submit a rating for two responses |
| `/rlhf_data` | GET | Get the current RLHF dataset |
| `/download_rlhf` | POST | Save and download the RLHF dataset |

## Tech Stack

- **Backend**: FastAPI
- **Frontend**: HTMX, Hyperscript, TailwindCSS
- **LLM Integration**: LangChain
- **Template Engine**: Jinja2

## License

Apache 2.0

## Usage

1. **Start the application** using the command in the "Running the Application" section.

2. **Navigate to the web interface** by opening `http://localhost:8000` in your browser.

3. **Configure LLM providers**:
   - Select providers and models from the dropdown menus for LLM 1 and LLM 2.
   - The application supports OpenAI, Google, DeepSeek, and Ollama models.

4. **Generate responses**:
   - Enter a prompt in the input area.
   - Click "Generate with LLM 1" or "Generate with LLM 2" to see responses.
   - You can generate responses from both models to compare them.

5. **Rate responses**:
   - After generating responses from both models, decide which one you prefer.
   - Add optional notes to explain your reasoning.
   - Click "Choose LLM 1" or "Choose LLM 2" to record your preference.

6. **View the dataset**:
   - Click "View RLHF Dataset" to see all the ratings you've collected.
   - The dataset includes the prompt, responses, and your ratings.

7. **Export the dataset**:
   - Click "Download RLHF Dataset" to save the data as a JSONL file.
   - This data can be used for fine-tuning or evaluating models.

## Environment Variables

The following environment variables can be set to configure API keys for different LLM providers:

- `OPENAI_API_KEY`: Your OpenAI API key
- `GOOGLE_API_KEY`: Your Google AI API key
- `DEEPSEEK_API_KEY`: Your DeepSeek API key

For Ollama, ensure the Ollama service is running locally.
