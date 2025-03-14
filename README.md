# Ymir: AI Dataset Generation and Management Tools

Ymir is a comprehensive toolkit for creating, managing, and processing AI datasets with a focus on language models.

## Features

- **RLHF Dataset Builder**
  - Side-by-side comparison of responses from different LLMs
  - Collection and export of human preferences for model training
  - Support for multiple LLM providers (OpenAI, Google, DeepSeek, Ollama)

- **Knowledge Triplet Extraction**
  - Extract subject-predicate-object triplets from text
  - Create knowledge graphs and structured data from unstructured text
  - Manual addition and editing of triplets

- **Batch Dataset Processing**
  - Process large datasets using CSV files
  - Customize prompts with template variables
  - Track batch processing status in real-time

- **Document Processor**
  - Extract and process text from PDF documents
  - Automatically detect table of contents and chapter structure
  - Split PDFs by chapters and generate structured datasets

- **Modern, Responsive UI**
  - Built with FastAPI, HTMX, and Hyperscript
  - Clean design with custom styling
  - Interactive, single-page application experience

## Development Setup

After cloning the repository:

1. Ensure you have Python 3.11 or higher installed on your system.

2. Install uv (recommended) for dependency management:
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

To start the Ymir tool:

```bash
./main.py --reload
```

or

```bash
python main.py --reload
```

This will start the FastAPI server on port 8008 with hot reloading enabled.

## Environment Variables

Create a `.env` file with your API keys:

```
OPENAI_API_KEY=your_openai_api_key
GOOGLE_API_KEY=your_google_api_key
DEEPSEEK_API_KEY=your_deepseek_api_key
```

For Ollama, ensure the Ollama service is running locally.

## Main Features and Usage

### RLHF Dataset Builder
- Generate responses from different LLMs for the same prompt
- Compare and rate responses to build a preference dataset
- Export ratings as JSONL for model fine-tuning

### Knowledge Triplet Extraction
- Extract structured knowledge from text using LLMs
- Manually add and edit subject-predicate-object triplets
- Export knowledge graphs for downstream applications

### Batch Dataset Builder
- Upload CSV files with data for batch processing
- Create custom prompt templates with variable substitution
- Process large datasets efficiently and download results

### Document Processor
- Upload PDF documents for processing
- Automatically detect and extract chapters
- Generate structured datasets from documents

## Tech Stack

- **Backend**: FastAPI, LangChain
- **Frontend**: HTMX, Hyperscript, TailwindCSS
- **LLM Integration**: OpenAI, Google, DeepSeek, Ollama
- **Template Engine**: Jinja2
- **Document Processing**: PyPDF

## License

Apache 2.0

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

## Usage

1. **Start the application** using the command in the "Running the Application" section.

2. **Navigate to the web interface** by opening `http://localhost:8008` in your browser.

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

## Additional Features

### Knowledge Triplet Extraction
- Extract structured knowledge from text using LLMs
- Manually add and edit subject-predicate-object triplets
- Export knowledge graphs for downstream applications

### Batch Dataset Processing
- Upload CSV files with data for batch processing
- Create custom prompt templates with variable substitution
- Process large datasets efficiently and download results

### Document Processor
- Upload PDF documents for processing
- Automatically detect and extract chapters
- Generate structured datasets from documents
