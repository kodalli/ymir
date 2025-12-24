# Ymir

> *Named after Ymir, the primordial giant from Norse mythology whose body was used to create the world — and a nod to the Founder Ymir from Attack on Titan.*

Ymir is a tool for generating high-quality agentic training data. It simulates multi-turn conversations between AI agents and users, producing tool-calling trajectories that can be used to fine-tune language models for agentic tasks.

## Features

- **Scenario-Based Generation**: Define scenarios with custom tools/functions that agents can use
- **Simulated Actors**: AI-powered user simulation with configurable personas and goals
- **Tool-Calling Trajectories**: Generate realistic multi-turn conversations with tool invocations
- **Trajectory Annotation**: Review and quality-score generated trajectories
- **Multiple Export Formats**: Export to Hermes, APIGen, and other fine-tuning formats
- **Local LLM Support**: Works with Ollama for fully local generation

## Quick Start

1. Install dependencies:
   ```bash
   uv venv && source .venv/bin/activate
   uv pip install -e .
   ```

2. Start Ollama with your preferred model:
   ```bash
   ollama run qwen3:4b
   ```

3. Run Ymir:
   ```bash
   python main.py
   ```

4. Open `http://localhost:8008` and use the 4-step wizard to generate trajectories.

## How It Works

1. **Select a Scenario**: Choose a domain (e.g., medical scheduling) with predefined tools
2. **Configure Tools**: Enable/disable specific tools the agent can use
3. **Define the Actor**: Set up situation details, background, and goals for the simulated user
4. **Generate**: Watch the agent and user interact, with the agent calling tools to accomplish tasks

## Example Trajectory

```
User: "Hi, I'd like to schedule an appointment"
Agent: [calls search_patient(first_name="Mark", last_name="Nolan")]
Tool Result: {patient_id: "12345", ...}
Agent: "I found your record. What type of appointment do you need?"
User: "I've been having leg pain for about a week"
Agent: [calls get_available_slots(date="2024-01-15")]
...
```

## Tech Stack

- **Backend**: FastAPI, Python 3.11+
- **Frontend**: HTMX, Hyperscript, TailwindCSS
- **LLM**: Ollama (local)
- **Storage**: JSON file-based

## License

Apache 2.0
