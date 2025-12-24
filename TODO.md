# Ymir - Agentic Dataset Generator

## Completed

- [x] Phase 1: Clean slate - removed old RLHF, triplet, batch, document processing code
- [x] Phase 2: Core data models (Trajectory, Message, ToolCall, etc.)
- [x] Phase 3: Ollama LLM provider (local generation only)
- [x] Phase 4: Function registry with scheduling templates
- [x] Phase 5: Trajectory generator (multi-turn tool-calling)
- [x] Phase 6: Dataset converters (APIGen-MT, Hermes FC)
- [x] Phase 7: Annotation system (quality scoring, review queue)
- [x] Phase 8: Storage layer (file-based JSONL)
- [x] Phase 9: Routes and HTMX templates
- [x] Phase 10: Updated app.py and dependencies

## Project Structure

```
ymir/
├── core/           # Data models (Trajectory, Message, ToolCall)
├── generators/     # Trajectory generation
├── converters/     # Dataset import/export
├── annotation/     # Quality scoring and review
├── functions/      # Function/tool definitions
├── storage/        # Persistence layer
├── routes/         # API endpoints
├── templates/      # HTMX UI
└── llm/            # Ollama provider
```

## Usage

1. Start the server: `python -m ymir.app`
2. Open http://localhost:8008
3. Generate trajectories or import existing datasets
4. Review and annotate trajectories
5. Export approved data for training

## Recent Updates

- [x] **Theming System** - Survey Corps (Attack on Titan) theme with centralized color system
  - CSS variables as single source of truth for colors
  - Theme toggle button in header (palette icon)
  - Two themes: "survey-corps" (default) and "default" (red)
  - All templates migrated to use semantic color classes
  - localStorage persistence for theme preference
- [x] **Generation Wizard UI** - Complete 4-step wizard for creating agent training data
  - Step 1: Select Scenario (card-based selection)
  - Step 2: Configure Tools (toggle individual tools on/off)
  - Step 3: Configure Actor (single query or simulated with structured details)
  - Step 4: Generate (model config + execute)
- [x] **Simulated Actor** - Configure actors with structured situation details
  - Situation details field for patient/customer data
  - Background and goal fields
  - "Load Example" button with sample data
- [x] **Tool Filtering** - Generator now accepts `enabled_tools` to filter available tools

## Next Steps

- [ ] Test wizard with actual Ollama models
- [ ] Add batch generation UI
- [ ] Improve quality scoring heuristics
- [ ] Add more built-in scenario templates (beyond medical scheduling)
- [ ] Add more persona presets for other scenario categories
- [ ] Support for editing trajectories in annotation UI
