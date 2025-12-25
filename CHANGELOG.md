# CHANGELOG


## v0.0.0-alpha.13 (2025-12-25)

### Bug Fixes

- **docs**: Update README instructions for running Ymir and modify sidebar item in index.html
  ([`1002701`](https://github.com/kodalli/ymir/commit/1002701b31e273dda95921ff22c131aceccc4116))

- Changed the command to run Ymir from `python main.py` to `uv run python -m ymir.app` for better
  compatibility. - Updated the sidebar item label from "Annotate" to "Data" and changed the
  associated icon for clarity.

- **generation**: Fix wizard scenario selection and tool card layout
  ([`50e2937`](https://github.com/kodalli/ymir/commit/50e2937602bd19887e52bc91c44cff73e1dfe5d1))

- Add missing templates import causing backend errors - Remove duplicate hidden input causing DOM
  selector issues - Add CSS for dynamic scenario card selection state - Fix tool parameter layout to
  prevent text overlap

- **generation**: Improve wizard UX and fix validation bugs
  ([`70273b0`](https://github.com/kodalli/ymir/commit/70273b00dbbb90e9a82b14465726f7eb0e3be07f))

- Fix simulated actor validation by syncing form values before validation - Make generate step
  layout more compact with tighter spacing - Stack actor form fields vertically for better
  readability - Fix tool counter showing wrong count (hyperscript array handling)

- **generation**: Persist wizard state and improve tool card layout
  ([`1a3e716`](https://github.com/kodalli/ymir/commit/1a3e7164671d55909774be1869ea7319328a7166))

- Upgrade HTMX to 2.0.4 for proper script execution in swapped content - Use window namespace for
  wizard state to avoid re-declaration errors - Add localStorage persistence for wizard progress
  across navigation - Redesign tool cards with toggle on right and single-column list - Fix stepper
  click navigation to call goToStep directly - Add missing templates import in conversion.py

### Features

- **annotation**: Improve UX with smooth animations and voice-friendly demos
  ([`1bb0992`](https://github.com/kodalli/ymir/commit/1bb0992468968b74a0eeb1cc591919cdbfd0039d))

- Add CSS grid-based expand/collapse animation for table rows - Make demo conversation messages
  TTS-friendly (remove markdown formatting) - Add inline notes editing with toggle button - Remove
  fixed heights and scrolling from detail panel (page scrolls instead) - Remove table horizontal
  scroll constraint

- **annotation**: Redesign as data management table with filtering
  ([`f49e28c`](https://github.com/kodalli/ymir/commit/f49e28cbb1857dd5fc4b3258f02086f9be9819d8))

Replace single-trajectory view with paginated table supporting: - Status, scenario, and quality
  score range filtering - Issue type filtering (hallucination, goal drift, tool skip, suspicious) -
  Sortable columns with preserved filter state - Bulk approve/reject actions - Expandable row
  details with quick review - Agentic quality detection (hallucination, goal drift, tool skipping,
  suspicious patterns)

- **data**: Add healthcare front desk demo trajectories
  ([`fd4947b`](https://github.com/kodalli/ymir/commit/fd4947b65527fa026421a49369e42c0f815ef9ab))

Replace generic demo data with healthcare-specific scenarios: - Appointment scheduling with provider
  availability check - Insurance verification with copay lookup - Billing inquiry (includes
  hallucination example for QA)

Demo data seeds automatically on empty database with realistic multi-turn conversations including
  tool calls and results.


## v0.0.0-alpha.12 (2025-12-24)

### Bug Fixes

- **api**: Initialize database on app startup
  ([`30f3b1f`](https://github.com/kodalli/ymir/commit/30f3b1f8648e2a097aa99b824adc3531b7afebec))

Move db.initialize() to lifespan handler instead of calling it in each API endpoint. Database is now
  automatically initialized when the app starts and closed on shutdown.

- **dev**: Add websockets dep and fix route trailing slashes
  ([`dac06d4`](https://github.com/kodalli/ymir/commit/dac06d4456961dcd60227dd2d84fdee708491b70))

Add websockets library for WebSocket support in uvicorn. Fix HTMX links to include trailing slashes,
  eliminating 307 redirects.

- **generation**: Remove duplicate inputs in wizard step 3
  ([`b815e71`](https://github.com/kodalli/ymir/commit/b815e7110bed9c93c6b2658f51a31fee4651d5f5))

Fixes duplicate content rendering in the Configure Actor step by: - Removing duplicate hidden inputs
  (already in parent form) - Fixing radio button names to match JS expectations (mode_toggle) -
  Adding proper name attrs to textareas for syncFormValues()

- **ui**: Add fixed positioning to toast to prevent layout interference
  ([`e07f538`](https://github.com/kodalli/ymir/commit/e07f538cce9c9f6642967ae42b3c3b51a3e9980d))

Toast element was part of body's flex flow, taking up horizontal space and preventing main content
  from extending to full width.

### Chores

- Update .gitignore and add TODO.md for project tracking
  ([`4417a2e`](https://github.com/kodalli/ymir/commit/4417a2e25b8d27ba91180f4f462659115a3452ad))

Add .claude to .gitignore to exclude specific files. Create TODO.md to outline completed phases,
  project structure, usage instructions, and next steps for the agentic dataset generation project.

- **deps**: Simplify dependencies for new architecture
  ([`0b4e364`](https://github.com/kodalli/ymir/commit/0b4e364b21b4d51582448f957007c687ffd044b4))

Remove LangChain dependencies and add native ollama package. Update project description to reflect
  agentic dataset generation focus.

- **deps**: Update package versions and add upload times
  ([`77e12af`](https://github.com/kodalli/ymir/commit/77e12af0cf57e4e3489a5410afc65ca5b6be9a3c))

Increment revision to 2 in uv.lock. Update package metadata for annotated-types and anyio to include
  upload times. Remove deprecated LangChain dependencies and add new dependencies for ollama and
  pydantic.

### Code Style

- **ui**: Reduce page title sizes and spacing for compact layout
  ([`09d7e20`](https://github.com/kodalli/ymir/commit/09d7e207fe8d706764a17df9c41a1989a028a509))

Makes better use of screen real estate by reducing title sizes from text-3xl/4xl to text-xl/2xl and
  margins from mb-8 to mb-4.

### Documentation

- Rewrite README for agentic training focus
  ([`93c9327`](https://github.com/kodalli/ymir/commit/93c9327337e6bda5acf0de9f7ff1d9f1b61e5577))

Refocus project description on agentic training data generation. Remove outdated features (RLHF,
  triplets, batch processing, PDF). Add Ymir mythology and Attack on Titan reference.

### Features

- Add new core modules for agentic dataset generation
  ([`cddfe00`](https://github.com/kodalli/ymir/commit/cddfe00900d298867ff211d29bb1d0ec87e5a1b8))

Add annotation, converters, core, functions, generators, and storage modules to support multi-turn
  tool-calling trajectory generation.

- **data**: Add SQLite storage with Dataset management
  ([`45cb23a`](https://github.com/kodalli/ymir/commit/45cb23af07add128262b84981469d174ebdd03a2))

Replace JSONL file storage with SQLite for better scalability to 100k+ sessions. Introduces Dataset
  entity for flexible grouping of sessions across multiple scenarios.

New components: - database.py: async SQLite with FTS5 full-text search - session_store.py:
  SQLite-backed trajectory storage - dataset_store.py: Dataset CRUD and session relationships -
  datasets.py API: REST endpoints for dataset management

Also adds HuggingFace export format with train/validation splits.

- **dev**: Add WebSocket auto-reload and UI polish
  ([`b1163d5`](https://github.com/kodalli/ymir/commit/b1163d5d4f6330083bc52f87f4ecba47215a30c1))

Add browser auto-refresh on server restart via WebSocket endpoint. Also includes layout
  improvements: full-width content areas, grid layouts for tools/scenarios, and refined sidebar
  styling.

- **generation**: Add 4-step wizard UI for trajectory generation
  ([`13b0eb6`](https://github.com/kodalli/ymir/commit/13b0eb69fd6f2627dec63e280e6ba4b8ebb57e31))

Redesign the generation interface as a guided wizard to make it clearer how to create multi-turn
  agent training datasets.

- Add step-by-step wizard: Scenario → Tools → Actor → Generate - Allow toggling individual tools
  on/off for each scenario - Add 5 preset personas for medical scheduling (Elderly Patient, etc.) -
  Add tool filtering support to TrajectoryGenerator - Include terminology explanations (Agent vs
  Actor vs Scenario) - Add wizard CSS styles (stepper, toggles, persona cards)

- **generation**: Add simulated actor with situation details field
  ([`06ada66`](https://github.com/kodalli/ymir/commit/06ada66aeb2b7b2582629bfc0d2db4f00c1bd291))

- Add dedicated stepper endpoint for proper step navigation - Replace persona cards with direct
  input fields for simulated actor - Add situation details field for structured actor facts (name,
  phone, etc.) - Add 'Load Example' button to prefill actor fields with sample data - Fix show/hide
  toggle using Tailwind hidden class - Update generators to use situation field for user simulation
  - Change default model to qwen3:4b

- **routes**: Add new API routes for trajectory generation
  ([`eb0211a`](https://github.com/kodalli/ymir/commit/eb0211aa5bc11e68ec7b36d9ed3326395f6785c7))

Add annotation, conversion, export, functions, and generation routes to expose the new agentic
  dataset generation features.

- **templates**: Add new UI templates for trajectory generation
  ([`f7ff40c`](https://github.com/kodalli/ymir/commit/f7ff40cf510c73099f683929619bdc58287a80f0))

Add annotation, components, conversion, export, functions, and generation templates for the new
  feature set.

- **ui**: Add full-page rendering for direct URL access
  ([`22e1e0b`](https://github.com/kodalli/ymir/commit/22e1e0b55d9add7f8866f11234919da57ce2870e))

Previously, accessing routes directly (e.g., /functions/) only returned the page fragment without
  the sidebar layout. This adds a render_page helper that detects HTMX requests vs direct browser
  access and returns either the fragment or a complete page with the sidebar navigation.

Also updates navigation links to use proper hrefs and hx-push-url for browser history support.

- **ui**: Add theming system with Survey Corps theme
  ([`441f045`](https://github.com/kodalli/ymir/commit/441f04551f4397a6aef35eb97d7346c361fffdd2))

Implement CSS variable-based theming with two themes: - Survey Corps (default): Navy/indigo with
  dark slate backgrounds - Classic Red: Original red theme as fallback

Add theme infrastructure: - _base.css: Semantic color tokens - survey-corps.css/default.css: Theme
  definitions - theme-manager.js: Theme switching with localStorage persistence - Theme toggle
  button in header

Migrate all templates to use semantic color classes (text-theme-*, bg-surface-*, border-theme) for
  consistent theming support.

### Refactoring

- Remove deprecated prompt, rlhf, and triplet modules
  ([`42514ad`](https://github.com/kodalli/ymir/commit/42514ade7cdc82bef902326a39bff7d0b0b041cf))

Remove legacy modules that are being replaced by the new agentic trajectory generation architecture.

- Reorganize project structure into 5 packages
  ([`201d521`](https://github.com/kodalli/ymir/commit/201d5217a32803309ff15da530d43562d7045145))

Consolidate 9 top-level folders into 5 logical packages: - pipeline/ (generators + llm + annotation)
  - data/ (storage + converters + runtime) - api/ (renamed from routes) - core/ (unchanged) -
  functions/ (unchanged)

Also removes TODO.md as no longer needed.

- Update app initialization and main UI for new architecture
  ([`b24b046`](https://github.com/kodalli/ymir/commit/b24b0462643812debadc9ea89c401cb78c0a7d46))

Update app.py with new description and simplified setup. Rewire routes/__init__.py to use new route
  modules. Replace index.html with new dashboard for trajectory generation.

- **llm**: Remove deprecated LLM providers
  ([`b1dea99`](https://github.com/kodalli/ymir/commit/b1dea9908297551da4ba6fb0ab13993d0313f428))

Remove OpenAI, Google, DeepSeek LangChain integrations and helper modules in preparation for new
  Ollama-only architecture focused on local models.

- **llm**: Rewrite Ollama LLM with native client
  ([`97a8a5c`](https://github.com/kodalli/ymir/commit/97a8a5cf45fed7ac6c376808930038141bf21bf8))

Replace LangChain-based implementation with native ollama package. Add OllamaLLM class with
  sync/async generation methods.

- **routes**: Remove deprecated route handlers
  ([`8b2f251`](https://github.com/kodalli/ymir/commit/8b2f251fd9ab95e70719fd4b34879393b1de7a8e))

Remove batch, document, datasets, triplet, rlhf, and shared route handlers as part of the
  application architecture overhaul.

- **templates**: Remove deprecated HTML templates
  ([`100b3a9`](https://github.com/kodalli/ymir/commit/100b3a9e3b65ee7502443ae2a5a595a2ae40fcbb))

Remove batch, document, datasets, triplet, and rlhf templates corresponding to the removed route
  handlers.

- **ui**: Redesign functions page with scenario-centric layout
  ([`4eb0cf5`](https://github.com/kodalli/ymir/commit/4eb0cf51453d8209649954fd7968e10585b73554))

Scenarios are now expandable accordion cards that reveal their functions in API-docs style when
  clicked. Makes the parent-child relationship between scenarios and functions clear.


## v0.0.0-alpha.11 (2025-03-18)


## v0.0.0-alpha.10 (2025-03-15)

### Chores

- Organized templates in separate folders
  ([`28f4025`](https://github.com/kodalli/ymir/commit/28f40258f1c55c40065f43527bce6ab6f758da95))

### Features

- Add token statistics calculation endpoint and UI integration for enhanced prompt management,
  including token usage display and recommendations based on user input
  ([`d9ee60b`](https://github.com/kodalli/ymir/commit/d9ee60bb742b18f9275be779578cccd140476641))

- Refactor application structure by modularizing routes.
  ([`8dd7cf2`](https://github.com/kodalli/ymir/commit/8dd7cf2a8a3db2fb076a011156d8968f362b497a))


## v0.0.0-alpha.9 (2025-03-14)

### Features

- Implement CSV file parsing endpoint with robust error handling and preview functionality,
  enhancing user experience in data uploads
  ([`d28076b`](https://github.com/kodalli/ymir/commit/d28076b8eea9f6d10d52534e48d061969b7908b3))

- Introduce prompt configuration management with YAML support, enabling users to save and load
  prompt settings through the UI, enhancing flexibility in prompt customization
  ([`0e3ffd4`](https://github.com/kodalli/ymir/commit/0e3ffd452c4f31823744536067120c75a7862fee))


## v0.0.0-alpha.8 (2025-03-14)


## v0.0.0-alpha.7 (2025-03-12)

### Code Style

- Update color palette to use custom 'ymir' colors across templates and stylesheets
  ([`0d55de4`](https://github.com/kodalli/ymir/commit/0d55de4632bebccf34ada058e76725e4236f232c))

### Features

- Add progress tracking for PDF processing and enhance logging for better user feedback, currently
  loading bar disabled wip
  ([`27a70c5`](https://github.com/kodalli/ymir/commit/27a70c51b7b9bd0ff7305f376d4d80c9ed5cc0f0))

- Enhance TOC detection process with improved logging, user instructions, and UI updates for better
  user experience
  ([`dfb2967`](https://github.com/kodalli/ymir/commit/dfb29675041402b293ceb7c51f6c3eb11563c046))

- Implement async lifespan context manager for app startup/shutdown and enhance PDF upload handling
  with improved logging and user feedback
  ([`ab7c402`](https://github.com/kodalli/ymir/commit/ab7c402ec5701098d70e86279f801f6cb861d7e2))


## v0.0.0-alpha.6 (2025-03-12)

### Refactoring

- Updated styling of several templates to have hover effect, and fixed some styling issues.
  ([`f57770c`](https://github.com/kodalli/ymir/commit/f57770c73538288fd0aab211efad8f486288f701))


## v0.0.0-alpha.5 (2025-03-12)


## v0.0.0-alpha.4 (2025-03-12)

### Chores

- Configure alpha prerelease for semantic-release
  ([`993a400`](https://github.com/kodalli/ymir/commit/993a400183ed0ce7620efc481021d4405a65d5bb))

- Added prerelease configuration in pyproject.toml - Modified release workflow to trigger on master
  branch push - Enabled alpha prerelease token for version management

### Features

- Add Batch Dataset Builder with OpenAI Batch Processing
  ([`49dfdef`](https://github.com/kodalli/ymir/commit/49dfdeffe88f098947f34c3cbb60b0812252216b))

- Implemented new `/batch` endpoint for bulk dataset generation - Added HTML templates for batch
  processing UI with CSV upload and prompt configuration - Created batch processing functionality
  using OpenAI Batch API - Supported dynamic prompt template generation with CSV column placeholders
  - Added file upload, preview, and download capabilities for batch results - Integrated client-side
  JavaScript for interactive batch processing status tracking

- Add Document Processor with PDF chapter extraction and processing
  ([`01c5be1`](https://github.com/kodalli/ymir/commit/01c5be13b842143a3cd8d2d6c49b08ce10d473b4))

- Implemented new `/document` endpoint for PDF document processing - Added HTML templates for
  document upload, table of contents detection, and processing - Created utility functions for PDF
  chapter extraction and text processing - Supported PDF splitting, text extraction, and CSV dataset
  generation - Integrated drag-and-drop file upload with client-side validation - Added reusable
  button and upload button components

- Add PDF chapter extraction utility
  ([`d172ce6`](https://github.com/kodalli/ymir/commit/d172ce66c56551b4c1b361a068a133b1b8665f24))

- Implemented `split_pdf_by_chapters` function in `ymir/prompt/pdf.py` - Added advanced PDF parsing
  with automatic chapter start detection - Supported parallel processing of PDF chapter extraction -
  Integrated PDF utility into project's prompt module - Added `pypdf[full]` dependency for enhanced
  PDF processing capabilities

- Add support for 'o' models with system prompt handling
  ([`adef3f7`](https://github.com/kodalli/ymir/commit/adef3f7b536b59ce20886da5b1e3b9c83c4d824c))

- Updated OpenAI batch processing to handle 'o' models without system messages - Modified app.py to
  include system instructions in user prompt for 'o' models - Enhanced batch.html template with
  dynamic system prompt handling - Added client-side JavaScript to disable system prompt for 'o'
  models - Implemented form validation to prevent empty system prompts for non-'o' models

- Add text-to-triplets extraction with multi-provider LLM support
  ([`13d1117`](https://github.com/kodalli/ymir/commit/13d1117828ad7999360f1e3aa8222cf06f02308f))

- Implemented new `/extract_triplets` endpoint for knowledge graph extraction - Added support for
  multiple LLM providers (OpenAI, Anthropic, HuggingFace, local models) - Created dynamic triplet
  extraction using LangChain with configurable entity types - Updated templates and JavaScript to
  support new triplet extraction tool - Enhanced text processing with improved chunking and entity
  relationship detection

- Enhance LLM provider and model selection with dynamic updates
  ([`5d56e1d`](https://github.com/kodalli/ymir/commit/5d56e1da22c6f8cdf1c1343f36cedf1a860bc1ac))

- Updated app.py to return HTML options for model dropdown - Modified RLHF dataset builder methods
  for more flexible dataset handling - Added client-side JavaScript to handle provider and model
  changes dynamically - Improved UI interactions with model loading and selection events - Updated
  HTML templates to support HTMX-driven model updates

- Implement FastAPI-based RLHF Dataset Builder with web interface
  ([`f3038c8`](https://github.com/kodalli/ymir/commit/f3038c8c33164e4caa66226e298b96f3a9ead139))

- Added FastAPI backend with routes for LLM interaction and dataset management - Created HTML
  templates for RLHF dataset generation and comparison - Implemented client-side JavaScript for
  interactive UI - Added static CSS and JS files for enhanced

### Refactoring

- Consolidate FastAPI application and update project structure
  ([`6133aeb`](https://github.com/kodalli/ymir/commit/6133aeb9308a797d5861446ca8e0012c74822c46))

- Merged fastapi_app.py into app.py - Updated main.py to use new app module path - Simplified
  project configuration in pyproject.toml - Updated release workflow to use uv and semantic-release
  - Removed RLHF-specific references in project description

- Improve button component layout and icon positioning
  ([`4f9d297`](https://github.com/kodalli/ymir/commit/4f9d29778c469f71964e6bc5ceab20c7e19aa9fe))

- Restructured button and upload button templates for better icon and text alignment - Added
  flexible left and right icon positioning with fixed-width containers - Centered button text
  dynamically using flex layout - Improved loading indicator and icon rendering with consistent
  spacing - Enhanced visual balance and responsiveness of button components

- Separate RLHF content and improve page loading strategy
  ([`fd2b2a3`](https://github.com/kodalli/ymir/commit/fd2b2a33856ef4d4a4d858bbac5c3c5625ddb9c0))

- Created new `rlhf_content.html` template to separate RLHF page content - Updated `app.py` to
  render main layout without initial content - Modified `index.html` to load RLHF content via HTMX
  on page load - Simplified initial page rendering and content loading - Added hyperscript for
  dynamic navigation item active state

- Update LangChain imports and enhance triplet extraction
  ([`7a90f53`](https://github.com/kodalli/ymir/commit/7a90f531d6e12879b28cc08420aa0ee14120177a))

- Migrated LangChain imports to use core modules - Removed deprecated OpenAI model from supported
  models - Added automatic entity type detection for triplet extraction - Simplified LLM model
  initialization and invocation - Updated text-to-triplets extraction with more flexible
  configuration


## v0.0.0-alpha.2 (2025-02-13)

### Features

- Add batch processing, prompt creation, and RLHF dataset utilities
  ([`c08e0f9`](https://github.com/kodalli/ymir/commit/c08e0f9f37b9322e9d2e2aedc1205a79cb8d5dd7))

- Introduced OpenAIBatchProcessor for efficient batch processing of OpenAI API requests - Added
  prompt creation utilities for generating OpenAI and ShareGPT formatted datasets - Enhanced RLHF
  dataset builder with optional file loading - Updated dependencies in pyproject.toml - Added new
  modules for LLM invocation and prompt management


## v0.0.0-alpha.1 (2025-02-13)

### Bug Fixes

- Change to pep621
  ([`5ae988a`](https://github.com/kodalli/ymir/commit/5ae988a13572a36d4031cc2b1614b58a75f5fad1))

### Continuous Integration

- Refactor release workflow for improved semantic versioning and release management
  ([`7f10944`](https://github.com/kodalli/ymir/commit/7f1094428fc8490a7868d454685064f86626553a))

- Update semantic-release commands to use poetry - Enhance version tag generation and GitHub release
  creation - Improve release notes generation with proper GitHub Actions escaping - Add conditional
  checks for version and release steps

- Update release workflow to trigger on version tags and master branch
  ([`4753935`](https://github.com/kodalli/ymir/commit/47539358e34b0b7315635707ab769ef0f87b6094))

- Add push event triggers for version tags and master branch - Configure GitHub Actions permissions
  for release workflow

### Features

- Enhance RLHF dataset builder with conversation tracking and UI improvements
  ([`ccf85ea`](https://github.com/kodalli/ymir/commit/ccf85ea2396fa5b664be83780b4dca71c8b692b4))

- Add conversation tracking and chosen/rejected response fields in RLHFDatasetBuilder - Implement
  RLHF dataset view with Gradio Dataframe component - Add markdown conversion for reasoning content
  in dataset display - Create rating submission functionality with optional notes - Remove unused
  chat_arena method from RLHFDatasetBuilder

- Initialize project structure with core components for Ymir synthetic data generation tool
  ([`72f7633`](https://github.com/kodalli/ymir/commit/72f7633513d467732adf57784b2a16cdd72d3cd0))

- Add project configuration files (pyproject.toml, .pre-commit-config.yaml) - Create initial LLM
  module with support for OpenAI, Google, DeepSeek, and Ollama models - Implement RLHF dataset
  builder and Gradio-based UI for model comparison - Add development setup instructions and GitHub
  release workflow - Include Apache 2.0 license

### Refactoring

- Simplify Ymir app and LLM module structure
  ([`72f9d94`](https://github.com/kodalli/ymir/commit/72f9d94cecd3b5cde3ef5d5b30656b3e06068854))

- Streamline app.py with a more focused, side-by-side LLM comparison interface - Consolidate LLM
  configuration and retrieval logic in get_llm.py - Update type hints and configuration methods for
  LLM modules - Improve dynamic LLM provider and model selection - Add markdown conversion for
  reasoning content
