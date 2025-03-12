# CHANGELOG


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

### Bug Fixes

- Change to pep621
  ([`5ae988a`](https://github.com/kodalli/ymir/commit/5ae988a13572a36d4031cc2b1614b58a75f5fad1))

### Features

- Add batch processing, prompt creation, and RLHF dataset utilities
  ([`c08e0f9`](https://github.com/kodalli/ymir/commit/c08e0f9f37b9322e9d2e2aedc1205a79cb8d5dd7))

- Introduced OpenAIBatchProcessor for efficient batch processing of OpenAI API requests - Added
  prompt creation utilities for generating OpenAI and ShareGPT formatted datasets - Enhanced RLHF
  dataset builder with optional file loading - Updated dependencies in pyproject.toml - Added new
  modules for LLM invocation and prompt management


## v0.0.0-alpha.1 (2025-02-13)

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
