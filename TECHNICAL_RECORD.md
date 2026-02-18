# Qpyt-UI: Technical Record & Development Log

This document tracks architectural decisions, implemented features, and the roadmap for the **Qpyt-UI** project.

## TECHNICAL RECORD - Qpyt-UI
**Current Version**: V0.9.9 (Flux 2 integration & Klein 4B)
**Goal**: Asynchronous task orchestration, serialized worker, and real-time job monitoring.
- **Architecture**: Python-driven modular framework for generative AI interfaces.
- **Engines**: 
    - **Imaging**: Diffusers (SDXL, FLUX, SD3.5).
    - **Vision**: Florence-2 (Prompt generation from images).
    - **Translation**: MarianMT (Helsinki-NLP) for FR -> EN prompt support.
    - **Cognitive**: Qwen2.5 (LLM for prompt enhancement).
- **VRAM Strategy**: Intelligent offloading (CPU/GPU) with NF4/T5-NF4 for SD 3.5. **Task Serialization**: New `QueueManager` worker ensures only one heavy GPU task runs at a time, preventing OOM crashes during concurrent requests.
- **Environment Constraint**: **USE `.venv`**. All commands, package checks, and executions must be performed within the project's virtual environment (`.venv`). Do not use system-level Python.

## 2. Key Technical Milestones

### Core Framework (qp-)
- **Modular Brick System**: Clean separation between UI (Web Components) and Logic (FastAPI/Python).
- **Workflow Persistence**: JSON-based saving/loading of brick layouts (`/workflows` directory).
- **Internationalization**: Full interface and log migration to English.

### AI Implementation
- **SDXL Stability**: Resolved "meta tensor" issues using model-level CPU offloading and precise component crystallization.
- **Image-to-Prompt**: Dedicated `analyzer.py` using Florence-2 with deep-level patches for `transformers` 4.45+ compatibility.
- **Native Translation**: Integrated CPU-based French-to-English translation to bridge language gaps in prompting.
- **Cognitive Reasoning**: Added `llm_prompter.py` leveraging small-footprint LLMs (Qwen2.5) for high-quality prompt synthesis on CPU.

### Advanced Editing (V0.8 Updates)
- **Robust Inpainting**: 
    - Fixed mask transmission issues by capturing state before DOM re-renders.
    - Removed arbitrary dimension overrides to support non-square aspect ratios.
- **True Outpainting**:
    - Implemented dedicated `QpOutpaint` brick with directional expansion (Top/Bottom/Left/Right).
    - **Auto-Masking**: Automatically generates masks where expanded areas are White (to be filled) and original content is Black (protected).

### Photo Editor V2 & Workflow (V0.9 Updates)
- **QpFilter Overhaul**:
    - Transformed `QpFilter` into a Fullscreen Overlay logic.
    - Added **Debounce** (400ms) for real-time slider preview without backend flooding.
    - **Circular Workflow**: Images flow Generator -> Filter -> Save/Source seamlessly.
- **Base64 Save Engine**:
    - Updated `core/save_service.py` to handle Data URIs directly, enabling "Save to Disk" from in-memory edits.
- **Workflow Management**:
    - Implemented `DELETE` endpoint for saved workflows (`/workflows/delete`).
    - Added UI controls (Trash icon) in the Workflow Drawer.
- **Final Output Lightbox**:
    - Fixed image pathing issues in `qp-dashboard.js`.
    - Added direct-click interaction on image previews to open the fullscreen view.

### LoRA Manager & Reliability (V0.9.5 Updates)
- **LoRA Manager Brick**:
    - Dynamic loading, unloading, and weight blending.
    - **Dual Controls**: Synchronized numerical input and slider for precise weight management.
- **Safety & Architecture Check**:
    - Implemented **Smart Detection** in `core/generator.py` to prevent "Size Mismatch" errors.
    - Proactively validates LoRA architecture (SD 1.5 vs SDXL/Flux) against the active pipeline.
    - Populates non-intrusive UI warnings when incompatible LoRAs are skipped.
- **Session History Log (HTML)**:
    - Implemented `api/history_log.py` to generate a local HTML report of the current session.
    - Stores metadata, prompts, and crucially, the **list of applied LoRAs** for reproducibility.

### SD 3.5 Optimization & Stability (V0.9.6 Updates)
- **NF4 Transformer Quantization**: 
    - Integrated `BitsAndBytesConfig` for SD 3.5 Turbo to reduce VRAM from 20GB+ to ~12-14GB.
    - Leveraged 4-bit T5 encoder (`diffusers/t5-nf4`) as standard for SD 3.5 code paths.
- **Precision Migration (BFloat16)**:
    - Migrated Flux and SD 3.5 pipelines to `torch.bfloat16`. 
    - Prevents NaN/OOM issues in Transformer blocks compared to standard FP16.
- **Iterative GGUF/Safetensors Fallback**:
    - Implemented a `while` loop fallback system in `core/generator.py` for SD 3.5.
    - **Hybrid Loading for Flux**: Replaced iterative fallback with a robust component assembly (Transformer from file + T5/VAE from Hub) to ensure correct GPU offloading and native speed (~22s vs 1700s).
- **Unit Testing Infrastructure**:
    - Created `tests/test_loader.py` to verify model loading states and API stability.
    - Integrated environment-aware regression testing (using `.venv`).

### Task Queue System & Concurrent Stability (V0.9.7 Updates)
- **Serialized Job Queue**:
    - Implemented `api/queue_manager.py` (Singleton) with an `asyncio.Queue` and a dedicated background worker.
    - Serializes all heavy generation (SDXL, Flux, Upscale) and processing (Rembg) tasks to prevent VRAM overflow.
- **Asynchronous API Refactor**:
    - Converted `/generate`, `/inpaint`, `/outpaint`, `/upscale`, and `/rembg` into "Fire and Forget" endpoints.
    - Endpoints now return a `task_id` immediately, allowing the UI to remain responsive and submit multiple jobs.
- **Frontend Job Monitor**:
    - Created `qp-queue-monitor.js` (System Brick, Red) to track pending, running, and completed tasks.
    - Integrated **Auto-Polling** and **Cancel** functionality with real-time progress bar support.
- **Architecture Integration**:
    - Refactored `QpRender` base class to support `submitAndPollTask` lifecycle, maintaining standard generation flow while leveraging the backend queue.
- **Queue-Optimized Preset**:
    - Added `SDXL_Queue_Optimized.json` including the new Job Monitor and pre-configured for batch generation.

### Flux Native Performance (V0.9.9 Updates)
- **FLUX.2 Klein Supported Architecture**:
    - Identified and implemented support for the 4B Klein architecture (5 double blocks, 20 single blocks).
    - Added specialized `Flux2Pipeline` and `Flux2Transformer2DModel` integration.
- **Manual Key Mapping (Alpha Checkpoints)**:
    - Resolved critical weight loading issues for SFT/Alpha checkpoints by implementing a manual remapping for `x_embedder`, `context_embedder`, and `time_guidance_embed`.
    - Implemented zero-initialization for missing guidance weights to eliminate random latent noise.
- **Auto-Guidance Control**:
    - Added automated detection of distilled models to enforce `guidance_scale=1.0`, ensuring optimal image quality without manual user intervention.
- **Qwen2-based Text Encoding**:
    - Patched the text encoding logic to correctly leverage `Qwen2` architectures (mistakenly tagged as Mistral3 in some repos), ensuring accurate prompt-to-image alignment.

## 3. Memory & Performance Strategy
- **SDXL/Flux**: Use of `enable_model_cpu_offload()` to stay under 12GB VRAM.
- **Lazy Loading**: Secondary models (Florence, Translator) are only loaded on demand.
- **Automatic GC**: Explicit calls to `gc.collect()` and `cuda.empty_cache()` between major task shifts.

## 4. Roadmap (Future Actions)

### Advanced Imaging & Processing
- [x] **Tiled Upscaler**: High-resolution output using tiling to avoid memory crashes.
- [x] **Img2Img Brick**: Using reference images to guide generation.
- [x] **Inpainting/Outpainting**: Targeted area editing and canvas expansion.
- [x] **Background Removal (REMBG)**: Foreground extraction (using `rembg[gpu]` for CUDA acceleration).
- [x] **Vectorization**: Integrated SVG conversion module.
- [x] **Filters & Editing**: Contrast, Brightness, Saturation, Blur/Sharpen post-processing.
- [x] **LoRA Manager**: Dynamic loading and blending of multiple LoRAs with weight control.

### Workflow & Speed
- [x] **SD 3.5 Turbo**: Support for high-speed distilled versions of SD 3.5 with NF4 optimizations.
- [x] **Style Injection Bricks**: One-click components to append specific tokens/LoRAs to the prompt.
- [x] **LoRA Dashboard**: Integrated LoRA selection with architectural safety checks.
- [x] **Workflow Management**: Save/Load/Delete full brick layouts.
- [ ] **Hot Reloading UI**: Better frontend synchronization.

### Finalization
- [x] **Preset System**: Library of pre-configured workflows.
- [x] **Task Queue System**: Asynchronous worker and job monitoring for VRAM stability.

## 5. Version History
- **V0.1 - V0.2**: PoC and Cartridge system (Micro-Gradio).
- **V0.3**: Rebranding to Qpyt-UI & Bricks Architecture.
- **V0.4**: SDXL Integration & CUDA 12.8 support.
- **V0.5**: Image-to-Prompt, Translator, and Workflow Persistence.
- **V0.6**: LLM Prompt Enhancer (Qwen2.5 cognitive layer).
- **V0.7**: Lightning Generation.
- **V0.8**: Inpainting & Outpainting overhaul.
- **V0.9**: Photo Editor V2, Real-time Filters, Base64 Save System, Workflow Deletion.
- **V0.9.5**: LoRA Manager with Architecture Check, HTML Session History Logs, UI Safety Warnings.
- **V0.9.6**: SD 3.5 Turbo NF4 Optimization, BFloat16 precision, Iterative GGUF/Safetensors Fallback.
- **V0.9.7**: Task Queue System, Serialized Worker, Job Monitor Brick, and Asynchronous API Refactor.
- **V0.9.8**: Minor performance tweaks and Flux alignment.
- **V0.9.9**: FLUX.2 Klein Support, manual weight mapping, and auto-guidance logic.

---

## UI Design System & Bricks Coloring

To maintain visual clarity, the following color code must be strictly followed for all bricks:

| Category | Type | Color | Description | Examples |
| :--- | :--- | :--- | :--- | :--- |
| **Inputs / Aids** | `input` | **Purple** (`#a855f7`) | Entry points, prompts, and tools that help create the input for generators. | `qp-prompt`, `qp-styles`, `qp-image-input`, `qp-img2prompt`, `qp-translator` |
| **Generators** | `generator` | **Green** (`#10b981`) | Active tools that transform or generate content from an input. | `qp-render-*`, `qp-upscaler`, `qp-rembg`, `qp-img2img`, `qp-vectorize` |
| **Settings** | `setting` | **Red** (`#ef4444`) | Configuration bricks to adjust global or specific parameters. | `qp-settings` |
| **Outputs** | `output` | **Orange** (`#f59e0b`) | Final results, visualizers, or storage actions. | `qp-image-out`, `qp-save-to-disk` |

> [!IMPORTANT]
> A brick like `qp-vectorize` is a **Generator** because it transforms pixels into vectors, but it produces a result (SVG) that can be further processed or saved.
