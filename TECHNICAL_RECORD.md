# Qpyt-UI: Technical Record & Development Log

This document tracks architectural decisions, implemented features, and the roadmap for the **Qpyt-UI** project.

## TECHNICAL RECORD - Qpyt-UI
**Current Version**: V0.7 (Lightning Generation)
**Goal**: Advanced imaging bricks and speed optimization.
- **Architecture**: Python-driven modular framework for generative AI interfaces.
- **Engines**: 
    - **Imaging**: Diffusers (SDXL, FLUX, SD3.5).
    - **Vision**: Florence-2 (Prompt generation from images).
    - **Translation**: MarianMT (Helsinki-NLP) for FR -> EN prompt support.
    - **Cognitive**: Qwen2.5 (LLM for prompt enhancement).
- **VRAM Strategy**: Intelligent offloading (CPU/GPU) with automatic unloading of secondary models (Analyzer/Translator/LLM) during image generation.

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

## 3. Memory & Performance Strategy
- **SDXL/Flux**: Use of `enable_model_cpu_offload()` to stay under 12GB VRAM.
- **Lazy Loading**: Secondary models (Florence, Translator) are only loaded on demand.
- **Automatic GC**: Explicit calls to `gc.collect()` and `cuda.empty_cache()` between major task shifts.

## 4. Roadmap (Future Actions)

### Advanced Imaging & Processing
- [x] **Tiled Upscaler**: High-resolution output using tiling to avoid memory crashes.
- [x] **Img2Img Brick**: Using reference images to guide generation.
- [ ] **Inpainting/Outpainting**: Targeted area editing and canvas expansion.
- [x] **Background Removal (REMBG)**: Foreground extraction (using `rembg[gpu]` for CUDA acceleration).
- [ ] **Vectorization**: Integrated SVG conversion module.
- [ ] **LoRA Manager**: Dynamic loading and blending of multiple LoRAs with weight control.

### Workflow & Speed
- [ ] **SD 3.5 Turbo**: Support for high-speed distilled versions of SD 3.5.
- [x] **Style Injection Bricks**: One-click components to append specific tokens/LoRAs to the prompt.
- [ ] **LoRA Dashboard**: Sidebar or brick for managing downloaded LoRAs.
- [ ] **Image Retouching**: AI filter layer for post-processing cleanup.
- [ ] **Hot Reloading UI**: Better frontend synchronization.

### Finalization
- [ ] **Preset System**: Library of pre-configured workflows (moved to final phase).

## 5. Version History
- **V0.1 - V0.2**: PoC and Cartridge system (Micro-Gradio).
- **V0.3**: Rebranding to Qpyt-UI & Bricks Architecture.
- **V0.4**: SDXL Integration & CUDA 12.8 support.
- **V0.5**: Image-to-Prompt, Translator, and Workflow Persistence.
- **V0.6**: LLM Prompt Enhancer (Qwen2.5 cognitive layer).

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
