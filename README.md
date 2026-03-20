# Qpyt-UI V1.3.0 (Sequential Chaining & Flux2)

**Advanced Modular Interface for Generative AI**
*Powered by Diffusers, Florence-2, Qwen2.5, and Web Components.*

<img width="1899" height="872" alt="image" src="https://github.com/user-attachments/assets/e2c45ffb-9edf-4260-b5a5-6c078fca209a" />


## Overview
Qpyt-UI is a **Python-Driven Modular Framework** designed for high-performance interaction with Stable Diffusion (SDXL, FLUX, SD3.5) and auxiliary AI models. It uses a unique "Brick" system where every UI component is a standalone Web Component powered by a dedicated Python backend service.

## Key Features (V1.1)

### ✨ Visual & Auditory Experience [NEW]
- **Glassmorphism UI**: High-end translucent interface with vibrant gradients and premium typography.
- **AI Audio Cues**: Real-time sound feedback for generations, completions, and errors.
- **Unified Lightbox**: Fast, immersive fullscreen preview for all history and output bricks.

### 🎨 Generative Imaging
- **Multi-Engine Support**: Seamlessly switch between **SDXL**, **FLUX.1-Schnell**, **SD 3.5 Turbo**, and **FLUX.2-Klein**.
- **Lightning Generation**: Optimized NF4/T5-NF4 pipelines and specialized **Flux 2 Klein 4B** support for high-speed generation with reduced VRAM footprint.
- **Tiled Upscaler**: High-fidelity upscaling (up to 4K+) using tiled diffusion to minimize VRAM usage.

### 🛠️ Advanced Editing Tools
- **Deep Inpainting**: Context-aware masking and regeneration.
- **True Outpainting**: Infinite canvas expansion with directional sliders (Top/Bottom/Left/Right) and auto-masking.
- **Photo Editor V2**: Fullscreen overlay editor for post-processing:
    - Real-time Filters (Contrast, Brightness, Saturation, Blur).
    - Non-destructive workflow (Circular: Generator -> Editor -> Save).
- **Background Removal**: GPU-accelerated subject extraction using `rembg`.
- **Vectorization**: One-click raster-to-SVG conversion for graphic design workflows.

### 🧠 Cognitive & Utility Layers
- **Prompt Enhancement**: Integration of **Qwen2.5 (0.5B)** to expand simple terms into rich, descriptive prompts.
- **Translation**: Embedded offline translation (FR -> EN) using **MarianMT**, allowing native language prompting.
- **Image Analysis**: **Florence-2** integration for detailed image captioning and prompt reverse-engineering.

### 🧱 Workflow Management
- **Brick System**: Drag-and-drop modular interface.
- **LoRA Manager**: Dynamic loading and blending of multiple LoRAs with weight synchronization and **Architectural Safety Checks** (SD 1.5 vs SDXL/Flux).
- **Iterative Fallback**: Robust handling of incomplete GGUF/Safetensors models by automatically fetching missing components (VAE, CLIP, T5) from official repositories.
- **Task Queue System**: Asynchronous job orchestration through a serialized `QueueManager`, preventing VRAM overflow by ensuring only one heavy task runs at a time.
- **Robust Notifications**: Integrated email system with **Media Attachments** and real-time backend-to-UI error reporting.
- **Session History Log**: Automatic generation of HTML reports including full metadata and LoRA configurations.
- **Persistence**: Save, Load, and **Delete** complex node layouts as JSON workflows.
- **Sequential Auto-Chaining [NEW]**: Link bricks for automatic step-by-step execution (e.g. SDXL Generator ➔ Upscaler).
- **State Management**: Circular dependencies and history tracking.

## Architecture
- **Backend**: FastAPI + PyTorch/Diffusers.
- **Frontend**: Native Web Components (Vanilla JS + Shadow DOM) + Shoelace UI.
- **Communication**: REST API for heavy lifting, Custom Events for UI reactivity.

## 🧱 Component Library (Inventory)

### 📥 Inputs & Prompts
*These bricks define *what* you want to generate.*
*   **Source Image** (`qp-image-input`): Load an image for Img2Img or ControlNet.
*   **Prompt** (`qp-prompt`): The main text input for your creation.
*   **Styles Selector** (`qp-styles`): Apply predefined artistic styles (Cinematic, Anime, etc.).
*   **Translator (GPT/Qwen)** (`qp-translator`): Auto-translate your prompt to English.
*   **LLM Prompt Enhancer** (`qp-llm-prompter`): Use an LLM to expand a simple idea into a detailed prompt.
*   **Image-to-Prompt** (`qp-img2prompt`): Reverse-engineer a prompt from an image (using CLIP/BLIP).
*   **Prompt Helper** (`qp-prompt-helper`) **[NEW]**: Build complex prompts with categorized keywords and a local persistence database for your own terms.

### ⚡ Generators (AI Models)
*The engines that create the content.*
*   **SDXL Generator** (`qp-render-sdxl`): Stable Diffusion XL (High Quality).
*   **FLUX Generator** (`qp-render-flux`): Black Forest Labs FLUX.1 (Schnell/Dev).
*   **FLUX 2 Klein** (`qp-render-flux2`) **[NEW]**: Optimized Flux 2 architecture (4B) specifically for speed and quality on consumer hardware.
*   **SD3.5 Turbo** (`qp-render-sd35turbo`): Latest Stability AI model (Lightning fast).
*   **Music Generator** (`qp-music-gen`): AI Music generation (Facebook MusicGen).
*   **Img2Img Refiner** (`qp-img2img`): refine an existing image (denoising strength).
*   **Inpainting** (`qp-inpaint`): Fill masked areas of an image.
*   **Outpainting** (`qp-outpaint`): Extend the borders of an image.
*   **LoRA Manager** (`qp-lora-manager`): Load Lightweight fine-tuned models (Characters, Styles).

### 🛠️ Utilities & FX (Post-Processing & Control)
*Tools to analyze, modify, or guide the generation.*
*   **ControlNet** (`qp-controlnet`) **[NEW]**: Guide generation with Depth, Pose, or Canny edges.
*   **OpenPose Editor** (`qp-openpose-editor`) **[NEW]**: Draw custom skeletons for ControlNet.
*   **Depth Estimator** (`qp-depth-estimator`) **[NEW]**: Create depth maps from any image.
*   **Normal Map** (`qp-normal-map`): Generate 3D normal maps for textures/3D work.
*   **Tiled Upscaler** (`qp-upscaler`): Increase resolution with added detail.
*   **Background Removal** (`qp-rembg`): Remove background (transparent PNG).
*   **Vectorize (SVG)** (`qp-vectorize`): Convert bitmap images to SVG vectors.
*   **Photo Editing** (`qp-filter`): Adjust contrast, color, saturation, etc.

### ⚙️ System & Output
*Global settings and export options.*
*   **Global Settings** (`qp-settings`): Manage model paths, default sizes, device (CUDA).
*   **Final Output** (`qp-image-out`): Display the final result.
*   **Save to Disk** (`qp-save-to-disk`): Auto-save logic.
*   **Job Queue Monitor** (`qp-queue-monitor`): Watch background tasks.

## Getting Started

## 📦 Installation & Setup Guide

### 1. Prerequisites
*   **OS**: Windows 10/11 (Linux support experimental).
*   **Python**: Version 3.10 to 3.13.5 (Python 3.13.5 recommended for optimal performance).
*   **Visual Studio Build Tools**: Mandatory for Windows users (must include the "Desktop development with C++" workload). This is strictly required by Triton to compile kernels on the fly.
*   **GPU**: NVIDIA GeForce RTX 3060 or better (8GB+ VRAM) recommended for SDXL/Flux.
*   **Git**: Required for cloning the repository.

### 2. Download Project
```bash
git clone https://github.com/lecyberbill/Qpyt-UI.git
cd Qpyt-UI
```
## 🤖 Agentic Control (MCP)

Qpyt-UI now supports the **MCP (Model Context Protocol)**, allowing agents like Claude Desktop, Cursor, or Antigravity to programmatically control the application.

### 🚀 Launching the MCP Server
Use the provided script in the root directory:
```bash
./start_mcp.bat
```

### ⚙️ Configuration

#### Claude Desktop
Add the following to your `claude_desktop_config.json` file:
```json
{
  "mcpServers": {
    "qpyt-ui": {
      "command": "/path/to/Qpyt-UI/.venv/Scripts/python.exe",
      "args": ["api/mcp_server.py"],
      "cwd": "/path/to/Qpyt-UI",
      "env": {
        "PYTHONPATH": "/path/to/Qpyt-UI"
      }
    }
  }
}
```

#### LM Studio
1. Go to the **Settings** (Gear icon) -> **MCP** tab.
2. Click on **Edit mcp.json** and add:
```json
{
  "mcpServers": {
    "qpyt-ui": {
      "command": "/path/to/Qpyt-UI/.venv/Scripts/python.exe",
      "args": ["api/mcp_server.py"],
      "cwd": "/path/to/Qpyt-UI",
      "env": {
        "PYTHONPATH": "/path/to/Qpyt-UI"
      }
    }
  }
}
```

### 🛠️ Available MCP Tools
- `generate_image`: Generate an image with prompt, model, and dimensions.
- `list_available_models`: List locally available models (.safetensors).
- `analyze_image_tags`: Analyze an image to extract tags (Image-to-Prompt).
- `get_current_config`: Retrieve current application settings and paths.

---

### 3. Install Dependencies
It is highly recommended to use a virtual environment:
```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Configuration ⚙️
The project requires a configuration file to know where your models are stored.
1.  Locate `config.example.json` in the root folder.
2.  Copy it and rename it to `config.json` (or `config.local.json` if you prefer to keep it git-ignored).
3.  Edit `config.json` with a text editor:
    *   Set `MODELS_DIR` to your stable-diffusion checkpoints folder.
    *   Set `CONTROLNET_DIR`, `VAE_DIR`, etc. to appropriate paths.
    *   *Note: If the folders don't exist, the validation script will attempt to create them.*

### 5. Validate Installation ✅
Run the included verification script to ensure everything is set up correctly (make sure your **venv is activated**):
```bash
# Windows
.\venv\Scripts\activate
python verify_install.py

# Mac/Linux
source venv/bin/activate
python verify_install.py
```
*If you see any [FAIL] messages, correct the issues (missing folders, CUDA not found) before proceeding.*

### 6. Launch Application 🚀
Double-click `start.bat` (Windows) or run:
```bash
python api/main.py
```
Then open your browser at: **http://127.0.0.1:8000**

## Version History
*   **V1.3.0**: **Sequential Auto-Chaining**. Added global on/off toggle and visual link indicators for automated brick-to-brick execution (from left to right in the DOM).
*   **V1.2.0**: **MCP & LLM Assistant**. Integrated Model Context Protocol (MCP) for agentic control and added the multi-provider LLM Assistant with 17+ creative roles.
*   **V1.1.2**: **Native Clipboard Paste**. Integrated direct image pasting from clipboard into the `QpImageInput` brick.
*   **V1.0.0**: **Prompt Helper Integration**. Added a compact, categorized keyword assistant with local persistence database, searchable chips, and direct injection into the Prompt brick. 
*   **V0.9.9**: **FLUX.2 Klein Support**, specialized weight mapping for Alpha/SFT checkpoints, automated guidance scaling (distilled models), and Qwen2-based text encoder integration.
*   **V0.9.8**: **ControlNet Ecosystem** (Depth, Pose), **OpenPose Editor** (Canvas), **Depth Anything V2** Integration, and Memory Leak Fixes.
*   **V0.9.7**: **Task Queue System**, Serialized Worker, Job Monitor Brick, and Asynchronous API Refactor for VRAM stability.
*   **V0.9.6**: SD 3.5 Turbo NF4 Optimization, BFloat16 precision, Iterative GGUF/Safetensors Fallback, Preset System Integration.
*   **V0.9.5**: LoRA Manager with Architecture Check, HTML Session History Logs, UI Safety Warnings.
*   **V0.9**: Photo Editor V2, Real-time Filters, Base64 Save System, Workflow Deletion.
*   **V0.8**: Inpainting & Outpainting Overhaul.
*   **V0.7**: Lightning Generation & Turbo Models.
*   **V0.5**: Workflow Persistence & Cognitive Layers.

---
*See [TECHNICAL_RECORD.md](TECHNICAL_RECORD.md) for detailed architectural logs and development milestones.*
