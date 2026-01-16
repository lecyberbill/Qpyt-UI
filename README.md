# Qpyt-UI V0.9 TURBO

**Advanced Modular Interface for Generative AI**
*Powered by Diffusers, Florence-2, Qwen2.5, and Web Components.*

<img width="1899" height="872" alt="image" src="https://github.com/user-attachments/assets/e2c45ffb-9edf-4260-b5a5-6c078fca209a" />


## Overview
Qpyt-UI is a **Python-Driven Modular Framework** designed for high-performance interaction with Stable Diffusion (SDXL, FLUX, SD3.5) and auxiliary AI models. It uses a unique "Brick" system where every UI component is a standalone Web Component powered by a dedicated Python backend service.

## Key Features (V0.9)

### 🎨 Generative Imaging
- **Multi-Engine Support**: Seamlessly switch between **SDXL**, **FLUX.1-Schnell**, and **SD 3.5 Turbo**.
- **Lightning Generation**: Optimized pipelines for near-real-time results on consumer RTX cards.
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
- **Persistence**: Save, Load, and **Delete** complex node layouts as JSON workflows.
- **State Management**: Circular dependencies and history tracking.

## Architecture
- **Backend**: FastAPI + PyTorch/Diffusers.
- **Frontend**: Native Web Components (Vanilla JS + Shadow DOM) + Shoelace UI.
- **Communication**: REST API for heavy lifting, Custom Events for UI reactivity.

## Getting Started

1.  **Prerequisites**: Python 3.10+, NVIDIA GPU (8GB+ VRAM recommended), CUDA 11.8/12.x.
2.  **Installation**:
    ```bash
    git clone https://github.com/lecyberbill/Qpyt-UI.git
    cd Qpyt-UI
    pip install -r requirements.txt
    ```
3.  **Launch**:
    Double-click `start.bat` or run:
    ```bash
    python api/main.py
    ```
4.  **Access**: Open `http://127.0.0.1:8000` in your browser.

## Version History
*   **V0.9**: Photo Editor V2, Real-time Filters, Base64 Save System, Workflow Deletion.
*   **V0.8**: Inpainting & Outpainting Overhaul.
*   **V0.7**: Lightning Generation & Turbo Models.
*   **V0.5**: Workflow Persistence & Cognitive Layers.

---
*See [TECHNICAL_RECORD.md](TECHNICAL_RECORD.md) for detailed architectural logs and development milestones.*
