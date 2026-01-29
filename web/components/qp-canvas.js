class QpCanvas extends HTMLElement {
    static get observedAttributes() { return ['brick-id']; }

    constructor() {
        super();
        this.attachShadow({ mode: 'open' });
        this.canvasOpen = false;
        this.isDrawing = false;
        this.lastX = 0;
        this.lastY = 0;
        this.brushSize = 5;
        this.isWhiteOnBlack = true; // Default for ControlNet Scribble/Canny often needs white lines on black

        // Internal state for the generated image
        this.generatedImage = null;
    }

    connectedCallback() {
        this.render();
    }

    // --- Canvas Logic ---

    openCanvas() {
        this.canvasOpen = true;
        this.renderOverlay();
        // Initialize Canvas Context after DOM update
        setTimeout(() => this.initCanvas(), 50);
    }

    closeCanvas() {
        this.canvasOpen = false;
        const overlay = this.shadowRoot.querySelector('.canvas-overlay');
        if (overlay) overlay.remove();
        this.render(); // Update thumbnail
    }

    initCanvas() {
        const canvas = this.shadowRoot.getElementById('drawing-surface');
        if (!canvas) return;

        // Set canvas resolution strictly
        const rect = canvas.parentElement.getBoundingClientRect();
        // Default to a square or 512x512-ish aspect ratio if possible, or fill screen
        // Let's make it a fixed high-res canvas that scales down visually
        canvas.width = 1024;
        canvas.height = 1024; // Square canvas for simplicity

        this.ctx = canvas.getContext('2d');
        this.ctx.lineJoin = 'round';
        this.ctx.lineCap = 'round';

        // Fill background
        this.fillBackground();

        // If we have an existing image, draw it
        if (this.generatedImage) {
            const img = new Image();
            img.onload = () => {
                this.ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
            };
            img.src = this.generatedImage;
        }

        // Event Listeners
        canvas.addEventListener('mousedown', (e) => this.startDrawing(e, canvas));
        canvas.addEventListener('mousemove', (e) => this.draw(e, canvas));
        canvas.addEventListener('mouseup', () => this.stopDrawing());
        canvas.addEventListener('mouseout', () => this.stopDrawing());

        // Touch support
        canvas.addEventListener('touchstart', (e) => {
            e.preventDefault();
            this.startDrawing(e.touches[0], canvas);
        });
        canvas.addEventListener('touchmove', (e) => {
            e.preventDefault();
            this.draw(e.touches[0], canvas);
        });
        canvas.addEventListener('touchend', () => this.stopDrawing());
    }

    fillBackground() {
        if (!this.ctx) return;
        this.ctx.fillStyle = this.isWhiteOnBlack ? '#000000' : '#ffffff';
        this.ctx.fillRect(0, 0, this.ctx.canvas.width, this.ctx.canvas.height);
    }

    startDrawing(e, canvas) {
        this.isDrawing = true;
        const rect = canvas.getBoundingClientRect();

        // Map DOM coordinates to Canvas coordinates
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;

        this.lastX = (e.clientX - rect.left) * scaleX;
        this.lastY = (e.clientY - rect.top) * scaleY;
    }

    draw(e, canvas) {
        if (!this.isDrawing) return;

        const rect = canvas.getBoundingClientRect();
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;

        const x = (e.clientX - rect.left) * scaleX;
        const y = (e.clientY - rect.top) * scaleY;

        this.ctx.strokeStyle = this.isWhiteOnBlack ? '#ffffff' : '#000000';
        this.ctx.lineWidth = this.brushSize;

        this.ctx.beginPath();
        this.ctx.moveTo(this.lastX, this.lastY);
        this.ctx.lineTo(x, y);
        this.ctx.stroke();

        this.lastX = x;
        this.lastY = y;
    }

    stopDrawing() {
        this.isDrawing = false;
    }

    clearCanvas() {
        if (!this.ctx) return;
        this.fillBackground();
    }

    toggleMode() {
        this.isWhiteOnBlack = !this.isWhiteOnBlack;
        // WARNING: This clears the canvas if we just refill. 
        // Ideally we would invert pixels, but let's keep it simple: clear and swap.
        // User expects "Invert" to invert the current drawing usually.
        this.invertPixels();
        this.renderOverlay(); // Re-render to update toggle button text
        setTimeout(() => this.initCanvas(), 0); // Re-init context binding but preserve data? 
        // Actually renderOverlay destroys DOM. We need to save state first.
    }

    invertPixels() {
        if (!this.ctx) return;
        const component = this;
        const imageData = this.ctx.getImageData(0, 0, this.ctx.canvas.width, this.ctx.canvas.height);
        const data = imageData.data;

        for (let i = 0; i < data.length; i += 4) {
            data[i] = 255 - data[i];     // r
            data[i + 1] = 255 - data[i + 1]; // g
            data[i + 2] = 255 - data[i + 2]; // b
        }

        this.ctx.putImageData(imageData, 0, 0);
        // Save state immediately
        this.generatedImage = this.ctx.canvas.toDataURL('image/png');
    }

    changeBrushSize(val) {
        this.brushSize = val;
    }

    saveState() {
        const canvas = this.shadowRoot.getElementById('drawing-surface');
        if (canvas) {
            this.generatedImage = canvas.toDataURL('image/png');
        }
    }

    saveAndClose() {
        this.saveState();
        this.closeCanvas();
    }

    // --- Helpers ---

    sendToControlNet() {
        if (!this.generatedImage) return;

        // Try to find ControlNet brick
        let cnBrick = document.querySelector('qp-controlnet');

        if (!cnBrick) {
            // Ask existing qpyt_app to add one? 
            if (confirm("ControlNet module not found. Add one now?")) {
                window.qpyt_app.addBrick('qp-controlnet').then(() => {
                    // Slight delay for DOM to update
                    setTimeout(() => this.sendToControlNet(), 500);
                });
                return;
            } else {
                return;
            }
        }

        if (cnBrick) {
            // Existing QpControlNet in qp-bricks.js does not have setImage()
            // We set the property directly and force a re-render
            cnBrick.controlNetImage = this.generatedImage;

            // Auto-select a scribble/canny model if none selected
            if (!cnBrick.selectedControlNet && cnBrick.controlnets) {
                const scribble = cnBrick.controlnets.find(m => m.toLowerCase().includes('scribble'));
                if (scribble) cnBrick.selectedControlNet = scribble;
            }

            cnBrick.hasRendered = false;
            if (typeof cnBrick.render === 'function') cnBrick.render();

            this.notify("Sent to ControlNet!", "success");
        } else {
            console.warn("qp-controlnet instance not found");
            this.notify("ControlNet brick not found", "warning");
        }
        this.saveAndClose();
    }

    sendToImg2Img() {
        if (!this.generatedImage) return;
        const sourceBrick = document.querySelector('qp-image-input');
        if (sourceBrick) {
            sourceBrick.base64 = this.generatedImage;
            sourceBrick.previewUrl = this.generatedImage;
            sourceBrick.render();
            this.notify("Sent to Source Image!", "success");
            this.saveAndClose();
        } else {
            this.notify("No Source Brick found", "warning");
        }
    }

    notify(msg, type) {
        if (window.qpyt_app) window.qpyt_app.notify(msg, type);
    }

    // --- Rendering ---

    renderOverlay() {
        // Full screen overlay logic
        let overlay = this.shadowRoot.querySelector('.canvas-overlay');
        if (!overlay) {
            overlay = document.createElement('div');
            overlay.className = 'canvas-overlay';
            this.shadowRoot.appendChild(overlay);
        }

        overlay.innerHTML = `
            <style>
                .canvas-overlay {
                    position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
                    background: #0f172a;
                    z-index: 99999;
                    display: flex; flex-direction: column;
                }
                .toolbar {
                    padding: 1rem;
                    background: #1e293b;
                    border-bottom: 1px solid rgba(255,255,255,0.1);
                    display: flex; gap: 1rem; align-items: center; justify-content: space-between;
                }
                .viewport {
                    flex: 1;
                    display: flex; align-items: center; justify-content: center;
                    background: #334155;
                    overflow: hidden;
                    position: relative;
                }
                canvas {
                    background: ${this.isWhiteOnBlack ? 'black' : 'white'};
                    box-shadow: 0 0 20px rgba(0,0,0,0.5);
                    max-width: 95vw;
                    max-height: 85vh;
                    cursor: crosshair;
                }
                .tools { display: flex; gap: 0.5rem; align-items: center; }
            </style>

            <div class="toolbar">
                <div class="tools">
                    <sl-button circle onclick="this.getRootNode().host.closeCanvas()">
                        <sl-icon name="arrow-left"></sl-icon>
                    </sl-button>
                    <span style="font-weight:bold; color:white; margin-left:0.5rem;">Sketch Canvas</span>
                    
                    <div style="width: 1px; height: 20px; background: rgba(255,255,255,0.2); margin: 0 1rem;"></div>

                    <sl-tooltip content="Clear">
                        <sl-button variant="danger" outline circle onclick="this.getRootNode().host.clearCanvas()">
                            <sl-icon name="trash"></sl-icon>
                        </sl-button>
                    </sl-tooltip>

                    <sl-tooltip content="Toggle Mode (B/W)">
                         <sl-button variant="neutral" outline circle onclick="this.getRootNode().host.toggleMode()">
                            <sl-icon name="${this.isWhiteOnBlack ? 'circle-half' : 'circle'}"></sl-icon>
                        </sl-button>
                    </sl-tooltip>
                    
                     <div style="width: 1px; height: 20px; background: rgba(255,255,255,0.2); margin: 0 1rem;"></div>

                    <span style="color:#94a3b8; font-size:0.8rem;">Size:</span>
                    <input type="range" min="1" max="50" value="${this.brushSize}" 
                        oninput="this.getRootNode().host.changeBrushSize(this.value)"
                        style="width: 100px; accent-color: #f59e0b;">
                </div>

                <div class="tools">
                    <sl-button variant="warning" outline onclick="this.getRootNode().host.sendToImg2Img()">
                         <sl-icon slot="prefix" name="image"></sl-icon> To Source
                    </sl-button>
                    <sl-button variant="primary" onclick="this.getRootNode().host.sendToControlNet()">
                         <sl-icon slot="prefix" name="diagram-3"></sl-icon> To ControlNet
                    </sl-button>
                    <sl-button variant="success" onclick="this.getRootNode().host.saveAndClose()">
                        <sl-icon slot="prefix" name="check-lg"></sl-icon> Done
                    </sl-button>
                </div>
            </div>

            <div class="viewport">
                <canvas id="drawing-surface"></canvas>
            </div>
        `;
    }

    render() {
        const brickId = this.getAttribute('brick-id') || '';

        this.shadowRoot.innerHTML = `
            <style>
                :host { display: block; }
                .preview-box {
                    width: 100%; aspect-ratio: 16/9;
                    background: #1e293b;
                    border: 1px solid rgba(255,255,255,0.1);
                    border-radius: 8px;
                    display: flex; align-items: center; justify-content: center;
                    cursor: pointer;
                    overflow: hidden;
                }
                .preview-box:hover { border-color: #f59e0b; }
                .preview-box img { width: 100%; height: 100%; object-fit: contain; }
            </style>
            
            <qp-cartridge title="Sketch Canvas" type="input" icon="brush" brick-id="${brickId}">
                <div style="display:flex; flex-direction:column; gap:1rem;">
                    <div class="preview-box" onclick="this.getRootNode().host.openCanvas()">
                        ${this.generatedImage
                ? `<img src="${this.generatedImage}">`
                : `<div style="text-align:center; color:#64748b;">
                                 <sl-icon name="pen" style="font-size:2rem; margin-bottom:0.5rem;"></sl-icon><br>
                                 Click to Draw
                               </div>`
            }
                    </div>
                    
                    <sl-button variant="primary" outline style="width:100%" onclick="this.getRootNode().host.openCanvas()">
                        <sl-icon slot="prefix" name="fullscreen"></sl-icon> Open Canvas
                    </sl-button>
                </div>
            </qp-cartridge>
        `;
    }
}

customElements.define('qp-canvas', QpCanvas);
