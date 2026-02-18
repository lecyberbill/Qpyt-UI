// import { QpCartridge } from './qp-cartridge.js'; // Not needed, loaded via index.html

class QpSprite extends HTMLElement {
    static get observedAttributes() { return ['brick-id']; }

    constructor() {
        super();
        this.attachShadow({ mode: 'open' });

        // State
        this.prompt = "";
        this.negativePrompt = "blur, fuzzy, extra limbs, malformed, text, watermark";
        this.frames = 16;
        this.steps = 8;
        this.guidance = 7.5;
        this.seed = "";
        this.resultUrl = null;
        this.isGenerating = false;
        this.selectedModel = "";
        this.models = [];

        this.hasRendered = false;
    }

    connectedCallback() {
        this.fetchModels();
        this.render();
    }

    async fetchModels() {
        try {
            // Using 'sd15' as type for Sprite models (AnimateDiff is SD1.5 based)
            const res = await fetch('/models/sd15');
            const data = await res.json();
            if (data.status === 'success') {
                this.models = data.models;
                if (!this.selectedModel && this.models.length > 0) {
                    // Prefer a model with 'pixel' in name if available, else first
                    const pixelModel = this.models.find(m => m.toLowerCase().includes('pixel'));
                    this.selectedModel = pixelModel || this.models[0];
                }
                this.hasRendered = false;
                this.render();
            }
        } catch (e) {
            console.error("Failed to fetch sprite models", e);
        }
    }

    attributeChangedCallback() {
        this.render();
    }

    async generate() {
        if (this.isGenerating) return;

        // Fetch prompt from global qp-prompt brick
        const promptEl = document.querySelector('qp-prompt');
        if (!promptEl) {
            window.qpyt_app?.notify("Missing 'Prompt' brick!", "warning");
            return;
        }

        // Handle both possible APIs: getValue() or direct access
        let promptValue = "";
        let negativeValue = "";

        if (typeof promptEl.getValue === 'function') {
            const values = promptEl.getValue();
            promptValue = values.prompt;
            negativeValue = values.negative_prompt;
        }

        if (!promptValue.trim()) {
            window.qpyt_app?.notify("Please enter a prompt!", "warning");
            return;
        }

        this.prompt = promptValue.trim();
        this.negativePrompt = negativeValue || this.negativePrompt;

        this.isGenerating = true;
        this.render(); // Update button state

        try {
            const payload = {
                prompt: this.prompt,
                negative_prompt: this.negativePrompt,
                height: 256, // Fixed for sprites
                frames: this.frames,
                steps: this.steps,
                guidance: this.guidance,
                seed: this.seed ? parseInt(this.seed) : null,
                model_name: this.selectedModel,
                loras: []
            };

            // Collect LoRAs from qp-lora-manager
            const loraManager = document.querySelector('qp-lora-manager');
            if (loraManager && typeof loraManager.getValues === 'function') {
                const loraData = loraManager.getValues();
                if (loraData.loras && loraData.loras.length > 0) {
                    payload.loras = loraData.loras.filter(l => l.enabled).map(l => ({
                        path: l.path,
                        weight: l.weight,
                        enabled: l.enabled
                    }));
                }
            }

            const response = await fetch('/generate/sprite', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            const data = await response.json();

            if (data.status === 'queued') {
                window.qpyt_app?.notify("Sprite generation queued...", "primary");
                this.monitorTask(data.task_id);
            } else {
                window.qpyt_app?.notify("Error starting generation", "danger");
                this.isGenerating = false;
                this.render();
            }
        } catch (e) {
            console.error(e);
            window.qpyt_app?.notify("Generation failed", "danger");
            this.isGenerating = false;
            this.render();
        }
    }

    async monitorTask(taskId) {
        const checkStatus = async () => {
            try {
                const res = await fetch(`/queue/status/${taskId}`);
                const task = await res.json();

                if (task.status === 'COMPLETED') {
                    this.resultUrl = task.result.image_url;
                    this.isGenerating = false;
                    window.qpyt_app?.notify("Sprite generated!", "success");

                    // Signal global output for Final Output brick
                    window.dispatchEvent(new CustomEvent('qpyt-output', {
                        detail: {
                            url: this.resultUrl,
                            brickId: this.getAttribute('brick-id'),
                            params: {
                                seed: task.result.metadata?.seed, // Corrected path
                                prompt: this.prompt,
                                model: "AnimateDiff"
                            }
                        },
                        bubbles: true,
                        composed: true
                    }));

                    // Update Dashboard/History
                    const dashboard = document.querySelector('qp-dashboard');
                    if (dashboard) {
                        dashboard.addEntry({
                            id: taskId,
                            image_url: this.resultUrl,
                            status: 'success',
                            metadata: {
                                prompt: this.prompt,
                                seed: task.result.metadata?.seed, // Corrected path
                                width: 256,
                                height: 256,
                                model_name: "AnimateDiff",
                                num_inference_steps: this.steps,
                                guidance_scale: this.guidance
                            },
                            execution_time: task.result.execution_time || 0
                        });
                    }

                    this.render();
                } else if (task.status === 'FAILED') {
                    this.isGenerating = false;
                    window.qpyt_app?.notify(`Generation failed: ${task.message}`, "danger");
                    this.render();
                } else {
                    setTimeout(checkStatus, 1000);
                }
            } catch (e) {
                this.isGenerating = false;
                this.render();
            }
        };
        checkStatus();
    }

    render() {
        if (this.hasRendered) {
            // Update specific elements if already rendered to avoid losing focus
            const btn = this.shadowRoot.getElementById('btn-generate');
            if (btn) btn.loading = this.isGenerating;

            const resultArea = this.shadowRoot.getElementById('result-area');
            if (resultArea && this.resultUrl) {
                // Only update if URL changed or empty
                if (!resultArea.querySelector('img') || resultArea.querySelector('img').src !== this.resultUrl) {
                    resultArea.innerHTML = `<img src="${this.resultUrl}" style="width: 256px; height: 256px; image-rendering: pixelated; border-radius: 8px; border: 2px solid #a855f7;">`;
                }
            }
            return;
        }

        this.hasRendered = true;
        const brickId = this.getAttribute('brick-id') || '';

        this.shadowRoot.innerHTML = `
            <style>
                .controls { display: flex; flex-direction: column; gap: 0.8rem; }
                .row { display: flex; gap: 0.5rem; align-items: center; }
                .col { display: flex; flex-direction: column; gap: 0.5rem; }
                .preview { 
                    display: flex; justify-content: center; align-items: center; 
                    min-height: 260px; background: rgba(0,0,0,0.3); border-radius: 8px;
                    border: 1px dashed rgba(255,255,255,0.1);
                    margin-top: 0.5rem;
                }
            </style>
            
            <qp-cartridge title="Sprite Animator (256px)" type="generator" brick-id="${brickId}">
                <div class="controls">
                    <div style="font-size: 0.8rem; color: #94a3b8; font-style: italic;">
                        Uses the main <b>Prompt</b> brick.
                    </div>

                    <div class="col">
                        <sl-input type="number" id="seed-input" label="Seed" placeholder="Random" value="${this.seed}"></sl-input>
                        <sl-select id="frames-input" label="Animation Frames" value="${this.frames}">
                             <sl-option value="8">8 Frames</sl-option>
                             <sl-option value="16">16 Frames (Standard)</sl-option>
                             <sl-option value="24">24 Frames</sl-option>
                             <sl-option value="32">32 Frames</sl-option>
                        </sl-select>

                        <sl-select id="model-select" label="Base Model" value="${this.selectedModel}" hoist>
                            ${this.models.length === 0 ? '<sl-option value="" disabled>No compatible models found</sl-option>' : ''}
                            ${this.models.map(m => `<sl-option value="${m}">${m}</sl-option>`).join('')}
                        </sl-select>
                    </div>

                    <div class="row">
                         <div style="flex:1">
                            <label style="font-size:0.8rem; color:#ccc;">Steps: <span id="steps-val">${this.steps}</span></label>
                            <input type="range" id="steps-range" min="4" max="50" value="${this.steps}" style="width:100%">
                         </div>
                         <div style="flex:1">
                            <label style="font-size:0.8rem; color:#ccc;">Guidance: <span id="guidance-val">${this.guidance}</span></label>
                            <input type="range" id="guidance-range" min="1" max="20" step="0.5" value="${this.guidance}" style="width:100%">
                         </div>
                    </div>

                    <sl-button id="btn-generate" variant="primary" ?loading="${this.isGenerating}">
                        <sl-icon slot="prefix" name="play-circle"></sl-icon>
                        Generate Sprite
                    </sl-button>

                    <div id="result-area" class="preview">
                        ${this.resultUrl ?
                `<img src="${this.resultUrl}" style="width: 256px; height: 256px; image-rendering: pixelated; border-radius: 8px; border: 2px solid #a855f7;">`
                : '<span style="color: #64748b; font-size: 0.8rem;">Preview Area</span>'}
                    </div>
                </div>
            </qp-cartridge>
        `;

        // Bind events
        this.shadowRoot.getElementById('btn-generate').addEventListener('click', () => this.generate());

        // No prompt input anymore
        this.shadowRoot.getElementById('seed-input').addEventListener('sl-input', (e) => this.seed = e.target.value);

        this.shadowRoot.getElementById('frames-input').addEventListener('sl-change', (e) => this.frames = parseInt(e.target.value));

        const modelSelect = this.shadowRoot.getElementById('model-select');
        if (modelSelect) {
            modelSelect.addEventListener('sl-change', (e) => this.selectedModel = e.target.value);
        }

        const stepsRange = this.shadowRoot.getElementById('steps-range');
        stepsRange.addEventListener('input', (e) => {
            this.steps = parseInt(e.target.value);
            this.shadowRoot.getElementById('steps-val').textContent = this.steps;
        });

        const guidanceRange = this.shadowRoot.getElementById('guidance-range');
        guidanceRange.addEventListener('input', (e) => {
            this.guidance = parseFloat(e.target.value);
            this.shadowRoot.getElementById('guidance-val').textContent = this.guidance;
        });
    }

    // Save/Load state support
    getValues() {
        return {
            prompt: this.prompt,
            frames: this.frames,
            steps: this.steps,
            guidance: this.guidance,
            prompt: this.prompt,
            frames: this.frames,
            steps: this.steps,
            guidance: this.guidance,
            seed: this.seed,
            selectedModel: this.selectedModel
        };
    }

    setValues(v) {
        if (!v) return;
        if (v.prompt) this.prompt = v.prompt;
        if (v.frames) this.frames = v.frames;
        if (v.steps) this.steps = v.steps;
        if (v.guidance) this.guidance = v.guidance;
        if (v.guidance) this.guidance = v.guidance;
        if (v.seed) this.seed = v.seed;
        if (v.selectedModel) this.selectedModel = v.selectedModel;

        this.hasRendered = false;
        this.render();
    }
}

customElements.define('qp-sprite', QpSprite);
