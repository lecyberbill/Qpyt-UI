// Prompt Cartridge
class QpPrompt extends HTMLElement {
    static get observedAttributes() { return ['brick-id']; }
    constructor() {
        super();
        this.attachShadow({ mode: 'open' });
        this.negativePrompt = "";
        this.hasRendered = false;
    }
    connectedCallback() {
        this.render();
        this.fetchConfig();
    }
    async fetchConfig() {
        try {
            const res = await fetch('/config');
            const data = await res.json();
            if (data.settings && data.settings.NEGATIVE_PROMPT) {
                this.negativePrompt = data.settings.NEGATIVE_PROMPT;
                const negInput = this.shadowRoot.querySelector('#negative-input');
                if (negInput) negInput.value = this.negativePrompt;
            }
        } catch (e) { console.error("Failed to fetch config for negative prompt", e); }
    }
    attributeChangedCallback() { this.render(); }

    render() {
        if (this.hasRendered) return;
        this.hasRendered = true;
        const brickId = this.getAttribute('brick-id') || '';
        this.shadowRoot.innerHTML = `
            <qp-cartridge title="Prompt" type="input" brick-id="${brickId}">
                <div style="display: flex; flex-direction: column; gap: 1rem; height: 100%;">
                    <sl-textarea id="prompt-input" name="prompt" label="Positive Prompt" placeholder="What do you want to see?" resize="none" style="flex-grow: 1;"></sl-textarea>
                    <sl-textarea id="negative-input" name="negative_prompt" label="Negative Prompt" value="${this.negativePrompt}" placeholder="What do you want to avoid?" resize="none" style="height: 120px;"></sl-textarea>
                    <div style="margin-top: auto; color: #64748b; font-size: 0.8rem;">
                        ✍️ Be specific for better results.
                    </div>
                </div>
            </qp-cartridge>
        `;
    }
    getValue() {
        return {
            prompt: this.shadowRoot.querySelector('#prompt-input').value,
            negative_prompt: this.shadowRoot.querySelector('#negative-input').value
        };
    }
    get value() { return this.getValue(); }

    // Allow external injection
    setPrompt(text) {
        const input = this.shadowRoot.querySelector('#prompt-input');
        if (input) {
            input.value = text;
            input.focus();
        }
    }
}
customElements.define('qp-prompt', QpPrompt);

// Image Source Cartridge (Reusable)
class QpImageInput extends HTMLElement {
    static get observedAttributes() { return ['brick-id']; }
    constructor() {
        super();
        this.attachShadow({ mode: 'open' });
        this.previewUrl = "";
        this.base64 = "";
    }
    connectedCallback() { this.render(); }

    async handleUpload(e) {
        const file = e.target.files[0];
        if (!file) return;

        this.previewUrl = URL.createObjectURL(file);

        const reader = new FileReader();
        reader.onload = async (event) => {
            const img = new Image();
            img.onload = () => {
                const canvas = document.createElement('canvas');
                let w = img.width;
                let h = img.height;

                if (w > 1024 || h > 1024) {
                    const scale = 1024 / Math.max(w, h);
                    w *= scale;
                    h *= scale;
                }

                w = Math.round(w / 64) * 64;
                h = Math.round(h / 64) * 64;
                if (w < 64) w = 64;
                if (h < 64) h = 64;

                canvas.width = w;
                canvas.height = h;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(img, 0, 0, w, h);
                this.base64 = canvas.toDataURL('image/png');
                this.render();
                this.dispatchEvent(new CustomEvent('image-changed', {
                    detail: { base64: this.base64 },
                    bubbles: true,
                    composed: true
                }));
                window.qpyt_app?.notify(`Image optimized for SDXL (${w}x${h})`, "neutral");
            };
            img.src = event.target.result;
        };
        reader.readAsDataURL(file);
        this.render();
    }

    render() {
        const brickId = this.getAttribute('brick-id') || '';
        this.shadowRoot.innerHTML = `
            <style>
                .preview-box {
                    width: 100%;
                    aspect-ratio: 1/1;
                    background: rgba(0,0,0,0.2);
                    border: 2px dashed rgba(255,255,255,0.1);
                    border-radius: 0.8rem;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    overflow: hidden;
                    position: relative;
                    cursor: pointer;
                    transition: all 0.2s;
                }
                .preview-box:hover {
                    border-color: var(--sl-color-primary-500);
                    background: rgba(var(--sl-color-primary-500), 0.1);
                }
                .preview-box img {
                    width: 100%;
                    height: 100%;
                    object-fit: contain;
                }
                .file-input {
                    position: absolute;
                    inset: 0;
                    opacity: 0;
                    cursor: pointer;
                }
            </style>
            <qp-cartridge title="Source Image" type="input" brick-id="${brickId}">
                <div class="preview-box">
                    ${this.previewUrl ? `<img src="${this.previewUrl}">` : `
                        <div style="text-align: center; color: #64748b;">
                            <sl-icon name="image" style="font-size: 2rem;"></sl-icon>
                            <div style="font-size: 0.8rem; margin-top: 0.5rem;">Click to upload</div>
                        </div>
                    `}
                    <input type="file" class="file-input" accept="image/*" id="file-input">
                </div>
                ${this.previewUrl ? `
                    <sl-button variant="danger" size="small" outline style="margin-top: 0.5rem; width: 100%;" id="clear-btn">
                        <sl-icon slot="prefix" name="trash"></sl-icon>
                        Clear Image
                    </sl-button>
                ` : ''}
            </qp-cartridge>
        `;
        this.shadowRoot.getElementById('file-input')?.addEventListener('change', (e) => this.handleUpload(e));
        this.shadowRoot.getElementById('clear-btn')?.addEventListener('click', () => {
            this.previewUrl = "";
            this.base64 = "";
            this.render();
            this.dispatchEvent(new CustomEvent('image-changed', {
                detail: { base64: "" },
                bubbles: true,
                composed: true
            }));
        });
    }

    getImage() { return this.base64; }
}
customElements.define('qp-image-input', QpImageInput);

// Settings Cartridge
class QpSettings extends HTMLElement {
    static get observedAttributes() { return ['brick-id']; }
    constructor() {
        super();
        this.attachShadow({ mode: 'open' });
        this.formats = [];
        this.selectedFormat = "1024*1024";
        this.hasRendered = false;
    }
    connectedCallback() {
        this.render();
        this.fetchFormats();
    }
    async fetchFormats() {
        try {
            const res = await fetch('/config');
            const data = await res.json();
            if (data.settings && data.settings.FORMATS) {
                this.formats = data.settings.FORMATS;
                const select = this.shadowRoot.getElementById('format-select');
                if (select) {
                    select.innerHTML = this.formats.map(f => `<sl-option value="${f.dimensions}">${f.orientation}: ${f.dimensions}</sl-option>`).join('');
                } else {
                    this.hasRendered = false;
                    this.render();
                }
            }
        } catch (e) { console.error(e); }
    }

    handleFormatChange(e) {
        this.selectedFormat = e.target.value;
        this.render();
    }
    attributeChangedCallback() { this.render(); }

    render() {
        if (this.hasRendered) return;
        this.hasRendered = true;
        const brickId = this.getAttribute('brick-id') || '';
        this.shadowRoot.innerHTML = `
            <qp-cartridge title="Settings" type="logic" brick-id="${brickId}">
                <div style="display: flex; flex-direction: column; gap: 1rem;">
                    
                    <sl-select id="format-select" label="Image Format" value="${this.selectedFormat}" hoist @sl-change="${e => this.handleFormatChange(e)}">
                        ${this.formats.map(f => `<sl-option value="${f.dimensions}">${f.orientation}: ${f.dimensions}</sl-option>`).join('')}
                    </sl-select>

                    <sl-input id="gs-input" type="number" step="0.1" label="Guidance Scale" value="7.0"></sl-input>
                    <sl-input id="steps-input" type="number" label="Inference Steps" value="30"></sl-input>
                    <sl-input id="batch-input" type="number" label="Images to Generate" value="1" min="1"></sl-input>
                    <sl-input id="seed-input" type="number" label="Seed" placeholder="Random (leave empty)"></sl-input>
                </div>
            </qp-cartridge>
        `;
    }
    get values() {
        const [w, h] = this.selectedFormat.split('*').map(Number);
        return {
            width: w || 1024,
            height: h || 1024,
            guidance_scale: parseFloat(this.shadowRoot.querySelector('#gs-input').value),
            num_inference_steps: parseInt(this.shadowRoot.querySelector('#steps-input').value),
            batch_count: parseInt(this.shadowRoot.querySelector('#batch-input').value) || 1,
            seed: this.shadowRoot.querySelector('#seed-input').value ? parseInt(this.shadowRoot.querySelector('#seed-input').value) : null
        };
    }
    getValue() { return this.values; }
    getValues() { return this.values; }
}
customElements.define('qp-settings', QpSettings);

// Render Cartridge
class QpRender extends HTMLElement {
    constructor(modelType = 'sdxl') {
        super();
        this.modelType = modelType;
        this.attachShadow({ mode: 'open' });
        this.selectedModel = '';
        this.selectedSampler = "dpm++_2m_sde_karras";
        this.selectedVae = '';
        this.denoisingStrength = 0.5;
        this.vaes = [];
        this.samplers = [];
        this.models = [];
        this.isGenerating = false;
        this.lastImageUrl = '';
        this.currentStep = 0;
        this.totalSteps = 0;
        this.previewInterval = null;
        this.hasRendered = false;
    }

    connectedCallback() {
        this.render();
        this.fetchModels();
        this.fetchVaes();
        this.fetchSamplers();

        // React to image input changes (Img2Img toggle)
        document.addEventListener('image-changed', () => {
            this.render();
        });
    }

    async fetchVaes() {
        try {
            const res = await fetch('/vaes');
            const data = await res.json();
            if (data.status === 'success') {
                this.vaes = data.vaes;
                const select = this.shadowRoot.getElementById('vae-select');
                if (select) {
                    select.innerHTML = '<sl-option value="">Default</sl-option>' + this.vaes.map(v => `<sl-option value="${v}">${v}</sl-option>`).join('');
                }
            }
        } catch (e) { console.error(e); }
    }
    async fetchSamplers() {
        try {
            const res = await fetch('/samplers');
            const data = await res.json();
            if (data.status === 'success') {
                this.samplers = data.samplers;
                const select = this.shadowRoot.getElementById('sampler-select');
                if (select) {
                    select.innerHTML = this.samplers.map(s => `<sl-option value="${s.replace(/ /g, '_')}">${s}</sl-option>`).join('');
                } else {
                    this.hasRendered = false; // Re-render if select not found
                    this.render();
                }
            }
        } catch (e) { console.error(e); }
    }

    async fetchModels() {
        try {
            const response = await fetch(`/models/${this.modelType}`);
            const result = await response.json();
            if (result.status === 'success') {
                this.models = result.models;
                if (this.models.length > 0 && !this.selectedModel) {
                    this.selectedModel = this.models[0];
                }
                const select = this.shadowRoot.getElementById('model-select');
                if (select) {
                    select.innerHTML = this.models.map(m => `<sl-option value="${m}">${m}</sl-option>`).join('');
                    if (this.models.length === 0) {
                        select.innerHTML = '<sl-option value="" disabled>No models found</sl-option>';
                    }
                    select.value = this.selectedModel;
                }
            }
        } catch (e) {
            console.error("Failed to fetch models", e);
        }
    }

    async generate() {
        if (this.isGenerating || !window.qpyt_app) return;

        const promptEl = document.querySelector('qp-prompt');
        const settingsEl = document.querySelector('qp-settings');

        if (!promptEl) {
            window.qpyt_app.notify("Missing 'Prompt' brick!", "danger");
            return;
        }

        const { prompt, negative_prompt } = promptEl.getValue();
        const settings = settingsEl ? settingsEl.getValue() : {};
        const { batch_count = 1, ...genSettings } = settings;

        // Img2Img specific - Only send image if this is a dedicated Img2Img brick
        const imageSource = document.querySelector('qp-image-input');
        const isImg2ImgBrick = this.tagName === 'QP-IMG2IMG';
        const image = isImg2ImgBrick ? (imageSource?.getImage() || null) : null;

        this.isGenerating = true;
        this.hasRendered = false; // Force re-render for generating state
        this.render();

        if (this.previewInterval) clearInterval(this.previewInterval);
        this.previewInterval = setInterval(() => this.pollPreview(), 500);

        try {
            for (let i = 0; i < batch_count; i++) {
                if (!this.isGenerating) break;

                console.log(`[Batch] Generating ${i + 1}/${batch_count}`);
                this.currentStep = 0;
                this.totalSteps = 0;
                this.lastImageUrl = '';
                this.batchInfo = batch_count > 1 ? ` (${i + 1}/${batch_count})` : '';
                this.updateStatus();

                const response = await fetch('/generate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        prompt,
                        negative_prompt,
                        model_type: this.modelType,
                        model_name: this.selectedModel,
                        sampler_name: this.selectedSampler,
                        vae_name: this.selectedVae || null,
                        image: image,
                        denoising_strength: image ? this.denoisingStrength : undefined,
                        ...genSettings,
                        seed: genSettings.seed ? genSettings.seed + i : null
                    })
                });

                const result = await response.json();
                if (result.status === 'success') {
                    this.lastImageUrl = result.data.image_url;
                    if (window.qpyt_app) window.qpyt_app.lastImage = this.lastImageUrl;

                    // Signal global output
                    window.dispatchEvent(new CustomEvent('qpyt-output', {
                        detail: { url: this.lastImageUrl, brickId: this.getAttribute('brick-id') },
                        bubbles: true,
                        composed: true
                    }));

                    const dashboard = document.querySelector('qp-dashboard');
                    if (dashboard) {
                        dashboard.addEntry(result.data);
                    }
                } else {
                    window.qpyt_app.notify(`Error: ${result.message}`, "danger");
                    break;
                }
            }
            window.qpyt_app.notify("Generation complete", "success");
        } catch (e) {
            console.error(e);
            window.qpyt_app.notify("Connection error", "danger");
        } finally {
            if (this.previewInterval) {
                clearInterval(this.previewInterval);
                this.previewInterval = null;
            }
            this.isGenerating = false;
            this.hasRendered = false; // Force re-render for final state
            this.render();
        }
    }

    async stop() {
        if (!this.isGenerating) return;
        try {
            await fetch('/generate/stop', { method: 'POST' });
            window.qpyt_app.notify("Stop request sent", "warning");
        } catch (e) {
            console.error(e);
        }
    }

    async pollPreview() {
        if (!this.isGenerating) return;
        try {
            const res = await fetch('/generate/preview');
            const data = await res.json();
            if (data.status === 'success') {
                if (data.preview) {
                    this.lastImageUrl = data.preview;
                }
                this.currentStep = data.current_step;
                this.totalSteps = data.total_steps;
                this.updateStatus();
            }
        } catch (e) {
            console.error("Preview poll failed", e);
        }
    }

    updateStatus() {
        const area = this.shadowRoot.getElementById('preview-area');
        if (!area) return;

        if (this.isGenerating) {
            let overlay = area.querySelector('.gen-overlay');
            if (!overlay) {
                area.innerHTML = `
                    <div class="gen-overlay" style="position: absolute; top:0; left:0; width:100%; height:100%; display:flex; flex-direction:column; align-items:center; justify-content:center; background: rgba(0,0,0,0.4); z-index:10; border-radius: 0.8rem;">
                        <sl-spinner style="font-size: 2rem;"></sl-spinner>
                        <div class="progress-txt" style="font-weight: 600; color: #10b981; margin-top: 0.5rem; text-shadow: 0 1px 4px rgba(0,0,0,0.8);"></div>
                    </div>
                    <img class="preview-img" src="" alt="Preview" style="position: absolute; top:0; left:0; width:100%; height:100%; object-fit: contain; opacity: 0.7; z-index:5; display:none;">
                `;
                overlay = area.querySelector('.gen-overlay');
            }

            const txt = overlay.querySelector('.progress-txt');
            if (txt) {
                const batchLabel = this.batchInfo || '';
                txt.textContent = this.currentStep > 0 ? `Step ${this.currentStep}/${this.totalSteps}${batchLabel}` : `Initializing...${batchLabel}`;
            }

            const img = area.querySelector('.preview-img');
            if (img && this.lastImageUrl) {
                img.src = this.lastImageUrl;
                img.style.display = 'block';
            }
        } else {
            this.render();
        }
    }

    render() {
        if (this.hasRendered) return;
        this.hasRendered = true;
        const titleMap = {
            'sdxl': 'SDXL Generator',
            'flux': 'FLUX Generator',
            'sd3': 'SD3.5 Generator',
            'sd3_5_turbo': 'SD3.5 Turbo (Lightning)'
        };
        const title = titleMap[this.modelType] || 'Generator';
        const brickId = this.getAttribute('brick-id') || '';

        this.shadowRoot.innerHTML = `
            <style>
                .render-container {
                    display: flex;
                    flex-direction: column;
                    gap: 1.2rem;
                    height: 100%;
                    justify-content: flex-start;
                    align-items: center;
                }
                .status-area {
                    width: 100%;
                    aspect-ratio: 1.2;
                    background: rgba(0,0,0,0.2);
                    border: 2px dashed rgba(255,255,255,0.1);
                    border-radius: 1rem;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    color: #94a3b8;
                    gap: 1rem;
                    position: relative;
                    overflow: hidden;
                }
                .status-area img {
                    width: 100%;
                    height: 100%;
                    object-fit: contain;
                    border-radius: 0.8rem;
                }
                .gen-btn {
                    width: 100%;
                    --sl-color-primary-600: #10b981;
                    --sl-color-primary-500: #059669;
                }
                .stop-btn {
                    width: 100%;
                    --sl-color-danger-600: #ef4444;
                    --sl-color-danger-500: #dc2626;
                }
                sl-spinner {
                    font-size: 3rem;
                    --indicator-color: #10b981;
                    --track-color: rgba(255,255,255,0.1);
                }
            </style>
            <qp-cartridge title="${title}" type="output" brick-id="${brickId}">
                <div class="render-container">
                    <div style="display: flex; flex-direction: column; gap: 0.5rem; width: 100%;">
                        <sl-select id="model-select" label="Checkpoint (.safetensors)" value="${this.selectedModel}" hoist style="width: 100%;">
                            ${this.models.map(m => `<sl-option value="${m}">${m}</sl-option>`).join('')}
                            ${this.models.length === 0 ? '<sl-option value="" disabled>No models found</sl-option>' : ''}
                        </sl-select>

                        ${this.modelType !== 'flux' ? `
                        <sl-select label="Sampler" value="${this.selectedSampler}" id="sampler-select" size="small" hoist>
                            ${this.samplers.map(s => `<sl-option value="${s.replace(/ /g, '_')}">${s}</sl-option>`).join('')}
                        </sl-select>

                        <sl-select id="vae-select" label="VAE" value="${this.selectedVae}" hoist>
                            <sl-option value="">Default</sl-option>
                            ${this.vaes.map(v => `<sl-option value="${v}">${v}</sl-option>`).join('')}
                        </sl-select>
                    </div>
                    ` : ''}

                    ${(this.tagName === 'QP-IMG2IMG' || this.tagName === 'QP-UPSCALER') ? `
                        <div style="background: rgba(16, 185, 129, 0.1); padding: 1rem; border-radius: 0.8rem; border: 1px solid rgba(16, 185, 129, 0.2); margin-top: 1rem; width: 100%; box-sizing: border-box;">
                            <div style="display: flex; align-items: center; gap: 0.5rem; color: #10b981; font-size: 0.85rem; font-weight: 700; margin-bottom: 0.8rem;">
                                <sl-icon name="${this.tagName === 'QP-UPSCALER' ? 'aspect-ratio' : 'magic'}"></sl-icon> 
                                ${this.tagName === 'QP-UPSCALER' ? 'Upscale Influence (Denoising)' : 'Transformation Strength'}
                            </div>
                            <input type="range" id="denoising-slider" min="0" max="1" step="0.01" value="${this.denoisingStrength}" 
                                style="width: 100%; height: 6px; border-radius: 3px; background: rgba(255,255,255,0.1); appearance: none; cursor: pointer; outline: none;">
                            <div id="denoising-label" style="font-size: 0.75rem; color: #94a3b8; margin-top: 0.6rem; font-family: monospace; font-weight: 600;">
                                Current: ${this.denoisingStrength.toFixed(2)}
                            </div>

                            ${this.tagName === 'QP-UPSCALER' ? `
                                <div style="display: flex; gap: 0.5rem; margin-top: 1rem;">
                                    <sl-select id="upscale-factor" label="Scale Factor" value="${this.upscaleFactor || 2}" size="small" style="flex:1;" hoist>
                                        <sl-option value="1.5">1.5x</sl-option>
                                        <sl-option value="2">2x</sl-option>
                                        <sl-option value="4">4x</sl-option>
                                    </sl-select>
                                    <sl-select id="tile-size" label="Tile Size" value="${this.tileSize || 768}" size="small" style="flex:1;" hoist>
                                        <sl-option value="512">512</sl-option>
                                        <sl-option value="768">768</sl-option>
                                        <sl-option value="1024">1024</sl-option>
                                    </sl-select>
                                </div>
                            ` : ''}
                        </div>
                        <style>
                            #denoising-slider::-webkit-slider-thumb {
                                appearance: none;
                                width: 16px;
                                height: 16px;
                                border-radius: 50%;
                                background: #10b981;
                                cursor: pointer;
                                border: 2px solid #fff;
                                box-shadow: 0 0 5px rgba(0,0,0,0.5);
                            }
                            #denoising-slider::-moz-range-thumb {
                                width: 16px;
                                height: 16px;
                                border-radius: 50%;
                                background: #10b981;
                                cursor: pointer;
                                border: 2px solid #fff;
                                box-shadow: 0 0 5px rgba(0,0,0,0.5);
                            }
                        </style>
                    ` : ''}
                    
                    <div class="status-area" id="preview-area" style="${this.lastImageUrl ? 'cursor: pointer;' : ''}">
                        ${this.isGenerating ? '' : (this.lastImageUrl ? `
                            <img src="${this.lastImageUrl}" alt="Generated image">
                        ` : `
                            <sl-icon name="image" style="font-size: 3rem; opacity: 0.3;"></sl-icon>
                            <div style="font-size: 0.9rem;">Ready to generate</div>
                        `)}
                    </div>
                    
                    ${this.isGenerating ? `
                        <sl-button class="stop-btn" variant="danger" size="large" id="stop-btn">
                            <sl-icon slot="prefix" name="stop-fill"></sl-icon>
                            Stop
                        </sl-button>
                    ` : `
                        <sl-button class="gen-btn" variant="primary" size="large" id="gen-btn">
                            <sl-icon slot="prefix" name="play-fill"></sl-icon>
                            Generate
                        </sl-button>
                    `}
                </div>
            </qp-cartridge>
        `;

        if (this.isGenerating) this.updateStatus();

        this.shadowRoot.getElementById('gen-btn')?.addEventListener('click', () => this.generate());
        this.shadowRoot.getElementById('stop-btn')?.addEventListener('click', () => this.stop());
        this.shadowRoot.getElementById('model-select')?.addEventListener('sl-change', (e) => {
            this.selectedModel = e.target.value;
        });
        const samplerSelect = this.shadowRoot.getElementById('sampler-select');
        if (samplerSelect) {
            samplerSelect.addEventListener('sl-change', (e) => {
                this.selectedSampler = e.target.value;
            });
        }
        this.shadowRoot.getElementById('vae-select')?.addEventListener('sl-change', (e) => {
            this.selectedVae = e.target.value;
        });

        const denoisingSlider = this.shadowRoot.getElementById('denoising-slider');
        if (denoisingSlider) {
            const updateLabel = (val) => {
                const label = this.shadowRoot.getElementById('denoising-label');
                if (label) label.textContent = `Current Influence: ${parseFloat(val).toFixed(2)}`;
            };
            denoisingSlider.addEventListener('input', (e) => updateLabel(e.target.value));
            denoisingSlider.addEventListener('change', (e) => {
                this.denoisingStrength = parseFloat(e.target.value);
                updateLabel(e.target.value);
            });
        }

        this.shadowRoot.getElementById('denoising-label')?.addEventListener('click', () => {
            this.denoisingStrength = this.tagName === 'QP-UPSCALER' ? 0.2 : 0.5;
            this.render();
        });

        // Upscaler specific events
        this.shadowRoot.getElementById('upscale-factor')?.addEventListener('sl-change', (e) => {
            this.upscaleFactor = parseFloat(e.target.value);
        });
        this.shadowRoot.getElementById('tile-size')?.addEventListener('sl-change', (e) => {
            this.tileSize = parseInt(e.target.value);
        });

        this.shadowRoot.getElementById('preview-area')?.addEventListener('click', () => {
            if (this.lastImageUrl && !this.isGenerating) {
                const dashboard = document.querySelector('qp-dashboard');
                if (dashboard) dashboard.openLightbox(this.lastImageUrl);
            }
        });
    }
}

class QpRenderSdxl extends QpRender { constructor() { super('sdxl'); } }
class QpRenderFlux extends QpRender { constructor() { super('flux'); } }
// Decided to remove standard SD35 to keep only Turbo as requested

customElements.define('qp-render-sdxl', QpRenderSdxl);
customElements.define('qp-render-flux', QpRenderFlux);

class QpRenderSd35Turbo extends QpRender {
    constructor() {
        super('sd3_5_turbo');
        this.title = "SD3.5 Turbo (Lightning)";
        this.defaultSteps = 4;
        this.defaultGuidance = 1.0;
    }

    connectedCallback() {
        super.connectedCallback();
    }

    render() {
        super.render();
        const cartridge = this.shadowRoot.querySelector('qp-cartridge');
        if (cartridge) {
            cartridge.setAttribute('title', this.title);
            cartridge.setAttribute('icon', 'lightning-charge');
        }
    }
}
customElements.define('qp-render-sd35turbo', QpRenderSd35Turbo);

customElements.define('qp-render', class extends QpRenderSdxl { });

// Dedicated Img2Img Brick
class QpImg2Img extends QpRender {
    constructor() {
        super('sdxl'); // Default to SDXL
        this.modelType = 'img2img'; // Special type to force slider
        this.title = "Img2Img Refiner";
        this.icon = "magic";
    }

    render() {
        super.render();
        const cartridge = this.shadowRoot.querySelector('qp-cartridge');
        if (cartridge) {
            cartridge.setAttribute('title', this.title);
            cartridge.setAttribute('icon', this.icon);
        }
    }
}
customElements.define('qp-img2img', QpImg2Img);

// Tiled Upscaler Brick
class QpUpscaler extends QpRender {
    constructor() {
        super('sdxl'); // Default to SDXL for upscaling details
        this.modelType = 'upscale';
        this.title = "Tiled Upscaler";
        this.icon = "aspect-ratio";
        this.denoisingStrength = 0.2; // Default for upscalers
        this.upscaleFactor = 2;
        this.tileSize = 768;
    }

    async generate() {
        // Override generate to use /upscale endpoint
        if (this.isGenerating || !window.qpyt_app) return;

        const promptEl = document.querySelector('qp-prompt');
        const settingsEl = document.querySelector('qp-settings');

        const { prompt, negative_prompt } = promptEl ? promptEl.getValue() : { prompt: "", negative_prompt: "" };
        const settings = settingsEl ? settingsEl.getValue() : {};

        const imageSource = document.querySelector('qp-image-input');
        let image = imageSource?.getImage() || null;

        if (!image && window.qpyt_app?.lastImage) {
            image = window.qpyt_app.lastImage;
            console.log("[Upscaler] Auto-picking last generated image:", image);
        }

        if (!image) {
            window.qpyt_app.notify("Upscaler requires a Source Image or a previous generation!", "danger");
            return;
        }

        this.isGenerating = true;
        this.hasRendered = false;
        this.render();

        if (this.previewInterval) clearInterval(this.previewInterval);
        this.previewInterval = setInterval(() => this.pollPreview(), 500);

        try {
            this.currentStep = 0;
            this.totalSteps = 0;
            this.lastImageUrl = '';
            this.updateStatus();

            const response = await fetch('/upscale', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    prompt,
                    negative_prompt,
                    model_type: this.modelType,
                    model_name: this.selectedModel,
                    image: image,
                    upscale_factor: this.upscaleFactor,
                    denoising_strength: this.denoisingStrength,
                    tile_size: this.tileSize,
                    ...settings
                })
            });

            const result = await response.json();
            if (result.status === 'success') {
                this.lastImageUrl = result.data.image_url;
                if (window.qpyt_app) window.qpyt_app.lastImage = this.lastImageUrl;

                // Signal global output
                window.dispatchEvent(new CustomEvent('qpyt-output', {
                    detail: { url: this.lastImageUrl, brickId: this.getAttribute('brick-id') },
                    bubbles: true,
                    composed: true
                }));

                const dashboard = document.querySelector('qp-dashboard');
                if (dashboard) dashboard.addEntry(result.data);
                window.qpyt_app.notify("Upscale complete", "success");
            } else {
                window.qpyt_app.notify(`Error: ${result.message}`, "danger");
            }
        } catch (e) {
            console.error(e);
            window.qpyt_app.notify("Connection error", "danger");
        } finally {
            if (this.previewInterval) clearInterval(this.previewInterval);
            this.isGenerating = false;
            this.hasRendered = false;
            this.render();
        }
    }

    render() {
        super.render();
        const cartridge = this.shadowRoot.querySelector('qp-cartridge');
        if (cartridge) {
            cartridge.setAttribute('title', this.title);
            cartridge.setAttribute('icon', this.icon);
        }
    }
}
customElements.define('qp-upscaler', QpUpscaler);

// Final Output Brick
class QpImageOut extends HTMLElement {
    static get observedAttributes() { return ['brick-id']; }
    constructor() {
        super();
        this.attachShadow({ mode: 'open' });
        this.currentUrl = "";
    }

    connectedCallback() {
        this.currentUrl = window.qpyt_app?.lastImage || "";
        this.render();
        window.addEventListener('qpyt-output', (e) => {
            this.currentUrl = e.detail.url;
            this.render();
        });
    }

    sendToSource() {
        if (!this.currentUrl) return;
        const sourceBrick = document.querySelector('qp-image-input');
        if (sourceBrick) {
            // Trick: we use the /view/ path as base64 value, the backend now supports it
            sourceBrick.base64 = this.currentUrl;
            sourceBrick.previewUrl = this.currentUrl;
            sourceBrick.render();
            window.qpyt_app?.notify("Sent to Source Image!", "success");
        }
    }

    render() {
        const brickId = this.getAttribute('brick-id') || '';
        this.shadowRoot.innerHTML = `
            <style>
                .out-container {
                    display: flex;
                    flex-direction: column;
                    gap: 1rem;
                    align-items: center;
                }
                .image-box {
                    width: 100%;
                    aspect-ratio: 1;
                    background: rgba(0,0,0,0.3);
                    border: 1px solid rgba(255,255,255,0.1);
                    border-radius: 12px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    overflow: hidden;
                    position: relative;
                }
                .image-box img {
                    width: 100%;
                    height: 100%;
                    object-fit: contain;
                    cursor: pointer;
                }
                .empty-msg {
                    color: #475569;
                    font-size: 0.8rem;
                    text-align: center;
                    padding: 2rem;
                }
            </style>
            <qp-cartridge title="Final Output" type="output" brick-id="${brickId}">
                <div class="out-container">
                    <div class="image-box" id="preview-box">
                        ${this.currentUrl ? `
                            <img src="${this.currentUrl}" alt="Final image">
                        ` : `
                            <div class="empty-msg">
                                <sl-icon name="hourglass" style="font-size: 2rem; margin-bottom: 0.5rem;"></sl-icon>
                                <div>Waiting for generation...</div>
                            </div>
                        `}
                    </div>
                    
                    ${this.currentUrl ? `
                        <div style="display: flex; gap: 0.5rem; width: 100%;">
                            <sl-button variant="primary" size="small" id="btn-source" style="flex: 1;">
                                <sl-icon slot="prefix" name="arrow-left-right"></sl-icon>
                                Send to Source
                            </sl-button>
                            <sl-button variant="neutral" size="small" id="btn-open" outline style="flex: 1;">
                                <sl-icon slot="prefix" name="arrows-fullscreen"></sl-icon>
                                Fullscreen
                            </sl-button>
                        </div>
                    ` : ''}
                </div>
            </qp-cartridge>
        `;

        this.shadowRoot.getElementById('btn-source')?.addEventListener('click', () => this.sendToSource());
        this.shadowRoot.getElementById('btn-open')?.addEventListener('click', () => {
            const dashboard = document.querySelector('qp-dashboard');
            if (dashboard) dashboard.openLightbox(this.currentUrl);
        });
        this.shadowRoot.getElementById('preview-box')?.querySelector('img')?.addEventListener('click', () => {
            const dashboard = document.querySelector('qp-dashboard');
            if (dashboard) dashboard.openLightbox(this.currentUrl);
        });
    }
}
customElements.define('qp-image-out', QpImageOut);

// Translator (Placeholder)
class QpTranslator extends HTMLElement {
    static get observedAttributes() { return ['brick-id']; }
    constructor() {
        super();
        this.attachShadow({ mode: 'open' });
        this.isLoading = false;
        this.result = "";
    }

    connectedCallback() { this.render(); }

    async translate() {
        const input = this.shadowRoot.getElementById('fr-prompt').value.trim();
        if (!input) return;

        this.isLoading = true;
        this.render();

        try {
            const response = await fetch('/translate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: input })
            });
            const data = await response.json();
            if (data.status === 'success') {
                this.result = data.translated_text;
            } else {
                this.result = `Error: ${data.message}`;
            }
        } catch (e) {
            this.result = "Translation failed.";
        } finally {
            this.isLoading = false;
            this.render();
        }
    }

    inject() {
        if (!this.result) return;
        const promptBrick = document.querySelector('qp-prompt');
        if (promptBrick && typeof promptBrick.setPrompt === 'function') {
            promptBrick.setPrompt(this.result);
            window.qpyt_app?.notify("Prompt injected!", "success");
        } else {
            window.qpyt_app?.notify("No prompt module found.", "danger");
        }
    }

    render() {
        const brickId = this.getAttribute('brick-id');
        this.shadowRoot.innerHTML = `
            <style>
                .translator-container {
                    display: flex;
                    flex-direction: column;
                    gap: 1rem;
                    padding: 0.5rem;
                }
                .result-area {
                    background: rgba(0,0,0,0.2);
                    padding: 0.8rem;
                    border-radius: 8px;
                    border: 1px solid rgba(255,255,255,0.05);
                    font-size: 0.9rem;
                    color: #94a3b8;
                    min-height: 50px;
                    word-break: break-word;
                }
            </style>
            <qp-cartridge title="Translator (FR ➔ EN)" type="logic" brick-id="${brickId}">
                <div class="translator-container">
                    <sl-textarea id="fr-prompt" label="French Prompt" placeholder="Entrez votre prompt en français..." resize="auto" size="small"></sl-textarea>
                    
                    <sl-button variant="primary" id="btn-translate" ?loading="${this.isLoading}" size="small">
                        <sl-icon slot="prefix" name="translate"></sl-icon>
                        Translate
                    </sl-button>

                    ${this.result ? `
                        <div class="result-area">${this.result}</div>
                        <sl-button variant="neutral" id="btn-inject" size="small" outline>
                            <sl-icon slot="prefix" name="chat-left-text"></sl-icon>
                            Inject to Prompt
                        </sl-button>
                    ` : ''}
                </div>
            </qp-cartridge>
        `;

        this.shadowRoot.getElementById('btn-translate')?.addEventListener('click', () => this.translate());
        this.shadowRoot.getElementById('btn-inject')?.addEventListener('click', () => this.inject());
    }
}
customElements.define('qp-translator', QpTranslator);

// Image-to-Prompt (Florence-2) - Refactored to use QpImageInput
class QpImg2Prompt extends HTMLElement {
    static get observedAttributes() { return ['brick-id']; }
    constructor() {
        super();
        this.attachShadow({ mode: 'open' });
        this.isAnalyzing = false;
        this.result = "";
        this.selectedTask = "<DETAILED_CAPTION>";
    }
    connectedCallback() { this.render(); }
    attributeChangedCallback() { this.render(); }

    async analyze() {
        if (this.isAnalyzing) return;
        const imageSource = document.querySelector('qp-image-input');
        const base64 = imageSource?.getImage();
        if (!base64) {
            window.qpyt_app?.notify("Please upload an image in the Source Image brick first.", "warning");
            return;
        }
        this.isAnalyzing = true;
        this.render();
        try {
            const res_blob = await fetch(base64);
            const blob = await res_blob.blob();
            const file = new File([blob], "input.png", { type: "image/png" });
            const formData = new FormData();
            formData.append("file", file);
            formData.append("task", this.selectedTask);
            const res = await fetch("/analyze", { method: "POST", body: formData });
            const data = await res.json();
            if (data.status === "success") {
                this.result = data.prompt;
                window.qpyt_app?.notify("Analysis complete!", "success");
            } else {
                window.qpyt_app?.notify(`Analysis failed: ${data.message}`, "danger");
            }
        } catch (e) {
            console.error(e);
            window.qpyt_app?.notify("Analysis failed due to error", "danger");
        } finally {
            this.isAnalyzing = false;
            this.render();
        }
    }

    inject() {
        const promptEl = document.querySelector('qp-prompt');
        if (promptEl && this.result) {
            promptEl.setPrompt(this.result);
            window.qpyt_app?.notify("Prompt injected!", "success");
        }
    }

    render() {
        const brickId = this.getAttribute('brick-id') || '';
        this.shadowRoot.innerHTML = `
            <style>
                .analyzer-container { display: flex; flex-direction: column; gap: 1rem; }
            </style>
            <qp-cartridge title="Image Analysis" type="logic" brick-id="${brickId}">
                <div class="analyzer-container">
                    <sl-select id="task-select" value="${this.selectedTask}" label="Analysis Task" hoist>
                        <sl-option value="<DETAILED_CAPTION>">Detailed Caption</sl-option>
                        <sl-option value="<MORE_DETAILED_CAPTION>">More Detailed Caption</sl-option>
                        <sl-option value="<CAPTION>">Simple Caption</sl-option>
                        <sl-option value="<GENERATE_TAGS>">Generate Tags</sl-option>
                    </sl-select>
                    <sl-button variant="primary" id="analyze-btn" ?loading="${this.isAnalyzing}" style="width: 100%;">
                        <sl-icon slot="prefix" name="search"></sl-icon>
                        Analyze Source Image
                    </sl-button>
                    ${this.result ? `
                        <sl-textarea id="result-text" value="${this.result}" rows="4" label="Analysis Result"></sl-textarea>
                        <sl-button variant="success" outline id="inject-btn" style="width: 100%;">
                            <sl-icon slot="prefix" name="box-arrow-in-right"></sl-icon>
                            Inject to Prompt
                        </sl-button>
                    ` : ''}
                </div>
            </qp-cartridge>
        `;
        this.shadowRoot.getElementById('task-select')?.addEventListener('sl-change', (e) => { this.selectedTask = e.target.value; });
        this.shadowRoot.getElementById('result-text')?.addEventListener('sl-change', (e) => { this.result = e.target.value; });
        this.shadowRoot.getElementById('analyze-btn')?.addEventListener('click', () => this.analyze());
        this.shadowRoot.getElementById('inject-btn')?.addEventListener('click', () => this.inject());
    }
}
customElements.define('qp-img2prompt', QpImg2Prompt);

// LLM Prompt Enhancer
class QpLlmPrompter extends HTMLElement {
    static get observedAttributes() { return ['brick-id']; }
    constructor() {
        super();
        this.attachShadow({ mode: 'open' });
        this.isLoading = false;
        this.result = "";
    }

    connectedCallback() { this.render(); }

    async enhance() {
        const input = this.shadowRoot.getElementById('base-idea').value.trim();
        if (!input) return;

        this.isLoading = true;
        this.render();

        try {
            const response = await fetch('/enhance', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: input })
            });
            const data = await response.json();
            if (data.status === 'success') {
                this.result = data.prompt;
            } else {
                this.result = `Error: ${data.message}`;
            }
        } catch (e) {
            this.result = "Enhancement failed.";
        } finally {
            this.isLoading = false;
            this.render();
        }
    }

    inject() {
        if (!this.result) return;
        const promptBrick = document.querySelector('qp-prompt');
        if (promptBrick && typeof promptBrick.setPrompt === 'function') {
            promptBrick.setPrompt(this.result);
            window.qpyt_app?.notify("Enhanced prompt injected!", "success");
        } else {
            window.qpyt_app?.notify("No prompt module found.", "danger");
        }
    }

    render() {
        const brickId = this.getAttribute('brick-id');
        this.shadowRoot.innerHTML = `
            <style>
                .prompter-container {
                    display: flex;
                    flex-direction: column;
                    gap: 1rem;
                    padding: 0.5rem;
                }
                .result-area {
                    background: rgba(0,0,0,0.2);
                    padding: 0.8rem;
                    border-radius: 8px;
                    border: 1px solid rgba(255,255,255,0.05);
                    font-size: 0.9rem;
                    color: #94a3b8;
                    min-height: 80px;
                    max-height: 200px;
                    overflow-y: auto;
                    word-break: break-word;
                }
            </style>
            <qp-cartridge title="LLM Prompt Enhancer" type="input" brick-id="${brickId}">
                <div class="prompter-container">
                    <sl-textarea id="base-idea" label="Your Idea" placeholder="Enter a simple idea (e.g. a robot in a garden)..." resize="auto" size="small"></sl-textarea>
                    
                    <sl-button variant="primary" id="btn-enhance" ?loading="${this.isLoading}" size="small">
                        <sl-icon slot="prefix" name="stars"></sl-icon>
                        Magic Enhance
                    </sl-button>

                    ${this.result ? `
                        <div class="result-area">${this.result}</div>
                        <sl-button variant="neutral" id="btn-inject" size="small" outline>
                            <sl-icon slot="prefix" name="chat-left-text"></sl-icon>
                            Inject to Prompt
                        </sl-button>
                    ` : ''}
                </div>
            </qp-cartridge>
        `;

        this.shadowRoot.getElementById('btn-enhance')?.addEventListener('click', () => this.enhance());
        this.shadowRoot.getElementById('btn-inject')?.addEventListener('click', () => this.inject());
    }
}
customElements.define('qp-llm-prompter', QpLlmPrompter);
