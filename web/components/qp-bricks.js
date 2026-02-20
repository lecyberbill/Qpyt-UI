console.log("[QpBricks] Script version v36 loading...");
// Prompt Cartridge
class QpPrompt extends HTMLElement {
    static get observedAttributes() { return ['brick-id']; }
    constructor() {
        super();
        this.attachShadow({ mode: 'open' });
        this.negativePrompt = "";
        this.hasRendered = false;
        this.lockedFields = new Set();
    }
    connectedCallback() {
        this.render();
        this.fetchConfig();
    }

    toggleLock(field, btn) {
        if (this.lockedFields.has(field)) {
            this.lockedFields.delete(field);
            btn.classList.remove('active');
            btn.name = 'lock-open';
            this.shadowRoot.querySelector(`#${field}-input`).classList.remove('locked-input');
        } else {
            this.lockedFields.add(field);
            btn.classList.add('active');
            btn.name = 'lock';
            this.shadowRoot.querySelector(`#${field}-input`).classList.add('locked-input');
        }
    }

    render() {
        if (this.hasRendered) return;
        this.hasRendered = true;
        const brickId = this.getAttribute('brick-id') || '';
        this.shadowRoot.innerHTML = `
            <style>
                .field-container {
                    position: relative;
                    display: flex;
                    flex-direction: column;
                }
                .expand-btn {
                    position: absolute;
                    top: 0;
                    right: 0;
                    z-index: 10;
                }
            </style>
            <qp-cartridge title="Prompt" type="input" brick-id="${brickId}">
                <div style="display: flex; flex-direction: column; gap: 1rem; height: 100%; position: relative;">
                    <sl-tooltip content="Open Editor">
                        <sl-icon-button class="expand-btn" name="arrows-angle-expand" label="Expand"></sl-icon-button>
                    </sl-tooltip>
                    
                    <div class="field-container">
                        <sl-icon-button class="lock-btn" name="lock-open" id="lock-prompt" title="Lock Prompt"></sl-icon-button>
                        <sl-textarea id="prompt-input" name="prompt" label="Positive Prompt" placeholder="What do you want to see?" resize="none" style="flex-grow: 1;"></sl-textarea>
                    </div>

                    <div class="field-container">
                         <sl-icon-button class="lock-btn" name="lock-open" id="lock-negative" title="Lock Negative Prompt"></sl-icon-button>
                         <sl-textarea id="negative-input" name="negative_prompt" label="Negative Prompt" value="${this.negativePrompt}" placeholder="What do you want to avoid?" resize="none" style="height: 120px;"></sl-textarea>
                    </div>

                    <div style="margin-top: auto; color: #64748b; font-size: 0.8rem;">
                        ✍️ Be specific for better results.
                    </div>
                </div>
            </qp-cartridge>
            <!-- ... dialog omitted for brevity in replace, but keeping logic ... -->

            <sl-dialog label="Prompt Editor" class="prompt-dialog" style="--width: 80vw;">
                <div style="display: flex; flex-direction: column; gap: 1rem; height: 70vh;">
                     <sl-textarea id="dialog-prompt" label="Positive Prompt" style="flex-grow: 2; height: 100%;" resize="none"></sl-textarea>
                     <sl-textarea id="dialog-negative" label="Negative Prompt" style="flex-grow: 1; height: 100%;" resize="none"></sl-textarea>
                </div>
                <div slot="footer">
                    <sl-button variant="primary" class="save-btn">Apply & Close</sl-button>
                    <sl-button variant="default" class="close-btn">Cancel</sl-button>
                </div>
            </sl-dialog>
        `;

        const dialog = this.shadowRoot.querySelector('.prompt-dialog');
        const expandBtn = this.shadowRoot.querySelector('.expand-btn');
        const saveBtn = this.shadowRoot.querySelector('.save-btn');
        const closeBtn = this.shadowRoot.querySelector('.close-btn');

        const pInput = this.shadowRoot.querySelector('#prompt-input');
        const nInput = this.shadowRoot.querySelector('#negative-input');
        const dPInput = this.shadowRoot.querySelector('#dialog-prompt');
        const dNInput = this.shadowRoot.querySelector('#dialog-negative');

        if (expandBtn) {
            expandBtn.addEventListener('click', () => {
                dPInput.value = pInput.value;
                dNInput.value = nInput.value;
                dialog.show();
            });
        }

        this.shadowRoot.getElementById('lock-prompt').addEventListener('click', (e) => this.toggleLock('prompt', e.target));
        this.shadowRoot.getElementById('lock-negative').addEventListener('click', (e) => this.toggleLock('negative', e.target));

        if (saveBtn) {
            saveBtn.addEventListener('click', () => {
                if (!this.lockedFields.has('prompt')) pInput.value = dPInput.value;
                if (!this.lockedFields.has('negative')) nInput.value = dNInput.value;
                dialog.hide();
            });
        }

        if (closeBtn) {
            closeBtn.addEventListener('click', () => dialog.hide());
        }
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
        if (this.lockedFields.has('prompt')) return;
        const input = this.shadowRoot.querySelector('#prompt-input');
        if (input) {
            input.value = text;
            input.focus();
        }
    }
    setValues(values) {
        if (!values) return;
        const pInput = this.shadowRoot.querySelector('#prompt-input');
        const nInput = this.shadowRoot.querySelector('#negative-input');
        if (pInput && values.prompt !== undefined && !this.lockedFields.has('prompt')) pInput.value = values.prompt;
        if (nInput && values.negative_prompt !== undefined && !this.lockedFields.has('negative')) nInput.value = values.negative_prompt;
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

        // Enable Drag & Drop for the image output
        const img = this.shadowRoot.querySelector('img');
        if (img) {
            img.style.cursor = 'grab';
            img.setAttribute('draggable', 'true');
            img.addEventListener('dragstart', (e) => {
                // Ensure we send the Base64 data, not the blob URL
                if (this.base64) {
                    e.dataTransfer.setData('text/plain', this.base64);
                    e.dataTransfer.effectAllowed = 'copy';
                }
            });
        }
    }

    getImage() { return this.base64; }
    getValue() {
        return {
            previewUrl: this.previewUrl,
            base64: this.base64
        };
    }
    setValues(values) {
        if (!values) return;
        this.previewUrl = values.previewUrl || "";
        this.base64 = values.base64 || "";
        this.render();
    }
}
customElements.define('qp-image-input', QpImageInput);

// Settings Cartridge
class QpSettings extends HTMLElement {
    static get observedAttributes() { return ['brick-id']; }
    constructor() {
        super();
        this.attachShadow({ mode: 'open' });
        this.formats = [];
        this.selectedDimension = "1024*1024";
        this.selectedOutputFormat = "webp";
        this.guidanceScale = 7.0;
        this.inferenceSteps = 30;
        this.batchCount = 1;
        this.seed = null;
        this.hasRendered = false;
        this.lockedFields = new Set();
    }

    toggleLock(field, btn) {
        if (this.lockedFields.has(field)) {
            this.lockedFields.delete(field);
            btn.classList.remove('active');
            btn.name = 'lock-open';
            this.shadowRoot.querySelector(`#${field}-input`)?.classList.remove('locked-input');
            this.shadowRoot.querySelector(`#${field}-select`)?.classList.remove('locked-input');
        } else {
            this.lockedFields.add(field);
            btn.classList.add('active');
            btn.name = 'lock';
            this.shadowRoot.querySelector(`#${field}-input`)?.classList.add('locked-input');
            this.shadowRoot.querySelector(`#${field}-select`)?.classList.add('locked-input');
        }
    }

    setValues(values) {
        if (!values) return;
        if (values.width && values.height && !this.lockedFields.has('format')) {
            this.selectedDimension = `${values.width}*${values.height}`;
        }
        if (values.output_format && !this.lockedFields.has('output-format')) {
            this.selectedOutputFormat = values.output_format;
        }

        if (values.guidance_scale !== undefined && !this.lockedFields.has('gs')) this.guidanceScale = Number(values.guidance_scale);
        if (values.num_inference_steps !== undefined && !this.lockedFields.has('steps')) this.inferenceSteps = parseInt(values.num_inference_steps);
        if (values.batch_count !== undefined && !this.lockedFields.has('batch')) this.batchCount = parseInt(values.batch_count);
        if (values.seed !== undefined && values.seed !== "" && !this.lockedFields.has('seed')) this.seed = parseInt(values.seed);
        else if ((values.seed === "" || values.seed === null) && !this.lockedFields.has('seed')) this.seed = null;

        this.hasRendered = false;
        this.render();
    }

    render() {
        if (this.hasRendered) return;
        this.hasRendered = true;
        const brickId = this.getAttribute('brick-id') || '';
        this.shadowRoot.innerHTML = `
            <style>
                .field-container { position: relative; display: flex; flex-direction: column; }
            </style>
            <qp-cartridge title="Settings" type="setting" brick-id="${brickId}">
                <div style="display: flex; flex-direction: column; gap: 1rem;">
                    
                    <div class="field-container">
                        <sl-icon-button class="lock-btn" name="lock-open" id="lock-format" title="Lock Dimensions"></sl-icon-button>
                        <sl-select id="format-select" label="Image Dimensions" value="${this.selectedDimension}" hoist>
                            ${this.formats.map(f => `<sl-option value="${f.dimensions}">${f.orientation}: ${f.dimensions}</sl-option>`).join('')}
                            ${!this.formats.some(f => f.dimensions === this.selectedDimension) ? `<sl-option value="${this.selectedDimension}">Custom: ${this.selectedDimension}</sl-option>` : ''}
                        </sl-select>
                    </div>

                    <div class="field-container">
                        <sl-icon-button class="lock-btn" name="lock-open" id="lock-gs"></sl-icon-button>
                        <sl-input id="gs-input" type="number" step="0.1" label="Guidance Scale" value="${this.guidanceScale}"></sl-input>
                    </div>

                    <div class="field-container">
                        <sl-icon-button class="lock-btn" name="lock-open" id="lock-steps"></sl-icon-button>
                        <sl-input id="steps-input" type="number" label="Inference Steps" value="${this.inferenceSteps}"></sl-input>
                    </div>

                    <div class="field-container">
                        <sl-icon-button class="lock-btn" name="lock-open" id="lock-batch"></sl-icon-button>
                        <sl-input id="batch-input" type="number" label="Images to Generate" value="${this.batchCount}" min="1"></sl-input>
                    </div>

                    <div class="field-container">
                        <sl-icon-button class="lock-btn" name="lock-open" id="lock-seed"></sl-icon-button>
                        <sl-input id="seed-input" type="number" label="Seed" placeholder="Random (leave empty)" value="${this.seed !== null ? this.seed : ''}"></sl-input>
                    </div>

                    <div class="field-container">
                        <sl-icon-button class="lock-btn" name="lock-open" id="lock-output-format"></sl-icon-button>
                        <sl-select id="output-format-select" label="Output File Format" value="${this.selectedOutputFormat}" hoist>
                            <sl-option value="png">PNG (Lossless / Large)</sl-option>
                            <sl-option value="jpeg">JPEG (Compressed / Small)</sl-option>
                            <sl-option value="webp">WebP (Modern / Efficient)</sl-option>
                        </sl-select>
                    </div>
                </div>
            </qp-cartridge>
        `;

        this.shadowRoot.getElementById('format-select').addEventListener('sl-change', (e) => this.handleFormatChange(e));
        this.shadowRoot.getElementById('output-format-select').addEventListener('sl-change', (e) => this.handleOutputFormatChange(e));

        // Lock events
        ['format', 'gs', 'steps', 'batch', 'seed', 'output-format'].forEach(f => {
            this.shadowRoot.getElementById(`lock-${f}`).addEventListener('click', (e) => this.toggleLock(f, e.target));
        });

        // Bind inputs to class state
        this.shadowRoot.getElementById('gs-input').addEventListener('sl-input', (e) => this.guidanceScale = Number(e.target.value));
        this.shadowRoot.getElementById('steps-input').addEventListener('sl-input', (e) => this.inferenceSteps = parseInt(e.target.value));
        this.shadowRoot.getElementById('batch-input').addEventListener('sl-input', (e) => this.batchCount = parseInt(e.target.value));
        this.shadowRoot.getElementById('seed-input').addEventListener('sl-input', (e) => {
            const v = e.target.value;
            this.seed = (v === "" || v === null) ? null : parseInt(v);
        });
    }
    get values() {
        // Source of truth is now the class state (which is kept in sync with inputs via listeners)
        // But to be safe, we can read from inputs if they exist, or fallback to state.
        // Actually, let's trust our state + handle dynamic dimension/format which are also state.
        const [w, h] = this.selectedDimension.split('*').map(Number);
        return {
            width: w || 1024,
            height: h || 1024,
            guidance_scale: this.guidanceScale,
            num_inference_steps: this.inferenceSteps,
            batch_count: this.batchCount,
            seed: this.seed,
            output_format: this.selectedOutputFormat
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
        this.endpoint = '/generate';
        this.denoisingStrength = 0.5;
        this.useLowVram = false; // New property
        this.vaes = [];
        this.samplers = [];
        this.models = [];
        this.isGenerating = false;
        this.lastImageUrl = '';
        this.currentStep = 0;
        this.totalSteps = 0;
        this.previewInterval = null;
        this.controlnets = [];
        this.hasRendered = false;

        // Default resonant settings
        this.defaultSteps = 30;
        this.defaultGuidance = 7.0;
    }

    applyDefaultSettings() {
        const settingsEl = document.querySelector('qp-settings') || document.querySelector('mg-settings');
        if (settingsEl && typeof settingsEl.setValues === 'function') {
            console.log(`[QpRender] Applying Resonant Defaults: Steps=${this.defaultSteps}, CFG=${this.defaultGuidance}`);
            settingsEl.setValues({
                num_inference_steps: this.defaultSteps,
                guidance_scale: this.defaultGuidance
            });
            window.qpyt_app?.notify(`Settings optimized for ${this.title || this.modelType}`, "success");
        }
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

    async submitAndPollTask(endpoint, payload) {
        try {
            const response = await fetch(endpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if (!response.ok) {
                const text = await response.text();
                console.error(`[Queue Error] Server returned ${response.status}: ${text}`);
                let msg = "Failed to submit task";
                try {
                    const errJson = JSON.parse(text);
                    msg = errJson.message || errJson.detail || msg;
                    if (Array.isArray(msg)) msg = msg.map(m => m.msg || JSON.stringify(m)).join(", ");
                } catch (e) { }
                throw new Error(`${msg} (${response.status})`);
            }

            const submission = await response.json();
            if (submission.status === 'queued') {
                const taskId = submission.task_id;
                // Polling loop
                while (true) {
                    if (!this.isGenerating) {
                        // Attempt to cancel if we stopped
                        fetch(`/queue/cancel/${taskId}`, { method: 'POST' }).catch(() => { });
                        return null;
                    }
                    const statusResp = await fetch(`/queue/status/${taskId}`);
                    const task = await statusResp.json();

                    if (task.status === 'COMPLETED') {
                        if (task.result && typeof task.result === 'object') {
                            task.result.request_id = taskId;
                        }
                        return task.result;
                    }
                    if (task.status === 'FAILED' || task.status === 'CANCELLED') {
                        throw new Error(task.error || "Task failed or was cancelled");
                    }
                    // Wait 1s before next poll
                    await new Promise(r => setTimeout(r, 1000));
                }
            } else if (submission.status === 'success') {
                return submission.data;
            } else {
                throw new Error(submission.message || "Failed to submit task");
            }
        } catch (e) {
            console.error("[Queue Polling Error]", e);
            throw e;
        }
    }

    async generate() {
        if (!window.qpyt_app) return;

        // Multi-tasking support: allow multiple queued tasks
        if (!this._activeTasks) this._activeTasks = 0;
        this.isGenerating = true;
        console.log("[QpRender] Generate triggered. Model Type:", this.modelType);
        // Don't render yet, we'll do it after adding tasks

        const promptEl = document.querySelector('qp-prompt');
        let settingsEl = document.querySelector('qp-settings');
        const stylesEl = document.querySelector('qp-styles');
        const loraManager = document.querySelector('qp-lora-manager');
        const controlNet = document.querySelector('qp-controlnet');

        if (!promptEl) {
            window.qpyt_app.notify("Missing 'Prompt' brick!", "danger");
            return;
        }

        let { prompt, negative_prompt } = promptEl.getValue();

        // Apply Fooocus styles if available
        if (stylesEl && typeof stylesEl.applyStyles === 'function') {
            const styled = stylesEl.applyStyles(prompt, negative_prompt);
            prompt = styled.prompt;
            negative_prompt = styled.negative_prompt;
        }

        // Fallback to mg-settings if qp-settings is missing (Legacy compatibility)
        if (!settingsEl) {
            settingsEl = document.querySelector('mg-settings');
            if (settingsEl) console.log("[Bricks] Using legacy mg-settings");
            else console.warn("[Bricks] No settings brick found!");
        }

        const settings = settingsEl ? (typeof settingsEl.getValue === 'function' ? settingsEl.getValue() : (settingsEl.values || {})) : {};
        console.log("[Bricks] Captured Settings:", settings);
        const { batch_count = 1, ...genSettings } = settings;
        let successCount = 0;

        // Img2Img specific - Only send image if this is a dedicated Img2Img brick
        const imageSource = document.querySelector('qp-image-input');
        const isImg2ImgBrick = this.tagName === 'QP-IMG2IMG' || this.tagName === 'QP-INPAINT' || this.tagName === 'QP-OUTPAINT';
        const blenderSource = document.querySelector('qp-image-blender');
        const blenderData = blenderSource ? blenderSource.getValue() : null;

        const image = isImg2ImgBrick ? (imageSource?.getImage() || null) : null;

        if (isImg2ImgBrick && !image) {
            window.qpyt_app.notify("Missing source image! Please upload an image to the 'Source Image' brick first.", "warning");
            return;
        }

        const isInpaint = this.tagName === 'QP-INPAINT';
        const isOutpaint = this.tagName === 'QP-OUTPAINT';
        let mask = null;
        if (isInpaint || isOutpaint) {
            mask = typeof this.getMask === 'function' ? this.getMask() : null;
            // Debug mask length
            if (mask) console.log("[Inpaint] Mask captured, length:", mask.length);
            else console.log("[Inpaint] No mask captured!");
        }

        this.isGenerating = true;
        this.hasRendered = false; // Force re-render for generating state
        this.render();

        if (this.previewInterval) clearInterval(this.previewInterval);
        this.previewInterval = setInterval(() => this.pollPreview(), 500);

        // Refactored to submit ALL tasks in the batch immediately to the queue.
        // This ensures the batch stays together and is not interleaved by other concurrent bricks.
        if (batch_count > 1) this._activeTasks += (batch_count - 1);

        const runOne = async (index) => {
            const endpoint = isInpaint ? '/inpaint' : (isOutpaint ? '/outpaint' : '/generate');
            const payload = {
                prompt,
                negative_prompt,
                model_type: this.modelType,
                model_name: this.selectedModel,
                sampler_name: this.selectedSampler,
                vae_name: this.selectedVae || null,
                denominator: genSettings.denominator,
                image: image,
                mask: mask,
                image_a: blenderData ? blenderData.image_a : null,
                image_b: blenderData ? blenderData.image_b : null,
                weight_a: blenderData ? blenderData.weight_a : 0.5,
                weight_b: blenderData ? blenderData.weight_b : 0.5,
                ip_adapter_scale: blenderData ? blenderData.fidelity : 0.5,
                denoising_strength: (image || mask) ? this.denoisingStrength : undefined,
                ...genSettings,
                seed: genSettings.seed ? genSettings.seed + index : null,
                output_format: genSettings.output_format,
                loras: loraManager ? loraManager.getValues().loras : [],
                controlnet_image: controlNet ? controlNet.getImage() : null,
                controlnet_conditioning_scale: controlNet ? controlNet.getStrength() : 0.7,
                controlnet_model: controlNet ? controlNet.getModel() : null,
                low_vram: this.useLowVram, // Pass low_vram setting
                workflow: window.qpyt_app?.getCurrentWorkflowState() || null
            };

            try {
                console.log("[QpRender] Submitting task to:", endpoint, "Payload:", payload);
                const data = await this.submitAndPollTask(endpoint, payload);
                if (!data) {
                    console.warn("[QpRender] No data returned from poll task");
                    return;
                }

                console.log(`[Batch ${index + 1}/${batch_count}] Completed:`, data);
                successCount++;
                this.lastImageUrl = data.image_url;
                if (window.qpyt_app) window.qpyt_app.lastImage = this.lastImageUrl;

                // Update UI status for the last finished one
                this.batchInfo = batch_count > 1 ? ` (${successCount}/${batch_count})` : '';

                // Signal global output
                window.dispatchEvent(new CustomEvent('qpyt-output', {
                    detail: {
                        url: this.lastImageUrl,
                        brickId: this.getAttribute('brick-id'),
                        params: {
                            seed: data.metadata?.seed || null,
                            prompt: prompt,
                            model: this.selectedModel
                        }
                    },
                    bubbles: true,
                    composed: true
                }));

                const dashboard = document.querySelector('qp-dashboard');
                if (dashboard) dashboard.addEntry(data);

                // Show Warnings if any
                if (data.warnings && data.warnings.length > 0) {
                    data.warnings.forEach(w => window.qpyt_app.notify(w, "warning"));
                }
            } catch (e) {
                console.error(`[Batch ${index}] Error during generation:`, e);
                window.qpyt_app.notify(`Generation Error: ${e.message}`, "danger");
            } finally {
                this._activeTasks--;
                if (this._activeTasks <= 0) {
                    this._activeTasks = 0;
                    this.isGenerating = false;
                    if (this.previewInterval) {
                        clearInterval(this.previewInterval);
                        this.previewInterval = null;
                    }
                }
                this.render();
            }
        };

        try {
            const batchPromises = [];
            for (let i = 0; i < batch_count; i++) {
                if (!this.isGenerating) break;
                this._activeTasks++;
                batchPromises.push(runOne(i));
            }
            this.render(); // Initial render with all tasks added
            await Promise.all(batchPromises);

            if (successCount > 0) {
                window.qpyt_app.notify("Generation complete", "success");
            }
        }
        catch (e) {
            console.error(e);
            window.qpyt_app.notify("Connection error", "danger");
        } finally {
            this.hasRendered = false;
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

    getValue() {
        return {
            selectedModel: this.selectedModel,
            selectedSampler: this.selectedSampler,
            selectedVae: this.selectedVae,
            denoisingStrength: this.denoisingStrength,
            lastImageUrl: this.lastImageUrl,
            useLowVram: this.useLowVram // Save state
        };
    }
    setValues(values) {
        if (!values) return;
        if (values.selectedModel !== undefined) this.selectedModel = values.selectedModel;
        if (values.selectedSampler !== undefined) this.selectedSampler = values.selectedSampler;
        if (values.selectedVae !== undefined) this.selectedVae = values.selectedVae;
        if (values.denoisingStrength !== undefined) this.denoisingStrength = values.denoisingStrength;
        if (values.lastImageUrl !== undefined) this.lastImageUrl = values.lastImageUrl;
        if (values.useLowVram !== undefined) this.useLowVram = values.useLowVram; // Restore state
        this.hasRendered = false;
        this.render();
    }

    render() {
        if (this.hasRendered) return;
        this.hasRendered = true;
        const titleMap = {
            'sdxl': 'SDXL Generator',
            'flux': 'FLUX Generator',
            'sd3': 'SD3.5 Generator',
            'sd3_5_turbo': 'SD3.5 Turbo (Lightning)',

        };
        const title = this.title || titleMap[this.modelType] || 'Generator';
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
                .reset-btn {
                    width: 100%;
                    --sl-color-neutral-600: #475569;
                    --sl-color-neutral-500: #334155;
                    margin-top: 0.5rem;
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
            <qp-cartridge title="${title}" type="generator" brick-id="${brickId}">
                <div class="render-container">
                    <div style="display: flex; flex-direction: column; gap: 0.5rem; width: 100%;">
                        ${this.modelType !== 'qwen' ? `
                        <sl-select id="model-select" label="Checkpoint (.safetensors)" value="${this.selectedModel}" hoist style="width: 100%;">
                            ${this.models.map(m => `<sl-option value="${m}">${m}</sl-option>`).join('')}
                            ${this.models.length === 0 ? '<sl-option value="" disabled>No models found</sl-option>' : ''}
                        </sl-select>
                        ` : `
                        <div style="font-size: 0.9rem; color: #94a3b8; background: rgba(255,255,255,0.05); padding: 0.8rem; border-radius: 0.5rem; border: 1px solid rgba(255,255,255,0.1);">
                            <sl-icon name="info-circle" style="vertical-align: middle; margin-right: 0.5rem; color: #60a5fa;"></sl-icon>
                            Using <b>Qwen-Image-2512</b> (Auto-managed)
                        </div>
                        `}

                        ${this.modelType === 'flux' ? `
                            <sl-checkbox id="low-vram-check" ${this.useLowVram ? 'checked' : ''} style="margin-top: 0.5rem;">Low VRAM / FP8 Mode</sl-checkbox>
                            <div style="font-size: 0.8rem; color: #94a3b8; margin-left: 1.7rem;">Use NF4 (HF) or FP8 (Local) quantization to save VRAM.</div>
                        ` : ''}

                        ${(this.modelType !== 'flux' && this.modelType !== 'qwen') ? `
                        <sl-select label="Sampler" value="${this.selectedSampler}" id="sampler-select" size="small" hoist>
                            ${this.samplers.map(s => `<sl-option value="${s.replace(/ /g, '_')}">${s}</sl-option>`).join('')}
                        </sl-select>

                        <sl-select id="vae-select" label="VAE" value="${this.selectedVae}" hoist>
                            <sl-option value="">Default</sl-option>
                            ${this.vaes.map(v => `<sl-option value="${v}">${v}</sl-option>`).join('')}
                        </sl-select>
                        ` : ''}

                        ${(this.tagName === 'QP-IMG2IMG' || this.tagName === 'QP-UPSCALER' || this.tagName === 'QP-INPAINT' || this.tagName === 'QP-OUTPAINT') ? `
                            <div style="background: rgba(16, 185, 129, 0.1); padding: 1rem; border-radius: 0.8rem; border: 1px solid rgba(16, 185, 129, 0.2); margin-top: 1rem; width: 100%; box-sizing: border-box;">
                                <div style="display: flex; align-items: center; gap: 0.5rem; color: #10b981; font-size: 0.85rem; font-weight: 700; margin-bottom: 0.8rem;">
                                    <sl-icon name="${this.tagName === 'QP-UPSCALER' ? 'aspect-ratio' : 'magic'}"></sl-icon> 
                                    ${this.tagName === 'QP-UPSCALER' ? 'Upscale Influence (Denoising)' : (this.tagName.includes('PAINT') ? 'Inpainting Strength' : 'Transformation Strength')}
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
                            ${this.lastImageUrl ? `
                                <img src="${this.lastImageUrl}" alt="Generated image">
                            ` : `
                                <sl-icon name="image" style="font-size: 3rem; opacity: 0.3;"></sl-icon>
                                <div style="font-size: 0.9rem;">Ready to generate</div>
                            `}
                            ${this.isGenerating ? `
                                <div style="position: absolute; inset:0; background: rgba(0,0,0,0.4); display: flex; align-items: center; justify-content: center; z-index: 10;">
                                    <sl-spinner style="font-size: 2rem; --indicator-color: #10b981;"></sl-spinner>
                                    ${this._activeTasks > 1 ? `
                                        <div style="position:absolute; bottom:10px; right:10px; background:#10b981; color:white; padding:2px 8px; border-radius:10px; font-size:0.7rem; font-weight:800;">
                                            ${this._activeTasks} JOBS
                                        </div>
                                    ` : ''}
                                </div>
                            ` : ''}
                        </div>
                        
                        <div style="display: flex; flex-direction: column; gap: 0.5rem; width: 100%;">
                            <sl-button class="gen-btn" variant="primary" size="large" id="gen-btn" style="flex: 1;">
                                <sl-icon slot="prefix" name="play-fill"></sl-icon>
                                Generate
                            </sl-button>
                            
                            <div style="display: flex; gap: 0.5rem;">
                                <sl-button variant="neutral" size="small" id="apply-defaults-btn" outline style="flex: 1;">
                                    <sl-icon slot="prefix" name="magic"></sl-icon>
                                    RESTORE OPTIMAL DEFAULTS
                                </sl-button>

                                ${this.isGenerating ? `
                                    <sl-button class="stop-btn" variant="danger" size="small" id="stop-btn" outline>
                                        <sl-icon name="stop-fill"></sl-icon>
                                    </sl-button>
                                ` : ''}
                            </div>
                        </div>
                    </div>
                </div>
            </qp-cartridge>
        `;

        if (this.isGenerating) this.updateStatus();

        this.shadowRoot.getElementById('apply-defaults-btn')?.addEventListener('click', () => this.applyDefaultSettings());
        this.shadowRoot.getElementById('gen-btn')?.addEventListener('click', () => this.generate());
        this.shadowRoot.getElementById('stop-btn')?.addEventListener('click', () => this.stop());
        this.shadowRoot.getElementById('model-select')?.addEventListener('sl-change', (e) => {
            this.selectedModel = e.target.value;
        });

        // Flux Low VRAM Listener
        const lowVramCheck = this.shadowRoot.getElementById('low-vram-check');
        if (lowVramCheck) lowVramCheck.addEventListener('sl-change', (e) => this.useLowVram = e.target.checked);

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

class QpRenderSdxl extends QpRender { constructor() { super('sdxl'); this.defaultSteps = 30; this.defaultGuidance = 7.0; } }
class QpRenderFlux extends QpRender { constructor() { super('flux'); this.defaultSteps = 4; this.defaultGuidance = 1.0; } }
// Decided to remove standard SD35 to keep only Turbo as requested

customElements.define('qp-render-sdxl', QpRenderSdxl);
customElements.define('qp-render-flux', QpRenderFlux);

class QpRenderFluxKlein extends QpRender {
    constructor() {
        super('flux');
        this.defaultSteps = 4;
        this.defaultGuidance = 1.0;
    }

    connectedCallback() {
        this.title = "FLUX.2 Klein 4B";
        this.icon = "lightning-charge";
        super.connectedCallback();
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
customElements.define('qp-render-flux-klein', QpRenderFluxKlein);

class QpRenderSd35Turbo extends QpRender {
    constructor() {
        super('sd3_5_turbo');
        this.defaultSteps = 4;
        this.defaultGuidance = 1.0;
    }

    connectedCallback() {
        this.title = "SD3.5 Turbo (Lightning)";
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
    }

    connectedCallback() {
        this.title = "Img2Img Refiner";
        this.icon = "magic";
        super.connectedCallback();
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

// Inpainting / Outpainting Brick
class QpInpaint extends QpRender {
    constructor() {
        super('sdxl');
        this.brushSize = 40;
        this.isDrawing = false;
        this.denoisingStrength = 0.9;
        this.ctx = null;
        this.maskCanvas = null;
        this.savedMaskData = null;
        this.endpoint = '/inpaint';
    }

    connectedCallback() {
        this.title = "Inpainting";
        this.icon = "brush";
        super.connectedCallback();
        this._onImageChanged = (e) => this.syncWithSourceImage(e.detail?.base64);
        window.addEventListener('image-changed', this._onImageChanged);
    }

    disconnectedCallback() {
        if (this._onImageChanged) {
            window.removeEventListener('image-changed', this._onImageChanged);
        }
    }

    syncWithSourceImage(imgData = null) {
        if (!imgData) {
            const src = document.querySelector('qp-image-input');
            imgData = src?.getImage();
        }
        if (!imgData) return;

        const previewBox = this.shadowRoot.getElementById('preview-box');
        const editorBg = this.shadowRoot.getElementById('editor-bg');
        if (previewBox) previewBox.style.backgroundImage = `url(${imgData})`;
        if (editorBg) editorBg.src = imgData;

        const tempImg = new Image();
        tempImg.onload = () => {
            const ratio = tempImg.naturalWidth / tempImg.naturalHeight;
            if (previewBox) {
                previewBox.style.aspectRatio = ratio.toString();
                previewBox.style.maxHeight = "400px";
            }
            const settingsBrick = document.querySelector('qp-settings');
            if (settingsBrick) {
                settingsBrick.setValues({
                    width: tempImg.naturalWidth,
                    height: tempImg.naturalHeight
                });
            }
            this.syncEditorSize();
        };
        tempImg.src = imgData;
    }

    syncEditorSize() {
        const editorBg = this.shadowRoot.getElementById('editor-bg');
        const wrapper = this.shadowRoot.getElementById('canvas-wrapper');
        const canvas = this.maskCanvas || this.shadowRoot.getElementById('mask-canvas');
        if (!canvas || !editorBg) return;

        if (!editorBg.complete || editorBg.naturalWidth === 0) {
            setTimeout(() => this.syncEditorSize(), 100);
            return;
        }

        const w = editorBg.clientWidth;
        const h = editorBg.clientHeight;
        if (w === 0 || h === 0) {
            setTimeout(() => this.syncEditorSize(), 200);
            return;
        }

        if (canvas.width !== w || canvas.height !== h) {
            const temp = document.createElement('canvas');
            temp.width = canvas.width;
            temp.height = canvas.height;
            if (canvas.width > 0 && canvas.height > 0) {
                temp.getContext('2d').drawImage(canvas, 0, 0);
            }

            canvas.width = w;
            canvas.height = h;
            if (wrapper) {
                wrapper.style.width = w + 'px';
                wrapper.style.height = h + 'px';
            }

            this.ctx = canvas.getContext('2d', { willReadFrequently: true });
            this.ctx.lineJoin = 'round';
            this.ctx.lineCap = 'round';
            this.ctx.strokeStyle = 'white';
            this.ctx.lineWidth = this.brushSize;

            this.ctx.fillStyle = 'black';
            this.ctx.fillRect(0, 0, w, h);

            if (temp.width > 0) {
                this.ctx.drawImage(temp, 0, 0, w, h);
            } else if (this.savedMaskData) {
                const img = new Image();
                img.onload = () => this.ctx.drawImage(img, 0, 0, w, h);
                img.src = this.savedMaskData;
            }
        }
    }

    getMask() {
        if (!this.maskCanvas) return this.savedMaskData || null;
        return this.maskCanvas.toDataURL('image/png');
    }

    clearMask() {
        if (!this.ctx) return;
        this.ctx.fillStyle = 'black';
        this.ctx.fillRect(0, 0, this.maskCanvas.width, this.maskCanvas.height);
    }

    render() {
        if (this.maskCanvas && this.maskCanvas.width > 0) {
            this.savedMaskData = this.maskCanvas.toDataURL();
        }

        super.render();

        const renderContainer = this.shadowRoot.querySelector('.render-container');
        if (!renderContainer) return;

        // 1. Brick UI (inside the cartridge)
        if (!this.shadowRoot.querySelector('.inpaint-ui')) {
            const canvasUI = document.createElement('div');
            canvasUI.className = 'inpaint-ui';
            canvasUI.innerHTML = `
                <style>
                    .inpaint-ui {
                        display: flex;
                        flex-direction: column;
                        gap: 12px;
                        margin-bottom: 20px;
                        width: 100%;
                    }
                    .preview-wrapper {
                        position: relative;
                        width: 100%;
                        background: #020617;
                        border-radius: 12px;
                        border: 1px solid rgba(255,255,255,0.1);
                        overflow: hidden;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                        min-height: 150px;
                        background-size: contain;
                        background-repeat: no-repeat;
                        background-position: center;
                        box-shadow: inset 0 2px 10px rgba(0,0,0,0.5);
                    }
                    .mask-thumb {
                        position: absolute;
                        inset: 0;
                        width: 100%;
                        height: 100%;
                        object-fit: contain;
                        opacity: 0.7;
                        mix-blend-mode: screen;
                        pointer-events: none;
                        filter: drop-shadow(0 0 5px rgba(255,255,255,0.3));
                    }
                </style>
                <div class="preview-wrapper" id="preview-box">
                    <img class="mask-thumb" id="mask-thumb" src="${this.savedMaskData || ''}" style="${this.savedMaskData ? '' : 'display:none'}" />
                </div>
                <sl-button variant="primary" id="open-editor-btn" size="medium" style="width: 100%; margin-top: 4px;">
                    <sl-icon slot="prefix" name="brush"></sl-icon> Edit Inpaint Mask
                </sl-button>
            `;
            renderContainer.prepend(canvasUI);
        }

        // 2. Global Dialog (outside the cartridge to avoid overflow:hidden)
        if (!this.shadowRoot.querySelector('.editor-dialog')) {
            const dialogContainer = document.createElement('div');
            dialogContainer.innerHTML = `
                <style>
                    .editor-dialog {
                        --sl-z-index-dialog: 10001;
                    }
                    .editor-dialog::part(panel) {
                        width: 100vw;
                        height: 100vh;
                        max-width: none;
                        max-height: none;
                        background: #020617;
                        border: none;
                        border-radius: 0;
                        position: fixed;
                        inset: 0;
                    }
                    .editor-dialog::part(overlay) {
                        backdrop-filter: blur(8px);
                        background-color: rgba(0, 0, 0, 0.85);
                    }
                    .editor-body {
                        display: flex;
                        flex-direction: column;
                        height: calc(100vh - 80px);
                        gap: 16px;
                        padding: 0;
                    }
                    .main-canvas-container {
                        flex: 1;
                        position: relative;
                        background: #000;
                        overflow: auto;
                        display: flex;
                        align-items: center;
                        justify-content: center;
                    }
                    .canvas-wrapper {
                        position: relative;
                        margin: auto;
                        box-shadow: 0 0 50px rgba(0,0,0,1);
                    }
                    #mask-canvas {
                        position: absolute;
                        top: 0;
                        left: 0;
                        width: 100%;
                        height: 100%;
                        touch-action: none;
                        mix-blend-mode: screen;
                        opacity: 0.8;
                        cursor: crosshair;
                        z-index: 10;
                    }
                    #editor-bg {
                        display: block;
                        max-width: 95vw;
                        max-height: 85vh;
                        object-fit: contain;
                        z-index: 5;
                        user-select: none;
                        pointer-events: none;
                    }
                    .editor-controls {
                        display: flex;
                        flex-wrap: wrap;
                        align-items: center;
                        gap: 24px;
                        padding: 16px 32px;
                        background: #0f172a;
                        border-top: 1px solid rgba(255,255,255,0.1);
                    }
                    .brush-label {
                        font-size: 0.7rem;
                        color: #94a3b8;
                        font-weight: 700;
                        letter-spacing: 0.05em;
                        text-transform: uppercase;
                        min-width: 120px;
                    }
                </style>
                <sl-dialog label="Fullscreen Mask Editor" class="editor-dialog" id="editor-dialog">
                    <div class="editor-body">
                        <div class="main-canvas-container">
                            <div class="canvas-wrapper" id="canvas-wrapper">
                                <img id="editor-bg" />
                                <canvas id="mask-canvas"></canvas>
                            </div>
                        </div>
                        <div class="editor-controls">
                            <div class="brush-label">Brush Size: ${this.brushSize}px</div>
                            <sl-range value="${this.brushSize}" min="2" max="300" step="1" id="brush-range" style="width: 200px;"></sl-range>
                            <sl-button variant="neutral" outline id="clear-btn">
                                <sl-icon slot="prefix" name="trash"></sl-icon> Clear
                            </sl-button>
                            <div style="flex:1"></div>
                            <sl-button variant="success" id="close-editor-btn" size="large">
                                <sl-icon slot="prefix" name="check-lg"></sl-icon> Done
                            </sl-button>
                        </div>
                    </div>
                </sl-dialog>
            `;
            this.shadowRoot.appendChild(dialogContainer);
        }

        const dialog = this.shadowRoot.getElementById('editor-dialog');
        const openBtn = this.shadowRoot.getElementById('open-editor-btn');
        const doneBtn = this.shadowRoot.getElementById('close-editor-btn');
        const editorBg = this.shadowRoot.getElementById('editor-bg');
        const brushRange = this.shadowRoot.getElementById('brush-range');
        const clearBtn = this.shadowRoot.getElementById('clear-btn');
        const maskThumb = this.shadowRoot.getElementById('mask-thumb');
        this.maskCanvas = this.shadowRoot.getElementById('mask-canvas');

        this.syncWithSourceImage();

        openBtn.onclick = () => {
            this.syncWithSourceImage();
            dialog.show();
            setTimeout(() => this.syncEditorSize(), 300);
        };

        doneBtn.onclick = () => {
            if (this.maskCanvas) {
                this.savedMaskData = this.maskCanvas.toDataURL();
                if (maskThumb) {
                    maskThumb.src = this.savedMaskData;
                    maskThumb.style.display = 'block';
                }
            }
            dialog.hide();
        };

        editorBg.onload = () => this.syncEditorSize();

        // Remove old listeners if any to avoid stacking
        brushRange.oninput = (e) => {
            this.brushSize = e.target.value;
            const label = this.shadowRoot.querySelector('.brush-label');
            if (label) label.textContent = `Brush Size: ${this.brushSize}px`;
            if (this.ctx) this.ctx.lineWidth = this.brushSize;
        };

        clearBtn.onclick = () => this.clearMask();

        // Ensure brush range shows current value
        if (brushRange) brushRange.value = this.brushSize;

        const getCoords = (e) => {
            const rect = this.maskCanvas.getBoundingClientRect();
            return {
                x: (e.clientX - rect.left) * (this.maskCanvas.width / rect.width),
                y: (e.clientY - rect.top) * (this.maskCanvas.height / rect.height)
            };
        };

        this.maskCanvas.onpointerdown = (e) => {
            this.isDrawing = true;
            const { x, y } = getCoords(e);
            this.ctx.beginPath();
            this.ctx.moveTo(x, y);
            this.draw(e);
        };

        this.maskCanvas.onpointermove = (e) => {
            if (this.isDrawing) this.draw(e);
        };

        window.onpointerup = () => {
            this.isDrawing = false;
            if (this.ctx) this.ctx.beginPath();
        };
    }

    draw(e) {
        if (!this.ctx || !this.isDrawing) return;
        const rect = this.maskCanvas.getBoundingClientRect();
        const x = (e.clientX - rect.left) * (this.maskCanvas.width / rect.width);
        const y = (e.clientY - rect.top) * (this.maskCanvas.height / rect.height);
        this.ctx.lineTo(x, y);
        this.ctx.stroke();
        this.ctx.beginPath();
        this.ctx.moveTo(x, y);
    }
}

customElements.define('qp-inpaint', QpInpaint);

class QpOutpaint extends QpRender {
    constructor() {
        super('sdxl');
        this.expandTop = 0;
        this.expandBottom = 0;
        this.expandLeft = 0;
        this.expandRight = 0;
        this.step = 64;
        this.endpoint = '/outpaint';
    }

    connectedCallback() {
        this.title = "Outpainting";
        this.icon = "arrows-angle-expand";
        super.connectedCallback();
        // Listen for image availability to update preview
        window.addEventListener('image-changed', () => this.render());

        // Polling retry for initial load (in case source loads slower)
        if (!this.checkSrc()) {
            this.imgCheck = setInterval(() => {
                if (this.checkSrc()) {
                    clearInterval(this.imgCheck);
                    this.imgCheck = null;
                    this.hasRendered = false;
                    this.render();
                }
            }, 500);
        }
    }

    disconnectedCallback() {
        if (this.imgCheck) clearInterval(this.imgCheck);
        super.disconnectedCallback();
    }

    checkSrc() {
        const imageSource = document.querySelector('qp-image-input');
        return imageSource?.getImage() || window.qpyt_app?.lastImage;
    }

    getExpandedDims(w, h) {
        return {
            width: w + this.expandLeft + this.expandRight,
            height: h + this.expandTop + this.expandBottom
        };
    }

    async generate() {
        if (!window.qpyt_app) return;
        if (!this._activeTasks) this._activeTasks = 0;
        this._activeTasks++;
        this.isGenerating = true;
        this.render();

        const imageSource = document.querySelector('qp-image-input');
        let srcBase64 = imageSource?.getImage() || null;
        if (!srcBase64 && window.qpyt_app.lastImage) {
            srcBase64 = window.qpyt_app.lastImage; // Auto-pick last
        }

        if (!srcBase64) {
            window.qpyt_app.notify("Outpainting needs a source image!", "warning");
            return;
        }

        // Check expansion
        if (this.expandTop === 0 && this.expandBottom === 0 && this.expandLeft === 0 && this.expandRight === 0) {
            window.qpyt_app.notify("Please set at least one expansion value > 0", "warning");
            return;
        }

        this.isGenerating = true;
        this.hasRendered = false;
        this.render();

        try {
            // Prepared Payload
            const { image, mask, width, height } = await this.preparePayload(srcBase64);

            // Standard Gather Params
            const qpElements = customElements.get ? ['q-upscaler-v3', 'qp-prompt', 'qp-render-sdxl'].map(tag => `${tag}: ${customElements.get(tag) ? 'YES' : 'NO'}`) : 'N/A';
            console.log("[App] Custom Registry Check:", qpElements);
            const promptEl = document.querySelector('qp-prompt');
            const loraManager = document.querySelector('qp-lora-manager');
            const settingsEl = document.querySelector('qp-settings');
            const { prompt, negative_prompt } = promptEl ? promptEl.getValue() : { prompt: "", negative_prompt: "" };
            const settings = settingsEl ? settingsEl.getValue() : {};

            console.log(`[Outpaint] Sending request: ${width}x${height}`);

            // Force Denoising Strength to 1.0 for Outpainting
            const forceStrength = 1.0;

            const payload = {
                prompt,
                negative_prompt,
                model_type: this.modelType,
                model_name: this.selectedModel,
                image: image,
                mask: mask,
                width: width,
                height: height,
                ...settings,
                denoising_strength: forceStrength,
                loras: loraManager ? loraManager.getValues().loras : []
            };

            const data = await this.submitAndPollTask('/inpaint', payload);
            if (!data) return; // Interrupted

            console.log("[Outpaint] Completed:", data);
            this.lastImageUrl = data.image_url;

            // Show Warnings if any
            if (data.warnings && data.warnings.length > 0) {
                data.warnings.forEach(w => window.qpyt_app.notify(w, "warning"));
            }

            if (window.qpyt_app) window.qpyt_app.lastImage = this.lastImageUrl;
            // Signal global output
            window.dispatchEvent(new CustomEvent('qpyt-output', {
                detail: { url: this.lastImageUrl, brickId: this.getAttribute('brick-id') },
                bubbles: true,
                composed: true
            }));
            window.qpyt_app.notify("Outpainting complete!", "success");

        } catch (e) {
            console.error(e);
            window.qpyt_app.notify("Outpainting failed", "danger");
        } finally {
            this._activeTasks--;
            if (this._activeTasks <= 0) {
                this._activeTasks = 0;
                this.isGenerating = false;
            }
            this.hasRendered = false;
            this.render();
        }
    }

    preparePayload(srcBase64) {
        return new Promise((resolve) => {
            const img = new Image();
            img.onload = () => {
                const w = img.width;
                const h = img.height;
                const newW = w + this.expandLeft + this.expandRight;
                const newH = h + this.expandTop + this.expandBottom;

                // 1. Create Expanded Image Canvas
                const imgCanvas = document.createElement('canvas');
                imgCanvas.width = newW;
                imgCanvas.height = newH;
                const ctx = imgCanvas.getContext('2d');

                // Fill with neutral grey for new areas (helps diffusion start)
                ctx.fillStyle = "#808080";
                ctx.fillRect(0, 0, newW, newH);

                // Draw source image in center/offset
                ctx.drawImage(img, this.expandLeft, this.expandTop);

                const finalImageBase64 = imgCanvas.toDataURL('image/png');

                // 2. Create Mask Canvas
                const maskCanvas = document.createElement('canvas');
                maskCanvas.width = newW;
                maskCanvas.height = newH;
                const mCtx = maskCanvas.getContext('2d');

                // Fill White (Inpaint Area)
                mCtx.fillStyle = "white";
                mCtx.fillRect(0, 0, newW, newH);

                // Draw Black Rect (Protected Area)
                mCtx.fillStyle = "black";
                mCtx.fillRect(this.expandLeft, this.expandTop, w, h);

                const finalMaskBase64 = maskCanvas.toDataURL('image/png');

                resolve({
                    image: finalImageBase64,
                    mask: finalMaskBase64,
                    width: newW,
                    height: newH
                });
            };
            img.src = srcBase64;
        });
    }

    updateUI() {
        const update = (dir, val, prop) => {
            const el = this.shadowRoot.querySelector(`.vis-${dir}`);
            if (el) {
                el.textContent = val;
                el.style[prop] = (val / 2) + 'px';
            }
            const range = this.shadowRoot.getElementById(`r-${dir}`);
            if (range) {
                const label = range.previousElementSibling?.querySelectorAll('span')[1];
                if (label) label.textContent = `${val}px`;
            }
        };

        update('top', this.expandTop, 'height');
        update('btm', this.expandBottom, 'height');
        update('left', this.expandLeft, 'width');
        update('right', this.expandRight, 'width');
    }

    render() {
        if (this.hasRendered) {
            this.updateUI();
            return;
        }
        this.hasRendered = true;

        const brickId = this.getAttribute('brick-id') || '';

        // Compute preview dims if image available
        const srcImg = this.checkSrc();
        const hasImage = !!srcImg;

        this.shadowRoot.innerHTML = `
            <style>
                .exp-grid {
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 1rem;
                    margin-top: 1rem;
                }
                .preview-zone {
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    background: #0f172a;
                    border: 1px dashed #334155;
                    border-radius: 8px;
                    padding: 1rem;
                    min-height: 200px;
                    position: relative;
                }
                .box-center {
                    width: 60px; height: 60px;
                    background: #475569;
                    display: flex; align-items: center; justify-content: center;
                    color: white; font-size: 0.8rem;
                    border: 1px solid #94a3b8;
                    position: relative;
                    z-index: 2;
                }
                .indicator {
                    position: absolute;
                    background: rgba(16, 185, 129, 0.2);
                    border: 1px solid rgba(16, 185, 129, 0.5);
                    display: flex; align-items: center; justify-content: center;
                    font-size: 0.7rem; color: #10b981;
                    /* Transitions for smooth updates */
                    transition: all 0.1s ease-out;
                }
                /* Initial Positions */
                .vis-top { bottom: 100%; left: 0; right: 0; height: 0px; }
                .vis-btm { top: 100%; left: 0; right: 0; height: 0px; }
                .vis-left { right: 100%; top: 0; bottom: 0; width: 0px; }
                .vis-right { left: 100%; top: 0; bottom: 0; width: 0px; }

                /* Native Range Styling */
                .range-group {
                    display: flex;
                    flex-direction: column;
                    gap: 0.2rem;
                }
                .range-label {
                    font-size: 0.8rem;
                    color: #cbd5e1;
                    display: flex;
                    justify-content: space-between;
                }
                input[type=range] {
                    width: 100%;
                    accent-color: #3b82f6;
                }
            </style>
            <qp-cartridge title="${this.title}" icon="${this.icon}" type="generator" brick-id="${brickId}">
                <div style="padding: 0.5rem; display: flex; flex-direction: column; gap: 1rem;">
                    
                    <sl-select size="small" placeholder="Select Model" value="${this.selectedModel}" id="model-select" hoist>
                         ${this.models.length > 0
                ? this.models.map(m => `<sl-option value="${m}">${m}</sl-option>`).join('')
                : `<sl-option value="" disabled>Loading models...</sl-option>`
            }
                    </sl-select>

                    ${!hasImage ? `<div style="color:orange; font-size:0.9rem;">⚠️ No source image found</div>` : ''}

                    <div class="preview-zone">
                        <div class="box-center">
                            IMG
                            <div class="indicator vis-top"></div>
                            <div class="indicator vis-btm"></div>
                            <div class="indicator vis-left"></div>
                            <div class="indicator vis-right"></div>
                        </div>
                    </div>

                    <div class="exp-grid">
                        <div class="range-group">
                            <div class="range-label"><span>Top</span> <span>${this.expandTop}px</span></div>
                            <input type="range" min="0" max="512" step="64" value="${this.expandTop}" id="r-top">
                        </div>
                        <div class="range-group">
                            <div class="range-label"><span>Bottom</span> <span>${this.expandBottom}px</span></div>
                            <input type="range" min="0" max="512" step="64" value="${this.expandBottom}" id="r-btm">
                        </div>
                        <div class="range-group">
                            <div class="range-label"><span>Left</span> <span>${this.expandLeft}px</span></div>
                            <input type="range" min="0" max="512" step="64" value="${this.expandLeft}" id="r-left">
                        </div>
                        <div class="range-group">
                            <div class="range-label"><span>Right</span> <span>${this.expandRight}px</span></div>
                            <input type="range" min="0" max="512" step="64" value="${this.expandRight}" id="r-right">
                        </div>
                    </div>

                    <div style="display: flex; gap: 0.5rem; width: 100%; position: relative;">
                        <sl-button variant="primary" size="medium" style="flex: 3;" id="btn-gen">
                            <sl-icon slot="prefix" name="arrows-angle-expand"></sl-icon>
                            Outpaint
                        </sl-button>
                        <sl-button variant="danger" size="medium" style="flex: 1;" id="btn-stop" outline>
                            <sl-icon name="stop-fill"></sl-icon>
                        </sl-button>

                        ${this.isGenerating ? `
                            <div style="position: absolute; top: -30px; right: 0; background: #10b981; color: white; padding: 2px 8px; border-radius: 10px; font-size: 0.65rem; font-weight: 800; z-index: 10;">
                                ${this._activeTasks || 1} JOBS
                            </div>
                        ` : ''}
                    </div>
                </div>
            </qp-cartridge>
        `;

        // Initial UI Update
        this.updateUI();

        const bind = (id, prop) => {
            const el = this.shadowRoot.getElementById(id);
            if (el) {
                el.addEventListener('input', (e) => {
                    this[prop] = parseInt(e.target.value);
                    this.updateUI(); // Update UI without re-rendering DOM
                });
            }
        };
        bind('r-top', 'expandTop');
        bind('r-btm', 'expandBottom');
        bind('r-left', 'expandLeft');
        bind('r-right', 'expandRight');

        // Bind Model Select
        const modelSel = this.shadowRoot.getElementById('model-select');
        if (modelSel) {
            modelSel.addEventListener('sl-change', (e) => {
                this.selectedModel = e.target.value;
            });
        }

        this.shadowRoot.getElementById('btn-gen').onclick = () => this.generate();
        const stopBtn = this.shadowRoot.getElementById('btn-stop');
        if (stopBtn) stopBtn.onclick = () => this.isGenerating = false;
    }
}
customElements.define('qp-outpaint', QpOutpaint);

// Tiled Upscaler Brick
// Tiled Upscaler Brick
// Tiled Upscaler Brick
class UpscalerV3 extends HTMLElement {
    constructor() {
        super();
        console.log("[UpscalerV3] Constructor starting...");
        this.attachShadow({ mode: 'open' });

        this.modelType = 'upscale';
        this.selectedModel = '';
        this.selectedSampler = "dpm++_2m_sde_karras";
        this.denoisingStrength = 0.2;
        this.upscaleFactor = 2;
        this.tileSize = 768;

        this.vaes = [];
        this.samplers = [];
        this.models = [];
        this.isGenerating = false;
        this.lastImageUrl = '';
        this.currentStep = 0;
        this.totalSteps = 0;
        this.hasRendered = false;
        this.endpoint = "/upscale";
        console.log("[UpscalerV3] Constructor finished successfully.");
    }

    connectedCallback() {
        this.render();
        this.fetchModels();
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
                    select.value = this.selectedModel;
                }
            }
        } catch (e) {
            console.error(e);
        }
    }

    async submitAndPollTask(endpoint, payload) {
        try {
            const response = await fetch(endpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const submission = await response.json();
            if (submission.status === 'queued') {
                const taskId = submission.task_id;
                while (this.isGenerating) {
                    const statusResp = await fetch(`/queue/status/${taskId}`);
                    const task = await statusResp.json();
                    if (task.status === 'COMPLETED') return task.result;
                    if (task.status === 'FAILED') throw new Error(task.error);
                    await new Promise(r => setTimeout(r, 1000));
                }
            }
        } catch (e) { console.error(e); throw e; }
    }

    async generate() {
        if (!window.qpyt_app) return;
        this.isGenerating = true;
        this.render();

        try {
            const promptEl = document.querySelector('qp-prompt');
            const { prompt, negative_prompt } = promptEl ? promptEl.getValue() : { prompt: "", negative_prompt: "" };

            const imageSource = document.querySelector('qp-image-input');
            let image = imageSource?.getImage() || window.qpyt_app?.lastImage;

            if (!image) {
                window.qpyt_app.notify("Upscaler requires an image!", "danger");
                this.isGenerating = false;
                this.render();
                return;
            }

            const payload = {
                prompt, negative_prompt,
                model_type: 'upscale',
                model_name: this.selectedModel,
                image: image,
                upscale_factor: this.upscaleFactor,
                denoising_strength: this.denoisingStrength,
                tile_size: this.tileSize,
                output_format: "png"
            };

            const data = await this.submitAndPollTask('/upscale', payload);
            if (data) {
                this.lastImageUrl = data.image_url;
                if (window.qpyt_app) window.qpyt_app.lastImage = this.lastImageUrl;
                const dashboard = document.querySelector('qp-dashboard');
                if (dashboard) dashboard.addEntry(data);

                window.dispatchEvent(new CustomEvent('qpyt-output', {
                    detail: { url: this.lastImageUrl, brickId: this.getAttribute('brick-id') },
                    bubbles: true,
                    composed: true
                }));

                window.qpyt_app.notify("Upscale complete", "success");
            }
        } catch (e) {
            console.error(e);
            window.qpyt_app.notify("Upscale failed", "danger");
        } finally {
            this.isGenerating = false;
            this.render();
        }
    }

    render() {
        const brickId = this.getAttribute('brick-id') || '';
        this.shadowRoot.innerHTML = `
            <style>
                .upscale-container { display: flex; flex-direction: column; gap: 0.8rem; padding: 0.5rem; }
            </style>
            <qp-cartridge title="Tiled Upscaler" icon="aspect-ratio" type="generator" brick-id="${brickId}">
                <div class="upscale-container">
                    <sl-select id="model-select" label="Upscaler Model" value="${this.selectedModel}" hoist>
                        ${this.models.map(m => `<sl-option value="${m}">${m}</sl-option>`).join('')}
                    </sl-select>

                    <sl-input type="number" label="Scale Factor" value="${this.upscaleFactor}" min="1.1" step="0.1" id="scale-input"></sl-input>
                    <sl-input type="number" label="Denoise" value="${this.denoisingStrength}" min="0.05" max="1" step="0.05" id="denoise-input"></sl-input>

                    <sl-button variant="primary" id="btn-gen" ?loading="${this.isGenerating}">
                        <sl-icon slot="prefix" name="aspect-ratio"></sl-icon>
                        Upscale
                    </sl-button>
                </div>
            </qp-cartridge>
        `;

        this.shadowRoot.getElementById('btn-gen').addEventListener('click', () => this.generate());

        const scaleInput = this.shadowRoot.getElementById('scale-input');
        if (scaleInput) scaleInput.addEventListener('sl-change', (e) => this.upscaleFactor = parseFloat(e.target.value));

        const denoiseInput = this.shadowRoot.getElementById('denoise-input');
        if (denoiseInput) denoiseInput.addEventListener('sl-change', (e) => this.denoisingStrength = parseFloat(e.target.value));

        const modelSelect = this.shadowRoot.getElementById('model-select');
        if (modelSelect) modelSelect.addEventListener('sl-change', (e) => this.selectedModel = e.target.value);
    }
}
console.log("[UpscalerV3] Registering 'q-upscaler-v3'...");
customElements.define('q-upscaler-v3', UpscalerV3);

class QpLegacyUpscaler extends HTMLElement {
    constructor() { super(); this.attachShadow({ mode: 'open' }); }
    connectedCallback() {
        this.shadowRoot.innerHTML = `<div style="padding:1rem;color:orange;border:1px solid orange">⚠️ Legacy 'qp-upscaler' brick detected.<br>Please remove it and add 'Tiled Upscaler' from the library.</div>`;
    }
}
customElements.define('qp-upscaler', QpLegacyUpscaler);

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
                .image-box img, .image-box video {
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
                        ${this.currentUrl ? (this.currentUrl.toLowerCase().endsWith('.mp4') ? `
                            <video src="${this.currentUrl}" controls autoplay loop></video>
                        ` : `
                            <img src="${this.currentUrl}" alt="Final image">
                        `) : `
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

// Background Removal Brick
class QpRembg extends HTMLElement {
    static get observedAttributes() { return ['brick-id']; }
    constructor() {
        super();
        this.attachShadow({ mode: 'open' });
        this.isGenerating = false;
        this.lastImageUrl = "";
        this.endpoint = "/rembg";
    }

    connectedCallback() {
        this.render();
    }

    async submitAndPollTask(endpoint, payload) {
        try {
            const response = await fetch(endpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            const submission = await response.json();
            if (submission.status === 'queued') {
                const taskId = submission.task_id;
                while (true) {
                    if (!this.isGenerating) return null;
                    const statusResp = await fetch(`/queue/status/${taskId}`);
                    const task = await statusResp.json();
                    if (task.status === 'COMPLETED') {
                        if (task.result && typeof task.result === 'object') {
                            task.result.request_id = taskId;
                        }
                        return task.result;
                    }
                    if (task.status === 'FAILED' || task.status === 'CANCELLED') {
                        throw new Error(task.error || "Task failed");
                    }
                    await new Promise(r => setTimeout(r, 1000));
                }
            }
            return submission.data;
        } catch (e) {
            console.error("[Queue Polling Error]", e);
            throw e;
        }
    }

    async generate() {
        if (!window.qpyt_app) return;
        if (!this._activeTasks) this._activeTasks = 0;
        this._activeTasks++;
        this.isGenerating = true;
        this.render();

        const imageSource = document.querySelector('qp-image-input');
        let image = imageSource?.getImage() || null;

        if (!image && window.qpyt_app?.lastImage) {
            image = window.qpyt_app.lastImage;
            console.log("[REMBG] Auto-picking last generated image:", image);
        }

        if (!image) {
            window.qpyt_app.notify("REMBG requires a Source Image or a previous generation!", "danger");
            this.isGenerating = false;
            this.render();
            return;
        }

        try {
            const data = await this.submitAndPollTask('/rembg', { image });
            if (data) {
                this.lastImageUrl = data.image_url;
                if (window.qpyt_app) window.qpyt_app.lastImage = this.lastImageUrl;

                // Signal global output
                window.dispatchEvent(new CustomEvent('qpyt-output', {
                    detail: { url: this.lastImageUrl, brickId: this.getAttribute('brick-id') },
                    bubbles: true,
                    composed: true
                }));

                const dashboard = document.querySelector('qp-dashboard');
                if (dashboard) dashboard.addEntry(data);
                window.qpyt_app.notify("Background removed!", "success");
            }
        }
        catch (e) {
            console.error(e);
            window.qpyt_app.notify("Connection error", "danger");
        } finally {
            this._activeTasks--;
            if (this._activeTasks <= 0) {
                this._activeTasks = 0;
                this.isGenerating = false;
            }
            this.render();
        }
    }

    getValue() {
        return { lastImageUrl: this.lastImageUrl };
    }
    setValues(values) {
        if (!values) return;
        this.lastImageUrl = values.lastImageUrl || "";
        this.render();
    }

    render() {
        const brickId = this.getAttribute('brick-id') || '';
        this.shadowRoot.innerHTML = `
            <style>
                .rembg-container {
                    display: flex;
                    flex-direction: column;
                    gap: 1.2rem;
                    padding: 0.5rem;
                }
                .status-badge {
                    background: rgba(16, 185, 129, 0.1);
                    border: 1px solid rgba(16, 185, 129, 0.2);
                    padding: 0.8rem;
                    border-radius: 8px;
                    text-align: center;
                    font-size: 0.85rem;
                    color: #10b981;
                }
            </style>
            <qp-cartridge title="Background Removal" icon="scissors" type="generator" brick-id="${brickId}">
                <div class="rembg-container">
                    <div class="status-badge">
                        <sl-icon name="info-circle" style="margin-right:0.5rem"></sl-icon>
                        Input: Last generation or Source
                    </div>
                    
                    <div style="display: flex; gap: 0.5rem; width: 100%; position: relative;">
                        <sl-button variant="primary" size="medium" id="btn-gen" style="flex: 3;">
                            <sl-icon slot="prefix" name="scissors"></sl-icon>
                            Extract Foreground
                        </sl-button>
                        <sl-button variant="danger" size="medium" id="btn-stop" outline style="flex: 1;">
                            <sl-icon name="stop-fill"></sl-icon>
                        </sl-button>

                        ${this.isGenerating ? `
                            <div style="position: absolute; top: -30px; right: 0; background: #10b981; color: white; padding: 2px 8px; border-radius: 10px; font-size: 0.65rem; font-weight: 800; z-index: 10;">
                                ${this._activeTasks || 1} JOBS
                            </div>
                        ` : ''}
                    </div>
                    
                    ${this.isGenerating ? `
                        <div style="font-size: 0.75rem; color: #94a3b8; text-align: center; font-style: italic;">
                            Processing Mask (U2NET)...
                        </div>
                    ` : ''}
                </div>
            </qp-cartridge>
        `;

        this.shadowRoot.getElementById('btn-gen').onclick = () => this.generate();
        const stopBtn = this.shadowRoot.getElementById('btn-stop');
        if (stopBtn) stopBtn.onclick = () => this.isGenerating = false;
    }
}
customElements.define('qp-rembg', QpRembg);

// SVG Vectorization Brick
class QpVectorize extends HTMLElement {
    static get observedAttributes() { return ['brick-id']; }
    constructor() {
        super();
        this.attachShadow({ mode: 'open' });
        this.isGenerating = false;
        this.lastSvgUrl = "";
        this.selectedMode = "spline";
        this.colorPrecision = 8;
        this.colorMode = "color";
    }

    connectedCallback() {
        this.render();
    }

    async generate() {
        if (!window.qpyt_app) return;
        if (!this._activeTasks) this._activeTasks = 0;
        this._activeTasks++;
        this.isGenerating = true;
        this.render();

        if (typeof ImageTracer === 'undefined') {
            window.qpyt_app?.notify("ImageTracer library not loaded yet.", "warning");
            return;
        }

        const imageSource = document.querySelector('qp-image-input');
        let image = imageSource?.getImage() || null;

        if (!image && window.qpyt_app?.lastImage) {
            image = window.qpyt_app.lastImage;
        }

        if (!image) {
            window.qpyt_app.notify("Vectorization requires an image!", "danger");
            return;
        }

        this.isGenerating = true;
        this.render();

        const options = {
            ltres: this.selectedMode === 'spline' ? 1 : 10,
            qtres: this.selectedMode === 'spline' ? 1 : 10,
            pathomit: 8,
            colorsampling: 1,
            numberofcolors: parseInt(this.colorPrecision),
            mincolorratio: 0,
            colorquantcycles: 3,
            scale: 1,
            lcpr: 0,
            qcpr: 0,
            desc: false,
            viewbox: true
        };

        if (this.colorMode === 'binary') {
            options.colorsampling = 0;
            options.numberofcolors = 2;
        }

        setTimeout(() => {
            try {
                ImageTracer.imageToSVG(image, (svgString) => {
                    const blob = new Blob([svgString], { type: 'image/svg+xml' });
                    if (this.lastSvgUrl) URL.revokeObjectURL(this.lastSvgUrl);
                    this.lastSvgUrl = URL.createObjectURL(blob);

                    window.dispatchEvent(new CustomEvent('qpyt-output', {
                        detail: { url: this.lastSvgUrl, brickId: this.getAttribute('brick-id'), params: { extension: 'svg' } },
                        bubbles: true,
                        composed: true
                    }));

                    const dashboard = document.querySelector('qp-dashboard');
                    if (dashboard) {
                        dashboard.addEntry({
                            request_id: 'vec-' + Date.now(),
                            image_url: this.lastSvgUrl,
                            prompt: "SVG Vectorization",
                            metadata: { mode: this.selectedMode, colors: this.colorPrecision },
                            status: 'success'
                        });
                    }

                    window.qpyt_app.notify("SVG Vectorization complete!", "success");
                    this.isGenerating = false;
                    this.render();
                }, options);
            } catch (e) {
                console.error("[Vectorize] Error:", e);
                window.qpyt_app.notify("Vectorization failed", "danger");
            } finally {
                this._activeTasks--;
                if (this._activeTasks <= 0) {
                    this._activeTasks = 0;
                    this.isGenerating = false;
                }
                this.render();
            }
        }, 50);
    }

    getValue() {
        return {
            lastSvgUrl: this.lastSvgUrl,
            selectedMode: this.selectedMode,
            colorPrecision: this.colorPrecision,
            colorMode: this.colorMode
        };
    }
    setValues(values) {
        if (!values) return;
        if (values.lastSvgUrl !== undefined) this.lastSvgUrl = values.lastSvgUrl;
        if (values.selectedMode !== undefined) this.selectedMode = values.selectedMode;
        if (values.colorPrecision !== undefined) this.colorPrecision = values.colorPrecision;
        if (values.colorMode !== undefined) this.colorMode = values.colorMode;
        this.render();
    }

    render() {
        const brickId = this.getAttribute('brick-id') || '';
        this.shadowRoot.innerHTML = `
            <style>
                .vec-container {
                    display: flex;
                    flex-direction: column;
                    gap: 1.2rem;
                    padding: 0.5rem;
                }
                .hint {
                    font-size: 0.75rem;
                    color: #94a3b8;
                    margin-top: -0.5rem;
                }
                sl-range::part(base) {
                    --track-color-active: #f59e0b;
                }
                .preview-svg {
                    width: 100%;
                    max-height: 200px;
                    background: white;
                    border-radius: 8px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    overflow: hidden;
                    border: 1px solid rgba(255,255,255,0.1);
                }
                .preview-svg img {
                    max-width: 100%;
                    max-height: 100%;
                    object-fit: contain;
                }
            </style>
            <qp-cartridge title="Vectorize (SVG)" icon="vector-pen" type="generator" brick-id="${brickId}">
                <div class="vec-container">
                    <sl-select id="mode-select" label="Drawing Mode" value="${this.selectedMode}" hoist>
                        <sl-option value="spline">Splines (Smooth curves)</sl-option>
                        <sl-option value="polygon">Polygons (Sharp lines)</sl-option>
                    </sl-select>

                    <sl-select id="type-select" label="Color Type" value="${this.colorMode}" hoist>
                        <sl-option value="color">Full Color</sl-option>
                        <sl-option value="binary">Black & White</sl-option>
                    </sl-select>

                    <sl-range id="prec-range" label="Color Precision" min="2" max="64" step="1" value="${this.colorPrecision}" 
                              help-text="${this.colorPrecision} colors targets. More colors = more paths."></sl-range>
                    
                    <div style="display: flex; gap: 0.5rem; width: 100%; position: relative;">
                        <sl-button variant="primary" size="medium" id="btn-gen" style="flex: 3;">
                            <sl-icon slot="prefix" name="vector-pen"></sl-icon>
                            Vectorize
                        </sl-button>
                        <sl-button variant="danger" size="medium" id="btn-stop" outline style="flex: 1;">
                            <sl-icon name="stop-fill"></sl-icon>
                        </sl-button>

                        ${this.isGenerating ? `
                            <div style="position: absolute; top: -30px; right: 0; background: #10b981; color: white; padding: 2px 8px; border-radius: 10px; font-size: 0.65rem; font-weight: 800; z-index: 10;">
                                ${this._activeTasks || 1} JOBS
                            </div>
                        ` : ''}
                    </div>

                    ${this.lastSvgUrl ? `
                        <div class="preview-svg">
                            <img src="${this.lastSvgUrl}" alt="SVG Preview">
                        </div>
                        <sl-button variant="success" size="small" outline download="vectorized_image.svg" href="${this.lastSvgUrl}" style="width: 100%;">
                            <sl-icon slot="prefix" name="download"></sl-icon>
                            Download SVG
                        </sl-button>
                    ` : ''}
                </div>
            </qp-cartridge>
        `;

        this.shadowRoot.getElementById('mode-select')?.addEventListener('sl-change', (e) => {
            this.selectedMode = e.target.value;
            this.render();
        });
        this.shadowRoot.getElementById('type-select')?.addEventListener('sl-change', (e) => {
            this.colorMode = e.target.value;
            this.render();
        });
        this.shadowRoot.getElementById('prec-range')?.addEventListener('sl-change', (e) => {
            this.colorPrecision = e.target.value;
            this.render(); // Re-render for help-text
        });
        this.shadowRoot.getElementById('btn-gen').onclick = () => this.generate();
        const stopBtn = this.shadowRoot.getElementById('btn-stop');
        if (stopBtn) stopBtn.onclick = () => this.isGenerating = false;
    }
}
customElements.define('qp-vectorize', QpVectorize);

// Save to Disk Brick
class QpSaveToDisk extends HTMLElement {
    static get observedAttributes() { return ['brick-id']; }
    constructor() {
        super();
        this.attachShadow({ mode: 'open' });
        this.isSaving = false;
        this.exportPath = "C:/QpytExports";
        this.filenamePattern = "gen_{seed}_{date}_{time}";
        this.selectedFormat = "png";
        this.lastParams = {};
    }

    connectedCallback() {
        this.render();
        window.addEventListener('qpyt-output', (e) => {
            if (e.detail.params) {
                this.lastParams = e.detail.params;
                console.log("[SaveToDisk] Captured params:", this.lastParams);
            }
        });
    }

    async doSave() {
        if (this.isSaving || !window.qpyt_app) return;

        const pathInput = this.shadowRoot.getElementById('path-input');
        const patternInput = this.shadowRoot.getElementById('pattern-input');
        const formatSelect = this.shadowRoot.getElementById('format-select');

        this.exportPath = pathInput.value;
        this.filenamePattern = patternInput.value;
        this.selectedFormat = formatSelect.value;

        const imageSource = document.querySelector('qp-image-input');
        let imageUrl = window.qpyt_app?.lastImage || "";

        if (!imageUrl) {
            window.qpyt_app.notify("No image to save!", "warning");
            return;
        }

        this.isSaving = true;
        this.render();

        try {
            const response = await fetch('/save-to-disk', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    image_url: imageUrl,
                    custom_path: this.exportPath,
                    pattern: this.filenamePattern,
                    output_format: this.selectedFormat,
                    params: this.lastParams
                })
            });

            const result = await response.json();
            if (result.status === 'success') {
                window.qpyt_app.notify("Image saved successfully!", "success");
            } else {
                window.qpyt_app.notify(`Error: ${result.message}`, "danger");
            }
        } catch (e) {
            console.error(e);
            window.qpyt_app.notify("Connection error", "danger");
        } finally {
            this.isSaving = false;
            this.render();
        }
    }

    getValue() {
        return {
            exportPath: this.exportPath,
            filenamePattern: this.filenamePattern,
            selectedFormat: this.selectedFormat,
            lastParams: this.lastParams
        };
    }
    setValues(values) {
        if (!values) return;
        if (values.exportPath !== undefined) this.exportPath = values.exportPath;
        if (values.filenamePattern !== undefined) this.filenamePattern = values.filenamePattern;
        if (values.selectedFormat !== undefined) this.selectedFormat = values.selectedFormat;
        if (values.lastParams !== undefined) this.lastParams = values.lastParams;
        this.render();
    }

    render() {
        const brickId = this.getAttribute('brick-id') || '';
        this.shadowRoot.innerHTML = `
            <style>
                .save-container {
                    display: flex;
                    flex-direction: column;
                    gap: 1.2rem;
                    padding: 0.5rem;
                }
                .hint {
                    font-size: 0.75rem;
                    color: #94a3b8;
                    margin-top: -0.5rem;
                    line-height: 1.4;
                }
                .hint code {
                    color: #f59e0b;
                    background: rgba(245, 158, 11, 0.1);
                    padding: 0 4px;
                    border-radius: 4px;
                }
            </style>
            <qp-cartridge title="Save to Disk" icon="download" type="output" brick-id="${brickId}">
                <div class="save-container">
                    <sl-input id="path-input" label="Export Directory" value="${this.exportPath}" help-text="Full path on your PC"></sl-input>
                    
                    <sl-input id="pattern-input" label="Filename Pattern" value="${this.filenamePattern}"></sl-input>
                    <div class="hint">
                        Tokens: <code>{seed}</code>, <code>{date}</code>, <code>{time}</code>, <code>{uuid}</code>, <code>{ext}</code>
                    </div>

                    <sl-select id="format-select" label="Output Format" value="${this.selectedFormat}" hoist>
                        <sl-option value="png">PNG</sl-option>
                        <sl-option value="jpeg">JPEG</sl-option>
                        <sl-option value="webp">WebP</sl-option>
                    </sl-select>

                    <sl-button variant="primary" size="medium" id="btn-save" 
                               ?loading="${this.isSaving}" style="width: 100%;">
                        <sl-icon slot="prefix" name="save"></sl-icon>
                        Save Current Result
                    </sl-button>
                </div>
            </qp-cartridge>
        `;

        this.shadowRoot.getElementById('btn-save').addEventListener('click', () => this.doSave());
    }
}
customElements.define('qp-save-to-disk', QpSaveToDisk);

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
        const btn = this.shadowRoot.getElementById('btn-translate');
        if (btn) btn.loading = true;
        this.updateResultArea();

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
            const btnFinal = this.shadowRoot.getElementById('btn-translate');
            if (btnFinal) btnFinal.loading = false;
            this.updateResultArea();
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

    getValue() {
        const input = this.shadowRoot.getElementById('fr-prompt');
        return {
            inputValue: input ? input.value : "",
            result: this.result
        };
    }
    setValues(values) {
        if (!values) return;
        this.result = values.result || "";

        // Update Result Area (Dynamic DOM)
        this.updateResultArea();

        const input = this.shadowRoot.getElementById('fr-prompt');
        if (input && values.inputValue !== undefined) input.value = values.inputValue;
    }

    render() {
        if (this.shadowRoot.getElementById('fr-prompt')) return; // Already rendered

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
                    margin-bottom: 0.5rem;
                }
            </style>
            <qp-cartridge title="Translator (FR ➔ EN)" type="input" brick-id="${brickId}">
                <div class="translator-container">
                    <sl-textarea id="fr-prompt" label="French Prompt" placeholder="Entrez votre prompt en français..." resize="auto" size="small"></sl-textarea>
                    
                    <sl-button variant="primary" id="btn-translate" ?loading="${this.isLoading}" size="small">
                        <sl-icon slot="prefix" name="translate"></sl-icon>
                        Translate
                    </sl-button>

                    <div id="result-container"></div>
                </div>
            </qp-cartridge>
        `;

        this.shadowRoot.getElementById('btn-translate')?.addEventListener('click', () => this.translate());
        this.updateResultArea();
    }

    updateResultArea() {
        const container = this.shadowRoot.getElementById('result-container');
        if (!container) return;

        if (this.result) {
            container.innerHTML = `
                <div class="result-area">${this.result}</div>
                <sl-button variant="neutral" id="btn-inject" size="small" outline style="width: 100%;">
                    <sl-icon slot="prefix" name="chat-left-text"></sl-icon>
                    Inject to Prompt
                </sl-button>
            `;
            container.querySelector('#btn-inject')?.addEventListener('click', () => this.inject());
        } else {
            container.innerHTML = '';
        }
    }
}
customElements.define('qp-translator', QpTranslator);

// Styles Selector Brick
class QpStyles extends HTMLElement {
    static get observedAttributes() { return ['brick-id']; }
    constructor() {
        super();
        this.attachShadow({ mode: 'open' });
        this.allStyles = [];
        this.selectedKeys = ["", "", "", ""];
    }

    connectedCallback() {
        this.fetchStyles();
    }

    async fetchStyles() {
        try {
            const res = await fetch('/styles');
            this.allStyles = await res.json();
            this.render();
        } catch (e) {
            console.error("Failed to fetch styles", e);
        }
    }

    getValue() {
        return { selectedKeys: [...this.selectedKeys] };
    }

    setValues(values) {
        if (values && values.selectedKeys) {
            this.selectedKeys = [...values.selectedKeys];

            // Check if already rendered
            if (this.shadowRoot.getElementById('style-select-0')) {
                // Update values directly without destroying DOM
                [0, 1, 2, 3].forEach(i => {
                    const el = this.shadowRoot.getElementById(`style-select-${i}`);
                    if (el) el.value = this.selectedKeys[i];
                });
            } else {
                this.render();
            }
        }
    }

    applyStyles(prompt, negative_prompt) {
        let finalPrompt = prompt || "";
        let finalNegative = negative_prompt || "";

        console.log(`[Styles] Initial Prompt: "${finalPrompt}"`);

        for (const key of this.selectedKeys) {
            if (!key || key === "styles_nom_Aucun_style") continue;
            // Match with underscores to handle Shoelace's automatic space replacement
            const style = this.allStyles.find(s => s.key.replace(/ /g, '_') === key);
            if (style) {
                console.log(`[Styles] Applying style: ${style.name}`);
                if (style.prompt) {
                    if (style.prompt.includes('{prompt}')) {
                        finalPrompt = style.prompt.replace('{prompt}', finalPrompt);
                    } else {
                        // Concatenate as prefix if placeholder is missing
                        finalPrompt = finalPrompt ? `${style.prompt}, ${finalPrompt}` : style.prompt;
                    }
                }
                if (style.negative_prompt) {
                    finalNegative = finalNegative ? `${finalNegative}, ${style.negative_prompt}` : style.negative_prompt;
                }
            }
        }

        console.log(`[Styles] Final Prompt: "${finalPrompt}"`);
        return { prompt: finalPrompt, negative_prompt: finalNegative };
    }

    render() {
        if (this.shadowRoot.getElementById('selectors-grid')) return; // Already rendered

        const brickId = this.getAttribute('brick-id') || '';
        // Sanitize values for Shoelace (no spaces in sl-option values)
        const optionsHtml = this.allStyles.map(s => `<sl-option value="${s.key.replace(/ /g, '_')}">${s.name}</sl-option>`).join('');

        this.shadowRoot.innerHTML = `
            <style>
                .styles-container {
                    display: flex;
                    flex-direction: column;
                    gap: 0.8rem;
                    padding: 0.5rem;
                }
                sl-select::part(listbox) {
                    max-height: 300px;
                }
            </style>
            <qp-cartridge title="Styles (Fooocus)" type="input" icon="palette" brick-id="${brickId}">
                <div class="styles-container" id="selectors-grid">
                    ${[0, 1, 2, 3].map(i => `
                        <sl-select id="style-select-${i}" value="${this.selectedKeys[i]}" size="small" clearable placeholder="Select a style..." hoist>
                            <sl-option value="">None</sl-option>
                            ${optionsHtml}
                        </sl-select>
                    `).join('')}
                    <div style="font-size: 0.7rem; color: #94a3b8; font-style: italic; margin-top: 0.2rem;">
                        Combine up to 4 styles (nested logic).
                    </div>
                </div>
            </qp-cartridge>
        `;

        [0, 1, 2, 3].forEach(i => {
            const el = this.shadowRoot.getElementById(`style-select-${i}`);
            el?.addEventListener('sl-change', (e) => {
                this.selectedKeys[i] = e.target.value;
            });
            // Also listen for clear button
            el?.addEventListener('sl-clear', () => {
                this.selectedKeys[i] = "";
            });
        });
    }
}
customElements.define('qp-styles', QpStyles);

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

    getValue() {
        return {
            result: this.result,
            selectedTask: this.selectedTask
        };
    }
    setValues(values) {
        if (!values) return;
        this.result = values.result || "";
        this.selectedTask = values.selectedTask || "<DETAILED_CAPTION>";
        this.render();
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
            <qp-cartridge title="Image Analysis" type="input" brick-id="${brickId}">
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

// ControlNet Cartridge
class QpControlNet extends HTMLElement {
    static get observedAttributes() { return ['brick-id']; }
    constructor() {
        super();
        this.attachShadow({ mode: 'open' });
        this.controlnets = [];
        this.selectedControlNet = '';
        this.controlNetImage = null; // Base64
        this.controlNetStrength = 0.7;
        this.hasRendered = false;
    }

    connectedCallback() {
        this.fetchControlNets();
        this.render();
    }

    async fetchControlNets() {
        try {
            const res = await fetch('/config/controlnets');
            const data = await res.json();
            if (data.status === 'success') {
                this.controlnets = data.models;
                // Auto-select depth of default
                const depth = this.controlnets.find(m => m.toLowerCase().includes('depth'));
                if (depth && !this.selectedControlNet) this.selectedControlNet = depth;
                else if (this.controlnets.length > 0 && !this.selectedControlNet) this.selectedControlNet = this.controlnets[0];

                this.hasRendered = false;
                this.render();
            }
        } catch (e) { console.error(e); }
    }

    attributeChangedCallback() { this.render(); }

    render() {
        if (this.hasRendered) return;
        this.hasRendered = true;
        const brickId = this.getAttribute('brick-id') || '';

        this.shadowRoot.innerHTML = `
            <qp-cartridge title="ControlNet" type="tool" brick-id="${brickId}">
                <div style="display: flex; flex-direction: column; gap: 1rem;">
                     <!-- ControlNet Checkpoint -->
                    <sl-select id="cn-model-select" label="ControlNet Model" value="${this.selectedControlNet}" size="small" hoist>
                            <sl-option value="">None</sl-option>
                            ${this.controlnets.map(m => `<sl-option value="${m}">${m}</sl-option>`).join('')}
                    </sl-select>

                    <!-- Control Image Dropzone -->
                    <div style="display: flex; gap: 0.5rem; margin-bottom: 0.5rem;">
                         <sl-button size="small" id="btn-import-source" style="flex: 1;">
                             <sl-icon slot="prefix" name="box-arrow-in-down"></sl-icon> Use Source
                         </sl-button>
                         <sl-button size="small" id="btn-upload-disk" style="flex: 1;">
                             <sl-icon slot="prefix" name="folder"></sl-icon> From Disk
                         </sl-button>
                         <input type="file" id="file-input" accept="image/*" style="display: none;">
                    </div>

                    <!-- Control Image Dropzone -->
                    <div id="cn-dropzone" style="
                        width: 100%; aspect-ratio: 16/9; 
                        border: 2px dashed ${this.controlNetImage ? '#a855f7' : '#475569'}; 
                        background: ${this.controlNetImage ? 'transparent' : 'rgba(0,0,0,0.2)'};
                        border-radius: 0.5rem; display: flex; align-items: center; justify-content: center;
                        position: relative; cursor: pointer; overflow: hidden; transition: all 0.2s;">
                        
                        ${this.controlNetImage ?
                `<img src="${this.controlNetImage}" style="width: 100%; height: 100%; object-fit: contain;">
                                <div style="position: absolute; top: 5px; right: 5px; background: rgba(0,0,0,0.6); border-radius: 50%; padding: 4px; cursor: pointer;" id="cn-clear">
                                <sl-icon name="x" style="color: white;"></sl-icon>
                                </div>`
                :
                `<div style="text-align: center; color: #94a3b8; pointer-events: none;">
                                <sl-icon name="cloud-upload" style="font-size: 1.5rem; margin-bottom: 0.5rem;"></sl-icon><br>
                                Drag & Drop or Click Buttons<br>
                                <span style="font-size: 0.7rem; opacity: 0.7;">(Depth Map, Canny, etc.)</span>
                                </div>`
            }
                    </div>

                    <!-- Strength Slider -->
                    <div>
                        <div style="display: flex; justify-content: space-between; font-size: 0.8rem; color: #cbd5e1; margin-bottom: 0.5rem;">
                            <span>Control Strength</span>
                            <span id="cn-strength-val">${this.controlNetStrength.toFixed(2)}</span>
                        </div>
                        <input type="range" id="cn-strength" min="0" max="2" step="0.05" value="${this.controlNetStrength}" 
                            style="width: 100%; height: 6px; border-radius: 3px; background: rgba(168, 85, 247, 0.2); appearance: none; cursor: pointer;">
                    </div>
                </div>
            </qp-cartridge>
        `;

        const cnSelect = this.shadowRoot.getElementById('cn-model-select');
        if (cnSelect) cnSelect.addEventListener('sl-change', (e) => { this.selectedControlNet = e.target.value; });

        const cnStrength = this.shadowRoot.getElementById('cn-strength');
        if (cnStrength) cnStrength.addEventListener('input', (e) => {
            this.controlNetStrength = parseFloat(e.target.value);
            this.shadowRoot.getElementById('cn-strength-val').textContent = this.controlNetStrength.toFixed(2);
        });

        // Button Handlers
        this.shadowRoot.getElementById('btn-import-source')?.addEventListener('click', () => {
            const srcBrick = document.querySelector('qp-image-input');
            if (srcBrick) {
                const img = srcBrick.getImage();
                if (img) {
                    this.controlNetImage = img;
                    this.hasRendered = false; this.render();
                } else {
                    window.qpyt_app?.notify("No image in Source Image brick!", "warning");
                }
            } else {
                window.qpyt_app?.notify("Source Image brick not found!", "danger");
            }
        });

        this.shadowRoot.getElementById('btn-upload-disk')?.addEventListener('click', () => {
            this.shadowRoot.getElementById('file-input').click();
        });

        this.shadowRoot.getElementById('file-input')?.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = (evt) => {
                    this.controlNetImage = evt.target.result;
                    this.hasRendered = false; this.render();
                };
                reader.readAsDataURL(file);
            }
        });

        // DnD for ControlNet
        const cnDrop = this.shadowRoot.getElementById('cn-dropzone');
        if (cnDrop) {
            cnDrop.addEventListener('dragover', (e) => { e.preventDefault(); cnDrop.style.borderColor = '#a855f7'; });
            cnDrop.addEventListener('dragleave', (e) => { e.preventDefault(); cnDrop.style.borderColor = this.controlNetImage ? '#a855f7' : '#475569'; });
            cnDrop.addEventListener('drop', (e) => {
                e.preventDefault();
                const file = e.dataTransfer.files[0];
                if (file) {
                    const reader = new FileReader();
                    reader.onload = (evt) => {
                        this.controlNetImage = evt.target.result;
                        this.hasRendered = false; this.render();
                    };
                    reader.readAsDataURL(file);
                } else {
                    const url = e.dataTransfer.getData('text/plain') || e.dataTransfer.getData('text/uri-list');
                    if (url) {
                        this.controlNetImage = url;
                        this.hasRendered = false; this.render();
                    }
                }
            });
            const cnClear = this.shadowRoot.getElementById('cn-clear');
            if (cnClear) {
                cnClear.onclick = (e) => {
                    e.stopPropagation();
                    this.controlNetImage = null;
                    this.hasRendered = false; this.render();
                }
            }
        }
    }

    // Public API
    getModel() { return this.selectedControlNet; }
    getImage() { return this.controlNetImage; }
    getStrength() { return this.controlNetStrength; }
}
customElements.define('qp-controlnet', QpControlNet);

// Video Generator Cartridge
class QpVideo extends HTMLElement {
    constructor() {
        super();
        this.attachShadow({ mode: 'open' });
        this.selectedModel = "THUDM/CogVideoX-2b";
        this.numFrames = 49;
        this.fps = 8;
        this.numSteps = 50;
        this.guidanceScale = 6.0;
        this.seed = null;
        this.lowVram = true;
        this.isGenerating = false;
        this.lastVideoUrl = '';
        this.hasRendered = false;
    }

    connectedCallback() {
        this.render();
    }

    render() {
        if (this.hasRendered) return;
        this.hasRendered = true;
        const brickId = this.getAttribute('brick-id') || '';

        this.shadowRoot.innerHTML = `
            <style>
                .preview-video { width: 100%; border-radius: 0.8rem; background: #000; aspect-ratio: 16/9; }
                .gen-overlay {
                    position: absolute; top:0; left:0; width:100%; height:100%;
                    display:flex; flex-direction:column; align-items:center; justify-content:center;
                    background: rgba(0,0,0,0.6); z-index:10; border-radius: 0.8rem;
                }
            </style>
            <qp-cartridge title="CogVideo" type="generator" brick-id="${brickId}">
                <div style="display: flex; flex-direction: column; gap: 1rem;">
                    
                    <div id="video-area" style="position: relative;">
                        ${this.isGenerating ? `
                            <div class="gen-overlay">
                                <sl-spinner style="font-size: 2rem;"></sl-spinner>
                                <div style="color: #10b981; margin-top: 0.5rem; font-weight: 600;">Generating Video...</div>
                            </div>
                        ` : ''}
                        ${this.lastVideoUrl ? `
                            <video class="preview-video" src="${this.lastVideoUrl}" controls autoplay loop></video>
                        ` : `
                            <div class="preview-video" style="display:flex; align-items:center; justify-content:center; color: #475569;">
                                <sl-icon name="camera-reels" style="font-size: 3rem;"></sl-icon>
                            </div>
                        `}
                    </div>

                    <sl-select label="CogVideo Model" value="${this.selectedModel}" id="model-select">
                        <sl-option value="THUDM/CogVideoX-2b">CogVideoX 2B (Fast)</sl-option>
                        <sl-option value="THUDM/CogVideoX-5b">CogVideoX 5B (High Quality)</sl-option>
                    </sl-select>

                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem;">
                        <sl-input type="number" label="Frames" value="${this.numFrames}" id="frames-input"></sl-input>
                        <sl-input type="number" label="FPS" value="${this.fps}" id="fps-input"></sl-input>
                    </div>

                    <sl-input type="number" label="Steps" value="${this.numSteps}" id="steps-input"></sl-input>
                    <sl-input type="number" label="Guidance" step="0.1" value="${this.guidanceScale}" id="gs-input"></sl-input>
                    <sl-input type="number" label="Seed" placeholder="Random" value="${this.seed || ''}" id="seed-input"></sl-input>
                    
                    <sl-checkbox ${this.lowVram ? 'checked' : ''} id="lowvram-check">Enable Low VRAM mode</sl-checkbox>

                    <sl-button variant="success" id="gen-btn" ${this.isGenerating ? 'loading' : ''}>
                        <sl-icon slot="prefix" name="play-circle"></sl-icon>
                        Generate Video
                    </sl-button>
                </div>
            </qp-cartridge>
        `;

        this.shadowRoot.getElementById('model-select').addEventListener('sl-change', (e) => this.selectedModel = e.target.value);
        this.shadowRoot.getElementById('frames-input').addEventListener('sl-input', (e) => this.numFrames = parseInt(e.target.value));
        this.shadowRoot.getElementById('fps-input').addEventListener('sl-input', (e) => this.fps = parseInt(e.target.value));
        this.shadowRoot.getElementById('steps-input').addEventListener('sl-input', (e) => this.numSteps = parseInt(e.target.value));
        this.shadowRoot.getElementById('gs-input').addEventListener('sl-input', (e) => this.guidanceScale = parseFloat(e.target.value));
        this.shadowRoot.getElementById('seed-input').addEventListener('sl-input', (e) => this.seed = e.target.value ? parseInt(e.target.value) : null);
        this.shadowRoot.getElementById('lowvram-check').addEventListener('sl-change', (e) => this.lowVram = e.target.checked);

        this.shadowRoot.getElementById('gen-btn').addEventListener('click', () => this.generate());
    }

    async generate() {
        const promptEl = document.querySelector('qp-prompt');
        if (!promptEl) {
            window.qpyt_app?.notify("Prompt brick required!", "danger");
            return;
        }

        const { prompt } = promptEl.getValue();
        this.isGenerating = true;
        this.hasRendered = false;
        this.render();

        try {
            const resp = await fetch('/generate/video', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    prompt,
                    model_name: this.selectedModel,
                    num_frames: this.numFrames,
                    fps: this.fps,
                    num_inference_steps: this.numSteps,
                    guidance_scale: this.guidanceScale,
                    seed: this.seed,
                    low_vram: this.lowVram
                })
            });

            const submission = await resp.json();
            if (submission.status === 'queued') {
                const taskId = submission.task_id;
                this.pollTask(taskId);
            }
        } catch (e) {
            window.qpyt_app?.notify(`Video error: ${e.message}`, "danger");
            this.isGenerating = false;
            this.hasRendered = false;
            this.render();
        }
    }

    async pollTask(taskId) {
        try {
            const res = await fetch(`/queue/status/${taskId}`);
            const task = await res.json();

            if (task.status === 'COMPLETED') {
                this.lastVideoUrl = task.result.image_url;
                this.isGenerating = false;
                this.hasRendered = false;
                this.render();

                // Signal global output
                window.dispatchEvent(new CustomEvent('qpyt-output', {
                    detail: {
                        url: this.lastVideoUrl,
                        brickId: this.getAttribute('brick-id'),
                        params: {
                            prompt: "CogVideo Generation",
                            model: this.selectedModel
                        }
                    },
                    bubbles: true,
                    composed: true
                }));

                const dashboard = document.querySelector('qp-dashboard');
                if (dashboard) {
                    dashboard.addEntry({
                        image_url: this.lastVideoUrl,
                        status: 'success',
                        execution_time: task.result.execution_time,
                        metadata: task.result.metadata
                    });
                }

                window.qpyt_app?.notify("Video generated!", "success");
            } else if (task.status === 'FAILED') {
                throw new Error(task.error || "Generation failed");
            } else {
                setTimeout(() => this.pollTask(taskId), 2000);
            }
        } catch (e) {
            window.qpyt_app?.notify(e.message, "danger");
            this.isGenerating = false;
            this.hasRendered = false;
            this.render();
        }
    }
}
customElements.define('qp-video', QpVideo);

