class QpNormalMap extends HTMLElement {
    static get observedAttributes() { return ['brick-id']; }
    constructor() {
        super();
        this.attachShadow({ mode: 'open' });
        this.isGenerating = false;
        this.lastImageUrl = "";
        this.strength = 2.0;
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

        const imageSource = document.querySelector('qp-image-input');
        let image = imageSource?.getImage() || null;

        // Auto-pick from last generated (Depth Map flow)
        if (!image && window.qpyt_app?.lastImage) {
            image = window.qpyt_app.lastImage;
            console.log("[Normal] Auto-picking last generated image:", image);
        }

        if (!image) {
            window.qpyt_app.notify("Normal Map requires a Depth Map (Source/Last Gen)!", "danger");
            this._activeTasks--;
            this.isGenerating = false;
            this.render();
            return;
        }

        try {
            const response = await fetch('/normal', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    image,
                    strength: this.strength
                })
            });

            const submission = await response.json();
            if (submission.status === 'queued') {
                const taskId = submission.task_id;
                while (true) {
                    if (!this.isGenerating) break;
                    const statusResp = await fetch(`/queue/status/${taskId}`);
                    const task = await statusResp.json();

                    if (task.status === 'COMPLETED') {
                        const data = task.result;
                        if (data && typeof data === 'object') {
                            data.request_id = taskId;
                        }
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
                        window.qpyt_app.notify("Normal Map generated!", "success");
                        break;
                    }
                    if (task.status === 'FAILED' || task.status === 'CANCELLED') {
                        throw new Error(task.error || "Normal Map task failed");
                    }
                    await new Promise(r => setTimeout(r, 500)); // Fast poll for CPU tasks
                }
            } else {
                throw new Error(submission.message || "Failed to queue task");
            }
        }
        catch (e) {
            console.error(e);
            window.qpyt_app.notify("Normal Map Error: " + e.message, "danger");
        } finally {
            this._activeTasks--;
            if (this._activeTasks <= 0) {
                this._activeTasks = 0;
                this.isGenerating = false;
            }
            this.render();
        }
    }

    updateStrength(val) {
        this.strength = parseFloat(val);
        const disp = this.shadowRoot.getElementById('strength-val');
        if (disp) disp.innerText = this.strength.toFixed(1);
    }

    getValue() {
        return { lastImageUrl: this.lastImageUrl, strength: this.strength };
    }
    setValues(values) {
        if (!values) return;
        this.lastImageUrl = values.lastImageUrl || "";
        if (values.strength !== undefined) this.strength = values.strength;
        this.render();
    }

    render() {
        const brickId = this.getAttribute('brick-id') || '';
        this.shadowRoot.innerHTML = `
            <style>
                .normal-container {
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
                .control-row {
                    display: flex; flex-direction: column; gap: 0.5rem;
                }
                .slider-row {
                    display: flex; align-items: center; gap: 0.5rem; color: #cbd5e1; font-size: 0.85rem;
                }
                input[type=range] { flex: 1; accent-color: #10b981; }
            </style>
            <qp-cartridge title="Normal Map" icon="box" type="generator" brick-id="${brickId}">
                <div class="normal-container">
                    <div class="status-badge">
                        <sl-icon name="info-circle" style="margin-right:0.5rem"></sl-icon>
                        Input: Depth Map (Source/Last)
                    </div>

                    <div class="control-row">
                        <div class="slider-row">
                            <span>Strength</span>
                            <input type="range" min="0.1" max="5.0" step="0.1" value="${this.strength}" 
                                oninput="this.getRootNode().host.updateStrength(this.value)">
                            <span id="strength-val" style="width:30px; text-align:right;">${this.strength.toFixed(1)}</span>
                        </div>
                    </div>
                    
                    <div style="display: flex; gap: 0.5rem; width: 100%; position: relative;">
                        <sl-button variant="primary" size="medium" id="btn-gen" style="flex: 3;" ?loading="${this.isGenerating}">
                            <sl-icon slot="prefix" name="box"></sl-icon>
                            Generate Normals
                        </sl-button>
                        <sl-button variant="danger" size="medium" id="btn-stop" outline style="flex: 1;" ${!this.isGenerating ? 'disabled' : ''}>
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

        const genBtn = this.shadowRoot.getElementById('btn-gen');
        if (genBtn) genBtn.onclick = () => this.generate();

        const stopBtn = this.shadowRoot.getElementById('btn-stop');
        if (stopBtn) stopBtn.onclick = () => this.isGenerating = false;
    }
}
customElements.define('qp-normal-map', QpNormalMap);
