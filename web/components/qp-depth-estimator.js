class QpDepthEstimator extends HTMLElement {
    static get observedAttributes() { return ['brick-id']; }
    constructor() {
        super();
        this.attachShadow({ mode: 'open' });
        this.isGenerating = false;
        this.lastImageUrl = "";
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

        if (!image && window.qpyt_app?.lastImage) {
            image = window.qpyt_app.lastImage;
            console.log("[Depth] Auto-picking last generated image:", image);
        }

        if (!image) {
            window.qpyt_app.notify("Depth Estimator requires a Source Image or a previous generation!", "danger");
            this._activeTasks--;
            this.isGenerating = false;
            this.render();
            return;
        }

        try {
            const response = await fetch('/depth', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image })
            });

            const submission = await response.json();
            if (submission.status === 'queued') {
                const taskId = submission.task_id;
                while (true) {
                    if (!this.isGenerating) break; // User stopped manually
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
                        window.qpyt_app.notify("Depth Map generated!", "success");
                        break;
                    }
                    if (task.status === 'FAILED' || task.status === 'CANCELLED') {
                        throw new Error(task.error || "Depth task failed");
                    }
                    await new Promise(r => setTimeout(r, 1000));
                }
            } else {
                throw new Error(submission.message || "Failed to queue task");
            }
        }
        catch (e) {
            console.error(e);
            window.qpyt_app.notify("Depth Error: " + e.message, "danger");
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
                .depth-container {
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
            <qp-cartridge title="Depth Estimator" icon="layers" type="generator" brick-id="${brickId}">
                <div class="depth-container">
                    <div class="status-badge">
                        <sl-icon name="info-circle" style="margin-right:0.5rem"></sl-icon>
                        Input: Last generation or Source
                    </div>
                    
                    <div style="display: flex; gap: 0.5rem; width: 100%; position: relative;">
                        <sl-button variant="primary" size="medium" id="btn-gen" style="flex: 3;" ?loading="${this.isGenerating}">
                            <sl-icon slot="prefix" name="layers"></sl-icon>
                            Generate Depth Map
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
                    
                    ${this.isGenerating ? `
                        <div style="font-size: 0.75rem; color: #94a3b8; text-align: center; font-style: italic;">
                            Running Depth-Anything-V2...
                        </div>
                    ` : ''}
                </div>
            </qp-cartridge>
        `;

        const genBtn = this.shadowRoot.getElementById('btn-gen');
        if (genBtn) genBtn.onclick = () => this.generate();

        const stopBtn = this.shadowRoot.getElementById('btn-stop');
        if (stopBtn) stopBtn.onclick = () => this.isGenerating = false;
    }
}
customElements.define('qp-depth-estimator', QpDepthEstimator);
