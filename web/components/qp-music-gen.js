class QpMusicGen extends HTMLElement {
    static get observedAttributes() { return ['brick-id']; }
    constructor() {
        super();
        this.attachShadow({ mode: 'open' });
        this.isGenerating = false;
        this.lastAudioUrl = "";
        this.duration = 10;
        this.guidance = 3.0;
    }

    connectedCallback() {
        this.render();
    }

    async generate() {
        if (!window.qpyt_app) return;

        // 1. Get Prompt from QpPrompt brick
        const promptEl = document.querySelector('qp-prompt');
        if (!promptEl) {
            window.qpyt_app.notify("Missing 'Prompt' brick! Please add one from Library.", "danger");
            return;
        }

        const { prompt } = promptEl.getValue();
        if (!prompt || prompt.trim() === "") {
            window.qpyt_app.notify("Please enter a text prompt first!", "warning");
            return;
        }

        if (!this._activeTasks) this._activeTasks = 0;
        this._activeTasks++;
        this.isGenerating = true;
        this.render();

        try {
            const response = await fetch('/generate/audio', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    prompt: prompt,
                    duration: this.duration,
                    guidance_scale: this.guidance
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
                        this.lastAudioUrl = data.image_url;

                        const dashboard = document.querySelector('qp-dashboard');
                        if (dashboard) dashboard.addEntry({
                            ...data,
                            image_url: null,
                            type: 'audio',
                            preview: this.lastAudioUrl
                        });
                        window.qpyt_app.notify("Music generated!", "success");
                        break;
                    }
                    if (task.status === 'FAILED' || task.status === 'CANCELLED') {
                        throw new Error(task.error || "MusicGen task failed");
                    }
                    // Slower polling for music
                    await new Promise(r => setTimeout(r, 2000));
                }
            } else {
                throw new Error(submission.message || "Failed to queue task");
            }
        }
        catch (e) {
            console.error(e);
            window.qpyt_app.notify("Music Error: " + e.message, "danger");
        } finally {
            this._activeTasks--;
            if (this._activeTasks <= 0) {
                this._activeTasks = 0;
                this.isGenerating = false;
            }
            this.render();
        }
    }

    updateSetting(key, val) {
        if (key === 'duration') this.duration = parseInt(val);
        if (key === 'guidance') this.guidance = parseFloat(val);

        // Update display text
        if (key === 'duration') {
            const el = this.shadowRoot.getElementById('dur-val');
            if (el) {
                const mins = Math.floor(this.duration / 60);
                const secs = this.duration % 60;
                el.innerText = mins > 0 ? `${mins}m ${secs}s` : `${secs}s`;
            }
        }
    }

    render() {
        const brickId = this.getAttribute('brick-id') || '';
        const mins = Math.floor(this.duration / 60);
        const secs = this.duration % 60;
        const durText = mins > 0 ? `${mins}m ${secs}s` : `${secs}s`;

        this.shadowRoot.innerHTML = `
            <style>
                .music-container {
                    display: flex;
                    flex-direction: column;
                    gap: 1rem;
                    padding: 0.5rem;
                }
                .control-row {
                    display: flex; flex-direction: column; gap: 0.5rem;
                }
                .label { color: #94a3b8; font-size: 0.8rem; font-weight: 600; }
                
                .slider-row {
                    display: flex; align-items: center; gap: 0.5rem; color: #cbd5e1; font-size: 0.85rem;
                }
                input[type=range] { flex: 1; accent-color: #a855f7; }
                
                audio {
                    width: 100%;
                    margin-top: 0.5rem;
                    border-radius: 8px;
                }
                
                .info-badge {
                    background: rgba(168, 85, 247, 0.1);
                    border: 1px solid rgba(168, 85, 247, 0.2);
                    padding: 0.8rem;
                    border-radius: 8px;
                    text-align: center;
                    font-size: 0.85rem;
                    color: #a855f7;
                }
                
                .warning-text {
                    font-size: 0.7rem;
                    color: #eab308;
                    font-style: italic;
                    text-align: center;
                }
            </style>
            <qp-cartridge title="Music Generator" icon="music-note-beamed" type="generator" brick-id="${brickId}" style="--brick-color: #a855f7;">
                <div class="music-container">
                    
                    <div class="info-badge">
                        <sl-icon name="chat-left-text" style="margin-right:0.5rem"></sl-icon>
                        Input: Text Prompt Brick
                    </div>

                    <div class="control-row">
                        <div class="slider-row">
                            <span>Duration</span>
                            <input type="range" min="10" max="300" step="10" value="${this.duration}" 
                                oninput="this.getRootNode().host.updateSetting('duration', this.value)">
                            <span id="dur-val" style="width:50px; text-align:right;">${durText}</span>
                        </div>
                        ${this.duration > 60 ? '<div class="warning-text">Long generation may take several minutes.</div>' : ''}
                    </div>

                    <div style="display: flex; gap: 0.5rem; width: 100%; position: relative;">
                        <sl-button variant="primary" size="medium" id="btn-gen" style="flex: 3; --sl-color-primary-600: #a855f7; --sl-color-primary-500: #9333ea;" ?loading="${this.isGenerating}">
                            <sl-icon slot="prefix" name="music-note"></sl-icon>
                            Generate
                        </sl-button>
                        <sl-button variant="danger" size="medium" id="btn-stop" outline style="flex: 1;" ${!this.isGenerating ? 'disabled' : ''}>
                            <sl-icon name="stop-fill"></sl-icon>
                        </sl-button>
                    </div>

                    ${this.lastAudioUrl ? `
                        <audio controls src="${this.lastAudioUrl}" autoplay></audio>
                        <div style="text-align:center; font-size:0.75rem; color:#94a3b8;">
                            <a href="${this.lastAudioUrl}" download target="_blank" style="color:inherit;">Download MP3</a>
                        </div>
                    ` : ''}

                    ${this.isGenerating ? `
                        <div style="font-size: 0.75rem; color: #94a3b8; text-align: center; font-style: italic;">
                            Composing long track... Please wait...
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
customElements.define('qp-music-gen', QpMusicGen);
