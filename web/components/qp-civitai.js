class QpCivitai extends HTMLElement {
    constructor() {
        super();
        this.attachShadow({ mode: 'open' });
        this.images = [];
        this.loading = false;
        this.cursor = null;
        this.history = []; // Track cursors for "Previous" logic
        this.nsfwLevel = 'None';
    }

    connectedCallback() {
        this.fetchImages();
    }

    async fetchImages(cursor = null, isBack = false) {
        this.loading = true;
        this.render();
        try {
            let url = `https://api.civitai.com/v1/images?limit=15&sort=Newest&hasMeta=true&nsfwLevel=${this.nsfwLevel}`;
            if (cursor) url += `&cursor=${cursor}`;

            const response = await fetch(url);
            const data = await response.json();

            this.images = data.items || [];
            this.currentCursor = cursor;
            this.nextCursor = data.metadata?.nextCursor;

            console.log("[Civitai] Fetched", this.images.length, "images with level", this.nsfwLevel, "cursor", cursor);
        } catch (e) {
            console.error("[Civitai Error]", e);
        } finally {
            this.loading = false;
            this.render();
        }
    }

    usePrompt(prompt) {
        if (!prompt) return;
        const promptBrick = document.querySelector('qp-prompt');
        if (promptBrick && typeof promptBrick.setPrompt === 'function') {
            promptBrick.setPrompt(prompt);
            window.qpyt_app?.notify("Prompt imported from Civitai!", "success");
        } else {
            window.qpyt_app?.notify("Global Prompt brick not found", "warning");
        }
    }

    showFullPrompt(prompt) {
        const dialog = this.shadowRoot.getElementById('prompt-viewer');
        const text = this.shadowRoot.getElementById('prompt-full-text');
        if (dialog && text) {
            text.textContent = prompt;
            dialog.show();
        }
    }

    openLightbox(image) {
        const dialog = this.shadowRoot.getElementById('lightbox');
        const img = this.shadowRoot.getElementById('lightbox-img');
        const meta = this.shadowRoot.getElementById('lightbox-meta');

        if (dialog && img) {
            img.src = image.url;
            if (meta) {
                const m = image.meta || {};
                meta.innerHTML = `
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; font-size: 0.8rem; color: #94a3b8; margin-top: 1rem;">
                        <div><strong>Model:</strong> ${image.modelId || 'N/A'}</div>
                        <div><strong>Steps:</strong> ${m.steps || 'N/A'}</div>
                        <div><strong>Sampler:</strong> ${m.sampler || 'N/A'}</div>
                        <div><strong>CFG:</strong> ${m.cfgScale || 'N/A'}</div>
                    </div>
                `;
            }
            dialog.show();
        }
    }

    render() {
        const brickId = this.getAttribute('brick-id') || '';

        this.shadowRoot.innerHTML = `
            <style>
                :host { display: block; height: 100%; }
                .civitai-container {
                    display: flex;
                    flex-direction: column;
                    height: 100%;
                    gap: 1rem;
                }
                .grid-view {
                    display: flex;
                    flex-direction: column;
                    gap: 0.8rem;
                    overflow-y: auto;
                    flex: 1;
                    padding-right: 5px;
                    scrollbar-width: thin;
                }
                .image-item {
                    background: rgba(255, 255, 255, 0.03);
                    border: 1px solid rgba(255, 255, 255, 0.05);
                    border-radius: 12px;
                    padding: 0.8rem;
                    display: flex;
                    flex-direction: column;
                    gap: 0.5rem;
                    transition: all 0.2s;
                }
                .image-item:hover {
                    background: rgba(255, 255, 255, 0.05);
                    border-color: #6366f1;
                }
                .prompt-snippet {
                    font-size: 0.8rem;
                    color: #94a3b8;
                    font-style: italic;
                    cursor: pointer;
                    white-space: nowrap;
                    overflow: hidden;
                    text-overflow: ellipsis;
                    background: rgba(0,0,0,0.2);
                    padding: 4px 8px;
                    border-radius: 4px;
                }
                .prompt-snippet:hover {
                    color: #6366f1;
                    background: rgba(99, 102, 241, 0.1);
                }
                .thumb-container {
                    width: 100%;
                    height: 160px;
                    border-radius: 8px;
                    overflow: hidden;
                    background: #0f172a;
                    cursor: pointer;
                    position: relative;
                }
                .thumb-container img {
                    width: 100%;
                    height: 100%;
                    object-fit: cover;
                }
                .actions {
                    display: flex;
                    gap: 0.5rem;
                }
                .pagination {
                    display: flex;
                    justify-content: space-between;
                    padding-top: 0.5rem;
                    border-top: 1px solid rgba(255,255,255,0.1);
                }
                sl-dialog::part(panel) {
                    background: #1e293b;
                    border: 1px solid rgba(255,255,255,0.1);
                }
                .full-prompt-area {
                    background: rgba(0,0,0,0.3);
                    padding: 1rem;
                    border-radius: 8px;
                    color: #e2e8f0;
                    font-family: monospace;
                    font-size: 0.9rem;
                    max-height: 300px;
                    overflow-y: auto;
                    white-space: pre-wrap;
                }
            </style>

            <qp-cartridge title="Civitai Explorer" type="input" brick-id="${brickId}">
                <div class="civitai-container">
                    ${this.loading ? `
                        <div style="display:flex; justify-content:center; align-items:center; flex:1;">
                            <sl-spinner style="font-size: 2rem;"></sl-spinner>
                        </div>
                    ` : `
                        <div class="grid-view">
                            ${this.images.length === 0 ? '<div style="text-align:center; padding: 2rem; color: #475569;">No images found</div>' : ''}
                            ${this.images.map((img, idx) => `
                                <div class="image-item" data-index="${idx}">
                                    <div class="prompt-snippet" title="Click to view full prompt">
                                        ${img.meta?.prompt || 'No prompt available'}
                                    </div>
                                    <div class="thumb-container">
                                        <img src="${img.url}" loading="lazy">
                                    </div>
                                    <sl-button variant="primary" size="small" class="use-btn" outline>
                                        <sl-icon slot="prefix" name="magic"></sl-icon>
                                        Use Prompt
                                    </sl-button>
                                </div>
                            `).join('')}
                        </div>
                        <div class="pagination">
                            <div style="display:flex; gap: 0.5rem; align-items: center;">
                                <sl-button size="small" id="refresh-btn">
                                    <sl-icon name="arrow-clockwise"></sl-icon>
                                </sl-button>
                                <sl-select id="nsfw-select" size="small" value="${this.nsfwLevel}" style="width: 100px;">
                                    <sl-option value="None">Safe</sl-option>
                                    <sl-option value="Soft">Soft</sl-option>
                                    <sl-option value="Mature">Mature</sl-option>
                                    <sl-option value="X">X</sl-option>
                                </sl-select>
                            </div>
                            <div style="display:flex; gap: 0.5rem;">
                                <sl-button size="small" variant="neutral" id="prev-btn" ${this.history.length === 0 ? 'disabled' : ''}>
                                    <sl-icon name="arrow-left"></sl-icon>
                                </sl-button>
                                <sl-button size="small" variant="neutral" id="next-btn" ${!this.nextCursor ? 'disabled' : ''}>
                                    <sl-icon name="arrow-right"></sl-icon>
                                </sl-button>
                            </div>
                        </div>
                    `}
                </div>
            </qp-cartridge>

            <sl-dialog id="lightbox" label="Image Detail" style="--width: 90vw;">
                <div style="display: flex; flex-direction: column; align-items: center; max-height: 80vh;">
                    <img id="lightbox-img" src="" style="max-width: 100%; max-height: 70vh; object-fit: contain; border-radius: 8px;">
                    <div id="lightbox-meta" style="width: 100%;"></div>
                </div>
                <sl-button slot="footer" variant="primary" onclick="this.closest('sl-dialog').hide()">Close</sl-button>
            </sl-dialog>

            <sl-dialog id="prompt-viewer" label="Full Prompt">
                <div class="full-prompt-area" id="prompt-full-text"></div>
                <sl-button slot="footer" variant="primary" id="copy-use-btn">Use This Prompt</sl-button>
            </sl-dialog>
        `;

        if (this.loading) return;

        // Listeners
        this.shadowRoot.querySelectorAll('.image-item').forEach((item, idx) => {
            const imgData = this.images[idx];

            item.querySelector('.prompt-snippet').onclick = () => this.showFullPrompt(imgData.meta?.prompt);
            item.querySelector('.thumb-container').onclick = () => this.openLightbox(imgData);
            item.querySelector('.use-btn').onclick = () => this.usePrompt(imgData.meta?.prompt);
        });

        this.shadowRoot.getElementById('next-btn')?.addEventListener('click', () => {
            if (this.nextCursor) {
                this.history.push(this.currentCursor);
                this.fetchImages(this.nextCursor);
            }
        });

        this.shadowRoot.getElementById('prev-btn')?.addEventListener('click', () => {
            if (this.history.length > 0) {
                const prevCursor = this.history.pop();
                this.fetchImages(prevCursor, true);
            }
        });

        this.shadowRoot.getElementById('refresh-btn')?.addEventListener('click', () => {
            this.history = [];
            this.fetchImages();
        });

        this.shadowRoot.getElementById('nsfw-select')?.addEventListener('sl-change', (e) => {
            this.nsfwLevel = e.target.value;
            this.fetchImages();
        });

        const copyUseBtn = this.shadowRoot.getElementById('copy-use-btn');
        if (copyUseBtn) {
            copyUseBtn.onclick = () => {
                const text = this.shadowRoot.getElementById('prompt-full-text').textContent;
                this.usePrompt(text);
                this.shadowRoot.getElementById('prompt-viewer').hide();
            }
        }
    }
}

customElements.define('qp-civitai', QpCivitai);
