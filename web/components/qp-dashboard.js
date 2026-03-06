class QpDashboard extends HTMLElement {
    constructor() {
        super();
        this.attachShadow({ mode: 'open' });
        this.history = [];
        this.hasRendered = false;
    }

    connectedCallback() {
        this.render();
        window.addEventListener('qpyt-status-update', (e) => this.addEntry(e.detail));
    }

    addEntry(data) {
        // Le backend renvoie request_id (UUID), on s'assure de l'utiliser
        const id = data.request_id || data.id;
        const existingIndex = this.history.findIndex(item => (item.request_id || item.id) === id);

        if (existingIndex !== -1) {
            this.history[existingIndex] = { ...this.history[existingIndex], ...data };
        } else {
            this.history.unshift({
                ...data,
                id: id, // Stockage pour compatibilité
                timestamp: new Date().toLocaleTimeString()
            });
        }

        if (this.history.length > 20) this.history.pop();
        this.renderHistory();
    }


    openLightbox(item) {
        if (!item) return;
        const dialog = this.shadowRoot.querySelector('#lightbox');
        const img = this.shadowRoot.querySelector('#lightbox-img');
        const video = this.shadowRoot.querySelector('#lightbox-video');
        const promptArea = this.shadowRoot.querySelector('#lightbox-prompt');
        const metaArea = this.shadowRoot.querySelector('#lightbox-meta');

        if (dialog) {
            const url = typeof item === 'string' ? item : item.image_url;
            if (!url) return;

            const isVideo = url.toLowerCase().endsWith('.mp4');

            // Force hide if already open to reset transition/stacking
            if (dialog.open) dialog.hide();

            // Small delay to ensure clean state
            setTimeout(() => {
                if (isVideo) {
                    if (video) {
                        video.src = url;
                        video.style.display = 'block';
                    }
                    if (img) img.style.display = 'none';
                } else {
                    if (img) {
                        img.src = url;
                        img.style.display = 'block';
                        img.src = url; // Re-assign for good measure
                    }
                    if (video) {
                        video.src = '';
                        video.style.display = 'none';
                    }
                }

                if (promptArea) {
                    promptArea.textContent = (typeof item === 'object' ? (item.metadata?.prompt || item.prompt) : '') || 'No prompt info';
                }

                if (metaArea) {
                    const m = (typeof item === 'object' ? (item.metadata || {}) : {});
                    if (Object.keys(m).length === 0) {
                        metaArea.style.display = 'none';
                    } else {
                        metaArea.style.display = 'grid';
                        metaArea.innerHTML = `
                            <div><strong>Seed:</strong> ${m.seed || 'N/A'}</div>
                            <div><strong>Model:</strong> ${m.model_name || 'N/A'}</div>
                            <div><strong>Steps:</strong> ${m.num_inference_steps || 'N/A'}</div>
                            <div><strong>Guidance:</strong> ${m.guidance_scale || 'N/A'}</div>
                            <div><strong>Res:</strong> ${m.width || '?'}x${m.height || '?'}</div>
                            <div><strong>Time:</strong> ${item.execution_time ? item.execution_time.toFixed(1) + 's' : 'N/A'}</div>
                        `;
                    }
                }
                dialog.show();
            }, 10);
        }
    }

    render() {
        this.shadowRoot.innerHTML = `
            <style>
                :host { display: block; }
                
                .dash-trigger {
                    position: fixed;
                    bottom: 30px;
                    left: 30px;
                    width: 50px;
                    height: 50px;
                    border-radius: 12px;
                    background: rgba(30, 41, 59, 0.8);
                    backdrop-filter: blur(10px);
                    color: white;
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    cursor: pointer;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    z-index: 9999;
                    font-size: 20px;
                    transition: all 0.2s;
                }

                .dash-trigger:hover {
                    background: #6366f1;
                    transform: scale(1.05);
                }

                .history-list {
                    display: flex;
                    flex-direction: column;
                    gap: 0.8rem;
                    padding: 0.5rem 0;
                }

                .history-item {
                    background: rgba(255, 255, 255, 0.03);
                    border: 1px solid rgba(255, 255, 255, 0.05);
                    border-radius: 10px;
                    padding: 0.8rem;
                    font-size: 0.85rem;
                    transition: border-color 0.2s;
                    display: flex;
                    gap: 0.8rem;
                    cursor: pointer;
                }

                .history-item:hover {
                    border-color: #6366f1;
                    background: rgba(99, 102, 241, 0.05);
                }

                .thumb-container {
                    width: 60px;
                    height: 60px;
                    border-radius: 6px;
                    background: #0f172a;
                    overflow: hidden;
                    flex-shrink: 0;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    border: 1px solid rgba(255,255,255,0.1);
                }

                .thumb-container img {
                    width: 100%;
                    height: 100%;
                    object-fit: cover;
                }

                .item-info {
                    flex: 1;
                    min-width: 0;
                }

                .status-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 0.3rem;
                }

                .status-badge {
                    padding: 2px 6px;
                    border-radius: 4px;
                    font-size: 0.6rem;
                    font-weight: bold;
                    text-transform: uppercase;
                }

                .status-pending { background: rgba(234, 179, 8, 0.2); color: #eab308; }
                .status-success { background: rgba(16, 185, 129, 0.2); color: #10b981; }
                .status-error { background: rgba(239, 68, 68, 0.2); color: #ef4444; }

                .prompt-text {
                    color: #94a3b8;
                    font-style: italic;
                    white-space: nowrap;
                    overflow: hidden;
                    text-overflow: ellipsis;
                    margin-bottom: 4px;
                }

                .meta {
                    font-size: 0.7rem;
                    color: #64748b;
                    display: flex;
                    justify-content: space-between;
                }

                sl-drawer::part(panel) {
                    background: #0f172a;
                    border-right: 1px solid rgba(255, 255, 255, 0.1);
                }

                #lightbox::part(panel) {
                    background: #1e293b;
                    border: 1px solid rgba(255,255,255,0.1);
                    max-width: 900px;
                    width: 90vw;
                }

                #lightbox-img {
                    width: 100%;
                    height: auto;
                    border-radius: 8px;
                    margin-bottom: 1rem;
                }

                #lightbox-meta {
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 0.5rem;
                    margin-top: 1rem;
                    font-size: 0.8rem;
                    color: #94a3b8;
                }
            </style>

            <button class="dash-trigger" id="open-dash" title="History & Status">
                <sl-icon name="list-ul"></sl-icon>
            </button>

            <sl-drawer label="Generation History" placement="start">
                <div id="history-container"></div>
            </sl-drawer>

            <sl-dialog id="lightbox" label="Generation Details" hoist>
                <div id="lightbox-content" style="position: relative;">
                    <img id="lightbox-img" src="" style="width:100%; border-radius:8px; margin-bottom:1rem; display:none;">
                    <video id="lightbox-video" src="" controls autoplay loop style="width:100%; border-radius:8px; display:none;"></video>
                </div>
                <div id="lightbox-prompt" style="background: rgba(0,0,0,0.3); padding: 1rem; border-radius: 8px; border: 1px solid rgba(255,255,255,0.05); color: #e2e8f0; font-style: italic; line-height: 1.4; max-height: 150px; overflow-y: auto;"></div>
                <div id="lightbox-meta"></div>
                <sl-button slot="footer" variant="primary" id="close-lightbox">Close</sl-button>
            </sl-dialog>
        `;

        this.shadowRoot.getElementById('close-lightbox').addEventListener('click', () => {
            this.shadowRoot.getElementById('lightbox').hide();
        });

        this.shadowRoot.getElementById('open-dash').addEventListener('click', () => this.toggleDrawer());

        this.renderHistory();
        this.hasRendered = true;
    }

    renderHistory() {
        const container = this.shadowRoot.getElementById('history-container');
        if (!container) return;

        container.innerHTML = `
            <div class="history-list">
                ${this.history.length === 0 ? '<div style="text-align: center; color: #475569; margin-top: 2rem;">No history available</div>' : ''}
                ${this.history.map((item, idx) => `
                    <div class="history-item" data-index="${idx}">
                        <div class="thumb-container">
                            ${item.thumbnail_url ? `<img src="${item.thumbnail_url}">` : (item.image_url ? `<img src="${item.image_url}">` : '<sl-icon name="image" style="opacity: 0.2;"></sl-icon>')}
                            ${item.image_url?.toLowerCase().endsWith('.mp4') ? `
                                <div style="position: absolute; bottom: 4px; right: 4px; background: rgba(0,0,0,0.6); border-radius: 4px; padding: 2px; display: flex;">
                                    <sl-icon name="camera-reels" style="color: white; font-size: 0.8rem;"></sl-icon>
                                </div>
                            ` : ''}
                        </div>
                        <div class="item-info">
                            <div class="status-header">
                                <span class="status-badge status-${item.status || 'success'}">${item.status === 'pending' ? '⏳' : item.status === 'error' ? '❌' : '✅'}</span>
                                <span style="font-size: 0.7rem; color: #475569;">${item.timestamp}</span>
                            </div>
                            <div class="prompt-text">"${item.metadata?.prompt || item.prompt || 'No prompt'}"</div>
                            <div class="meta">
                                <span>${item.execution_time ? `${item.execution_time.toFixed(1)}s` : ''}</span>
                                <span>
                                    ${item.metadata?.seed ? `Seed: ${item.metadata.seed}` : ''}
                                    ${item.metadata?.width ? `[${item.metadata.width}x${item.metadata.height}]` : ''}
                                </span>
                            </div>
                        </div>
                    </div>
                `).join('')}
            </div>
        `;

        // Re-attach listeners for history items
        container.querySelectorAll('.history-item').forEach(itemCard => {
            itemCard.addEventListener('click', () => {
                const idx = itemCard.dataset.index;
                const data = this.history[idx];
                if (data && data.image_url) {
                    this.openLightbox(data);
                }
            });
        });
    }

    toggleDrawer(open) {
        const drawer = this.shadowRoot.querySelector('sl-drawer');
        if (drawer) {
            if (open === undefined) {
                // Toggle logic
                if (drawer.open) drawer.hide();
                else drawer.show();
            } else {
                if (open) drawer.show();
                else drawer.hide();
            }
        }
    }
}

customElements.define('qp-dashboard', QpDashboard);
