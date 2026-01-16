class QpDashboard extends HTMLElement {
    constructor() {
        super();
        this.attachShadow({ mode: 'open' });
        this.history = [];
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
        this.render();
    }

    toggleDrawer(open) {
        const drawer = this.shadowRoot.querySelector('sl-drawer');
        if (drawer) {
            if (open) drawer.show();
            else drawer.hide();
        }
    }

    openLightbox(imageUrl) {
        const dialog = this.shadowRoot.querySelector('#lightbox');
        const img = this.shadowRoot.querySelector('#lightbox-img');
        if (dialog && img) {
            img.src = imageUrl;
            dialog.show();
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
                }

                .history-item:hover {
                    border-color: #6366f1;
                }

                .thumb-container {
                    width: 60px;
                    height: 60px;
                    border-radius: 6px;
                    background: #0f172a;
                    overflow: hidden;
                    flex-shrink: 0;
                    cursor: pointer;
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
                    background: transparent;
                    box-shadow: none;
                    max-width: 90vw;
                }

                #lightbox img {
                    width: 100%;
                    height: auto;
                    border-radius: 12px;
                    box-shadow: 0 0 50px rgba(0,0,0,0.8);
                }
            </style>

            <button class="dash-trigger" id="open-dash" title="History & Status">
                <sl-icon name="list-ul"></sl-icon>
            </button>

            <sl-drawer label="Generation History" placement="start">
                <div class="history-list">
                    ${this.history.length === 0 ? '<div style="text-align: center; color: #475569; margin-top: 2rem;">No history available</div>' : ''}
                    ${this.history.map(item => `
                        <div class="history-item">
                            <div class="thumb-container">
                                ${item.image_url ? `<img src="${item.image_url}" class="history-thumb" data-url="${item.image_url}">` : '<sl-icon name="image" style="opacity: 0.2;"></sl-icon>'}
                            </div>
                            <div class="item-info">
                                <div class="status-header">
                                    <span class="status-badge status-${item.status || 'success'}">${item.status === 'pending' ? '⏳' : item.status === 'error' ? '❌' : '✅'}</span>
                                    <span style="font-size: 0.7rem; color: #475569;">${item.timestamp}</span>
                                </div>
                                <div class="prompt-text">"${item.prompt}"</div>
                                <div class="meta">
                                    <span>${item.execution_time ? `${item.execution_time.toFixed(1)}s` : ''}</span>
                                    <span>${(item.request_id || '').substring(0, 4)}</span>
                                </div>
                            </div>
                        </div>
                    `).join('')}
                </div>
                </div>
            </sl-drawer>

            <sl-dialog id="lightbox" no-header>
                <img id="lightbox-img" src="">
                <sl-button slot="footer" variant="primary" id="close-lightbox">Close</sl-button>
            </sl-dialog>
        `;

        this.shadowRoot.getElementById('open-dash').addEventListener('click', () => this.toggleDrawer());
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
