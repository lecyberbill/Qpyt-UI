class QpLibrary extends HTMLElement {
    constructor() {
        super();
        this.attachShadow({ mode: 'open' });
        this.availableBricks = [];
    }

    connectedCallback() {
        this.loadConfig();
    }

    async loadConfig() {
        try {
            const resp = await fetch('/config');
            const config = await resp.json();
            this.availableBricks = config.library || [];
            this.render();
        } catch (e) {
            console.error("Failed to load library config", e);
        }
    }

    addBrick(type) {
        if (window.qpyt_app) {
            window.qpyt_app.addBrick(type);
            this.toggleDrawer(false);
        }
    }

    toggleDrawer(open) {
        const drawer = this.shadowRoot.querySelector('sl-drawer');
        if (drawer) {
            const shouldOpen = open === undefined ? !drawer.open : open;

            if (typeof drawer.show === 'function') {
                if (shouldOpen) drawer.show();
                else drawer.hide();
            } else {
                if (shouldOpen) {
                    drawer.setAttribute('open', '');
                    drawer.open = true;
                } else {
                    drawer.removeAttribute('open');
                    drawer.open = false;
                }
            }
        }
    }

    render() {
        // Categorize bricks
        const groups = {
            inputs: {
                label: "Inputs & Prompts",
                icon: "chat-left-text",
                ids: ["qp-image-input", "qp-prompt", "qp-styles", "qp-translator", "qp-llm-prompter", "qp-img2prompt", "qp-civitai", "qp-image-blender"]
            },
            generators: {
                label: "Generators (AI)",
                icon: "magic",
                ids: ["qp-render-sdxl", "qp-render-flux", "qp-render-flux-klein", "qp-render-sd35turbo", "qp-sprite", "qp-render-zimage", "qp-img2img", "qp-inpaint", "qp-outpaint", "qp-lora-manager", "qp-controlnet", "qp-video"]
            },
            audio: {
                label: "Generators (Audio)",
                icon: "music-note-beamed",
                ids: ["qp-music-gen"]
            },
            utilities: {
                label: "Utilities & FX",
                icon: "layers",
                ids: ["q-upscaler-v3", "qp-rembg", "qp-depth-estimator", "qp-normal-map", "qp-openpose-editor", "qp-vectorize", "qp-filter", "qp-canvas", "qp-batch-runner"]
            },
            system: {
                label: "Export & Settings",
                icon: "gear",
                ids: ["qp-settings", "qp-image-out", "qp-save-to-disk", "qp-queue-monitor", "qp-monitor"]
            }
        };

        const getGroupBricks = (groupId) => {
            return this.availableBricks.filter(b => groups[groupId].ids.includes(b.id));
        };

        this.shadowRoot.innerHTML = `
            <style>
                :host { display: block; }
                .lib-trigger {
                    position: fixed;
                    bottom: 30px;
                    right: 30px;
                    width: 60px;
                    height: 60px;
                    border-radius: 50%;
                    background: #6366f1;
                    color: white;
                    border: none;
                    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.4), 0 0 20px rgba(99, 102, 241, 0.4);
                    cursor: pointer;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    z-index: 9999;
                    font-size: 28px;
                    transition: transform 0.2s, background 0.2s;
                }
                .lib-trigger:hover { transform: scale(1.1); background: #4f46e5; }
                
                .brick-list { 
                    display: grid; 
                    gap: 0.8rem; 
                    padding: 1rem 0.5rem;
                    max-height: calc(100vh - 150px);
                    overflow-y: auto;
                }
                
                .brick-item {
                    display: flex; align-items: center; gap: 1rem; padding: 1rem;
                    border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 12px;
                    cursor: pointer; transition: all 0.2s ease; background: rgba(255, 255, 255, 0.03); color: white;
                }
                .brick-item:hover { background: rgba(255, 255, 255, 0.1); transform: translateY(-2px); }
                
                /* Category specific hover effects */
                .brick-item[data-type="input"]:hover { border-color: #a855f7; }
                .brick-item[data-type="generator"]:hover { border-color: #10b981; }
                .brick-item[data-type="setting"]:hover { border-color: #ef4444; }
                .brick-item[data-type="tool"]:hover { border-color: #a855f7; }
                .brick-item[data-type="output"]:hover { border-color: #f59e0b; }
                
                .icon { font-size: 1.4rem; }
                .brick-item[data-type="input"] .icon { color: #a855f7; }
                .brick-item[data-type="generator"] .icon { color: #10b981; }
                .brick-item[data-type="setting"] .icon { color: #ef4444; }
                .brick-item[data-type="tool"] .icon { color: #a855f7; }
                .brick-item[data-type="output"] .icon { color: #f59e0b; }

                .details { flex: 1; }
                .details b { display: block; margin-bottom: 2px; font-size: 0.95rem; }
                
                .type-badge { 
                    font-size: 0.55rem; text-transform: uppercase; letter-spacing: 1px; color: white; font-weight: 800; 
                    padding: 2px 6px; border-radius: 4px; display: inline-block;
                }
                .type-badge[data-type="input"] { background: #a855f7; }
                .type-badge[data-type="generator"] { background: #10b981; }
                .type-badge[data-type="setting"] { background: #ef4444; }
                .type-badge[data-type="tool"] { background: #a855f7; }
                .type-badge[data-type="output"] { background: #f59e0b; }

                sl-tab-group {
                    --indicator-color: #6366f1;
                }
                sl-tab::part(base) {
                    padding: 12px 8px;
                    font-size: 0.85rem;
                }
                sl-drawer::part(body) {
                    padding: 0 1rem;
                }
                sl-drawer:not(:defined) { display: none; }
            </style>

            <button class="lib-trigger" id="open-drawer" title="Add module">+</button>

            <sl-drawer label="Qpyt-UI Library" placement="end" style="--size: 420px;">
                <sl-tab-group>
                    ${Object.entries(groups).map(([id, group]) => `
                        <sl-tab slot="nav" panel="${id}">
                            <sl-icon name="${group.icon}" style="margin-right: 6px;"></sl-icon>
                            ${group.label.split(' ')[0]}
                        </sl-tab>
                    `).join('')}

                    ${Object.entries(groups).map(([id, group]) => `
                        <sl-tab-panel name="${id}">
                            <div class="brick-list">
                                ${getGroupBricks(id).length === 0 ? '<p style="color: #64748b; font-size: 0.8rem; text-align: center; padding: 2rem;">Coming soon...</p>' : ''}
                                ${getGroupBricks(id).map(brick => `
                                    <div class="brick-item" data-id="${brick.id}" data-type="${brick.type}">
                                        <sl-icon class="icon" name="${brick.icon}"></sl-icon>
                                        <div class="details">
                                            <b>${brick.label}</b>
                                            <div class="type-badge" data-type="${brick.type}">${brick.type}</div>
                                        </div>
                                    </div>
                                `).join('')}
                            </div>
                        </sl-tab-panel>
                    `).join('')}
                </sl-tab-group>
            </sl-drawer>
        `;

        this.shadowRoot.getElementById('open-drawer').addEventListener('click', () => this.toggleDrawer());
        this.shadowRoot.querySelectorAll('.brick-item').forEach(item => {
            item.addEventListener('click', () => {
                console.log("[Library] Clicking brick:", item.dataset.id);
                this.addBrick(item.dataset.id)
            });
        });
    }
}
customElements.define('qp-library', QpLibrary);
