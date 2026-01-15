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
            if (typeof drawer.show === 'function') {
                if (open) drawer.show();
                else drawer.hide();
            } else {
                if (open) {
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
                .brick-list { display: grid; gap: 1rem; padding: 1rem 0; }
                .brick-item {
                    display: flex; align-items: center; gap: 1rem; padding: 1.2rem;
                    border: 1px solid rgba(255, 255, 255, 0.1); border-radius: 12px;
                    cursor: pointer; transition: all 0.2s ease; background: rgba(255, 255, 255, 0.03); color: white;
                }
                .brick-item:hover { background: rgba(255, 255, 255, 0.1); transform: translateY(-2px); }
                .brick-item[data-type="input"]:hover { border-color: #6366f1; }
                .brick-item[data-type="logic"]:hover { border-color: #a855f7; }
                .brick-item[data-type="output"]:hover { border-color: #10b981; }
                
                .icon { font-size: 1.5rem; }
                .brick-item[data-type="input"] .icon { color: #6366f1; }
                .brick-item[data-type="logic"] .icon { color: #a855f7; }
                .brick-item[data-type="output"] .icon { color: #10b981; }

                .details { flex: 1; }
                .details b { display: block; margin-bottom: 2px; font-size: 1rem; }
                .type-badge { 
                    font-size: 0.6rem; text-transform: uppercase; letter-spacing: 1px; color: white; font-weight: 800; 
                    padding: 2px 6px; border-radius: 4px; display: inline-block;
                }
                .type-badge[data-type="input"] { background: #6366f1; }
                .type-badge[data-type="logic"] { background: #a855f7; }
                .type-badge[data-type="output"] { background: #10b981; }
                sl-drawer:not(:defined) { display: none; }
            </style>

            <button class="lib-trigger" id="open-drawer" title="Add module">+</button>

            <sl-drawer label="Qpyt-UI Library" placement="end">
                <div class="brick-list">
                    ${this.availableBricks.map(brick => `
                        <div class="brick-item" data-id="${brick.id}" data-type="${brick.type}">
                            <sl-icon class="icon" name="${brick.icon}"></sl-icon>
                            <div class="details">
                                <b>${brick.label}</b>
                                <div class="type-badge" data-type="${brick.type}">${brick.type}</div>
                            </div>
                        </div>
                    `).join('')}
                </div>
                <div slot="footer">
                    <sl-button variant="neutral" outline id="close-drawer" style="width: 100%;">Close</sl-button>
                </div>
            </sl-drawer>
        `;

        this.shadowRoot.getElementById('open-drawer').addEventListener('click', () => this.toggleDrawer(true));
        this.shadowRoot.getElementById('close-drawer')?.addEventListener('click', () => this.toggleDrawer(false));
        this.shadowRoot.querySelectorAll('.brick-item').forEach(item => {
            item.addEventListener('click', () => this.addBrick(item.dataset.id));
        });
    }
}
customElements.define('qp-library', QpLibrary);
