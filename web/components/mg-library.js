class MgLibrary extends HTMLElement {
    constructor() {
        super();
        this.attachShadow({ mode: 'open' });
        this.availableBricks = [
            { id: 'mg-prompt', label: 'Prompt Input', type: 'input', icon: 'chat-left-text' },
            { id: 'mg-settings', label: 'Image Settings', type: 'logic', icon: 'gear' },
            { id: 'mg-render', label: 'Render Engine', type: 'output', icon: 'play-circle' }
        ];
    }

    connectedCallback() {
        this.render();
    }

    addBrick(id) {
        const workflow = document.querySelector('.workflow-area');
        if (workflow) {
            if (workflow.querySelector(id)) {
                alert('Ce module est déjà présent dans le workflow.');
                return;
            }
            const newBrick = document.createElement(id);
            workflow.appendChild(newBrick);
            this.toggleDrawer(false);
        }
    }

    toggleDrawer(open) {
        const drawer = this.shadowRoot.querySelector('sl-drawer');
        if (drawer) {
            // Si le composant n'est pas encore "upgradé", .show() peut échouer
            if (typeof drawer.show === 'function') {
                if (open) drawer.show();
                else drawer.hide();
            } else {
                // Fallback direct sur l'attribut/propriété
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
                :host {
                    display: block;
                }

                /* Floating Action Button */
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

                .lib-trigger:hover {
                    transform: scale(1.1);
                    background: #4f46e5;
                }

                .lib-trigger:active {
                    transform: scale(0.9);
                }

                /* List Area */
                .brick-list {
                    display: grid;
                    gap: 1rem;
                    padding: 1rem 0;
                }

                .brick-item {
                    display: flex;
                    align-items: center;
                    gap: 1rem;
                    padding: 1.2rem;
                    border: 1px solid rgba(255, 255, 255, 0.1);
                    border-radius: 12px;
                    cursor: pointer;
                    transition: all 0.2s ease;
                    background: rgba(255, 255, 255, 0.03);
                    color: white;
                }

                .brick-item:hover {
                    background: rgba(255, 255, 255, 0.1);
                    border-color: #6366f1;
                    transform: translateY(-2px);
                }

                .icon {
                    font-size: 1.5rem;
                    color: #6366f1;
                }

                .details {
                    flex: 1;
                }

                .details b {
                    display: block;
                    margin-bottom: 2px;
                    font-size: 1rem;
                }

                .type-badge {
                    font-size: 0.7rem;
                    text-transform: uppercase;
                    letter-spacing: 1px;
                    color: #94a3b8;
                    font-weight: 600;
                }

                /* Ensure drawer is hidden before upgrade */
                sl-drawer:not(:defined) {
                    display: none;
                }
            </style>

            <button class="lib-trigger" id="open-drawer" title="Ajouter un module">
                +
            </button>

            <sl-drawer label="Bibliothèque de Modules" placement="end">
                <div class="brick-list">
                    ${this.availableBricks.map(brick => `
                        <div class="brick-item" data-id="${brick.id}">
                            <sl-icon class="icon" name="${brick.icon}"></sl-icon>
                            <div class="details">
                                <b>${brick.label}</b>
                                <div class="type-badge">${brick.type}</div>
                            </div>
                        </div>
                    `).join('')}
                </div>
                <div slot="footer">
                    <sl-button variant="neutral" outline id="close-drawer" style="width: 100%;">
                        Fermer le tiroir
                    </sl-button>
                </div>
            </sl-drawer>
        `;

        this.shadowRoot.getElementById('open-drawer').addEventListener('click', () => this.toggleDrawer(true));
        this.shadowRoot.getElementById('close-drawer').addEventListener('click', () => this.toggleDrawer(false));

        this.shadowRoot.querySelectorAll('.brick-item').forEach(item => {
            item.addEventListener('click', () => this.addBrick(item.dataset.id));
        });
    }
}

customElements.define('mg-library', MgLibrary);
