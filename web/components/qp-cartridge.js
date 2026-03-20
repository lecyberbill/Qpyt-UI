class QpCartridge extends HTMLElement {
    static get observedAttributes() {
        return ['title', 'type', 'collapsed', 'brick-id', 'chained'];
    }

    constructor() {
        super();
        this.attachShadow({ mode: 'open' });
    }

    connectedCallback() {
        this.render();
    }

    attributeChangedCallback(name, oldValue, newValue) {
        if (oldValue !== newValue) {
            this.render();
        }
    }

    toggleCollapse() {
        const isCollapsed = this.hasAttribute('collapsed');
        if (isCollapsed) {
            this.removeAttribute('collapsed');
        } else {
            this.setAttribute('collapsed', '');
        }
    }

    handleDelete(e) {
        e.stopPropagation();
        const brickId = this.getAttribute('brick-id');
        if (brickId && window.qpyt_app) {
            window.qpyt_app.removeBrick(brickId);
        }
    }

    render() {
        const title = this.getAttribute('title') || 'Module';
        const type = this.getAttribute('type') || 'input';
        const isCollapsed = this.hasAttribute('collapsed');
        const brickId = this.getAttribute('brick-id');

        const colors = {
            input: { bg: '#1e293b', border: '#a855f7', text: '#a855f7', glow: 'rgba(168, 85, 247, 0.3)' }, // Purple
            generator: { bg: '#1e293b', border: '#10b981', text: '#10b981', glow: 'rgba(16, 185, 129, 0.3)' }, // Green
            setting: { bg: '#1e293b', border: '#ef4444', text: '#ef4444', glow: 'rgba(239, 68, 68, 0.3)' }, // Red
            output: { bg: '#1e293b', border: '#f59e0b', text: '#f59e0b', glow: 'rgba(245, 158, 11, 0.3)' }  // Orange
        };

        const theme = colors[type] || colors.input;

        this.shadowRoot.innerHTML = `
            <style>
                :host {
                    display: block;
                    margin-right: -10px;
                    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
                    flex-shrink: 0;
                    --brick-color: ${theme.border};
                    --brick-glow: ${theme.glow};
                    position: relative;
                }

                :host([chained]) .cartridge {
                    border-color: #06b6d4 !important; /* Cyan for chaining */
                    box-shadow: 0 0 20px rgba(6, 182, 212, 0.4) !important;
                }

                .chain-badge {
                    display: none;
                    position: absolute;
                    top: 8px; /* Moved inside */
                    right: 8px; /* Moved inside */
                    background: #06b6d4;
                    color: white;
                    width: 24px;
                    height: 24px;
                    border-radius: 50%;
                    align-items: center;
                    justify-content: center;
                    font-size: 14px;
                    z-index: 100;
                    box-shadow: 0 2px 10px rgba(0,0,0,0.5);
                    animation: pulse-cyan 2s infinite;
                }

                :host([chained]) .chain-badge {
                    display: flex;
                }

                @keyframes pulse-cyan {
                    0% { transform: scale(1); box-shadow: 0 0 0 0 rgba(6, 182, 212, 0.7); }
                    70% { transform: scale(1.1); box-shadow: 0 0 0 10px rgba(6, 182, 212, 0); }
                    100% { transform: scale(1); box-shadow: 0 0 0 0 rgba(6, 182, 212, 0); }
                }

                .cartridge {
                    display: flex;
                    height: 500px;
                    background: var(--cartridge-bg, ${theme.bg});
                    border: var(--cartridge-border, 1px solid ${theme.border});
                    border-radius: var(--cartridge-radius, 1rem);
                    overflow: hidden;
                    box-shadow: var(--cartridge-shadow, 0 10px 30px -10px rgba(0,0,0,0.5), inset 0 0 20px ${theme.glow});
                    backdrop-filter: var(--cartridge-blur, none);
                    -webkit-backdrop-filter: var(--cartridge-blur, none);
                    transition: width 0.4s cubic-bezier(0.4, 0, 0.2, 1), transform 0.3s ease;
                    width: ${isCollapsed ? '60px' : '350px'};
                    position: relative;
                }

                .side-label {
                    width: 60px;
                    background: var(--cartridge-header-bg, rgba(0, 0, 0, 0.3));
                    border-right: 1px solid rgba(255, 255, 255, 0.1);
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: space-between;
                    padding: 1rem 0;
                    cursor: pointer;
                    user-select: none;
                    transition: background 0.3s;
                }

                .side-label:hover {
                    background: var(--cartridge-header-hover, rgba(255, 255, 255, 0.05));
                }

                .title-wrapper {
                    flex: 1;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                }

                .title-rotated {
                    transform: rotate(-90deg);
                    white-space: nowrap;
                    font-weight: 800;
                    text-transform: uppercase;
                    letter-spacing: 2px;
                    color: var(--cartridge-title-color, ${theme.text});
                    font-size: 0.9rem;
                }

                .delete-btn {
                    color: #94a3b8;
                    font-size: 1.2rem;
                    transition: color 0.2s, transform 0.2s;
                    margin-top: auto;
                    border: none;
                    background: none;
                    cursor: pointer;
                    padding: 5px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                }

                .delete-btn:hover {
                    color: #ef4444;
                    transform: scale(1.2);
                }

                .content {
                    flex: 1;
                    padding: 1.5rem;
                    opacity: ${isCollapsed ? '0' : '1'};
                    transform: translateX(${isCollapsed ? '20px' : '0'});
                    transition: all 0.3s ease;
                    display: flex;
                    flex-direction: column;
                    pointer-events: ${isCollapsed ? 'none' : 'auto'};
                    overflow-y: auto;
                }

            </style>

            <div class="cartridge" id="main-cartridge" part="base">
                <div class="chain-badge">🔗</div>
                <div class="side-label drag-handle" id="toggle-btn" style="cursor: grab;" part="header">
                    <!-- Drag Handle Icon -->
                    <sl-icon name="grip-vertical" style="font-size: 20px; color: rgba(255,255,255,0.6); margin-top: 10px;"></sl-icon>
                    
                    <div class="title-wrapper">
                        <div class="title-rotated">${title}</div>
                    </div>
                    ${brickId ? `
                        <button class="delete-btn" id="del-btn" title="Delete this module">
                            <sl-icon name="trash"></sl-icon>
                        </button>
                    ` : ''}
                </div>
                <div class="content" part="content">
                    <slot></slot>
                </div>
            </div>
        `;

        this.shadowRoot.getElementById('toggle-btn').addEventListener('click', (e) => {
            if (e.target.closest('#del-btn')) return;
            this.toggleCollapse();
        });

        const delBtn = this.shadowRoot.getElementById('del-btn');
        if (delBtn) {
            delBtn.addEventListener('click', (e) => this.handleDelete(e));
        }
    }
}

customElements.define('qp-cartridge', QpCartridge);
