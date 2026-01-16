class QpCartridge extends HTMLElement {
    static get observedAttributes() {
        return ['title', 'type', 'collapsed', 'brick-id'];
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
                }

                .cartridge {
                    display: flex;
                    height: 500px;
                    background: ${theme.bg};
                    border: 1px solid ${theme.border};
                    border-radius: 1rem;
                    overflow: hidden;
                    box-shadow: 0 10px 30px -10px rgba(0,0,0,0.5), inset 0 0 20px ${theme.glow};
                    transition: width 0.4s cubic-bezier(0.4, 0, 0.2, 1);
                    width: ${isCollapsed ? '60px' : '350px'};
                    position: relative;
                }

                .side-label {
                    width: 60px;
                    background: rgba(0, 0, 0, 0.3);
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
                    background: rgba(255, 255, 255, 0.05);
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
                    color: ${theme.text};
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

                .arrow {
                    position: absolute;
                    right: -15px;
                    top: 50%;
                    transform: translateY(-50%);
                    width: 30px;
                    height: 30px;
                    background: ${theme.bg};
                    border-right: 1px solid ${theme.border};
                    border-top: 1px solid ${theme.border};
                    transform: translateY(-50%) rotate(45deg);
                    z-index: 2;
                    display: ${isCollapsed ? 'none' : 'block'};
                }
            </style>

            <div class="cartridge" id="main-cartridge">
                <div class="side-label" id="toggle-btn">
                    <div class="title-wrapper">
                        <div class="title-rotated">${title}</div>
                    </div>
                    ${brickId ? `
                        <button class="delete-btn" id="del-btn" title="Delete this module">
                            <sl-icon name="trash"></sl-icon>
                        </button>
                    ` : ''}
                </div>
                <div class="content">
                    <slot></slot>
                </div>
                <div class="arrow"></div>
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
