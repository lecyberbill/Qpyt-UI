class MgCartridge extends HTMLElement {
    static get observedAttributes() {
        return ['title', 'type', 'collapsed'];
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

    render() {
        const title = this.getAttribute('title') || 'Module';
        const type = this.getAttribute('type') || 'input';
        const isCollapsed = this.hasAttribute('collapsed');

        // Color mapping based on type
        const colors = {
            input: { bg: '#1e293b', border: '#6366f1', text: '#6366f1', glow: 'rgba(99, 102, 241, 0.3)' },
            logic: { bg: '#1e293b', border: '#a855f7', text: '#a855f7', glow: 'rgba(168, 85, 247, 0.3)' },
            output: { bg: '#1e293b', border: '#10b981', text: '#10b981', glow: 'rgba(16, 185, 129, 0.3)' }
        };

        const theme = colors[type] || colors.input;

        this.shadowRoot.innerHTML = `
            <style>
                :host {
                    display: block;
                    margin-right: -10px; /* Overlap effect like cards */
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

                /* Vertical Label Section */
                .side-label {
                    width: 60px;
                    background: rgba(0, 0, 0, 0.3);
                    border-right: 1px solid rgba(255, 255, 255, 0.1);
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    cursor: pointer;
                    user-select: none;
                    transition: background 0.3s;
                }

                .side-label:hover {
                    background: rgba(255, 255, 255, 0.05);
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

                /* Content Area */
                .content {
                    flex: 1;
                    padding: 1.5rem;
                    opacity: ${isCollapsed ? '0' : '1'};
                    transform: translateX(${isCollapsed ? '20px' : '0'});
                    transition: all 0.3s ease;
                    display: flex;
                    flex-direction: column;
                    pointer-events: ${isCollapsed ? 'none' : 'auto'};
                }

                /* Arrow decorator */
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

            <div class="cartridge">
                <div class="side-label" id="toggle-btn">
                    <div class="title-rotated">${title}</div>
                </div>
                <div class="content">
                    <slot></slot>
                </div>
                <div class="arrow"></div>
            </div>
        `;

        this.shadowRoot.getElementById('toggle-btn').addEventListener('click', () => this.toggleCollapse());
    }
}

customElements.define('mg-cartridge', MgCartridge);
