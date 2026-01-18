class QpLoraManager extends HTMLElement {
    constructor() {
        super();
        this.attachShadow({ mode: 'open' });
        this.availableLoras = [];
        this.selectedLoras = []; // [{ path: string, weight: float, enabled: bool }]
        this.hasRendered = false;
    }

    connectedCallback() {
        this.fetchLoras();
    }

    async fetchLoras() {
        try {
            const resp = await fetch('/config/loras');
            const data = await resp.json();
            this.availableLoras = data.loras || [];
            this.render();
        } catch (e) {
            console.error("Failed to fetch LoRAs:", e);
        }
    }

    setValues(values) {
        if (values && values.loras) {
            this.selectedLoras = values.loras;
            this.render();
        }
    }

    getValues() {
        return {
            loras: this.selectedLoras
        }
    }

    addLora(path) {
        if (!path) return;
        if (this.selectedLoras.find(l => l.path === path)) return;

        this.selectedLoras.push({
            path: path,
            weight: 1.0,
            enabled: true
        });
        this.render();
    }

    removeLora(index) {
        this.selectedLoras.splice(index, 1);
        this.render();
    }

    updateLora(index, key, value) {
        if (this.selectedLoras[index]) {
            this.selectedLoras[index][key] = value;
        }
    }

    render() {
        const brickId = this.getAttribute('brick-id') || '';

        this.shadowRoot.innerHTML = `
            <style>
                :host { display: block; flex-shrink: 0; }
                
                .lora-manager-container {
                    display: flex;
                    flex-direction: column;
                    gap: 1.2rem;
                    height: 100%;
                }

                .selector-group {
                    display: flex;
                    flex-direction: column;
                    gap: 0.5rem;
                    padding-bottom: 1rem;
                    border-bottom: 1px solid rgba(255, 255, 255, 0.05);
                }

                .lora-list {
                    display: flex;
                    flex-direction: column;
                    gap: 1rem;
                    flex-grow: 1;
                    overflow-y: auto;
                    padding-right: 5px;
                }

                /* Custom scrollbar */
                .lora-list::-webkit-scrollbar {
                    width: 4px;
                }
                .lora-list::-webkit-scrollbar-track {
                    background: transparent;
                }
                .lora-list::-webkit-scrollbar-thumb {
                    background: rgba(255, 255, 255, 0.1);
                    border-radius: 10px;
                }

                .lora-item {
                    background: rgba(255, 255, 255, 0.03);
                    border: 1px solid rgba(255, 255, 255, 0.05);
                    border-radius: 12px;
                    padding: 1rem;
                    display: flex;
                    flex-direction: column;
                    gap: 0.8rem;
                    transition: all 0.2s ease;
                }
                .lora-item:hover {
                    background: rgba(255, 255, 255, 0.05);
                    border-color: rgba(16, 185, 129, 0.3);
                }

                .lora-top {
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                    gap: 0.5rem;
                }
                .lora-name {
                    font-size: 0.85rem;
                    font-weight: 600;
                    color: #e2e8f0;
                    white-space: nowrap;
                    overflow: hidden;
                    text-overflow: ellipsis;
                    max-width: 170px;
                }

                .weight-control {
                    display: flex;
                    flex-direction: column;
                    gap: 0.2rem;
                }
                .weight-label {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    font-size: 0.75rem;
                    color: #94a3b8;
                    gap: 0.5rem;
                }
                
                .weight-input-field {
                    width: 60px;
                    --sl-input-height-small: 22px;
                    --sl-input-font-size-small: 0.75rem;
                }
                
                .weight-slider {
                    --track-height: 4px;
                    --thumb-size: 14px;
                }
                
                .weight-input-field::part(base) {
                    background: rgba(0,0,0,0.3);
                    border-color: rgba(255,255,255,0.1);
                    font-weight: 700;
                    color: #10b981;
                }

                sl-range {
                    --track-color-active: #10b981;
                }
                
                .empty-msg {
                    padding: 2rem;
                    text-align: center;
                    color: #64748b;
                    font-size: 0.85rem;
                    font-style: italic;
                    background: rgba(0,0,0,0.2);
                    border-radius: 12px;
                    border: 1px dashed rgba(255,255,255,0.05);
                }

                sl-select::part(combobox) {
                    background: rgba(0, 0, 0, 0.3);
                    border-color: rgba(255, 255, 255, 0.1);
                }
            </style>

            <qp-cartridge title="LoRA Manager" type="generator" brick-id="${brickId}">
                <div class="lora-manager-container">
                    <div class="selector-group">
                        <sl-select placeholder="Select a LoRA..." id="lora-selector" clearable size="small">
                            ${this.availableLoras.map(p => {
            const filename = p.split(/[\\\/]/).pop().replace('.safetensors', '').replace('.ckpt', '');
            return `<sl-option value="${p}">${filename}</sl-option>`;
        }).join('')}
                        </sl-select>
                        <sl-button variant="primary" size="small" id="add-btn" outline>
                            <sl-icon slot="prefix" name="plus-lg"></sl-icon> Add LoRA
                        </sl-button>
                    </div>

                    <div class="lora-list">
                        ${this.selectedLoras.length === 0 ? `
                            <div class="empty-msg">No LoRAs active.</div>
                        ` : this.selectedLoras.map((lora, idx) => {
            const filename = lora.path.split(/[\\\/]/).pop().replace('.safetensors', '').replace('.ckpt', '');
            return `
                                <div class="lora-item">
                                    <div class="lora-top">
                                        <sl-switch ?checked="${lora.enabled}" class="enable-toggle" data-index="${idx}" size="small"></sl-switch>
                                        <div class="lora-name" title="${lora.path}">${filename}</div>
                                        <sl-icon-button name="trash" class="remove-btn" data-index="${idx}" style="font-size: 1rem;"></sl-icon-button>
                                    </div>
                                    <div class="weight-control">
                                        <div class="weight-label">
                                            <span>Weight</span>
                                            <sl-input type="number" step="0.05" value="${lora.weight}" class="weight-input" data-index="${idx}" size="small" class="weight-input-field"></sl-input>
                                        </div>
                                        <sl-range min="0" max="2" step="0.05" value="${lora.weight}" class="weight-slider" data-index="${idx}"></sl-range>
                                    </div>
                                </div>
                            `;
        }).join('')}
                    </div>
                </div>
            </qp-cartridge>
        `;

        // Event listeners
        const addBtn = this.shadowRoot.getElementById('add-btn');
        const selector = this.shadowRoot.getElementById('lora-selector');

        addBtn?.addEventListener('click', () => {
            const path = selector.value;
            if (path) {
                this.addLora(path);
                selector.value = '';
            }
        });

        this.shadowRoot.querySelectorAll('.remove-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const idx = parseInt(btn.dataset.index);
                this.removeLora(idx);
            });
        });

        this.shadowRoot.querySelectorAll('.enable-toggle').forEach(sw => {
            sw.addEventListener('sl-change', () => {
                const idx = parseInt(sw.dataset.index);
                this.updateLora(idx, 'enabled', sw.checked);
            });
        });

        this.shadowRoot.querySelectorAll('.weight-slider').forEach(range => {
            range.addEventListener('sl-input', () => {
                const idx = parseInt(range.dataset.index);
                const val = parseFloat(range.value);
                this.updateLora(idx, 'weight', val);

                // Sync numerical input
                const input = this.shadowRoot.querySelector(`.weight-input[data-index="${idx}"]`);
                if (input) input.value = val.toString();
            });
        });

        this.shadowRoot.querySelectorAll('.weight-input').forEach(input => {
            input.addEventListener('sl-input', () => {
                const idx = parseInt(input.dataset.index);
                const val = parseFloat(input.value) || 0;
                this.updateLora(idx, 'weight', val);

                // Sync slider
                const slider = this.shadowRoot.querySelector(`.weight-slider[data-index="${idx}"]`);
                if (slider) slider.value = val;
            });
        });
    }
}

customElements.define('qp-lora-manager', QpLoraManager);
