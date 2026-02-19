/**
 * Logic_Zone: SAFE | Delta_Initial: 0.25 | Resolution: convergent
 * [WFGY-Metadata]
 * Component: QpImageBlender
 * Purpose: Stable version using native styled sliders to avoid Shoelace hydration issues in Shadow DOM.
 */
class QpImageBlender extends HTMLElement {
    static get observedAttributes() { return ['brick-id']; }

    constructor() {
        super();
        this.attachShadow({ mode: 'open' });
        this.imageA = { url: "", base64: "" };
        this.imageB = { url: "", base64: "" };
        this.weightA = 0.5;
        this.weightB = 0.5;
        this.fidelity = 0.5;
    }

    connectedCallback() {
        this.render();
    }

    async handleUpload(e, slot) {
        const file = e.target.files[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = (event) => {
            const img = new Image();
            img.onload = () => {
                const canvas = document.createElement('canvas');
                let w = img.width;
                let h = img.height;

                if (w > 1024 || h > 1024) {
                    const scale = 1024 / Math.max(w, h);
                    w *= scale; h *= scale;
                }
                w = Math.round(w / 64) * 64;
                h = Math.round(h / 64) * 64;

                canvas.width = w; canvas.height = h;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(img, 0, 0, w, h);

                const data = {
                    url: URL.createObjectURL(file),
                    base64: canvas.toDataURL('image/png')
                };

                if (slot === 'A') this.imageA = data;
                else this.imageB = data;

                this.render();
                window.qpyt_app?.notify(`Image ${slot} uploaded`, "success");
            };
            img.src = event.target.result;
        };
        reader.readAsDataURL(file);
    }

    getValue() {
        return {
            image_a: this.imageA.base64,
            image_b: this.imageB.base64,
            weight_a: this.weightA,
            weight_b: this.weightB,
            fidelity: this.fidelity
        };
    }

    render() {
        const brickId = this.getAttribute('brick-id') || '';
        this.shadowRoot.innerHTML = `
            <style>
                :host { display: block; }
                .blender-container {
                    display: flex;
                    flex-direction: column;
                    gap: 1rem;
                    color: white;
                    font-family: 'Outfit', sans-serif;
                }
                .slots {
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 0.8rem;
                }
                .slot {
                    aspect-ratio: 1/1;
                    background: rgba(15, 23, 42, 0.6);
                    border: 2px dashed rgba(255,255,255,0.1);
                    border-radius: 0.8rem;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    position: relative;
                    cursor: pointer;
                    overflow: hidden;
                    transition: all 0.2s;
                }
                .slot:hover { border-color: #6366f1; background: rgba(99, 102, 241, 0.1); }
                .slot img { width: 100%; height: 100%; object-fit: contain; }
                .slot-label {
                    position: absolute;
                    top: 6px;
                    left: 6px;
                    background: rgba(0,0,0,0.8);
                    padding: 3px 8px;
                    border-radius: 6px;
                    text-transform: uppercase;
                    font-weight: 800;
                    color: #fff;
                    font-size: 0.6rem;
                    z-index: 10;
                    letter-spacing: 0.05em;
                }
                .file-input { position: absolute; inset: 0; opacity: 0; cursor: pointer; }
                
                .clear-btn {
                    position: absolute;
                    bottom: 8px;
                    right: 8px;
                    background: rgba(244, 63, 94, 0.2);
                    color: #fb7185;
                    border: 1px solid rgba(244, 63, 94, 0.3);
                    border-radius: 50%;
                    width: 28px;
                    height: 28px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    z-index: 11;
                    cursor: pointer;
                    transition: all 0.2s;
                }
                .clear-btn:hover { background: #f43f5e; color: white; transform: scale(1.1); }

                .controls {
                    background: rgba(30, 41, 59, 0.5);
                    padding: 1.2rem;
                    border-radius: 1rem;
                    border: 1px solid rgba(255,255,255,0.08);
                    display: flex;
                    flex-direction: column;
                    gap: 1.2rem;
                    box-shadow: inset 0 2px 4px rgba(0,0,0,0.2);
                }
                
                .control-row {
                    display: flex;
                    flex-direction: column;
                    gap: 0.6rem;
                }

                .control-header {
                    display: flex;
                    justify-content: space-between;
                    font-size: 0.75rem;
                    text-transform: uppercase;
                    letter-spacing: 0.08em;
                    color: #94a3b8;
                    font-weight: 700;
                }
                .val-badge {
                    color: #818cf8;
                    font-family: monospace;
                    font-size: 0.85rem;
                }
                
                /* Custom Premium Native Slider */
                input[type=range] {
                    -webkit-appearance: none;
                    width: 100%;
                    background: transparent;
                }
                input[type=range]:focus { outline: none; }
                
                /* Track */
                input[type=range]::-webkit-slider-runnable-track {
                    width: 100%;
                    height: 6px;
                    cursor: pointer;
                    background: rgba(255,255,255,0.1);
                    border-radius: 3px;
                }
                input[type=range]::-webkit-slider-thumb {
                    height: 18px;
                    width: 18px;
                    border-radius: 50%;
                    background: #6366f1;
                    cursor: pointer;
                    -webkit-appearance: none;
                    margin-top: -6px;
                    box-shadow: 0 0 10px rgba(99, 102, 241, 0.5);
                    border: 2px solid white;
                    transition: all 0.2s;
                }
                input[type=range]:active::-webkit-slider-thumb {
                    transform: scale(1.2);
                    background: #a855f7;
                    box-shadow: 0 0 15px rgba(168, 85, 247, 0.8);
                }
            </style>
            
            <qp-cartridge title="Image Blender (IP)" type="input" brick-id="${brickId}">
                <div class="blender-container">
                    <div class="slots">
                        <div class="slot" id="slot-a">
                            <div class="slot-label">Source A</div>
                            ${this.imageA.url ? `
                                <img src="${this.imageA.url}">
                                <div class="clear-btn" id="clear-a" title="Clear">×</div>
                            ` : `
                                <div style="font-size: 1.5rem;">🖼️</div>
                                <span style="font-size: 0.7rem; margin-top: 5px;">Input A</span>
                            `}
                            <input type="file" class="file-input" id="file-a" accept="image/*">
                        </div>
                        <div class="slot" id="slot-b">
                            <div class="slot-label">Source B</div>
                            ${this.imageB.url ? `
                                <img src="${this.imageB.url}">
                                <div class="clear-btn" id="clear-b" title="Clear">×</div>
                            ` : `
                                <div style="font-size: 1.5rem;">🎨</div>
                                <span style="font-size: 0.7rem; margin-top: 5px;">Input B</span>
                            `}
                            <input type="file" class="file-input" id="file-b" accept="image/*">
                        </div>
                    </div>
                    
                    <div class="controls">
                        <div class="control-row">
                            <div class="control-header">
                                <span>Weight Image A</span>
                                <span class="val-badge" id="val-a">${this.weightA.toFixed(2)}</span>
                            </div>
                            <input type="range" min="0" max="200" value="${this.weightA * 100}" id="slider-a">
                        </div>
                        
                        <div class="control-row">
                            <div class="control-header">
                                <span>Weight Image B</span>
                                <span class="val-badge" id="val-b">${this.weightB.toFixed(2)}</span>
                            </div>
                            <input type="range" min="0" max="200" value="${this.weightB * 100}" id="slider-b">
                        </div>

                        <div class="control-row">
                            <div class="control-header">
                                <span>Global Intensity</span>
                                <span class="val-badge" id="val-f">${this.fidelity.toFixed(2)}</span>
                            </div>
                            <input type="range" min="0" max="200" value="${this.fidelity * 100}" id="slider-f">
                        </div>
                    </div>
                </div>
            </qp-cartridge>
        `;

        // Event binding
        this.shadowRoot.getElementById('file-a')?.addEventListener('change', (e) => this.handleUpload(e, 'A'));
        this.shadowRoot.getElementById('file-b')?.addEventListener('change', (e) => this.handleUpload(e, 'B'));

        this.shadowRoot.getElementById('clear-a')?.addEventListener('click', (e) => {
            e.stopPropagation(); this.imageA = { url: "", base64: "" }; this.render();
        });
        this.shadowRoot.getElementById('clear-b')?.addEventListener('click', (e) => {
            e.stopPropagation(); this.imageB = { url: "", base64: "" }; this.render();
        });

        const updateVal = (id, prop, targetBadge) => {
            const slider = this.shadowRoot.getElementById(id);
            slider?.addEventListener('input', (e) => {
                const val = e.target.value / 100;
                this[prop] = val;
                this.shadowRoot.getElementById(targetBadge).textContent = val.toFixed(2);
            });
        };

        updateVal('slider-a', 'weightA', 'val-a');
        updateVal('slider-b', 'weightB', 'val-b');
        updateVal('slider-f', 'fidelity', 'val-f');
    }
}

customElements.define('qp-image-blender', QpImageBlender);
