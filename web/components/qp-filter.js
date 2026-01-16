class QpFilter extends HTMLElement {
    static get observedAttributes() { return ['brick-id']; }

    constructor() {
        super();
        this.attachShadow({ mode: 'open' });
        this.isProcessing = false;

        // Initial Settings
        this.defaultSettings = {
            contrast: 1.0, saturation: 1.0, color_boost: 1.0,
            blur_radius: 0, sharpness_factor: 1.0,
            rotation_angle: 0, mirror_type: "none", special_filter: "none",
            vibrance: 0.0, hue_angle: 0,
            color_shift_r: 0, color_shift_g: 0, color_shift_b: 0
        };
        this.settings = { ...this.defaultSettings };

        this.originalImage = null; // Base64 loaded
        this.previewImage = null; // Base64 result
        this.editorOpen = false;
    }

    connectedCallback() {
        this.render();
    }

    // --- Logic ---

    async loadImage() {
        // 1. Try last generation result
        let imgToLoad = window.qpyt_app?.lastImage;

        // 2. If none, checks Source Image Input
        if (!imgToLoad) {
            const sourceBrick = document.querySelector('qp-image-input');
            // 'base64' is the internal prop, 'previewUrl' is the display
            if (sourceBrick && (sourceBrick.base64 || sourceBrick.previewUrl)) {
                imgToLoad = sourceBrick.base64 || sourceBrick.previewUrl;
            }
        }

        if (imgToLoad) {
            this.originalImage = imgToLoad;
            this.previewImage = this.originalImage;
            this.settings = { ...this.defaultSettings }; // Reset settings on new load
            this.render();
            // If editor is open, reload it
            if (this.editorOpen) this.renderEditor();
            this.notify("Image loaded", "success");
        } else {
            this.notify("No image found (History or Source)", "warning");
        }
    }

    async apply() {
        if (!this.originalImage) await this.loadImage(); // Auto-load if applied without load
        if (!this.originalImage) return;

        this.isProcessing = true;
        if (this.editorOpen) this.updateEditorState(true);

        try {
            const resp = await fetch('/filter', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    image: this.originalImage,
                    settings: this.settings
                })
            });
            const res = await resp.json();
            if (res.status === 'success') {
                this.previewImage = res.image;
            } else {
                this.notify("Filter failed: " + res.message, "danger");
            }
        } catch (e) {
            console.error(e);
            this.notify("Network error", "danger");
        } finally {
            this.isProcessing = false;
            this.render(); // Updates the thumbnail in brick
            if (this.editorOpen) this.renderEditor(); // Updates the preview in editor
        }
    }

    reset() {
        this.settings = { ...this.defaultSettings };
        this.previewImage = this.originalImage;
        this.render();
        if (this.editorOpen) this.renderEditor();
    }

    saveAndClose() {
        if (this.previewImage && window.qpyt_app) {
            window.qpyt_app.lastImage = this.previewImage;
            window.dispatchEvent(new CustomEvent('qpyt-output', { detail: { url: this.previewImage } }));
            this.notify("Changes saved", "success");
        }
        this.closeEditor();
    }

    sendToSource() {
        if (!this.previewImage) return;
        const sourceBrick = document.querySelector('qp-image-input');
        if (sourceBrick) {
            sourceBrick.base64 = this.previewImage;
            sourceBrick.previewUrl = this.previewImage;
            sourceBrick.render(); // Ensure source updates
            this.notify("Sent to Source Image!", "success");
        } else {
            this.notify("No Source Brick found", "warning");
        }
    }

    notify(msg, type) {
        if (window.qpyt_app) window.qpyt_app.notify(msg, type);
    }

    updateSetting(key, val) {
        this.settings[key] = val;
        // Auto-apply with debounce
        if (this.editorOpen) {
            this.debouncedApply();
        }
    }

    // Debounce utility
    debounce(func, wait) {
        let timeout;
        return (...args) => {
            clearTimeout(timeout);
            timeout = setTimeout(() => func.apply(this, args), wait);
        };
    }

    // Initialize debounced function in constructor if possible, or lazy init here
    get debouncedApply() {
        if (!this._debouncedApply) {
            this._debouncedApply = this.debounce(() => {
                // Check if already processing to avoid stacks? 
                // Actually fetch queueing is fine, but let's try to proceed
                this.apply();
            }, 400); // 400ms wait
        }
        return this._debouncedApply;
    }

    // --- Editor Modal Management ---

    openEditor() {
        if (!this.originalImage) {
            this.loadImage().then(() => {
                if (this.originalImage) {
                    this.editorOpen = true;
                    this.renderEditor();
                }
            });
            return;
        }
        this.editorOpen = true;
        this.renderEditor();
    }

    closeEditor() {
        this.editorOpen = false;
        const dialog = this.shadowRoot.querySelector('.editor-overlay');
        if (dialog) dialog.remove();
        this.render(); // Ensure brick state matches
    }

    updateEditorState(loading) {
        const btn = this.shadowRoot.getElementById('editor-apply-btn');
        if (btn) btn.loading = loading;
    }

    renderEditor() {
        // Remove existing if any
        let overlay = this.shadowRoot.querySelector('.editor-overlay');
        if (!overlay) {
            overlay = document.createElement('div');
            overlay.className = 'editor-overlay';
            this.shadowRoot.appendChild(overlay);
        }

        const slider = (label, key, min, max, step) => `
            <div class="control-row">
                <span class="label">${label}</span>
                <input type="range" min="${min}" max="${max}" step="${step}" 
                    value="${this.settings[key]}" 
                    oninput="this.getRootNode().host.updateSetting('${key}', parseFloat(this.value)); this.nextElementSibling.innerText = this.value">
                <span class="val">${this.settings[key]}</span>
            </div>
        `;

        overlay.innerHTML = `
            <style>
                .editor-overlay {
                    position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
                    background: rgba(0,0,0,0.85);
                    backdrop-filter: blur(5px);
                    z-index: 99999;
                    display: flex;
                    flex-direction: column;
                }
                .editor-header {
                    padding: 1rem 2rem;
                    background: #1e293b;
                    border-bottom: 1px solid rgba(255,255,255,0.1);
                    display: flex; justify-content: space-between; align-items: center;
                }
                .editor-body {
                    flex: 1;
                    display: flex;
                    overflow: hidden;
                }
                .preview-pane {
                    flex: 1;
                    padding: 2rem;
                    display: flex; align-items: center; justify-content: center;
                    background: #0f172a;
                }
                .preview-pane img {
                     max-width: 100%; max-height: 100%; 
                     box-shadow: 0 0 20px rgba(0,0,0,0.5);
                     border-radius: 4px;
                }
                .controls-pane {
                    width: 350px;
                    background: #1e293b;
                    border-left: 1px solid rgba(255,255,255,0.1);
                    display: flex; flex-direction: column;
                }
                .control-content {
                    flex: 1;
                    overflow-y: auto;
                    padding: 1rem;
                }
                .control-row { display: flex; align-items: center; gap: 0.5rem; font-size: 0.8rem; margin-bottom: 0.8rem; }
                .label { width: 80px; color: #94a3b8; }
                input[type=range] { flex: 1; accent-color: #f59e0b; }
                .val { width: 35px; text-align: right; color: #cbd5e1; font-family: monospace; }
                
                sl-tab-group { --indicator-color: #f59e0b; height: 100%; }
                sl-tab-panel { height: 100%; }
                
                .action-bar {
                    padding: 1rem;
                    border-top: 1px solid rgba(255,255,255,0.1);
                    display: flex; gap: 0.5rem; flex-direction: column;
                }
            </style>
            
            <div class="editor-header">
                <div style="font-weight:bold; font-size:1.2rem; color:#f59e0b;">
                    <sl-icon name="sliders" style="vertical-align:middle; margin-right:0.5rem;"></sl-icon>
                    Photo Editor
                </div>
                <sl-icon-button name="x-lg" style="font-size:1.5rem;" onclick="this.getRootNode().host.closeEditor()"></sl-icon-button>
            </div>

            <div class="editor-body">
                <div class="preview-pane">
                    ${this.previewImage
                ? `<img src="${this.previewImage}">`
                : `<div style="color:#64748b">No Image</div>`
            }
                </div>
                <div class="controls-pane">
                    <div class="control-content">
                        <sl-tab-group>
                            <sl-tab slot="nav" panel="basic">Basic</sl-tab>
                            <sl-tab slot="nav" panel="color">Color</sl-tab>
                            <sl-tab slot="nav" panel="effects">FX</sl-tab>

                            <sl-tab-panel name="basic">
                                <div style="display:flex; flex-direction:column; gap:0.5rem; padding-top:1rem;">
                                    ${slider("Contrast", "contrast", 0.5, 2.0, 0.1)}
                                    ${slider("Saturation", "saturation", 0.0, 2.0, 0.1)}
                                    ${slider("Brightness", "color_boost", 0.5, 1.5, 0.05)}
                                    ${slider("Sharpness", "sharpness_factor", 0.0, 5.0, 0.5)}
                                    ${slider("Blur", "blur_radius", 0, 10, 1)}
                                </div>
                            </sl-tab-panel>

                            <sl-tab-panel name="color">
                                <div style="display:flex; flex-direction:column; gap:0.5rem; padding-top:1rem;">
                                   ${slider("Vibrance", "vibrance", -1.0, 1.0, 0.1)}
                                   ${slider("Hue Shift", "hue_angle", -180, 180, 10)}
                                   <div style="height:1px; background:rgba(255,255,255,0.1); margin:0.5rem 0;"></div>
                                   ${slider("Shift R", "color_shift_r", -255, 255, 5)}
                                   ${slider("Shift G", "color_shift_g", -255, 255, 5)}
                                   ${slider("Shift B", "color_shift_b", -255, 255, 5)}
                                </div>
                            </sl-tab-panel>

                            <sl-tab-panel name="effects">
                                <div style="display:flex; flex-direction:column; gap:1rem; padding-top:1rem;">
                                     <sl-select size="small" label="Special Filter" value="${this.settings.special_filter}" 
                                        onsl-change="this.getRootNode().host.updateSetting('special_filter', this.value)">
                                        <sl-option value="none">None</sl-option>
                                        <sl-option value="sepia">Sepia</sl-option>
                                        <sl-option value="contour">Contour</sl-option>
                                        <sl-option value="negative">Negative</sl-option>
                                        <sl-option value="emboss">Emboss</sl-option>
                                        <sl-option value="vignette">Vignette</sl-option>
                                        <sl-option value="pixelize">Pixelize</sl-option>
                                    </sl-select>
                                     <div style="height:1px; background:rgba(255,255,255,0.1);"></div>
                                     ${slider("Rotation", "rotation_angle", -180, 180, 90)}
                                     <sl-select size="small" label="Mirror" value="${this.settings.mirror_type}"
                                        onsl-change="this.getRootNode().host.updateSetting('mirror_type', this.value)">
                                        <sl-option value="none">None</sl-option>
                                        <sl-option value="horizontal">Horizontal</sl-option>
                                        <sl-option value="vertical">Vertical</sl-option>
                                    </sl-select>
                                </div>
                            </sl-tab-panel>
                        </sl-tab-group>
                    </div>

                    <div class="action-bar">
                         <sl-button variant="primary" id="editor-apply-btn" onclick="this.getRootNode().host.apply()" ?loading="${this.isProcessing}">
                            <sl-icon slot="prefix" name="magic"></sl-icon> Apply Effects
                        </sl-button>
                        <div style="display:flex; gap:0.5rem;">
                             <sl-button variant="success" style="flex:1;" onclick="this.getRootNode().host.saveAndClose()">
                                <sl-icon slot="prefix" name="check-lg"></sl-icon> Save & Close
                            </sl-button>
                            <sl-button variant="warning" outline style="flex:1;" onclick="this.getRootNode().host.sendToSource()">
                                <sl-icon slot="prefix" name="arrow-left"></sl-icon> Use Source
                            </sl-button>
                        </div>
                         <sl-button variant="neutral" outline size="small" onclick="this.getRootNode().host.reset()">
                            Reset Defaults
                        </sl-button>
                    </div>
                </div>
            </div>
        `;
    }

    // --- Main Brick UI ---

    render() {
        const brickId = this.getAttribute('brick-id') || '';

        this.shadowRoot.innerHTML = `
        <style>
            :host { display: block; }
            .container { display: flex; flex-direction: column; gap: 1rem; align-items:center; }
            
            .preview-thumb {
                width: 100%; aspect-ratio: 16/9;
                background: #0f172a;
                border-radius: 8px;
                overflow: hidden;
                display: flex; align-items: center; justify-content: center;
                border: 1px solid rgba(255,255,255,0.1);
                cursor: pointer;
                transition: border-color 0.2s;
            }
            .preview-thumb:hover { border-color: #f59e0b; }
            .preview-thumb img { width: 100%; height: 100%; object-fit: cover; }
        </style>

        <qp-cartridge title="Photo Editor" type="output" icon="sliders" brick-id="${brickId}">
            <div class="container">
                <div class="preview-thumb" onclick="this.getRootNode().host.openEditor()">
                    ${this.previewImage
                ? `<img src="${this.previewImage}">`
                : `<div style="color:#64748b; font-size:0.8rem; display:flex; flex-direction:column; align-items:center;">
                             <sl-icon name="image" style="font-size:2rem; margin-bottom:0.5rem;"></sl-icon>
                             Click to Load & Edit
                           </div>`
            }
                </div>

                <sl-button variant="primary" style="width:100%" onclick="this.getRootNode().host.openEditor()">
                    <sl-icon slot="prefix" name="pencil-square"></sl-icon> Open Editor
                </sl-button>
                
                ${this.previewImage ? `
                     <sl-button variant="neutral" outline style="width:100%" size="small" onclick="this.getRootNode().host.sendToSource()">
                        <sl-icon slot="prefix" name="arrow-up-circle"></sl-icon> Send to Source
                    </sl-button>
                ` : ''}
            </div>
        </qp-cartridge>
        `;
    }
}
customElements.define('qp-filter', QpFilter);
