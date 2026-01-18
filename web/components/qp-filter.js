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

        this.originalImage = null; // Absolute original (for reset)
        this.workingImage = null; // Current committed state
        this.previewImage = null; // (Unused in new logic, alias to workingImage)
        this.editorOpen = false;
    }

    connectedCallback() {
        this.render();
    }

    // --- Logic ---

    async loadImage(forceSource = false) {
        let imgToLoad = null;

        // 1. Try Source if forced or if no lastImage
        if (forceSource) {
            const sourceBrick = document.querySelector('qp-image-input');
            if (sourceBrick && (sourceBrick.base64 || sourceBrick.previewUrl)) {
                imgToLoad = sourceBrick.base64 || sourceBrick.previewUrl;
            } else {
                this.notify("No Source Image found", "warning");
                return;
            }
        } else {
            // Default behavior: Prefer Last Gen, then Source
            imgToLoad = window.qpyt_app?.lastImage;
            if (!imgToLoad) {
                const sourceBrick = document.querySelector('qp-image-input');
                if (sourceBrick && (sourceBrick.base64 || sourceBrick.previewUrl)) {
                    imgToLoad = sourceBrick.base64 || sourceBrick.previewUrl;
                }
            }
        }

        if (imgToLoad) {
            this.originalImage = imgToLoad;
            this.workingImage = imgToLoad;
            this.settings = { ...this.defaultSettings };
            this.render();
            if (this.editorOpen) this.renderEditor();
            this.notify(forceSource ? "Source Image Loaded" : "Image loaded", "success");
        } else {
            this.notify("No image found (History or Source)", "warning");
        }
    }

    async apply() {
        if (!this.workingImage) return;

        this.isProcessing = true;
        if (this.editorOpen) this.updateEditorState(true);

        try {
            const resp = await fetch('/filter', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    image: this.workingImage,
                    settings: this.settings
                })
            });
            const res = await resp.json();
            if (res.status === 'success') {
                // COMMIT: Update working image, reset sliders
                this.workingImage = res.image;
                this.settings = { ...this.defaultSettings };
                this.notify("Effects applied (Baked)", "success");
            } else {
                this.notify("Filter failed: " + res.message, "danger");
            }
        } catch (e) {
            console.error(e);
            this.notify("Network error", "danger");
        } finally {
            this.isProcessing = false;
            this.render();
            if (this.editorOpen) {
                this.renderEditor(); // Full re-render to reset sliders visually
            }
        }
    }

    reset() {
        this.workingImage = this.originalImage;
        this.settings = { ...this.defaultSettings };
        this.render();
        if (this.editorOpen) this.renderEditor();
    }

    async saveAndClose() {
        // Auto-bake if there are visual changes driven by settings
        if (this.hasUnsavedChanges()) {
            this.notify("Auto-baking pending changes...", "neutral");
            await this.apply();
        }

        if (this.workingImage && window.qpyt_app) {
            window.qpyt_app.lastImage = this.workingImage;
            window.dispatchEvent(new CustomEvent('qpyt-output', { detail: { url: this.workingImage } }));
            this.notify("Changes saved", "success");
        }
        this.closeEditor();
    }

    hasUnsavedChanges() {
        for (const key in this.defaultSettings) {
            if (this.settings[key] !== this.defaultSettings[key]) {
                return true;
            }
        }
        return false;
    }

    sendToSource() {
        if (!this.workingImage) return;
        const sourceBrick = document.querySelector('qp-image-input');
        if (sourceBrick) {
            sourceBrick.base64 = this.workingImage;
            sourceBrick.previewUrl = this.workingImage;
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
        // Real-time CSS Update
        if (this.editorOpen) {
            this.updateCSSPreview();
        }
    }

    get cssFilter() {
        const s = this.settings;
        const filters = [];
        // Basic CSS filters
        if (s.contrast !== 1.0) filters.push(`contrast(${s.contrast})`);
        if (s.color_boost !== 1.0) filters.push(`brightness(${s.color_boost})`);
        if (s.blur_radius > 0) filters.push(`blur(${s.blur_radius}px)`);
        if (s.hue_angle !== 0) filters.push(`hue-rotate(${s.hue_angle}deg)`);

        // Link to SVG Filter for Matrix operations (Saturation, Vibrance, RGB Shift, Special)
        filters.push('url(#dynamic-filter)');

        return filters.join(' ');
    }

    get cssTransform() {
        const s = this.settings;
        const transforms = [];
        if (s.rotation_angle !== 0) transforms.push(`rotate(${s.rotation_angle}deg)`);
        if (s.mirror_type === 'horizontal') transforms.push('scaleX(-1)');
        if (s.mirror_type === 'vertical') transforms.push('scaleY(-1)');
        return transforms.join(' ');
    }

    updateCSSPreview() {
        const img = this.shadowRoot.querySelector('.preview-pane img');
        if (img) {
            img.style.filter = this.cssFilter;
            img.style.transform = this.cssTransform;
        }

        // Update SVG Matrix
        const filterEl = this.shadowRoot.getElementById('dynamic-filter');
        if (filterEl) {
            const s = this.settings;

            // 1. Saturation / Vibrance (Approximated)
            // Standard Saturation Matrix
            const sat = s.saturation;
            const v = s.vibrance * 0.5;
            const S = sat + v;

            // Luminance constants for RGB
            const Lr = 0.2126;
            const Lg = 0.7152;
            const Lb = 0.0722;

            const sr = (1 - S) * Lr;
            const sg = (1 - S) * Lg;
            const sb = (1 - S) * Lb;

            let m00 = sr + S; let m01 = sg; let m02 = sb;
            let m10 = sr; let m11 = sg + S; let m12 = sb;
            let m20 = sr; let m21 = sg; let m22 = sb + S;

            // 2. RGB Shift (Additive offsets)
            const offR = s.color_shift_r / 255;
            const offG = s.color_shift_g / 255;
            const offB = s.color_shift_b / 255;

            let m04 = offR;
            let m14 = offG;
            let m24 = offB;

            // 3. Special Filters (Naive Overrides)
            if (s.special_filter === 'sepia') {
                m00 = 0.393; m01 = 0.769; m02 = 0.189; m04 = 0;
                m10 = 0.349; m11 = 0.686; m12 = 0.168; m14 = 0;
                m20 = 0.272; m21 = 0.534; m22 = 0.131; m24 = 0;
            }
            else if (s.special_filter === 'negative') {
                m00 = -1; m01 = 0; m02 = 0; m04 = 1;
                m10 = 0; m11 = -1; m12 = 0; m14 = 1;
                m20 = 0; m21 = 0; m22 = -1; m24 = 1;
            }

            // 4. Construct Filter Chain
            let svgContent = '';

            // Base Color Matrix
            svgContent += `<feColorMatrix in="SourceGraphic" result="color" type="matrix" values="
                 ${m00} ${m01} ${m02} 0 ${m04}
                 ${m10} ${m11} ${m12} 0 ${m14}
                 ${m20} ${m21} ${m22} 0 ${m24}
                 0 0 0 1 0
             "/>`;

            let lastResult = "color";

            // Convolution Filters
            if (s.special_filter === 'contour') {
                svgContent += `<feConvolveMatrix in="${lastResult}" result="contour" order="3" kernelMatrix="-1 -1 -1 -1 8 -1 -1 -1 -1" preserveAlpha="true"/>`;
                lastResult = "contour";
            }
            else if (s.special_filter === 'emboss') {
                svgContent += `<feConvolveMatrix in="${lastResult}" result="emboss" order="3" kernelMatrix="-2 -1 0 -1 1 1 0 1 2" preserveAlpha="true"/>`;
                lastResult = "emboss";
            }

            // Apply to innerHTML
            filterEl.innerHTML = svgContent;

            // Vignette (handled via CSS Overlay)
            const vOverlay = this.shadowRoot.getElementById('vignette-overlay');
            if (vOverlay) {
                vOverlay.style.opacity = (s.special_filter === 'vignette') ? "1" : "0";
            }
        }
    }

    // --- Editor Modal Management ---

    openEditor() {
        if (!this.workingImage) {
            this.loadImage().then(() => {
                if (this.workingImage) {
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

    updatePreviewImage() {
        // Only update the image element to avoid resetting sliders
        const container = this.shadowRoot.querySelector('.preview-pane');
        if (container) {
            container.innerHTML = this.previewImage
                ? `<img src="${this.previewImage}">`
                : `<div style="color:#64748b">No Image</div>`;
        }
        this.updateEditorState(false);
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
                <div class="preview-pane" style="position:relative;">
                    <!-- SVG Filter Definition -->
                    <svg width="0" height="0" style="position:absolute; pointer-events:none;">
                        <defs>
                            <filter id="dynamic-filter" color-interpolation-filters="sRGB">
                                <!-- Replaced by JS -->
                            </filter>
                        </defs>
                    </svg>
                    
                    ${this.workingImage
                ? `<img src="${this.workingImage}" style="filter:${this.cssFilter}; transform:${this.cssTransform}; transition: transform 0.1s;">`
                : `<div style="color:#64748b">No Image</div>`
            }
                    <!-- Vignette Overlay -->
                    <div id="vignette-overlay" style="position:absolute; top:0; left:0; width:100%; height:100%; background:radial-gradient(circle, transparent 50%, black 140%); opacity:0; pointer-events:none; transition:opacity 0.2s;"></div>
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
                                     <sl-select size="small" label="Special Filter" value="${this.settings.special_filter}" id="special-filter-select">
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
                                     <sl-select size="small" label="Mirror" value="${this.settings.mirror_type}" id="mirror-select">
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
                            <sl-icon slot="prefix" name="magic"></sl-icon> Apply (Bake)
                        </sl-button>
                        <div style="font-size:0.75rem; color:#64748b; text-align:center;">
                            'Apply' will bake the filters permanently and reset sliders.
                        </div>
                        <div style="display:flex; gap:0.5rem;">
                             <sl-button variant="success" style="flex:1;" onclick="this.getRootNode().host.saveAndClose()">
                                <sl-icon slot="prefix" name="check-lg"></sl-icon> Save & Close
                            </sl-button>
                            <sl-button variant="warning" outline style="flex:1;" onclick="this.getRootNode().host.loadImage(true)">
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



        // Attach Event Listeners for Custom Elements (sl-select) with improved reliability
        // We search within overlay to be sure
        setTimeout(() => {
            const specialSelect = overlay.querySelector('#special-filter-select');
            if (specialSelect) {
                console.log("[QP-Filter] Attaching listener to Special Filter");
                specialSelect.addEventListener('sl-change', (e) => {
                    console.log("[QP-Filter] Special Filter Changed:", e.target.value);
                    this.updateSetting('special_filter', e.target.value);
                });
            } else {
                console.error("[QP-Filter] Special Filter Element NOT FOUND");
            }

            const mirrorSelect = overlay.querySelector('#mirror-select');
            if (mirrorSelect) {
                mirrorSelect.addEventListener('sl-change', (e) => {
                    console.log("[QP-Filter] Mirror Changed:", e.target.value);
                    this.updateSetting('mirror_type', e.target.value);
                });
            }
        }, 0);

        // Initialize SVG Filter
        this.updateCSSPreview();
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
                    ${this.workingImage
                ? `<img src="${this.workingImage}">`
                : `<div style="color:#64748b; font-size:0.8rem; display:flex; flex-direction:column; align-items:center;">
                             <sl-icon name="image" style="font-size:2rem; margin-bottom:0.5rem;"></sl-icon>
                             Click to Load & Edit
                           </div>`
            }
                </div>

                <sl-button variant="primary" style="width:100%" onclick="this.getRootNode().host.openEditor()">
                    <sl-icon slot="prefix" name="pencil-square"></sl-icon> Open Editor
                </sl-button>
                
                <sl-button variant="neutral" outline style="width:100%" size="small" onclick="this.getRootNode().host.loadImage(true)">
                    <sl-icon slot="prefix" name="box-arrow-in-down"></sl-icon> Load Source Image
                </sl-button>
                
                ${this.workingImage ? `
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
