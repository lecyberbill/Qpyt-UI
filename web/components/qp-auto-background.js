/**
 * QpAutoBackground - Smart Background Replacement for Qpyt-UI
 * 
 * Logic_Zone: TRANSIT (0.4) | Delta_Initial: 0.3 | Resolution: convergent
 * Purpose: Removes background and generates a new one based on prompt.
 */
class QpAutoBackground extends HTMLElement {
    constructor() {
        super();
        this.attachShadow({ mode: 'open' });
        this.isProcessing = false;
    }

    connectedCallback() {
        this.render();
    }

    render() {
        const brickId = this.getAttribute('brick-id') || '';
        this.shadowRoot.innerHTML = `
            <style>
                .container {
                    display: flex;
                    flex-direction: column;
                    gap: 1.2rem;
                }
                .action-btn {
                    width: 100%;
                    --sl-color-primary-600: #8b5cf6;
                }
                .hint {
                    font-size: 0.75rem;
                    color: #64748b;
                    line-height: 1.4;
                    background: rgba(255,255,255,0.03);
                    padding: 0.8rem;
                    border-radius: 0.5rem;
                }
                .warning-box {
                    display: none;
                    background: rgba(245, 158, 11, 0.1);
                    border: 1px solid #f59e0b;
                    color: #f59e0b;
                    padding: 0.6rem;
                    border-radius: 0.4rem;
                    font-size: 0.75rem;
                    margin-bottom: 0.5rem;
                }
            </style>
            <qp-cartridge title="Auto-Background" type="generator" brick-id="${brickId}">
                <div class="container">
                    <div id="dim-warning" class="warning-box">
                        ⚠️ <b>Resolution Mismatch:</b> The source image aspect ratio doesn't match your settings. Image may be distorted.
                    </div>

                    <sl-button variant="primary" size="large" class="action-btn" id="run-btn" ${this.isProcessing ? 'loading' : ''}>
                        <sl-icon slot="prefix" name="magic"></sl-icon>
                        Update Background
                    </sl-button>

                    <div class="hint">
                        ✨ This brick uses the <b>Global Prompt</b> to generate a new background. 
                        <br><br>
                        💡 <i>Tip: Set your Generator resolution to match the Source Image to avoid stretching.</i>
                    </div>
                </div>
            </qp-cartridge>
        `;

        this.shadowRoot.getElementById('run-btn').addEventListener('click', () => this.execute());
        this.checkResolution();
    }

    async checkResolution() {
        // Periodic check or on-click check for the warning
        const imageSource = document.querySelector('qp-image-input');
        const imgData = imageSource?.getImage();
        if (!imgData) return;

        try {
            const dims = await this.getImageDimensions(imgData);
            const settingsEl = document.querySelector('qp-settings') || document.querySelector('mg-settings');
            const settings = settingsEl?.getValue() || {};
            const targetW = settings.width || 1024;
            const targetH = settings.height || 1024;

            const sourceRatio = dims.width / dims.height;
            const targetRatio = targetW / targetH;

            const warningBox = this.shadowRoot.getElementById('dim-warning');
            if (Math.abs(sourceRatio - targetRatio) > 0.05) {
                warningBox.style.display = 'block';
                warningBox.innerHTML = `⚠️ <b>Distortion Risk:</b> Source is ${dims.width}x${dims.height} but settings are ${targetW}x${targetH}.`;
            } else {
                warningBox.style.display = 'none';
            }
        } catch (e) {}

        // Re-check in 3 seconds to keep UI reactive
        setTimeout(() => this.checkResolution(), 3000);
    }

    getImageDimensions(base64) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.onload = () => resolve({ width: img.width, height: img.height });
            img.onerror = reject;
            img.src = base64;
        });
    }

    async execute() {
        if (!window.qpyt_app || this.isProcessing) return;

        const imageSource = document.querySelector('qp-image-input');
        const imgData = imageSource?.getImage();
        
        if (!imgData) {
            window.qpyt_app.notify("Source image is empty! Add an image in 'Source Image'.", "warning");
            return;
        }

        // Final Resolution Warning before start
        const dims = await this.getImageDimensions(imgData);
        const settingsEl = document.querySelector('qp-settings') || document.querySelector('mg-settings');
        const settings = settingsEl?.getValue() || {};
        const targetW = settings.width || 1024;
        const targetH = settings.height || 1024;

        if (Math.abs((dims.width/dims.height) - (targetW/targetH)) > 0.05) {
            if (!confirm(`Warning: The source image (${dims.width}x${dims.height}) does not match the output resolution (${targetW}x${targetH}). The result will be distorted (stretched). Continue anyway?`)) {
                return;
            }
        }

        this.isProcessing = true;
        this.render();
        window.qpyt_audio?.play('start');

        try {
            window.qpyt_app.notify("Analyzing image and removing background...", "info");
            const rembgRes = await fetch('/rembg', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: imgData })
            });
            const rembgData = await rembgRes.json();
            
            if (rembgData.status !== 'queued') throw new Error("Queue error (RemBG)");

            const foregroundUrl = await this.pollTask(rembgData.task_id);
            if (!foregroundUrl) throw new Error("Background removal failed");

            window.qpyt_app.notify("Generating new background...", "info");
            
            const payload = await this.prepareInpaintPayload(imgData, foregroundUrl);
            
            const inpaintRes = await fetch('/inpaint', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            if (!inpaintRes.ok) {
                const errText = await inpaintRes.text();
                let errMsg = "Inpainting submission failed";
                try {
                    const json = JSON.parse(errText);
                    errMsg = json.message || json.detail || errMsg;
                } catch(e) {}
                throw new Error(`${errMsg} (Status: ${inpaintRes.status})`);
            }

            const inpaintData = await inpaintRes.json();
            
            if (inpaintData.status === 'queued') {
                window.qpyt_app.notify("Generation task added to queue...", "success");
                const finalUrl = await this.pollTask(inpaintData.task_id);
                if (finalUrl) {
                    window.qpyt_app.notify("Background update complete!", "success");
                    window.qpyt_audio?.play('finish');
                    // Important: Broadcast result for history and other bricks
                    window.dispatchEvent(new CustomEvent('qpyt-output', { detail: { url: finalUrl } }));
                    if (window.qpyt_app) window.qpyt_app.lastImage = finalUrl;
                } else {
                    throw new Error("Generation failed (Inpaint)");
                }
            } else {
                throw new Error(inpaintData.message || "Inpainting submission failed (Unexpected status)");
            }

        } catch (e) {
            console.error("[AutoBackground] Error:", e);
            window.qpyt_app.notify(`Background Error: ${e.message}`, "danger");
        } finally {
            this.isProcessing = false;
            this.render();
        }
    }

    async pollTask(taskId) {
        while (true) {
            const res = await fetch(`/queue/status/${taskId}`);
            const task = await res.json();
            if (task.status === 'COMPLETED') return task.result.image_url;
            if (task.status === 'FAILED') return null;
            await new Promise(r => setTimeout(r, 1000));
        }
    }

    async prepareInpaintPayload(originalBase64, foregroundUrl) {
        const promptEl = document.querySelector('qp-prompt');
        const settingsEl = document.querySelector('qp-settings') || document.querySelector('mg-settings');
        const stylesEl = document.querySelector('qp-styles');
        
        let prompt = "a professional studio background";
        if (promptEl) {
            const val = promptEl.getValue();
            prompt = val.prompt;
            if (stylesEl) prompt = stylesEl.applyStyles(prompt, "").prompt;
        }

        const settings = settingsEl?.getValue() || {};
        const gen = document.querySelector('qp-render-flux, qp-render-sdxl, qp-render-flux-klein, qp-render-sd35turbo');
        
        // Auto-sync dimensions to source to avoid distortion
        const dims = await this.getImageDimensions(originalBase64);
        
        return {
            prompt: prompt,
            model_type: gen?.modelType || 'flux2', 
            model_name: gen?.selectedModel || null,
            image: originalBase64,
            mask: foregroundUrl, 
            width: dims.width,
            height: dims.height,
            denoising_strength: 0.95,
            invert_mask: true,
            ...settings
        };
    }
}

customElements.define('qp-auto-background', QpAutoBackground);
