/**
 * QpGridExplore - Automation Brick for Qpyt-UI
 * 
 * Logic_Zone: SAFE (<0.35) | Delta_Initial: 0.2 | Resolution: convergent
 * Purpose: Automates parameter sweeping (CFG, Steps, Denoise) with minimal UI.
 */
class QpGridExplore extends HTMLElement {
    constructor() {
        super();
        this.attachShadow({ mode: 'open' });
        this.parameter = 'guidance_scale';
        this.mode = 'contrast'; // contrast (3) vs deep (7)
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
                    gap: 1rem;
                }
                .label {
                    font-size: 0.85rem;
                    font-weight: 600;
                    color: #cbd5e1;
                    margin-bottom: 0.3rem;
                    display: block;
                }
                .launch-btn {
                    width: 100%;
                    --sl-color-primary-600: #10b981;
                }
                .hint {
                    font-size: 0.75rem;
                    color: #94a3b8;
                    margin-top: 0.5rem;
                }
            </style>
            <qp-cartridge title="Grid Explorer" type="generator" brick-id="${brickId}">
                <div class="container">
                    <div>
                        <span class="label">Parameter to Test</span>
                        <sl-select id="param-select" value="${this.parameter}" hoist>
                            <sl-option value="guidance_scale">Guidance Scale (CFG)</sl-option>
                            <sl-option value="num_inference_steps">Inference Steps</sl-option>
                            <sl-option value="denoising_strength">Denoising (Style/Inpaint)</sl-option>
                        </sl-select>
                    </div>

                    <div>
                        <span class="label">Sweep Intensity</span>
                        <sl-select id="mode-select" value="${this.mode}" hoist>
                            <sl-option value="contrast">Contrast (3 images)</sl-option>
                            <sl-option value="deep">Precision (7 images)</sl-option>
                        </sl-select>
                    </div>

                    <sl-button variant="primary" size="large" class="launch-btn" id="launch-btn">
                        <sl-icon slot="prefix" name="layers-half"></sl-icon>
                        Launch Exploration
                    </sl-button>

                    <div class="hint">
                        💡 This brick will use the nearest generator's settings to create a comparative series.
                    </div>
                </div>
            </qp-cartridge>
        `;

        this.shadowRoot.getElementById('param-select').addEventListener('sl-change', (e) => this.parameter = e.target.value);
        this.shadowRoot.getElementById('mode-select').addEventListener('sl-change', (e) => this.mode = e.target.value);
        this.shadowRoot.getElementById('launch-btn').addEventListener('click', () => this.launch());
    }

    async launch() {
        if (!window.qpyt_app) return;

        // 1. Find the target generator
        const generator = this.findTargetGenerator();
        if (!generator) {
            window.qpyt_app.notify("No generator found on the dashboard!", "danger");
            return;
        }

        // 2. Prepare values
        const values = this.getRange();
        
        window.qpyt_app.notify(`Launching exploration: ${values.length} images queued...`, "info");
        window.qpyt_audio?.play('start');

        let queued = 0;
        for (const val of values) {
            try {
                const payload = this.capturePayload(generator, val);
                const endpoint = generator.tagName.includes('INPAINT') ? '/inpaint' : (generator.tagName.includes('OUTPAINT') ? '/outpaint' : '/generate');

                const response = await fetch(endpoint, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });

                if (response.ok) queued++;
            } catch (e) {
                console.error("[GridExplore] Submission failed", e);
            }
        }

        window.qpyt_app.notify(`Exploration ready: ${queued} tasks added to the queue.`, "success");
        window.qpyt_audio?.play('finish');
    }

    findTargetGenerator() {
        return document.querySelector('qp-render-sdxl, qp-render-flux, qp-render-flux-klein, qp-render-sd35turbo, qp-img2img, qp-inpaint, qp-outpaint');
    }

    getRange() {
        if (this.parameter === 'guidance_scale') {
            return this.mode === 'contrast' ? [3.0, 7.0, 12.0] : [1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0];
        }
        if (this.parameter === 'num_inference_steps') {
            return this.mode === 'contrast' ? [15, 30, 50] : [10, 20, 30, 40, 50, 60, 80];
        }
        if (this.parameter === 'denoising_strength') {
            return this.mode === 'contrast' ? [0.3, 0.6, 0.9] : [0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 1.0];
        }
        return [1.0];
    }

    capturePayload(gen, sweepVal) {
        const promptEl = document.querySelector('qp-prompt');
        const settingsEl = document.querySelector('qp-settings') || document.querySelector('mg-settings');
        const stylesEl = document.querySelector('qp-styles');
        const loraManager = document.querySelector('qp-lora-manager');
        const controlNet = document.querySelector('qp-controlnet');
        const imageSource = document.querySelector('qp-image-input');
        const blenderSource = document.querySelector('qp-image-blender');

        if (!promptEl) return {};

        let { prompt, negative_prompt } = promptEl.getValue();
        if (stylesEl && typeof stylesEl.applyStyles === 'function') {
            const styled = stylesEl.applyStyles(prompt, negative_prompt);
            prompt = styled.prompt;
            negative_prompt = styled.negative_prompt;
        }

        const settings = settingsEl?.getValue() || {};
        const blenderData = blenderSource?.getValue() || null;
        
        const payload = {
            prompt,
            negative_prompt,
            model_type: gen.modelType || 'sdxl',
            model_name: gen.selectedModel,
            sampler_name: gen.selectedSampler || 'euler_a',
            vae_name: gen.selectedVae || null,
            ...settings,
            loras: loraManager ? loraManager.getValues().loras : [],
            controlnet_image: controlNet ? controlNet.getImage() : null,
            controlnet_conditioning_scale: controlNet ? controlNet.getStrength() : 0.7,
            controlnet_model: controlNet ? controlNet.getModel() : null,
            workflow: window.qpyt_app?.getCurrentWorkflowState() || null
        };

        const isImg2Img = gen.tagName.includes('IMG2IMG') || (gen.modelType === 'flux2' && imageSource?.getImage());
        if (isImg2Img || gen.tagName.includes('PAINT')) {
            payload.image = imageSource?.getImage() || null;
            payload.denoising_strength = gen.denoisingStrength;
        }
        if (gen.tagName.includes('PAINT')) {
            payload.mask = typeof gen.getMask === 'function' ? gen.getMask() : null;
        }
        if (blenderData) {
            payload.image_a = blenderData.image_a;
            payload.image_b = blenderData.image_b;
            payload.weight_a = blenderData.weight_a;
            payload.weight_b = blenderData.weight_b;
            payload.ip_adapter_scale = blenderData.fidelity;
        }

        payload[this.parameter] = sweepVal;
        
        return payload;
    }
}

customElements.define('qp-grid-explore', QpGridExplore);
