// Logic_Zone: TRANSIT | Delta_Initial: 0.55 | Resolution: convergent
// [WFGY-Metadata] Batch Runner Expansion and UX fixes
/**
 * Batch Runner Brick
 * Test multiple prompts and seeds systematically in a grid.
 */
class QpBatchRunner extends HTMLElement {
    static get observedAttributes() { return ['brick-id']; }

    constructor() {
        super();
        this.attachShadow({ mode: 'open' });
        this.prompts = ["A beautiful landscape", "A futuristic city"];
        this.seeds = ["1234, 5678"];
        this.results = []; // Array of {url, prompt, seed, status}
        this.isGenerating = false;
        this.targetBrickId = null;
        this.columns = 3;
    }

    connectedCallback() {
        this.render();
        // Initial scan for generators after a short delay for component upgrade
        setTimeout(() => this.updateGeneratorList(), 500);
    }

    attributeChangedCallback() {
        this.render();
    }

    getValue() {
        return {
            prompts: this.prompts,
            seeds: this.getSeedList(),
            target_brick_id: this.targetBrickId,
            columns: this.columns
        };
    }

    getSeedList() {
        return this.seeds.join(', ').split(',').map(s => s.trim()).filter(s => s && !isNaN(s)).map(Number);
    }

    async runBatch() {
        if (this.isGenerating) return;

        const targetEl = document.querySelector(`[brick-id="${this.targetBrickId}"]`);
        if (!targetEl || typeof targetEl.submitAndPollTask !== 'function') {
            window.qpyt_app?.notify("Please select a valid Generator brick first", "warning");
            return;
        }

        const seedList = this.getSeedList();
        if (seedList.length === 0) seedList.push(Math.floor(Math.random() * 1000000));

        const promptList = this.prompts.map(p => p.trim()).filter(p => p);
        if (promptList.length === 0) {
            window.qpyt_app?.notify("Please enter at least one prompt", "warning");
            return;
        }

        this.isGenerating = true;
        this.results = [];
        this.render();

        const total = promptList.length * seedList.length;
        let count = 0;

        // Force target brick into generating state to prevent polling cancellation
        const originalState = targetEl.isGenerating;
        targetEl.isGenerating = true;
        if (typeof targetEl.render === 'function') targetEl.render();

        try {
            // Context capture
            const settingsEl = document.querySelector('qp-settings') || document.querySelector('mg-settings');
            const genSettings = settingsEl ? settingsEl.getValue() : {};
            const loraManager = document.querySelector('qp-lora-manager');
            const stylesEl = document.querySelector('qp-styles');

            // Use a labeled loop to allow breaking out of both prompts and seeds
            mainLoop: for (const prompt of promptList) {
                for (const seed of seedList) {
                    if (!this.isGenerating) break mainLoop;

                    const resultEntry = { prompt, seed, status: 'pending', url: null };
                    this.results.push(resultEntry);
                    this.render();

                    try {
                        let activePrompt = prompt;
                        let activeNeg = genSettings.negative_prompt || "";

                        if (stylesEl && typeof stylesEl.applyStyles === 'function') {
                            const styled = stylesEl.applyStyles(activePrompt, activeNeg);
                            activePrompt = styled.prompt;
                            activeNeg = styled.negative_prompt;
                        }

                        // Construction of a robust payload
                        // Capture latest specialized params from target brick properties (in case they changed)
                        const payload = {
                            ...genSettings,
                            prompt: activePrompt,
                            negative_prompt: activeNeg,
                            seed: seed,
                            model_type: targetEl.modelType || null,
                            model_name: targetEl.selectedModel || null,
                            sampler_name: targetEl.selectedSampler || null,
                            vae_name: targetEl.selectedVae || null,
                            duration: targetEl.duration || genSettings.duration || undefined,
                            guidance_scale: targetEl.guidance_scale || targetEl.guidance || genSettings.guidance_scale || undefined,
                            steps: targetEl.steps || genSettings.num_inference_steps || undefined,
                            frames: targetEl.frames || genSettings.frames || undefined,
                            loras: loraManager ? loraManager.getValues().loras : [],
                            workflow: window.qpyt_app?.getCurrentWorkflowState() || null
                        };

                        const endpoint = targetEl.endpoint || '/generate';
                        console.log(`[Batch Runner] (${count + 1}/${total}) Submitting to ${endpoint}`);
                        const data = await targetEl.submitAndPollTask(endpoint, payload);

                        // Double check if we were stopped during the await
                        if (!this.isGenerating) {
                            resultEntry.status = 'error';
                            break mainLoop;
                        }

                        if (data && (data.image_url || data.preview)) {
                            resultEntry.url = data.image_url || data.preview;
                            resultEntry.status = 'success';

                            // Signal for Dashboard History
                            window.dispatchEvent(new CustomEvent('qpyt-status-update', {
                                detail: {
                                    image_url: resultEntry.url,
                                    status: 'success',
                                    request_id: data.request_id || `batch-${Date.now()}-${Math.random()}`,
                                    prompt: activePrompt,
                                    metadata: {
                                        seed: seed,
                                        prompt: activePrompt,
                                        model_name: targetEl.selectedModel || targetEl.modelType || "Batch",
                                        num_inference_steps: genSettings.num_inference_steps || targetEl.steps,
                                        guidance_scale: genSettings.guidance_scale || targetEl.guidance_scale || targetEl.guidance,
                                        width: targetEl.width || genSettings.width,
                                        height: targetEl.height || genSettings.height
                                    }
                                }
                            }));

                            // Trigger legacy qpyt-output for other bricks
                            window.dispatchEvent(new CustomEvent('qpyt-output', {
                                detail: {
                                    url: resultEntry.url,
                                    brickId: this.getAttribute('brick-id'),
                                    params: { seed, prompt: activePrompt, model: targetEl.modelType || targetEl.tagName.toLowerCase() }
                                },
                                bubbles: true,
                                composed: true
                            }));
                        } else {
                            resultEntry.status = 'error';
                        }
                    } catch (e) {
                        console.error("[Batch Runner] Task failed:", e);
                        resultEntry.status = 'error';
                    }
                    count++;
                    this.render();
                }
            }
        } finally {
            this.isGenerating = false;
            targetEl.isGenerating = originalState; // Restore target brick state
            if (typeof targetEl.render === 'function') targetEl.render();
            this.render();
            window.qpyt_app?.notify(`Batch complete: ${count}/${total} tasks processed`, "success");
        }
    }

    stopBatch() {
        console.log("[Batch Runner] Stop requested.");
        this.isGenerating = false;
        // Stop the target brick too so its polling exits
        const targetEl = document.querySelector(`[brick-id="${this.targetBrickId}"]`);
        if (targetEl) targetEl.isGenerating = false;
        this.render();
    }

    findGenerators() {
        // Broaden detection: look for anything that has a brick-id and the required API
        return Array.from(document.querySelectorAll('*'))
            .filter(el => {
                const tag = el.tagName.toLowerCase();
                const isKnownGen = (tag.startsWith('qp-render') ||
                    tag === 'qp-img2img' ||
                    tag === 'qp-inpaint' ||
                    tag === 'qp-outpaint' ||
                    tag === 'q-upscaler-v3' ||
                    tag === 'qp-rembg' ||
                    tag === 'qp-sprite' ||
                    tag === 'qp-music-gen');

                // Crucial check: Batch Runner needs submitAndPollTask
                const hasApi = typeof el.submitAndPollTask === 'function';
                return el.hasAttribute('brick-id') && (isKnownGen || hasApi);
            })
            .map(el => {
                const id = el.getAttribute('brick-id');
                const tag = el.tagName.toLowerCase();
                let label = tag.replace('qp-render-', '').replace('qp-', '').toUpperCase();
                if (tag === 'q-upscaler-v3') label = "UPSCALER V3";
                return { id, name: label, canBatch: typeof el.submitAndPollTask === 'function' };
            });
    }

    updateGeneratorList() {
        const generators = this.findGenerators();
        console.log("[Batch Runner] Found generators:", generators);
        const tSelect = this.shadowRoot.getElementById('target-select');
        if (!tSelect) return;

        // Update options without full re-render
        let html = '<sl-option value="">-- Select Generator --</sl-option>';
        if (generators.length === 0) {
            html += '<sl-option value="" disabled>No generators found</sl-option>';
        } else {
            html += generators.map(g => `
                <sl-option value="${g.id}" ${!g.canBatch ? 'disabled' : ''}>
                    ${g.name} [${g.id}] ${!g.canBatch ? '(Not batchable)' : ''}
                </sl-option>
            `).join('');
        }
        tSelect.innerHTML = html;
        tSelect.value = this.targetBrickId || '';
    }

    render() {
        const brickId = this.getAttribute('brick-id') || '';
        const generators = this.findGenerators();

        this.shadowRoot.innerHTML = `
            <style>
                .batch-container {
                    display: flex;
                    flex-direction: column;
                    gap: 1rem;
                }
                .grid-view {
                    display: grid;
                    grid-template-columns: repeat(${this.columns}, 1fr);
                    gap: 8px;
                    margin-top: 1rem;
                }
                .result-card {
                    aspect-ratio: 1/1;
                    background: #0f172a;
                    border-radius: 8px;
                    overflow: hidden;
                    border: 1px solid rgba(255,255,255,0.1);
                    position: relative;
                    cursor: pointer;
                }
                .result-card img {
                    width: 100%;
                    height: 100%;
                    object-fit: cover;
                }
                .result-card .badge {
                    position: absolute;
                    bottom: 4px;
                    right: 4px;
                    font-size: 0.6rem;
                    background: rgba(0,0,0,0.6);
                    color: white;
                    padding: 2px 4px;
                    border-radius: 4px;
                }
                .loader {
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    height: 100%;
                    color: #6366f1;
                }
                .config-section {
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 1rem;
                }
                .controls-row {
                    display: flex; 
                    gap: 1rem; 
                    align-items: flex-end;
                    flex-wrap: wrap;
                }
                sl-select {
                    min-width: 200px;
                }
                textarea {
                    width: 100%;
                    background: #1e293b;
                    color: #e2e8f0;
                    border: 1px solid rgba(255,255,255,0.1);
                    border-radius: 4px;
                    padding: 8px;
                    font-family: inherit;
                    font-size: 0.8rem;
                    resize: vertical;
                }
                #viewer::part(panel) {
                    background: #1e293b;
                    border: 1px solid rgba(255,255,255,0.1);
                    max-width: 90vw;
                }
                .viewer-content img {
                    width: 100%;
                    height: auto;
                    border-radius: 8px;
                }
                .viewer-meta {
                    margin-top: 1rem;
                    padding: 1rem;
                    background: rgba(0,0,0,0.3);
                    border-radius: 8px;
                    font-size: 0.85rem;
                    color: #94a3b8;
                }
            </style>
            <qp-cartridge title="Batch Runner" type="tool" brick-id="${brickId}">
                <div class="batch-container">
                    <div class="config-section">
                        <div>
                            <sl-label>Prompts (one per line)</sl-label>
                            <textarea id="prompts-input" rows="4" placeholder="Enter prompts...">${this.prompts.join('\n')}</textarea>
                        </div>
                        <div>
                            <sl-label>Seeds (comma separated)</sl-label>
                            <textarea id="seeds-input" rows="4" placeholder="1234, 5678...">${this.seeds.join(', ')}</textarea>
                        </div>
                    </div>

                    <div class="controls-row">
                        <sl-select id="target-select" label="Target Generator" value="${this.targetBrickId || ''}" style="flex: 2;" hoist>
                            <sl-icon name="arrow-clockwise" slot="prefix" id="refresh-list" style="cursor: pointer;"></sl-icon>
                            <sl-option value="">-- Select Generator --</sl-option>
                            ${generators.length === 0 ? '<sl-option value="" disabled>No generators found</sl-option>' : ''}
                            ${generators.map(g => `<sl-option value="${g.id}">${g.name} [${g.id}]</sl-option>`).join('')}
                        </sl-select>

                        <sl-input id="cols-input" type="number" label="Cols" value="${this.columns}" min="1" max="6" style="width: 70px;"></sl-input>

                        <sl-button id="run-btn" variant="${this.isGenerating ? 'danger' : 'primary'}" style="flex: 1;">
                            <sl-icon slot="prefix" name="${this.isGenerating ? 'stop-fill' : 'play-fill'}"></sl-icon>
                            ${this.isGenerating ? 'Stop Batch' : 'Run Batch'}
                        </sl-button>
                    </div>

                    <div class="grid-view">
                        ${this.results.map(res => `
                            <div class="result-card" data-url="${res.url || ''}">
                                ${res.url ? `<img src="${res.url}">` : `
                                    <div class="loader">
                                        ${res.status === 'pending' ? '<sl-spinner style="font-size: 1.5rem;"></sl-spinner>' : '<sl-icon name="exclamation-triangle"></sl-icon>'}
                                    </div>
                                `}
                                <div class="badge">${res.seed}</div>
                            </div>
                        `).join('')}
                    </div>
                </div>
            </qp-cartridge>

            <sl-dialog id="viewer" label="Batch Result Viewer" class="viewer-dialog">
                <div class="viewer-content">
                    <img id="viewer-img" src="">
                    <div class="viewer-meta" id="viewer-meta"></div>
                </div>
                <sl-button slot="footer" variant="primary" onclick="this.closest('sl-dialog').hide()">Close</sl-button>
            </sl-dialog>
        `;

        const pInput = this.shadowRoot.getElementById('prompts-input');
        const sInput = this.shadowRoot.getElementById('seeds-input');
        const tSelect = this.shadowRoot.getElementById('target-select');
        const cInput = this.shadowRoot.getElementById('cols-input');
        const runBtn = this.shadowRoot.getElementById('run-btn');
        const refreshBtn = this.shadowRoot.getElementById('refresh-list');

        pInput?.addEventListener('input', () => { this.prompts = pInput.value.split('\n'); });
        sInput?.addEventListener('input', () => { this.seeds = [sInput.value]; });
        tSelect?.addEventListener('sl-change', () => { this.targetBrickId = tSelect.value; });

        // Auto-refresh when clicking select to find new bricks
        tSelect?.addEventListener('mousedown', () => {
            console.log("[Batch Runner] Select clicked, re-scanning for bricks...");
            this.updateGeneratorList();
        });

        refreshBtn?.addEventListener('click', (e) => {
            e.stopPropagation();
            this.updateGeneratorList();
        });
        cInput?.addEventListener('sl-input', () => {
            this.columns = parseInt(cInput.value) || 3;
            const grid = this.shadowRoot.querySelector('.grid-view');
            if (grid) grid.style.gridTemplateColumns = `repeat(${this.columns}, 1fr)`;
        });

        runBtn?.addEventListener('click', () => {
            if (this.isGenerating) this.stopBatch();
            else this.runBatch();
        });

        this.shadowRoot.querySelectorAll('.result-card').forEach((card, idx) => {
            card.addEventListener('click', () => {
                const res = this.results[idx];
                if (res && res.url) {
                    const dialog = this.shadowRoot.getElementById('viewer');
                    const img = this.shadowRoot.getElementById('viewer-img');
                    const meta = this.shadowRoot.getElementById('viewer-meta');
                    img.src = res.url;
                    meta.innerHTML = `
                        <div style="color: #e2e8f0; margin-bottom: 0.5rem;"><strong>Prompt:</strong> ${res.prompt}</div>
                        <div><strong>Seed:</strong> ${res.seed}</div>
                    `;
                    dialog.show();
                }
            });
        });
    }
}

customElements.define('qp-batch-runner', QpBatchRunner);
