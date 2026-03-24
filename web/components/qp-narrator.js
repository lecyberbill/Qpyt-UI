/**
 * QpNarrator - Storytelling Brick for Qpyt-UI
 * 
 * Logic_Zone: TRANSIT (0.5) | Delta_Initial: 0.4 | Resolution: convergent
 * Purpose: Generates a 4-panel story with prompts and captions via LLM.
 */
class QpNarrator extends HTMLElement {
    constructor() {
        super();
        this.attachShadow({ mode: 'open' });
        this.isThinking = false;
        this.isGenerating = false;
        this.panels = []; // { caption, prompt }
        this.lastImageUrl = "";
    }

    connectedCallback() {
        this.render();
        this.setupObserver();
    }

    disconnectedCallback() {
        this.observer?.disconnect();
    }

    setupObserver() {
        setTimeout(() => {
            const parent = this.parentElement;
            if (!parent) return;
            this.observer = new MutationObserver(() => this.render());
            this.observer.observe(parent, { childList: true });
        }, 100);
    }

    render() {
        const brickId = this.getAttribute('brick-id') || '';
        this.shadowRoot.innerHTML = `
            <style>
                .narrator-container { display: flex; flex-direction: column; gap: 1rem; }
                .output-preview {
                    width: 100%; aspect-ratio: 16/9; background: #0f172a; border-radius: 8px; overflow: hidden; border: 1px solid rgba(255,255,255,0.1);
                    margin-bottom: 0.5rem;
                }
                .output-preview img { width: 100%; height: 100%; object-fit: contain; }
                .panel-list { display: flex; flex-direction: column; gap: 0.8rem; margin-top: 0.5rem; max-height: 400px; overflow-y: auto; }
                .panel-item {
                    background: rgba(255,255,255,0.05);
                    padding: 0.6rem;
                    border-radius: 8px;
                    border-left: 3px solid #ec4899;
                }
                .panel-caption { font-weight: 800; color: #f472b6; margin-bottom: 0.2rem; font-size: 0.8rem; }
                .panel-prompt { 
                    opacity: 0.7; 
                    font-style: italic; 
                    word-break: break-word;
                    white-space: normal;
                    font-size: 0.75rem;
                }
                .gen-all-btn { width: 100%; margin-top: 0.8rem; }
                .welcome-guide {
                    margin-top: 1rem; padding: 1rem; border: 1px dashed rgba(255, 255, 255, 0.1); border-radius: 12px; background: rgba(0,0,0,0.2);
                }
                .welcome-step { display: flex; gap: 0.8rem; margin-bottom: 0.6rem; font-size: 0.8rem; line-height: 1.3; color: #94a3b8; align-items: flex-start; }
                .step-num { 
                    width: 18px; height: 18px; background: #6366f1; color: white; border-radius: 50%; 
                    display: flex; align-items: center; justify-content: center; font-size: 0.65rem; font-weight: bold; flex-shrink: 0;
                }
                .warning-notice {
                    background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.2);
                    color: #ef4444; padding: 0.8rem; border-radius: 8px; font-size: 0.75rem; margin-bottom: 0.5rem;
                }
            </style>
            <qp-cartridge title="Story Narrator" icon="book" type="input" brick-id="${brickId}">
                <div class="narrator-container">
                    ${!document.querySelector('qp-llm-assistant') ? `
                        <div class="warning-notice">
                            ⚠️ <b>Assistant Missing</b><br>
                            Add an "LLM Assistant" brick to use this feature.
                        </div>
                    ` : ''}

                    ${this.panels.length > 0 ? `
                        <div class="output-preview">
                            <img src="${this.lastImageUrl || ''}" style="display: ${this.lastImageUrl ? 'block' : 'none'}">
                        </div>
                    ` : ''}
                    
                    <sl-button variant="primary" size="large" class="narrate-btn" id="run-btn" 
                        ${this.isThinking || !document.querySelector('qp-llm-assistant') ? 'loading' : ''}
                        ${!document.querySelector('qp-llm-assistant') ? 'disabled' : ''}>
                        <sl-icon slot="prefix" name="book"></sl-icon>
                        Imagine the Story
                    </sl-button>

                    ${this.panels.length === 0 ? `
                        <div class="welcome-guide">
                            <div class="welcome-step"><div class="step-num">1</div><div><b>Prepare</b>: Use the LLM Assistant (Narrator preset) to refine your story.</div></div>
                            <div class="welcome-step"><div class="step-num">2</div><div><b>Imagine</b>: Click "Imagine the Story" above. The narrator will read your chat.</div></div>
                            <div class="welcome-step"><div class="step-num">3</div><div><b>Generate</b>: Once the storyboard appears, click "Generate 4 Images".</div></div>
                        </div>
                    ` : `
                        <div class="panel-list">
                            ${this.panels.map((p, i) => `
                                <div class="panel-item">
                                    <div class="panel-caption">Panel ${i + 1}: ${p.caption}</div>
                                    <div class="panel-prompt" title="${p.prompt}">${p.prompt}</div>
                                </div>
                            `).join('')}
                        </div>
                        <sl-button variant="success" size="small" class="gen-all-btn" id="gen-all-btn" ${this.isGenerating ? 'loading' : ''}>
                            <sl-icon slot="prefix" name="play-circle"></sl-icon>
                            Generate 4 Images
                        </sl-button>
                    `}
                    
                    <div style="font-size: 0.7rem; color: #64748b; font-style: italic; margin-top: 1rem; text-align: center;">
                        Transform a simple idea into a coherent storyboard.
                    </div>
                </div>
            </qp-cartridge>
        `;

        this.shadowRoot.getElementById('run-btn').addEventListener('click', () => this.generateStory());
        this.shadowRoot.getElementById('gen-all-btn')?.addEventListener('click', () => this.generateAllImages());
    }

    async generateStory() {
        if (!window.qpyt_app || this.isThinking) return;
        this.isThinking = true;
        this.render();
        window.qpyt_audio?.play('start');

        try {
            const assistant = document.querySelector('qp-llm-assistant');
            if (!assistant) return;

            const assistantData = assistant.getValue();
            const provider = assistantData.provider || 'local';
            const chatHistory = assistantData.messages || [];
            
            const promptEl = document.querySelector('qp-prompt');
            const storyIdea = promptEl ? promptEl.getValue().prompt : "A mysterious adventure";

            const messages = chatHistory.length > 0 
                ? [...chatHistory, { role: 'user', content: "Now, generate the 4-panel JSON storyboard for this story." }]
                : [{ role: 'user', content: storyIdea }];

            const systemPrompt = `You are an AI screenwriter. For the provided idea or conversation, you MUST create a storyboard of EXACTLY 4 panels. 
            CHARACTER CONSISTENCY IS CRITICAL: Describe the main character's physical traits (color, clothes, features) identically in EVERY prompt.
            Strictly NO other text, NO preamble, NO conversational noise.
            JSON response format only: 
            [
              {"caption": "short description", "prompt": "precise SD prompt with same character details"},
              {"caption": "short description", "prompt": "precise SD prompt with same character details"},
              {"caption": "short description", "prompt": "precise SD prompt with same character details"},
              {"caption": "short description", "prompt": "precise SD prompt with same character details"}
            ]`;
            
            const response = await fetch('/prompt/assistant', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    provider: provider, 
                    messages: messages,
                    system_prompt: systemPrompt,
                    temperature: 0.7
                })
            });

            const data = await response.json();
            if (data.status === 'success') {
                let content = data.content.trim();
                content = content.replace(/```json/gi, '').replace(/```/gi, '').trim();
                const firstBracket = content.indexOf('[');
                const lastBracket = content.lastIndexOf(']');
                if (firstBracket !== -1 && lastBracket !== -1) content = content.substring(firstBracket, lastBracket + 1);

                this.panels = JSON.parse(content);
                window.qpyt_app.notify("Story successfully imagined!", "success");
            }
        } catch (e) {
            console.error("[Narrator] Story gen failed", e);
            window.qpyt_app.notify(`Narrator error: ${e.message}`, "danger");
        } finally {
            this.isThinking = false;
            this.render();
        }
    }

    async generateAllImages() {
        if (!window.qpyt_app || this.panels.length === 0 || this.isGenerating) return;

        const gen = document.querySelector('qp-render-flux, qp-render-sdxl, qp-render-flux-klein, qp-render-sd35turbo');
        if (!gen) {
            window.qpyt_app.notify("No active generator found!", "danger");
            return;
        }

        this.isGenerating = true;
        this.render();
        window.qpyt_app.notify(`Launching story production: 4 images...`, "info");
        
        try {
            for (let i = 0; i < this.panels.length; i++) {
                const panel = this.panels[i];
                window.qpyt_app.notify(`Generating Panel ${i+1}/4...`, "info");

                const settingsEl = document.querySelector('qp-settings') || document.querySelector('mg-settings');
                const settings = settingsEl?.getValue() || {};
                const loraManager = document.querySelector('qp-lora-manager');
                
                const payload = {
                    prompt: panel.prompt,
                    model_type: gen.modelType || 'flux',
                    model_name: gen.selectedModel,
                    ...settings,
                    loras: loraManager ? loraManager.getValues().loras : [],
                    workflow: window.qpyt_app?.getCurrentWorkflowState() || null
                };

                const response = await fetch('/generate', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });
                
                if (!response.ok) throw new Error("Failed to queue task");
                const res = await response.json();
                const taskId = res.task_id;

                let completed = false;
                while (!completed) {
                    const statusResp = await fetch(`/queue/status/${taskId}`);
                    const task = await statusResp.json();

                    if (task.status === 'COMPLETED') {
                        completed = true;
                        this.lastImageUrl = task.result.image_url;
                        if (window.qpyt_app) window.qpyt_app.lastImage = this.lastImageUrl;
                        
                        const dashboard = document.querySelector('qp-dashboard');
                        if (dashboard) {
                            dashboard.addEntry({
                                ...task.result,
                                request_id: taskId,
                                status: 'success',
                                prompt: panel.prompt
                            });
                        }

                        window.dispatchEvent(new CustomEvent('qpyt-output', {
                            detail: { url: this.lastImageUrl, brickId: this.getAttribute('brick-id') },
                            bubbles: true, composed: true
                        }));
                        window.qpyt_audio?.play('finish');
                    } else if (task.status === 'FAILED') {
                        throw new Error(`Panel ${i+1} failed: ${task.error}`);
                    }
                    if (!completed) await new Promise(r => setTimeout(r, 1500));
                }
            }
            window.qpyt_app.notify("Full story completed!", "success");
        } catch (e) {
            console.error("[Narrator] Generation failed", e);
            window.qpyt_app.notify(`Generation error: ${e.message}`, "danger");
        } finally {
            this.isGenerating = false;
            this.render();
        }
    }
}

customElements.define('qp-narrator', QpNarrator);
