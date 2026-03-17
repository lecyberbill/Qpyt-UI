class QpLlmAssistant extends HTMLElement {
    constructor() {
        super();
        this.attachShadow({ mode: 'open' });
        this.messages = [];
        this.providers = [
            { id: 'local', label: 'Local (Qwen)' },
            { id: 'ollama', label: 'Ollama' },
            { id: 'lm-studio', label: 'LM Studio' },
            { id: 'gemini', label: 'Google Gemini' },
            { id: 'openai', label: 'ChatGPT (OpenAI)' },
            { id: 'claude', label: 'Claude (Anthropic)' },
            { id: 'grok', label: 'Grok (xAI)' }
        ];
        this.selectedProvider = 'local';
        this.systemPrompt = "";
        this.roles = [];
        this.showContext = false;
        this.isTyping = false;
    }

    async connectedCallback() {
        await this.fetchPresets();
        this.render();
    }

    async fetchPresets() {
        try {
            const res = await fetch('/static/llm_assistant_presets.json');
            const data = await res.json();
            this.roles = data.roles || [];
            // Default to first role (Standard)
            if (this.roles.length > 0) {
                this.systemPrompt = this.roles[0].prompt;
            }
        } catch (e) {
            console.error("Failed to load LLM presets", e);
        }
    }

    render() {
        const brickId = this.getAttribute('brick-id') || '';
        this.shadowRoot.innerHTML = `
            <style>
                :host { display: block; height: 100%; }
                .assistant-container {
                    display: flex;
                    flex-direction: column;
                    gap: 0.8rem;
                    height: 100%;
                    min-height: 400px;
                    max-height: 600px;
                }
                .chat-history {
                    flex-grow: 1;
                    overflow-y: auto;
                    padding: 0.5rem;
                    background: rgba(0,0,0,0.2);
                    border-radius: 8px;
                    border: 1px solid rgba(255,255,255,0.05);
                    display: flex;
                    flex-direction: column;
                    gap: 0.8rem;
                    scrollbar-width: thin;
                }
                .message {
                    padding: 0.6rem 0.8rem;
                    border-radius: 8px;
                    max-width: 90%;
                    font-size: 0.85rem;
                    line-height: 1.4;
                    position: relative;
                }
                .message-user {
                    align-self: flex-end;
                    background: var(--sl-color-primary-600);
                    color: white;
                }
                .message-assistant {
                    align-self: flex-start;
                    background: rgba(255,255,255,0.05);
                    border: 1px solid rgba(255,255,255,0.1);
                    color: #e2e8f0;
                }
                .context-area {
                    background: rgba(255,255,255,0.03);
                    border: 1px solid rgba(255,255,255,0.1);
                    border-radius: 8px;
                    padding: 0.8rem;
                    display: ${this.showContext ? 'flex' : 'none'};
                    flex-direction: column;
                    gap: 0.5rem;
                }
                .inject-btn {
                    margin-top: 0.5rem;
                    --sl-spacing-x-small: 4px;
                }
                .typing-indicator {
                    font-size: 0.75rem;
                    color: #64748b;
                    font-style: italic;
                    margin-bottom: 4px;
                }
                .header-actions {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    gap: 0.5rem;
                }
            </style>
            <qp-cartridge title="LLM Assistant" type="input" brick-id="${brickId}">
                <div class="assistant-container">
                    <div class="header-actions">
                        <sl-select id="provider-select" value="${this.selectedProvider}" size="small" style="flex: 1;" hoist>
                            ${this.providers.map(p => `<sl-option value="${p.id}">${p.label}</sl-option>`).join('')}
                        </sl-select>
                        <sl-tooltip content="Toggle Role/Context">
                            <sl-icon-button name="${this.showContext ? 'chevron-up' : 'gear'}" id="toggle-context"></sl-icon-button>
                        </sl-tooltip>
                        <sl-tooltip content="Clear Chat">
                            <sl-icon-button name="trash" id="clear-chat"></sl-icon-button>
                        </sl-tooltip>
                    </div>

                    <div class="context-area" id="context-pane">
                        <sl-select id="role-select" label="Role Preset" size="small" hoist>
                            ${this.roles.map(r => `<sl-option value="${r.id}">${r.label}</sl-option>`).join('')}
                        </sl-select>
                        <sl-textarea id="system-prompt-input" label="System Prompt (Context)" 
                            size="small" resize="none" value="${this.systemPrompt}"
                            placeholder="IA's personality and rules..."></sl-textarea>
                    </div>

                    <div class="chat-history" id="chat-history">
                        ${this.messages.length === 0 ? '<div style="text-align:center; color:#475569; margin-top:2rem; font-size:0.8rem;">Start chatting to refine your prompt...</div>' : ''}
                        ${this.messages.map((m, idx) => `
                            <div class="message message-${m.role}">
                                ${m.content}
                                ${m.role === 'assistant' ? `
                                    <div style="display:flex; justify-content: flex-end;">
                                        <sl-button variant="success" size="small" outline class="inject-btn" data-index="${idx}">
                                            <sl-icon slot="prefix" name="arrow-right-short"></sl-icon>
                                            Send to Prompt
                                        </sl-button>
                                    </div>
                                ` : ''}
                            </div>
                        `).join('')}
                        ${this.isTyping ? '<div class="typing-indicator">AI is searching for inspiration...</div>' : ''}
                    </div>

                    <div style="display: flex; gap: 0.5rem;">
                        <sl-input id="user-input" placeholder="An idea? A modification?" style="flex: 1;" @keydown="${e => e.key === 'Enter' && this.handleSend()}"></sl-input>
                        <sl-button variant="primary" id="send-btn" circle>
                            <sl-icon name="send"></sl-icon>
                        </sl-button>
                    </div>
                </div>
            </qp-cartridge>
        `;

        this.attachListeners();
        this.scrollToBottom();
    }

    attachListeners() {
        const root = this.shadowRoot;
        root.getElementById('toggle-context').onclick = () => {
            this.showContext = !this.showContext;
            this.render();
        };
        root.getElementById('clear-chat').onclick = () => {
            if (confirm("Clear this conversation?")) {
                this.messages = [];
                this.render();
            }
        };
        root.getElementById('send-btn').onclick = () => this.handleSend();
        root.getElementById('user-input').addEventListener('keydown', (e) => {
            if (e.key === 'Enter') this.handleSend();
        });

        root.getElementById('provider-select').addEventListener('sl-change', (e) => {
            this.selectedProvider = e.target.value;
        });

        root.getElementById('role-select').addEventListener('sl-change', (e) => {
            const role = this.roles.find(r => r.id === e.target.value);
            if (role) {
                this.systemPrompt = role.prompt;
                root.getElementById('system-prompt-input').value = role.prompt;
            }
        });

        root.getElementById('system-prompt-input').addEventListener('sl-input', (e) => {
            this.systemPrompt = e.target.value;
        });

        root.querySelectorAll('.inject-btn').forEach(btn => {
            btn.onclick = () => {
                const idx = btn.dataset.index;
                const text = this.messages[idx].content;
                this.injectToMainPrompt(text);
            };
        });
    }

    async handleSend() {
        const input = this.shadowRoot.getElementById('user-input');
        const text = input.value.trim();
        if (!text || this.isTyping) return;

        this.messages.push({ role: 'user', content: text });
        input.value = '';
        this.isTyping = true;
        this.render();

        try {
            const res = await fetch('/prompt/assistant', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    provider: this.selectedProvider,
                    messages: this.messages,
                    system_prompt: this.systemPrompt
                })
            });
            const data = await res.json();
            if (data.status === 'success') {
                this.messages.push({ role: 'assistant', content: data.content });
            } else {
                this.messages.push({ role: 'assistant', content: "Error: " + data.message });
            }
        } catch (e) {
            this.messages.push({ role: 'assistant', content: "Failed to connect to backend." });
        } finally {
            this.isTyping = false;
            this.render();
        }
    }

    injectToMainPrompt(text) {
        // Find the main prompt brick
        const promptBrick = document.querySelector('qp-prompt');
        if (promptBrick && typeof promptBrick.setPrompt === 'function') {
            promptBrick.setPrompt(text);
            window.qpyt_app?.notify("Prompt injected!", "success");
        } else {
            // Fallback: try direct DOM access if component not found by tag
            const fallback = document.getElementById('prompt-input');
            if (fallback) fallback.value = text;
            window.qpyt_app?.notify("Prompt set (fallback)", "info");
        }
    }

    scrollToBottom() {
        setTimeout(() => {
            const history = this.shadowRoot.getElementById('chat-history');
            if (history) history.scrollTop = history.scrollHeight;
        }, 50);
    }

    getValue() {
        return {
            messages: this.messages,
            provider: this.selectedProvider,
            systemPrompt: this.systemPrompt,
            showContext: this.showContext
        };
    }

    setValues(values) {
        if (!values) return;
        this.messages = values.messages || [];
        this.selectedProvider = values.provider || 'local';
        this.systemPrompt = values.systemPrompt || "";
        this.showContext = !!values.showContext;
        this.render();
    }
}

customElements.define('qp-llm-assistant', QpLlmAssistant);
