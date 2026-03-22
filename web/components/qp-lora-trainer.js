class QpLoraTrainer extends HTMLElement {
    static get observedAttributes() { return ['brick-id']; }
    
    constructor() {
        super();
        this.attachShadow({ mode: 'open' });
        this.status = "idle";
        this.progress = 0;
        this.taskId = null;
        this.lastLoss = 0;
        this.currentStep = 0;
        this.datasetPath = "";
        this.loraName = "my_new_lora";
        this.conceptName = "ohwx style";
        this.steps = 500;
        this.lr = 0.0001;
    }

    connectedCallback() {
        this.render();
    }

    attributeChangedCallback(name, oldValue, newValue) {
        if (oldValue !== newValue && name === 'brick-id') {
            this.render();
        }
    }

    async startTraining() {
        const payload = {
            input_dir: this.shadowRoot.getElementById('dataset-path').value,
            output_name: this.shadowRoot.getElementById('lora-name').value,
            concept_name: this.shadowRoot.getElementById('concept-name').value,
            steps: parseInt(this.shadowRoot.getElementById('steps').value),
            lr: parseFloat(this.shadowRoot.getElementById('lr').value),
            rank: 16,
            batch_size: 1
        };

        if (!payload.input_dir) {
            alert("Please specify a dataset directory.");
            return;
        }

        this.status = "submitting";
        this.render();

        try {
            const resp = await fetch('/lora/train', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });
            const data = await resp.json();
            if (data.status === "success") {
                this.taskId = data.task_id;
                this.status = "training";
                this.pollStatus();
            } else {
                this.status = "error";
                alert("Error: " + data.message);
            }
        } catch (e) {
            this.status = "error";
            console.error(e);
        }
        this.render();
    }

    async pollStatus() {
        if (!this.taskId || this.status !== "training") return;

        try {
            const resp = await fetch(`/queue/status/${this.taskId}`);
            const data = await resp.json();
            
            this.progress = data.progress || 0;
            if (data.status === "COMPLETED") {
                this.status = "completed";
                this.taskId = null;
                alert("Training Complete! Your LoRA is ready.");
            } else if (data.status === "FAILED") {
                this.status = "error";
                this.error = data.error;
            } else if (data.status === "RUNNING") {
                if (data.result && data.result.loss) this.lastLoss = data.result.loss;
                if (data.result && data.result.steps) this.currentStep = data.result.steps;
            }

            this.render();
            if (this.status === "training") {
                setTimeout(() => this.pollStatus(), 2000);
            }
        } catch (e) {
            console.error("Polling error", e);
        }
    }

    render() {
        const brickId = this.getAttribute('brick-id') || '';
        this.shadowRoot.innerHTML = `
            <style>
                :host { 
                    display: block; 
                }
                * { box-sizing: border-box; }
                .trainer-container {
                    display: flex;
                    flex-direction: column;
                    gap: 0.8rem;
                }
                .field { margin-bottom: 0.2rem; }
                .progress-section { 
                    margin-top: 1rem; 
                    padding: 0.8rem; 
                    background: rgba(0,0,0,0.3); 
                    border-radius: 8px; 
                    border: 1px solid rgba(255,255,255,0.05);
                }
                .status-badge { 
                    font-size: 0.65rem; 
                    padding: 2px 8px; 
                    border-radius: 10px; 
                    background: #334155; 
                    text-transform: uppercase; 
                    width: fit-content;
                    color: white;
                }
                .stats { display: flex; justify-content: space-between; font-size: 0.75rem; margin-top: 5px; color: #94a3b8; }
                
                sl-input::part(base) {
                    background-color: rgba(0, 0, 0, 0.2);
                }
                
                .header-text { margin-bottom: 0.5rem; }
                .header-text b { color: #6366f1; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px; display: block; }
                .header-text small { display: block; color: #64748b; font-size: 0.75rem; }
            </style>

            <qp-cartridge title="LoRA Trainer" type="tool" icon="cpu" brick-id="${brickId}">
                <div class="trainer-container">
                    <div class="header-text">
                        <div class="status-badge">${this.status}</div>
                        <b>SDXL Training Session</b>
                        <small>Florence-2 Captioning & LoRA fine-tuning</small>
                    </div>

                    <div class="field">
                        <sl-input id="dataset-path" label="Dataset Directory" size="small" placeholder="C:\\images\\my_concept" 
                            help-text="Local path containing your training images (JPG/PNG)."></sl-input>
                    </div>

                    <div class="field">
                        <sl-input id="lora-name" label="LoRA Name" size="small" value="${this.loraName}" 
                            help-text="The filename of the final .safetensors file."></sl-input>
                    </div>

                    <div class="field">
                        <sl-input id="concept-name" label="Trigger Word" size="small" value="${this.conceptName}" 
                            help-text="Activation keyword to trigger this style in prompts."></sl-input>
                    </div>

                    <div class="field">
                        <sl-input id="steps" label="Steps" type="number" size="small" value="${this.steps}"
                            help-text="Total training iterations. 500-1000 is a good range."></sl-input>
                    </div>

                    <div class="field">
                        <sl-input id="lr" label="Learning Rate" type="number" size="small" step="0.00001" value="${this.lr}"
                            help-text="Learning speed. 0.0001 is the recommended value."></sl-input>
                    </div>

                    <div style="margin-top: 0.5rem">
                        <sl-button variant="primary" id="start-btn" size="small" style="width: 100%;" ?loading="${this.status === 'training' || this.status === 'submitting'}">
                            <sl-icon slot="prefix" name="play-fill"></sl-icon>
                            Start Training
                        </sl-button>
                    </div>

                    ${this.status === 'training' || this.status === 'completed' || this.status === 'error' ? `
                        <div class="progress-section">
                            <sl-progress-bar value="${this.progress}"></sl-progress-bar>
                            <div class="stats">
                                <span>${Math.round(this.progress)}%</span>
                                <span>Step ${this.currentStep}</span>
                            </div>
                            ${this.error ? `<div style="color: #ef4444; font-size: 0.7rem; margin-top: 8px;">Error: ${this.error}</div>` : ''}
                        </div>
                    ` : ''}
                </div>
            </qp-cartridge>
        `;

        this.shadowRoot.getElementById('start-btn').addEventListener('click', () => this.startTraining());
    }
}
customElements.define('qp-lora-trainer', QpLoraTrainer);
