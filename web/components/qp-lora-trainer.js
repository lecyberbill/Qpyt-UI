class QpLoraTrainer extends HTMLElement {
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
        this.steps = 1000;
        this.lr = 0.0001;
    }

    connectedCallback() {
        this.render();
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
                // Future: backend can return loss/step in 'result' or custom progress info
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
        this.shadowRoot.innerHTML = `
            <style>
                :host { display: block; width: 350px; background: #0f172a; border-radius: 16px; border: 1px solid rgba(255, 255, 255, 0.1); color: white; padding: 1rem; }
                .header { display: flex; align-items: center; gap: 10px; margin-bottom: 1.5rem; }
                .header sl-icon { font-size: 1.5rem; color: #6366f1; }
                .field { margin-bottom: 1rem; }
                .progress-section { margin-top: 1.5rem; padding: 1rem; background: rgba(0,0,0,0.2); border-radius: 8px; }
                .status-badge { font-size: 0.7rem; padding: 2px 8px; border-radius: 10px; background: #334155; text-transform: uppercase; float: right; }
                .stats { display: flex; justify-content: space-between; font-size: 0.8rem; margin-top: 5px; color: #94a3b8; }
            </style>

            <div class="header">
                <sl-icon name="cpu"></sl-icon>
                <div style="flex: 1">
                    <b style="display:block">LoRA Trainer</b>
                    <small style="color: #64748b">Stable Diffusion XL</small>
                </div>
                <div class="status-badge">${this.status}</div>
            </div>

            <div class="field">
                <sl-input id="dataset-path" label="Dataset Directory" placeholder="C:\\images\\my_concept" help-text="Local path containing your training images."></sl-input>
            </div>

            <div class="field">
                <sl-input id="lora-name" label="Output Name" value="${this.loraName}" help-text="The filename of your LoRA (.safetensors)"></sl-input>
            </div>

            <div class="field">
                <sl-input id="concept-name" label="Trigger Word (Concept)" value="${this.conceptName}" help-text="The unique keyword to activate this LoRA."></sl-input>
            </div>

            <div style="display: flex; gap: 10px;">
                <sl-input id="steps" label="Steps" type="number" value="${this.steps}" style="flex: 1"></sl-input>
                <sl-input id="lr" label="Learning Rate" type="number" step="0.00001" value="${this.lr}" style="flex: 1"></sl-input>
            </div>

            <div style="margin-top: 1.5rem">
                <sl-button variant="primary" id="start-btn" style="width: 100%;" ?loading="${this.status === 'training' || this.status === 'submitting'}">
                    <sl-icon slot="prefix" name="play-fill"></sl-icon>
                    Start Training
                </sl-button>
            </div>

            ${this.status === 'training' || this.status === 'completed' || this.status === 'error' ? `
                <div class="progress-section">
                    <sl-progress-bar value="${this.progress}"></sl-progress-bar>
                    <div class="stats">
                        <span>Progress: ${Math.round(this.progress)}%</span>
                        <span>Step: ${this.currentStep}</span>
                    </div>
                    ${this.error ? `<div style="color: #ef4444; font-size: 0.75rem; margin-top: 10px;">Error: ${this.error}</div>` : ''}
                </div>
            ` : ''}
        `;

        this.shadowRoot.getElementById('start-btn').addEventListener('click', () => this.startTraining());
    }
}
customElements.define('qp-lora-trainer', QpLoraTrainer);
