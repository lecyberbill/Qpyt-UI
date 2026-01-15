// Cartouche de Prompt
class MgPrompt extends HTMLElement {
    constructor() { super(); this.attachShadow({ mode: 'open' }); }
    connectedCallback() {
        this.shadowRoot.innerHTML = `
            <mg-cartridge title="Prompt" type="input">
                <sl-textarea id="prompt-input" name="prompt" label="Description" placeholder="Entrez votre idée..." resize="none" style="height: 350px;"></sl-textarea>
                <div style="margin-top: auto; color: #64748b; font-size: 0.8rem;">
                    ✍️ Soyez précis pour de meilleurs résultats.
                </div>
            </mg-cartridge>
        `;
    }
    get value() { return this.shadowRoot.querySelector('#prompt-input').value; }
}
customElements.define('mg-prompt', MgPrompt);

// Cartouche de Paramètres
class MgSettings extends HTMLElement {
    constructor() { super(); this.attachShadow({ mode: 'open' }); }
    connectedCallback() {
        this.shadowRoot.innerHTML = `
            <mg-cartridge title="Settings" type="logic">
                <div style="display: flex; flex-direction: column; gap: 1rem;">
                    <sl-input id="width-input" type="number" label="Largeur" value="512"></sl-input>
                    <sl-input id="height-input" type="number" label="Hauteur" value="512"></sl-input>
                    <sl-input id="gs-input" type="number" step="0.1" label="Guidance Scale" value="7.5"></sl-input>
                    <sl-input id="seed-input" type="number" label="Seed" placeholder="Aléatoire"></sl-input>
                </div>
            </mg-cartridge>
        `;
    }
    get values() {
        return {
            width: parseInt(this.shadowRoot.querySelector('#width-input').value),
            height: parseInt(this.shadowRoot.querySelector('#height-input').value),
            guidance_scale: parseFloat(this.shadowRoot.querySelector('#gs-input').value),
            seed: this.shadowRoot.querySelector('#seed-input').value ? parseInt(this.shadowRoot.querySelector('#seed-input').value) : null
        };
    }
}
customElements.define('mg-settings', MgSettings);

// Cartouche de Rendu
class MgRender extends HTMLElement {
    constructor() {
        super();
        this.attachShadow({ mode: 'open' });
        this.loading = false;
    }

    connectedCallback() {
        this.render();
    }

    async generate() {
        const promptComp = document.querySelector('mg-prompt');
        const settingsComp = document.querySelector('mg-settings');

        const prompt = promptComp?.value;
        const settings = settingsComp?.values;

        if (!prompt) { alert("Le prompt est requis !"); return; }

        this.loading = true;
        this.render();

        try {
            const response = await fetch('/generate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    prompt,
                    ...settings
                })
            });
            const result = await response.json();
            if (result.status === 'success') {
                this.imageUrl = result.data.image_url;
                this.time = result.data.execution_time.toFixed(2);
            }
        } catch (e) {
            console.error(e);
        } finally {
            this.loading = false;
            this.render();
        }
    }

    render() {
        this.shadowRoot.innerHTML = `
            <style>
                .preview {
                    flex: 1;
                    border: 2px dashed rgba(255,255,255,0.1);
                    border-radius: 1rem;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    overflow: hidden;
                    background: #0f172a;
                    margin-bottom: 1.5rem;
                    position: relative;
                }
                img { max-width: 100%; max-height: 100%; object-fit: contain; }
                .badge { font-size: 0.7rem; color: #10b981; margin-top: 0.5rem; text-align: center; font-weight: bold; }
            </style>
            <mg-cartridge title="Render" type="output">
                <div class="preview">
                    ${this.loading ? '<sl-spinner style="font-size: 2rem;"></sl-spinner>' : ''}
                    ${this.imageUrl && !this.loading ? `<img src="${this.imageUrl}">` : ''}
                    ${!this.imageUrl && !this.loading ? '<div style="color: #475569">Prêt à générer</div>' : ''}
                </div>
                <sl-button variant="success" size="large" ?loading="${this.loading}" id="gen-btn">
                    <sl-icon slot="prefix" name="play"></sl-icon>
                    Lancer la génération
                </sl-button>
                ${this.time ? `<div class="badge">Success: ${this.time}s</div>` : ''}
            </mg-cartridge>
        `;
        this.shadowRoot.getElementById('gen-btn')?.addEventListener('click', () => this.generate());
    }
}
customElements.define('mg-render', MgRender);
