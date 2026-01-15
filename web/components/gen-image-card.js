class GenImageCard extends HTMLElement {
    constructor() {
        super();
        this.attachShadow({ mode: 'open' });
        this.loading = false;
        this.resultImageUrl = null;
        this.executionTime = null;
    }

    connectedCallback() {
        this.render();
    }

    async handleGenerate(event) {
        event.preventDefault();

        // Lecture directe des valeurs depuis les composants Shoelace
        const promptEl = this.shadowRoot.querySelector('sl-textarea[name="prompt"]');
        const widthEl = this.shadowRoot.querySelector('sl-input[name="width"]');
        const heightEl = this.shadowRoot.querySelector('sl-input[name="height"]');
        const gsEl = this.shadowRoot.querySelector('sl-input[name="guidance_scale"]');
        const seedEl = this.shadowRoot.querySelector('sl-input[name="seed"]');

        const payload = {
            prompt: promptEl.value,
            width: parseInt(widthEl.value),
            height: parseInt(heightEl.value),
            guidance_scale: parseFloat(gsEl.value),
            seed: seedEl.value ? parseInt(seedEl.value) : null
        };

        this.loading = true;
        this.resultImageUrl = null;
        this.render();

        try {
            const response = await fetch('/generate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            const result = await response.json();

            if (result.status === 'success') {
                this.resultImageUrl = result.data.image_url;
                this.executionTime = result.data.execution_time.toFixed(2);
            } else {
                alert('Erreur: ' + (result.message || 'Inconnue'));
            }
        } catch (error) {
            console.error('Fetch error:', error);
            alert('Erreur de connexion au serveur');
        } finally {
            this.loading = false;
            this.render();
        }
    }

    render() {
        this.shadowRoot.innerHTML = `
            <style>
                :host {
                    display: block;
                    color: #f8fafc;
                }

                form {
                    display: grid;
                    gap: 1.5rem;
                }

                .grid-2 {
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 1rem;
                }

                .result-area {
                    margin-top: 2rem;
                    text-align: center;
                    min-height: 200px;
                    display: flex;
                    flex-direction: column;
                    justify-content: center;
                    align-items: center;
                    border: 2px dashed rgba(255, 255, 255, 0.1);
                    border-radius: 1rem;
                    overflow: hidden;
                    position: relative;
                }

                img {
                    max-width: 100%;
                    border-radius: 0.5rem;
                    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
                    transition: transform 0.3s ease;
                }

                img:hover {
                    transform: scale(1.02);
                }

                .execution-badge {
                    margin-top: 1rem;
                    background: rgba(16, 185, 129, 0.2);
                    color: #10b981;
                    padding: 0.25rem 0.75rem;
                    border-radius: 9999px;
                    font-size: 0.875rem;
                    font-weight: 500;
                }

                /* Customizing Shoelace inputs via parts if needed, 
                   but here we just use the standard components */
                sl-input, sl-textarea, sl-range {
                    --sl-color-primary-600: #6366f1;
                }

                .loading-overlay {
                    position: absolute;
                    top: 0;
                    left: 0;
                    right: 0;
                    bottom: 0;
                    background: rgba(15, 23, 42, 0.8);
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    z-index: 10;
                }
            </style>

            <form id="gen-form">
                <sl-textarea 
                    name="prompt" 
                    label="Prompt" 
                    placeholder="Entrez votre description ici..." 
                    resize="none"
                    required
                    help-text="Décrivez l'image que vous souhaitez créer.">
                </sl-textarea>

                <div class="grid-2">
                    <sl-input type="number" name="width" label="Largeur" value="512" min="256" max="1024"></sl-input>
                    <sl-input type="number" name="height" label="Hauteur" value="512" min="256" max="1024"></sl-input>
                </div>

                <div class="grid-2">
                    <sl-input type="number" name="guidance_scale" step="0.1" label="Guidance Scale" value="7.5" min="1" max="20"></sl-input>
                    <sl-input type="number" name="seed" label="Seed (Optionnel)" placeholder="Aléatoire"></sl-input>
                </div>

                <sl-button variant="primary" type="submit" size="large" ?loading="${this.loading}">
                    <sl-icon slot="prefix" name="magic"></sl-icon>
                    Générer l'Image
                </sl-button>
            </form>

            <div class="result-area">
                ${this.loading ? `
                    <div class="loading-overlay">
                        <sl-spinner style="font-size: 3rem;"></sl-spinner>
                    </div>
                ` : ''}

                ${this.resultImageUrl ? `
                    <img src="${this.resultImageUrl}" alt="Image générée">
                    <div class="execution-badge">Généré en ${this.executionTime}s</div>
                ` : !this.loading ? `
                    <p style="color: #64748b;">L'image s'affichera ici...</p>
                ` : ''}
            </div>
        `;

        this.shadowRoot.getElementById('gen-form').addEventListener('submit', (e) => this.handleGenerate(e));
    }
}

customElements.define('gen-image-card', GenImageCard);
