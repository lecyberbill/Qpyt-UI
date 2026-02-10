class QpMonitor extends HTMLElement {
    constructor() {
        super();
        this.attachShadow({ mode: 'open' });
        this.interval = null;
        this.stats = {
            cpu: { percent: 0 },
            ram: { percent: 0, used_mb: 0, total_mb: 0 },
            gpus: [],
            disk: { total_gb: 0, free_gb: 0, percent: 0 }
        };
    }

    connectedCallback() {
        this.render();
        this.startMonitoring();
    }

    disconnectedCallback() {
        this.stopMonitoring();
    }

    startMonitoring() {
        this.fetchStats();
        this.interval = setInterval(() => this.fetchStats(), 2000);
    }

    stopMonitoring() {
        if (this.interval) clearInterval(this.interval);
    }

    async fetchStats() {
        try {
            const res = await fetch('/monitor/stats');
            const data = await res.json();
            if (data.status === 'success') {
                this.stats = data;
                this.updateUI();
            }
        } catch (e) {
            console.error("Monitor poll failed", e);
        }
    }

    triggerModelRefresh() {
        if (window.qpyt_app && typeof window.qpyt_app.refreshModels === 'function') {
            window.qpyt_app.refreshModels();
            const btn = this.shadowRoot.getElementById('btn-refresh');
            if (btn) {
                btn.loading = true;
                setTimeout(() => btn.loading = false, 1000);
            }
        } else {
            // Fallback: Dispath event
            document.dispatchEvent(new CustomEvent('qpyt:refresh-models'));
            const btn = this.shadowRoot.getElementById('btn-refresh');
            if (btn) {
                btn.innerHTML = '<sl-icon slot="prefix" name="check"></sl-icon> Config Reloaded';
                setTimeout(() => {
                    btn.innerHTML = '<sl-icon slot="prefix" name="arrow-clockwise"></sl-icon> Refresh Models';
                    // Reload page as fallback if app.js helper missing
                    window.location.reload();
                }, 800);
            }
        }
    }

    getCirclePath(percent, radius = 35) {
        // SVG Circle Arc logic
        // percent 0-100
        const c = 2 * Math.PI * radius;
        const offset = c - (percent / 100) * c;
        return { c, offset };
    }

    renderGauge(label, value, color, subtext = "") {
        const radius = 30;
        const { c, offset } = this.getCirclePath(value, radius);

        return `
            <div class="gauge">
                <svg width="70" height="70" viewBox="0 0 70 70">
                    <circle cx="35" cy="35" r="${radius}" stroke="rgba(255,255,255,0.1)" stroke-width="5" fill="none"></circle>
                    <circle cx="35" cy="35" r="${radius}" 
                        stroke="${color}" 
                        stroke-width="5" 
                        fill="none" 
                        stroke-dasharray="${c}" 
                        stroke-dashoffset="${offset}"
                        style="transition: stroke-dashoffset 0.5s ease; transform: rotate(-90deg); transform-origin: 50% 50%;"
                    ></circle>
                    <text x="35" y="39" text-anchor="middle" fill="white" font-size="12" font-weight="bold">${value.toFixed(0)}%</text>
                </svg>
                <div class="label">${label}</div>
                <div class="subtext">${subtext}</div>
            </div>
        `;
    }

    updateUI() {
        // Update Gauges Row (CPU | RAM | Disk)
        const gaugesContainer = this.shadowRoot.querySelector('.gauges-row');
        if (gaugesContainer) {
            gaugesContainer.innerHTML = `
                ${this.renderGauge('CPU', this.stats.cpu.percent, '#3b82f6')}
                ${this.renderGauge('RAM', this.stats.ram.percent, '#f59e0b', `${Math.round(this.stats.ram.used_mb / 1024)}GB`)}
                ${this.renderGauge('DISK', this.stats.disk ? this.stats.disk.percent : 0, '#8b5cf6', `${this.stats.disk ? this.stats.disk.free_gb : 0}GB Free`)}
            `;
        }

        // Update GPUs
        const gpuContainer = this.shadowRoot.querySelector('.gpu-list');
        if (gpuContainer) {
            if (this.stats.gpus.length === 0) {
                gpuContainer.innerHTML = '<div class="no-gpu">No GPU Detected</div>';
            } else {
                gpuContainer.innerHTML = this.stats.gpus.map(gpu => `
                     <div class="gpu-box">
                        <div class="gpu-gauge">
                             ${this.renderGauge('VRAM', gpu.vram_percent, '#10b981')}
                        </div>
                        <div class="gpu-info">
                            <div class="gpu-name">GPU ${gpu.index}</div>
                            <div class="gpu-stat">Mem: ${gpu.vram_used_mb}MB / ${gpu.vram_total_mb}MB</div>
                            <div class="gpu-stat">Load: ${gpu.gpu_util_percent}%</div>
                        </div>
                     </div>
                `).join('');
            }
        }
    }

    render() {
        const brickId = this.getAttribute('brick-id') || '';
        this.shadowRoot.innerHTML = `
            <style>
                .content {
                    padding: 0.5rem;
                    display: flex;
                    flex-direction: column;
                    gap: 1rem;
                }
                .gauges-row {
                    display: flex;
                    justify-content: space-around;
                    align-items: flex-start;
                    padding: 5px 0;
                    border-bottom: 1px solid rgba(255,255,255,0.1);
                    padding-bottom: 1rem;
                }
                .gauge {
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    gap: 2px;
                }
                .label {
                    font-size: 0.7rem;
                    color: #94a3b8;
                    font-weight: 700;
                }
                .subtext {
                    font-size: 0.65rem;
                    color: #64748b;
                }
                .gpu-list {
                    display: flex;
                    flex-direction: column;
                    gap: 10px;
                }
                .gpu-box {
                    display: flex;
                    align-items: center;
                    background: rgba(0,0,0,0.2);
                    padding: 8px;
                    border-radius: 8px;
                    gap: 10px;
                }
                .gpu-info {
                    flex: 1;
                    display: flex;
                    flex-direction: column;
                    justify-content: center;
                }
                .gpu-name {
                    font-size: 0.75rem;
                    font-weight: 700;
                    color: #e2e8f0;
                    margin-bottom: 2px;
                }
                .gpu-stat {
                    font-size: 0.7rem;
                    color: #94a3b8;
                    font-family: monospace;
                }
                .no-gpu {
                    text-align: center;
                    color: #64748b;
                    font-style: italic;
                    font-size: 0.8rem;
                }
                sl-button {
                    width: 100%;
                }
            </style>
            <qp-cartridge title="System Monitor" icon="activity" type="setting" brick-id="${brickId}">
                <div class="content">
                    <div class="gauges-row">Checking stats...</div>
                    <div class="gpu-list"></div>
                    
                    <sl-button id="btn-refresh" size="small" variant="default" outline>
                        <sl-icon slot="prefix" name="arrow-clockwise"></sl-icon>
                        Refresh Models List
                    </sl-button>
                </div>
            </qp-cartridge>
        `;

        setTimeout(() => {
            const btn = this.shadowRoot.getElementById('btn-refresh');
            if (btn) btn.addEventListener('click', () => this.triggerModelRefresh());
        }, 0);
    }
}

customElements.define('qp-monitor', QpMonitor);
