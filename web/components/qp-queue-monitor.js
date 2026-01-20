class QpQueueMonitor extends HTMLElement {
    constructor() {
        super();
        this.attachShadow({ mode: 'open' });
        this.tasks = [];
        this.pollInterval = null;
    }

    connectedCallback() {
        this.render();
        this.startPolling();
    }

    disconnectedCallback() {
        this.stopPolling();
    }

    startPolling() {
        this.fetchTasks();
        this.pollInterval = setInterval(() => this.fetchTasks(), 2000);
    }

    stopPolling() {
        if (this.pollInterval) clearInterval(this.pollInterval);
    }

    async fetchTasks() {
        try {
            const resp = await fetch('/queue/list');
            const data = await resp.json();
            if (data.status === 'success') {
                this.tasks = data.tasks;
                this.updateUI();
            }
        } catch (e) {
            console.error("Queue poll failed", e);
        }
    }

    async cancelTask(tid) {
        try {
            const resp = await fetch(`/queue/cancel/${tid}`, { method: 'POST' });
            const data = await resp.json();
            if (data.status === 'success') {
                window.qpyt_app.notify("Task cancelled", "success");
                this.fetchTasks();
            }
        } catch (e) {
            console.error("Cancel failed", e);
        }
    }

    async clearHistory() {
        try {
            const resp = await fetch('/queue/clear', { method: 'POST' });
            const data = await resp.json();
            if (data.status === 'success') {
                window.qpyt_app.notify("Finished tasks cleared", "success");
                this.fetchTasks();
            }
        } catch (e) {
            console.error("Clear failed", e);
        }
    }

    formatTime(timestamp) {
        if (!timestamp) return "-";
        const date = new Date(timestamp * 1000);
        return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    }

    getStatusColor(status) {
        switch (status) {
            case 'RUNNING': return '#3b82f6'; // Blue
            case 'COMPLETED': return '#10b981'; // Green
            case 'FAILED': return '#ef4444'; // Red
            case 'CANCELLED': return '#6b7280'; // Gray
            default: return '#f59e0b'; // Amber for PENDING
        }
    }

    updateUI() {
        const container = this.shadowRoot.querySelector('.task-list');
        if (!container) return;

        if (this.tasks.length === 0) {
            container.innerHTML = '<div class="empty">No tasks in queue</div>';
        } else {
            container.innerHTML = this.tasks.map(task => `
                <div class="task-item" data-id="${task.task_id}">
                    <div class="task-info">
                        <span class="task-type">${task.type.toUpperCase()}</span>
                        <span class="task-status" style="color: ${this.getStatusColor(task.status)}">${task.status}</span>
                    </div>
                    <div class="task-meta">
                        ID: ${task.task_id.substring(0, 8)}... | ${this.formatTime(task.created_at)}
                    </div>
                    ${task.status === 'RUNNING' ? `
                        <div class="progress-bar">
                            <div class="progress-fill" style="width: ${task.progress}%"></div>
                        </div>
                    ` : ''}
                    ${task.status === 'PENDING' ? `
                        <sl-button variant="danger" size="small" outline onclick="this.getRootNode().host.cancelTask('${task.task_id}')">
                            <sl-icon slot="prefix" name="trash"></sl-icon> Cancel
                        </sl-button>
                    ` : ''}
                    ${task.error ? `<div class="error-msg">${task.error}</div>` : ''}
                </div>
            `).join('');
        }

        const hasFinished = this.tasks.some(t => ['COMPLETED', 'FAILED', 'CANCELLED'].includes(t.status));
        const currentBtn = this.shadowRoot.getElementById('btn-clear');

        if (hasFinished && !currentBtn) {
            this.render();
            return;
        } else if (!hasFinished && currentBtn) {
            this.render();
            return;
        }

        if (currentBtn) {
            currentBtn.onclick = () => this.clearHistory();
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
                    height: 100%;
                }
                .task-list {
                    display: flex;
                    flex-direction: column;
                    gap: 10px;
                    overflow-y: auto;
                    flex: 1;
                }
                .task-item {
                    background: rgba(255, 255, 255, 0.05);
                    border-radius: 8px;
                    padding: 10px;
                    border-left: 3px solid #ef4444;
                    transition: transform 0.2s;
                }
                .task-item:hover {
                    background: rgba(255, 255, 255, 0.08);
                    transform: translateX(2px);
                }
                .task-info {
                    display: flex;
                    justify-content: space-between;
                    font-size: 0.8rem;
                    font-weight: 700;
                    margin-bottom: 4px;
                }
                .task-type {
                    color: #94a3b8;
                    font-family: monospace;
                }
                .task-meta {
                    font-size: 0.7rem;
                    color: #64748b;
                    margin-bottom: 8px;
                }
                .progress-bar {
                    height: 4px;
                    background: rgba(0, 0, 0, 0.3);
                    border-radius: 2px;
                    overflow: hidden;
                    margin-top: 5px;
                }
                .progress-fill {
                    height: 100%;
                    background: #3b82f6;
                    box-shadow: 0 0 10px rgba(59, 130, 246, 0.5);
                    transition: width 0.3s ease;
                }
                .empty {
                    text-align: center;
                    color: #475569;
                    font-style: italic;
                    font-size: 0.8rem;
                    padding: 2rem 0;
                }
                .error-msg {
                    color: #ef4444;
                    font-size: 0.7rem;
                    margin-top: 5px;
                    background: rgba(239, 68, 68, 0.1);
                    padding: 4px;
                    border-radius: 4px;
                }
                sl-button {
                    width: 100%;
                    margin-top: 5px;
                }
            </style>
            <qp-cartridge title="Job Queue Monitor" icon="collection" type="setting" brick-id="${brickId}">
                <div class="content">
                    <div class="task-list">
                        <div class="empty">Scanning queue...</div>
                    </div>
                    ${this.tasks.some(t => ['COMPLETED', 'FAILED', 'CANCELLED'].includes(t.status)) ? `
                        <sl-button variant="default" size="small" id="btn-clear" outline>
                            <sl-icon slot="prefix" name="eraser"></sl-icon> Clear Finished
                        </sl-button>
                    ` : ''}
                </div>
            </qp-cartridge>
        `;
        this.updateUI();
    }
}

customElements.define('qp-queue-monitor', QpQueueMonitor);
