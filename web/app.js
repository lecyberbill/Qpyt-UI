class QpytApp {
    constructor() {
        this.workflow = document.getElementById('main-workflow');
        this.titleEl = document.getElementById('app-title');
        this.lastImage = null; // Global track for last generated image (URL)
    }

    async init() {
        try {
            const response = await fetch('/config');
            const config = await response.json();

            // Store current workflow and settings
            this.currentWorkflow = config.workflow;
            this.settings = config.settings || {};

            // Set dynamic title
            if (this.titleEl && config.title) {
                this.titleEl.textContent = config.title;
            }

            // Mount workflow bricks
            this.mountWorkflow(config.workflow);

            // Setup Settings Drawer
            this.setupSettings();

            // Setup Workflow Manager
            this.setupWorkflows();

            // Setup Drag & Drop
            this.setupDragAndDrop();

            console.log("Qpyt-UI initialized with config:", config);
        } catch (e) {
            console.error("Qpyt-UI failed to initialize:", e);
        }
    }

    setupWorkflows() {
        const trigger = document.getElementById('workflow-trigger');
        const drawer = document.getElementById('workflow-drawer');
        const saveBtn = document.getElementById('wf-save-btn');
        const nameInput = document.getElementById('wf-save-name');
        const listContainer = document.getElementById('workflow-list');
        const closeBtn = document.getElementById('wf-close-btn');

        if (!trigger || !drawer) return;

        trigger.addEventListener('click', () => {
            if (drawer.open) {
                drawer.hide();
            } else {
                this.fetchWorkflows();
                drawer.show();
            }
        });

        closeBtn?.addEventListener('click', () => drawer.hide());

        saveBtn?.addEventListener('click', async () => {
            const name = nameInput.value.trim();
            if (!name) {
                this.notify("Please enter a workflow name", "danger");
                return;
            }

            // Collect current workflow state from DOM
            const bricks = this.getCurrentWorkflowState();

            try {
                saveBtn.loading = true;
                const response = await fetch('/workflows/save', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ name, workflow: bricks })
                });
                const result = await response.json();
                if (result.status === 'success') {
                    this.notify(result.message, "success");
                    nameInput.value = '';
                    this.fetchWorkflows();
                } else {
                    this.notify(result.message, "danger");
                }
            } catch (e) {
                this.notify("Saving failed", "danger");
            } finally {
                saveBtn.loading = false;
            }
        });
    }

    async fetchWorkflows() {
        const wfContainer = document.getElementById('workflow-list');
        const prContainer = document.getElementById('preset-list');
        if (!wfContainer || !prContainer) return;

        try {
            const response = await fetch('/workflows');
            const data = await response.json(); // { user: [], system: [] }

            // 1. Render User Workflows
            if (data.user.length === 0) {
                wfContainer.innerHTML = '<p style="color: #475569; font-style: italic; font-size: 0.8rem;">No workflows saved yet.</p>';
            } else {
                wfContainer.innerHTML = data.user.map(name => `
                    <div style="display: flex; gap: 0.5rem; align-items: center; background: rgba(255,255,255,0.03); padding: 0.5rem; border-radius: 8px; border: 1px solid rgba(255,255,255,0.05);">
                        <sl-icon name="journal-bookmark" style="color: #6366f1;"></sl-icon>
                        <span style="flex: 1; font-size: 0.9rem;">${name}</span>
                        <sl-button size="small" variant="neutral" onclick="window.qpyt_app.loadWorkflow('${name}')">Load</sl-button>
                        <sl-button size="small" variant="danger" outline onclick="window.qpyt_app.deleteWorkflow('${name}')" title="Delete">
                            <sl-icon name="trash"></sl-icon>
                        </sl-button>
                    </div>
                `).join('');
            }

            // 2. Render System Presets
            if (data.system.length === 0) {
                prContainer.innerHTML = '<p style="color: #475569; font-style: italic; font-size: 0.8rem;">No factory presets found.</p>';
            } else {
                prContainer.innerHTML = data.system.map(name => `
                    <div style="display: flex; gap: 0.5rem; align-items: center; background: rgba(255,255,255,0.03); padding: 0.5rem; border-radius: 8px; border: 1px solid rgba(255,255,255,0.1); border-left: 3px solid #f59e0b;">
                        <sl-icon name="magic" style="color: #f59e0b;"></sl-icon>
                        <span style="flex: 1; font-size: 0.9rem; font-weight: 500;">${name.replace(/_/g, ' ')}</span>
                        <sl-button size="small" variant="primary" outline onclick="window.qpyt_app.loadWorkflow('${name}')">Apply</sl-button>
                    </div>
                `).join('');
            }
        } catch (e) {
            wfContainer.innerHTML = '<p style="color: #ef4444;">Failed to load lists.</p>';
        }
    }

    async deleteWorkflow(name) {
        if (!confirm(`Are you sure you want to delete workflow '${name}'?`)) return;

        try {
            const response = await fetch('/workflows/delete', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name })
            });
            const result = await response.json();
            if (result.status === 'success') {
                this.notify(`Workflow '${name}' deleted.`, "success");
                this.fetchWorkflows();
            } else {
                this.notify(result.message, "danger");
            }
        } catch (e) {
            this.notify("Deletion failed", "danger");
        }
    }

    async loadWorkflow(name) {
        try {
            const response = await fetch('/workflows/load', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ name })
            });
            const result = await response.json();
            if (result.status === 'success') {
                this.mountWorkflow(result.workflow);
                this.notify(`Workflow '${name}' loaded.`, "success");
                document.getElementById('workflow-drawer').hide();
            } else {
                this.notify(result.message, "danger");
            }
        } catch (e) {
            this.notify("Loading failed", "danger");
        }
    }

    setupSettings() {
        const trigger = document.getElementById('settings-trigger');
        const drawer = document.getElementById('settings-drawer');
        const saveBtn = document.getElementById('cfg-save-btn');
        const closeBtn = document.getElementById('cfg-close-btn');

        if (!trigger || !drawer) return;

        // Populate fields
        const fields = {
            'cfg-models-dir': this.settings.MODELS_DIR,
            'cfg-flux-models-dir': this.settings.FLUX_MODELS_DIR,
            'cfg-loras-dir': this.settings.LORAS_DIR,
            'cfg-output-dir': this.settings.OUTPUT_DIR,
            'cfg-default-model': this.settings.DEFAULT_MODEL
        };

        Object.entries(fields).forEach(([id, value]) => {
            const el = document.getElementById(id);
            if (el) el.value = value || '';
        });

        trigger.addEventListener('click', () => {
            if (drawer.open) drawer.hide();
            else drawer.show();
        });
        closeBtn?.addEventListener('click', () => drawer.hide());

        saveBtn?.addEventListener('click', async () => {
            const newData = {
                ...this.settings,
                MODELS_DIR: document.getElementById('cfg-models-dir').value,
                FLUX_MODELS_DIR: document.getElementById('cfg-flux-models-dir').value,
                LORAS_DIR: document.getElementById('cfg-loras-dir').value,
                OUTPUT_DIR: document.getElementById('cfg-output-dir').value,
                DEFAULT_MODEL: document.getElementById('cfg-default-model').value
            };

            try {
                saveBtn.loading = true;
                const response = await fetch('/config/save', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(newData)
                });
                const result = await response.json();
                if (result.status === 'success') {
                    this.notify("Configuration saved!", "success");
                    this.settings = newData;
                    drawer.hide();
                    // Optional: refresh page or re-init if paths changed drastically
                } else {
                    this.notify(`Error: ${result.message}`, "danger");
                }
            } catch (e) {
                this.notify("Save error", "danger");
            } finally {
                saveBtn.loading = false;
            }
        });
    }

    getCurrentWorkflowState() {
        if (!this.workflow) return [];
        return Array.from(this.workflow.children).map(el => {
            const id = el.getAttribute('brick-id');
            const type = el.tagName.toLowerCase();
            let props = {};
            if (typeof el.getValue === 'function') props = el.getValue();
            else if (typeof el.getValues === 'function') props = el.getValues();
            return { id, type, props };
        });
    }

    async addBrick(type) {
        // 1. Capture current values
        const currentStates = this.getCurrentWorkflowState();

        try {
            console.log(`Adding brick: ${type}`);
            const response = await fetch('/brick', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ type })
            });
            const result = await response.json();
            if (result.status === 'success') {
                // 2. Merge local states while PRESERVING local order
                // Identify new bricks (those in result but not in current)
                const newBricks = result.workflow.filter(
                    rb => !currentStates.find(lb => lb.id === rb.id)
                );

                // Construct merged list: Existing Local Order + New Bricks
                const mergedWorkflow = [
                    ...currentStates.map(state => {
                        // Update props from backend if needed, or keep local
                        const backendBrick = result.workflow.find(b => b.id === state.id);
                        return backendBrick ? { ...backendBrick, props: state.props } : null;
                    }).filter(b => b !== null), // Remove any deleted on backend (corner case)
                    ...newBricks
                ];

                this.mountWorkflow(mergedWorkflow);
                this.notify("Module added successfully", "success");
            } else {
                this.notify(`Error: ${result.message}`, "danger");
            }
        } catch (e) {
            console.error("Failed to add brick:", e);
            this.notify("Error while adding module", "danger");
        }
    }

    async removeBrick(brickId) {
        // 1. Capture current values
        const currentStates = this.getCurrentWorkflowState();

        try {
            console.log(`Removing brick: ${brickId}`);
            const response = await fetch(`/brick/${brickId}`, {
                method: 'DELETE'
            });
            const result = await response.json();
            if (result.status === 'success') {
                // 2. Merge local states
                const mergedWorkflow = result.workflow.map(brick => {
                    const saved = currentStates.find(s => s.id === brick.id);
                    if (saved) return { ...brick, props: saved.props };
                    return brick;
                });
                this.mountWorkflow(mergedWorkflow);
                this.notify("Module removed", "primary");
            } else {
                this.notify(`Error: ${result.message}`, "danger");
            }
        } catch (e) {
            console.error("Failed to remove brick:", e);
            this.notify("Error while removing module", "danger");
        }
    }

    setupDragAndDrop() {
        if (!this.workflow) return;

        // Prevent default drag behaviors on container
        this.workflow.addEventListener('dragover', (e) => {
            e.preventDefault();
            const afterElement = this.getDragAfterElement(this.workflow, e.clientX);
            const draggable = document.querySelector('.dragging');
            if (!draggable) return;
            if (afterElement == null) {
                this.workflow.appendChild(draggable);
            } else {
                this.workflow.insertBefore(draggable, afterElement);
            }
        });
    }

    getDragAfterElement(container, x) {
        // FIX: Select ALL bricks (identified by brick-id), not just draggable ones
        // Bricks are not draggable by default anymore.
        const draggableElements = [...container.querySelectorAll('[brick-id]:not(.dragging)')];

        return draggableElements.reduce((closest, child) => {
            const box = child.getBoundingClientRect();
            // We are dragging horizontally
            const offset = x - box.left - box.width / 2;
            if (offset < 0 && offset > closest.offset) {
                return { offset: offset, element: child };
            } else {
                return closest;
            }
        }, { offset: Number.NEGATIVE_INFINITY }).element;
    }

    mountWorkflow(bricks) {
        if (!this.workflow) return;

        try {
            // 1. Reuse existing elements map
            const existingMap = new Map();
            Array.from(this.workflow.children).forEach(child => {
                const id = child.getAttribute('brick-id');
                if (id) existingMap.set(id, child);
            });

            // 2. Process bricks list (Create or Reuse)
            bricks.forEach(brick => {
                if (!brick.type) return;

                let el = existingMap.get(brick.id);

                if (el) {
                    existingMap.delete(brick.id); // Mark as used
                } else {
                    // CREATE: New brick
                    el = document.createElement(brick.type);
                    el.setAttribute('brick-id', brick.id);
                    // Add listeners only once on creation
                    this.setupBrickListeners(el);
                }

                // Update Properties
                if (brick.props) {
                    if (typeof el.setValues === 'function') {
                        el.setValues(brick.props);
                    } else {
                        Object.assign(el, brick.props);
                    }
                }

                // Append moves the element if it exists in the DOM, likely triggering less destructive changes
                // than clear + re-add.
                this.workflow.appendChild(el);
            });

            // 3. Cleanup unused elements
            existingMap.forEach(child => child.remove());

        } catch (err) {
            console.error("[Workflow] Critical error during reconciliation:", err);
            this.notify("Error rendering workflow", "danger");
        }
    }

    setupBrickListeners(el) {
        // Dynamic Draggable Logic
        const addDrag = (e) => {
            const path = e.composedPath();
            const isHandle = path.some(node =>
                node.classList && node.classList.contains('drag-handle')
            );
            if (isHandle) el.setAttribute('draggable', 'true');
        };

        const removeDrag = () => el.removeAttribute('draggable');

        const cleanupDrag = () => {
            el.removeAttribute('draggable');
            el.classList.remove('dragging');
            el.style.opacity = '1';
        };

        const startDrag = (e) => {
            if (!el.getAttribute('draggable')) {
                e.preventDefault();
                return;
            }
            el.classList.add('dragging');
            el.style.opacity = '0.5';
        };

        el.addEventListener('mousedown', addDrag);
        el.addEventListener('mouseup', removeDrag);
        el.addEventListener('dragend', cleanupDrag);
        el.addEventListener('dragstart', startDrag);
    }

    notify(message, variant = 'primary', icon = 'info-circle', duration = 3000) {
        console.log(`[Notification] ${variant}: ${message}`);
        const toast = document.createElement('div');
        const color = variant === 'success' ? '#10b981' : (variant === 'danger' ? '#ef4444' : '#6366f1');

        toast.style.cssText = `
            position: fixed; top: 20px; right: 20px; 
            background: #1e293b; color: white; 
            padding: 1rem 1.5rem; border-radius: 0.5rem;
            border-left: 4px solid ${color};
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
            z-index: 10000; font-weight: 600;
            display: flex; align-items: center; gap: 0.5rem;
            transition: opacity 0.3s;
        `;
        toast.innerHTML = `<span>${message}</span>`;
        document.body.appendChild(toast);

        setTimeout(() => {
            toast.style.opacity = '0';
            setTimeout(() => toast.remove(), 300);
        }, duration);
    }
}

// Start the app immediately and expose to window
window.addEventListener('DOMContentLoaded', () => {
    window.qpyt_app = new QpytApp();
    window.qpyt_app.init();
});
