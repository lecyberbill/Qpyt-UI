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
            this.setupImageDrop();

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

    async updateModelSelectors(type = 'all', pathOverride = null) {
        // Fetch current lists for SDXL and Flux
        const sdxlPath = (type === 'sdxl' || type === 'all') ? (pathOverride || document.getElementById('cfg-models-dir')?.value || this.settings.MODELS_DIR) : null;
        const fluxPath = (type === 'flux' || type === 'all') ? (pathOverride || document.getElementById('cfg-flux-models-dir')?.value || this.settings.FLUX_MODELS_DIR) : null;

        const updateOne = async (modelType, path, selectId, settingKey) => {
            const select = document.getElementById(selectId);
            if (!select || !path) return;

            try {
                const res = await fetch(`/models/${modelType}?path=${encodeURIComponent(path)}`);
                const data = await res.json();

                if (data.status === 'success') {
                    // Update options
                    select.innerHTML = data.models.map(m => `<sl-option value="${m}">${m}</sl-option>`).join('');

                    // Force a small delay to let Shoelace register the new options
                    requestAnimationFrame(() => {
                        const currentValue = this.settings[settingKey];
                        if (data.models.includes(currentValue)) {
                            select.value = currentValue;
                        }
                    });
                    // Optional: select first one if nothing selected? No, stay neutral.
                    // This condition was originally an `else if` but is now separate due to the requestAnimationFrame.
                    // If the above `requestAnimationFrame` didn't set a value, this might still apply.
                    // However, the user's provided snippet removed the `else` and placed it outside the `if (data.status === 'success')` block.
                    // To maintain syntactic correctness and the spirit of the change, I'm placing it here as a separate check.
                    if (data.models.length > 0 && !select.value) {
                        // No action, as per original comment.
                    }
                }
            } catch (e) {
                console.error(`Failed to scan ${modelType} at ${path}`, e);
            }
        };

        if (type === 'all' || type === 'sdxl') await updateOne('sdxl', sdxlPath, 'cfg-default-model', 'DEFAULT_MODEL');
        if (type === 'all' || type === 'flux') await updateOne('flux', fluxPath, 'cfg-default-flux-model', 'DEFAULT_FLUX_MODEL');
    }

    setupSettings() {
        const trigger = document.getElementById('settings-trigger');
        const drawer = document.getElementById('settings-drawer');
        const saveBtn = document.getElementById('cfg-save-btn');
        const jsonSaveBtn = document.getElementById('cfg-json-save-btn');
        const jsonEditor = document.getElementById('cfg-json-editor');

        if (!trigger || !drawer) return;

        // Path inputs
        const sdxlInput = document.getElementById('cfg-models-dir');
        const fluxInput = document.getElementById('cfg-flux-models-dir');
        const loraInput = document.getElementById('cfg-loras-dir');
        const outputInput = document.getElementById('cfg-output-dir');

        // Scan buttons
        const scanSdxlBtn = document.getElementById('btn-scan-sdxl');
        const scanFluxBtn = document.getElementById('btn-scan-flux');

        // Initial population of path fields
        const populatePaths = () => {
            if (sdxlInput) sdxlInput.value = this.settings.MODELS_DIR || '';
            if (fluxInput) fluxInput.value = this.settings.FLUX_MODELS_DIR || '';
            if (loraInput) loraInput.value = this.settings.LORAS_DIR || '';
            if (outputInput) outputInput.value = this.settings.OUTPUT_DIR || '';

            if (jsonEditor) jsonEditor.value = JSON.stringify(this.settings, null, 4);
        };

        populatePaths();

        trigger.addEventListener('click', async () => {
            if (drawer.open) {
                drawer.hide();
            } else {
                populatePaths();
                await this.updateModelSelectors('all');
                drawer.show();
            }
        });

        // Manual Scan buttons
        scanSdxlBtn?.addEventListener('click', async () => {
            scanSdxlBtn.loading = true;
            await this.updateModelSelectors('sdxl', sdxlInput.value);
            scanSdxlBtn.loading = false;
        });
        scanFluxBtn?.addEventListener('click', async () => {
            scanFluxBtn.loading = true;
            await this.updateModelSelectors('flux', fluxInput.value);
            scanFluxBtn.loading = false;
        });

        // Live Scan on input change (also triggered on Enter)
        sdxlInput?.addEventListener('sl-change', () => this.updateModelSelectors('sdxl', sdxlInput.value));
        fluxInput?.addEventListener('sl-change', () => this.updateModelSelectors('flux', fluxInput.value));

        // Basic settings save
        saveBtn?.addEventListener('click', async () => {
            const newData = {
                ...this.settings,
                MODELS_DIR: sdxlInput.value,
                FLUX_MODELS_DIR: fluxInput.value,
                LORAS_DIR: loraInput.value,
                OUTPUT_DIR: outputInput.value,
                DEFAULT_MODEL: document.getElementById('cfg-default-model').value,
                DEFAULT_FLUX_MODEL: document.getElementById('cfg-default-flux-model').value
            };
            await this.saveConfiguration(newData, saveBtn);
        });

        // JSON Advanced save
        jsonSaveBtn?.addEventListener('click', async () => {
            try {
                const newData = JSON.parse(jsonEditor.value);
                await this.saveConfiguration(newData, jsonSaveBtn);
            } catch (e) {
                this.notify("Invalid JSON syntax: " + e.message, "danger");
            }
        });
    }

    async saveConfiguration(newData, btnEl) {
        try {
            if (btnEl) btnEl.loading = true;
            const response = await fetch('/config/save', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(newData)
            });
            const result = await response.json();
            if (result.status === 'success') {
                this.notify("Configuration updated successfully!", "success");
                this.settings = newData;
                // Sync UI
                if (document.getElementById('cfg-json-editor')) {
                    document.getElementById('cfg-json-editor').value = JSON.stringify(newData, null, 4);
                }
                // Update selectors from the new saved state
                await this.updateModelSelectors('all');
            } else {
                this.notify(`Error: ${result.message}`, "danger");
            }
        } catch (e) {
            this.notify("Save error", "danger");
        } finally {
            if (btnEl) btnEl.loading = false;
        }
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

    setupImageDrop() {
        // Global drag/drop for images to reconstruct workflow
        document.addEventListener('dragover', (e) => {
            e.preventDefault();
            e.dataTransfer.dropEffect = 'copy';
        });

        document.addEventListener('drop', async (e) => {
            // Check if we are dropping a file (not an internal brick reorder)
            if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
                e.preventDefault();
                const file = e.dataTransfer.files[0];
                if (file.type.startsWith('image/')) {
                    this.extractWorkflowFromImage(file);
                }
            }
        });
    }

    async extractWorkflowFromImage(file) {
        try {
            this.notify(`Extracting workflow from ${file.name}...`, "info");

            const formData = new FormData();
            formData.append('file', file);

            const response = await fetch('/workflows/extract', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();
            if (result.status === 'success') {
                this.notify("Workflow reconstructed successfully!", "success");
                this.mountWorkflow(result.workflow);
            } else {
                this.notify(result.message || "No workflow found in image", "warning");
            }
        } catch (e) {
            console.error("[Metadata Extension]", e);
            this.notify("Failed to extract metadata", "danger");
        }
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
                // 2. Add New Brick Safely (Client is Source of Truth)
                // We only care about the NEW brick created by the backend.
                // We do NOT want to sync with result.workflow because the server might have reset/desynced.
                const newBrickId = result.brick_id;
                const newBrick = result.workflow.find(b => b.id === newBrickId);

                if (newBrick) {
                    // Simple Priority Heuristic for "Smart Insert"
                    // qp-styles (priority 1) and inputs should be near top
                    const lowPriorityTypes = ['qp-styles', 'qp-prompt', 'qp-image-input'];
                    if (lowPriorityTypes.includes(newBrick.type)) {
                        currentStates.splice(0, 0, newBrick); // Insert at top
                    } else {
                        currentStates.push(newBrick); // Append at bottom
                    }
                    this.mountWorkflow(currentStates);

                    // 3. New: Apply Resonant Defaults if available
                    // We wait for the next tick to ensure the element is in the DOM and upgraded
                    setTimeout(() => {
                        const el = this.workflow.querySelector(`[brick-id="${newBrickId}"]`);
                        if (el && typeof el.applyDefaultSettings === 'function') {
                            console.log(`[App] Applying resonant defaults for: ${el.tagName}`);
                            el.applyDefaultSettings();
                        }
                    }, 50);

                }
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
                // 2. Remove locally
                const mergedWorkflow = currentStates.filter(b => b.id !== brickId);
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
            console.log("[App] Workflow Reconciliation Started. Bricks:", bricks.length);
            const tagsToCheck = ['q-upscaler-v3', 'qp-prompt', 'qp-render-sdxl', 'qp-render-flux', 'qp-render-flux-klein', 'qp-image-out'];
            const qpElements = customElements.get ? tagsToCheck.map(tag => `${tag}: ${customElements.get(tag) ? 'YES' : 'NO'}`) : 'N/A';
            console.log("[App] Custom Registry Check:", qpElements);

            bricks.forEach((brick, index) => {
                if (!brick || !brick.type) {
                    console.warn(`[App] Skipping invalid brick at index ${index}`, brick);
                    return;
                }

                let el = existingMap.get(brick.id);

                if (el) {
                    existingMap.delete(brick.id); // Mark as used
                } else {
                    const safeType = (brick.type || "").trim().toLowerCase();
                    const codes = safeType.split('').map(c => c.charCodeAt(0)).join(',');

                    // Validate standard custom element name
                    const isValidTagName = safeType.includes('-') && /^[a-z][a-z0-9-]*$/.test(safeType);

                    if (!isValidTagName) {
                        console.error(`[App] Illegal brick type: '${safeType}' (Codes: ${codes})`, brick);
                        el = document.createElement('div');
                        el.setAttribute('style', 'color:#ef4444; background:rgba(239,68,68,0.1); padding:1rem; border:1px solid #ef4444; border-radius:8px; width:200px;');
                        el.innerHTML = `<b>Invalid Module</b><br><small>Type: ${safeType || 'EMPTY'}</small>`;
                    } else {
                        try {
                            const isRegistered = !!(customElements.get && customElements.get(safeType));
                            console.log(`[App] Mounting Module: ${safeType} (ID: ${brick.id}, Registered: ${isRegistered}, Codes: ${codes})`);

                            if (isRegistered) {
                                el = document.createElement(safeType);
                            } else {
                                throw new Error("Element not registered in customElements");
                            }
                        } catch (ce_err) {
                            console.error(`[App] Failed to create '${safeType}':`, ce_err);

                            el = document.createElement('div');
                            el.setAttribute('style', 'color:#f59e0b; background:rgba(245,158,11,0.1); padding:1rem; border:1px solid #f59e0b; border-radius:8px; width:250px;');
                            el.innerHTML = `
                                <div style="font-weight:800; margin-bottom: 0.5rem;">⚠️ Creation Failed</div>
                                <div style="font-size:0.75rem; opacity:0.8;">
                                    Tag: <code>${safeType}</code><br>
                                    Error: ${ce_err.name || 'Error'}: ${ce_err.message}<br>
                                    Module may be missing or failed to load.
                                </div>
                            `;
                        }
                    }

                    if (el) {
                        el.setAttribute('brick-id', brick.id);
                        this.setupBrickListeners(el);
                    }
                }

                if (el) {
                    // Update Properties
                    if (brick.props) {
                        try {
                            if (typeof el.setValues === 'function') {
                                el.setValues(brick.props);
                            } else {
                                Object.assign(el, brick.props);
                            }
                        } catch (propErr) {
                            console.warn(`[App] Failed to apply props to ${brick.id}:`, propErr);
                        }
                    }
                    this.workflow.appendChild(el);
                }
            });

            // 3. Cleanup unused elements
            existingMap.forEach(child => {
                console.log(`[App] Removing unused brick: ${child.getAttribute('brick-id')}`);
                child.remove();
            });

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
