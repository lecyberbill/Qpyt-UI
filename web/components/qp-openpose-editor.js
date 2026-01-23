class QpOpenPoseEditor extends HTMLElement {
    static get observedAttributes() { return ['brick-id']; }

    constructor() {
        super();
        this.attachShadow({ mode: 'open' });

        // --- OpenPose COCO Constants ---
        this.joints = [
            { id: 0, name: 'Nose', color: '#ff0000', x: 0, y: 0 },
            { id: 1, name: 'Neck', color: '#ff5500', x: 0, y: 0 },
            { id: 2, name: 'R-Sho', color: '#ffaa00', x: 0, y: 0 },
            { id: 3, name: 'R-Elb', color: '#ffff00', x: 0, y: 0 },
            { id: 4, name: 'R-Wr', color: '#aaff00', x: 0, y: 0 },
            { id: 5, name: 'L-Sho', color: '#55ff00', x: 0, y: 0 },
            { id: 6, name: 'L-Elb', color: '#00ff00', x: 0, y: 0 },
            { id: 7, name: 'L-Wr', color: '#00ff55', x: 0, y: 0 },
            { id: 8, name: 'R-Hip', color: '#00ffaa', x: 0, y: 0 },
            { id: 9, name: 'R-Knee', color: '#00ffff', x: 0, y: 0 },
            { id: 10, name: 'R-Ank', color: '#00aaff', x: 0, y: 0 },
            { id: 11, name: 'L-Hip', color: '#0055ff', x: 0, y: 0 },
            { id: 12, name: 'L-Knee', color: '#0000ff', x: 0, y: 0 },
            { id: 13, name: 'L-Ank', color: '#5500ff', x: 0, y: 0 },
            { id: 14, name: 'R-Eye', color: '#aa00ff', x: 0, y: 0 },
            { id: 15, name: 'L-Eye', color: '#ff00ff', x: 0, y: 0 },
            { id: 16, name: 'R-Ear', color: '#ff00aa', x: 0, y: 0 },
            { id: 17, name: 'L-Ear', color: '#ff0055', x: 0, y: 0 },
        ];

        // Connections (Limb Indices)
        this.connections = [
            [1, 2], [1, 5], [2, 3], [3, 4], [5, 6], [6, 7], [1, 8], [8, 9], [9, 10], [1, 11], [11, 12], [12, 13], [1, 0], [0, 14], [14, 16], [0, 15], [15, 17]
        ];

        // Limb Colors (matching ControlNet standards roughly)
        // Usually OpenPose uses specific limb colors
        this.limbColors = [
            '#ff5500', '#55ff00', '#ffaa00', '#ffff00', '#aaff00', '#00ff00', '#00ffaa',
            '#00ffff', '#00aaff', '#0055ff', '#0000ff', '#5500ff', '#ff0000', '#aa00ff', '#ff00aa', '#ff00ff', '#ff0055'
        ];

        this.poses = []; // Array of skeletons (multi-person support foundation)
        this.activeJoint = null;
        this.activePoseIndex = -1;
        this.editorOpen = false;

        this.canvasWidth = 512;
        this.canvasHeight = 512;

        this.generatedImage = null; // Resulting image (black bg + skeletons)
    }

    connectedCallback() {
        this.render();
    }

    // --- Editor Logic ---

    initDefaultPose() {
        // Create a default standing pose centered
        const cx = this.canvasWidth / 2;
        const cy = this.canvasHeight / 2;
        const scale = this.canvasHeight / 3;

        // Helper to clone joints structure
        const newJoints = JSON.parse(JSON.stringify(this.joints));

        // Set coordinates (Simple standing pose)
        const setJ = (id, dx, dy) => {
            newJoints[id].x = cx + dx * scale;
            newJoints[id].y = cy + dy * scale - (scale * 0.5);
        };

        setJ(0, 0, 0); // Nose
        setJ(1, 0, 0.2); // Neck
        setJ(2, -0.2, 0.2); setJ(3, -0.3, 0.5); setJ(4, -0.35, 0.7); // R-Arm
        setJ(5, 0.2, 0.2); setJ(6, 0.3, 0.5); setJ(7, 0.35, 0.7); // L-Arm
        setJ(8, -0.15, 0.6); setJ(9, -0.15, 1.0); setJ(10, -0.15, 1.4); // R-Leg
        setJ(11, 0.15, 0.6); setJ(12, 0.15, 1.0); setJ(13, 0.15, 1.4); // L-Leg
        setJ(14, -0.05, -0.05); setJ(15, 0.05, -0.05); // Eyes
        setJ(16, -0.1, -0.02); setJ(17, 0.1, -0.02); // Ears

        this.poses.push(newJoints);
        this.drawCanvas();
    }

    openEditor() {
        // Create default pose if empty
        if (this.poses.length === 0) {
            this.initDefaultPose();
        }

        this.editorOpen = true;
        this.renderEditor();
        this.drawCanvas();
    }

    closeEditor() {
        this.editorOpen = false;
        const dialog = this.shadowRoot.querySelector('.editor-overlay');
        if (dialog) dialog.remove();
        this.render();
    }

    drawCanvas() {
        if (!this.editorOpen) return;
        const canvas = this.shadowRoot.getElementById('pose-canvas');
        if (!canvas) return;
        const ctx = canvas.getContext('2d');

        // Clear (Black Background)
        ctx.fillStyle = '#000000';
        ctx.fillRect(0, 0, canvas.width, canvas.height);

        // Draw Poses
        this.poses.forEach(joints => {
            // Draw Limbs
            this.connections.forEach((conn, i) => {
                const j1 = joints[conn[0]];
                const j2 = joints[conn[1]];
                if (j1.x !== 0 && j2.x !== 0) { // Check visibility (0,0 usually hidden)
                    ctx.beginPath();
                    ctx.moveTo(j1.x, j1.y);
                    ctx.lineTo(j2.x, j2.y);
                    ctx.strokeStyle = this.limbColors[i % this.limbColors.length];
                    ctx.lineWidth = 4;
                    ctx.stroke();
                }
            });

            // Draw Joints
            joints.forEach(j => {
                ctx.beginPath();
                ctx.arc(j.x, j.y, 4, 0, 2 * Math.PI);
                ctx.fillStyle = j.color;
                ctx.fill();
            });
        });
    }

    handleMouseDown(e) {
        const canvas = e.target;
        const rect = canvas.getBoundingClientRect();
        const x = (e.clientX - rect.left) * (canvas.width / rect.width);
        const y = (e.clientY - rect.top) * (canvas.height / rect.height);

        // Check collision with joints
        for (let pIdx = 0; pIdx < this.poses.length; pIdx++) {
            const joints = this.poses[pIdx];
            for (let j of joints) {
                const dist = Math.sqrt((j.x - x) ** 2 + (j.y - y) ** 2);
                if (dist < 10) { // Hit radius
                    this.activePoseIndex = pIdx;
                    this.activeJoint = j;
                    return;
                }
            }
        }
    }

    handleMouseMove(e) {
        if (!this.activeJoint) return;
        const canvas = e.target;
        const rect = canvas.getBoundingClientRect();
        const x = (e.clientX - rect.left) * (canvas.width / rect.width);
        const y = (e.clientY - rect.top) * (canvas.height / rect.height);

        this.activeJoint.x = x;
        this.activeJoint.y = y;
        this.drawCanvas();
    }

    handleMouseUp(e) {
        this.activeJoint = null;
    }

    clearAll() {
        this.poses = [];
        this.drawCanvas();
    }

    addPerson() {
        this.initDefaultPose();
    }

    sendToControlNet() {
        const canvas = this.shadowRoot.getElementById('pose-canvas');
        if (!canvas) return;

        this.generatedImage = canvas.toDataURL('image/png');

        const cnBrick = document.querySelector('qp-controlnet');
        if (cnBrick) {
            cnBrick.controlNetImage = this.generatedImage;
            cnBrick.hasRendered = false; cnBrick.render();
            // Try to auto-select OpenPose model if available
            if (cnBrick.controlnets) {
                const openposeModel = cnBrick.controlnets.find(m => m.toLowerCase().includes('openpose'));
                if (openposeModel) {
                    cnBrick.selectedControlNet = openposeModel;
                    // Trigger update manually if needed
                }
            }
            this.notify("Sent to ControlNet!", "success");
            this.closeEditor();
        } else {
            this.notify("ControlNet brick not found.", "warning");
        }
    }

    notify(msg, type) {
        if (window.qpyt_app) window.qpyt_app.notify(msg, type);
    }

    // --- UI Rendering ---

    renderEditor() {
        let overlay = this.shadowRoot.querySelector('.editor-overlay');
        if (!overlay) {
            overlay = document.createElement('div');
            overlay.className = 'editor-overlay';
            this.shadowRoot.appendChild(overlay);
        }

        overlay.innerHTML = `
            <style>
                .editor-overlay {
                    position: fixed; top: 0; left: 0; width: 100vw; height: 100vh;
                    background: rgba(0,0,0,0.9);
                    z-index: 99999;
                    display: flex; flex-direction: column;
                }
                .header { padding: 1rem; background: #1e293b; display: flex; justify-content: space-between; align-items:center; }
                .body { flex: 1; display: flex; align-items: center; justify-content: center; background: #0f172a; position: relative; }
                canvas { background: black; box-shadow: 0 0 20px rgba(0,0,0,0.5); cursor: crosshair; touch-action: none; }
                .controls { padding: 1rem; background: #1e293b; display: flex; gap: 1rem; justify-content: center; }
            </style>
            
            <div class="header">
                 <div style="font-weight:bold; font-size:1.2rem; color:#a855f7;">
                    <sl-icon name="person-standing"></sl-icon> OpenPose Editor
                </div>
                 <sl-icon-button name="x-lg" style="font-size:1.5rem;" id="close-btn"></sl-icon-button>
            </div>
            
            <div class="body">
                <canvas id="pose-canvas" width="${this.canvasWidth}" height="${this.canvasHeight}"></canvas>
            </div>
            
            <div class="controls">
                <sl-button variant="neutral" id="add-btn">
                    <sl-icon slot="prefix" name="person-plus"></sl-icon> Add Person
                </sl-button>
                <sl-button variant="danger" outline id="clear-btn">
                    <sl-icon slot="prefix" name="trash"></sl-icon> Clear
                </sl-button>
                <sl-button variant="success" id="send-btn">
                    <sl-icon slot="prefix" name="arrow-right-circle"></sl-icon> Send to ControlNet
                </sl-button>
            </div>
        `;

        const canvas = overlay.querySelector('canvas');
        canvas.addEventListener('mousedown', (e) => this.handleMouseDown(e));
        canvas.addEventListener('mousemove', (e) => this.handleMouseMove(e));
        window.addEventListener('mouseup', (e) => this.handleMouseUp(e));

        overlay.querySelector('#close-btn').onclick = () => this.closeEditor();
        overlay.querySelector('#add-btn').onclick = () => this.addPerson();
        overlay.querySelector('#clear-btn').onclick = () => this.clearAll();
        overlay.querySelector('#send-btn').onclick = () => this.sendToControlNet();
    }

    render() {
        const brickId = this.getAttribute('brick-id') || '';
        this.shadowRoot.innerHTML = `
            <style>
                .preview-box {
                    width: 100%; aspect-ratio: 1;
                    background: #000;
                    border: 1px solid #334155;
                    border-radius: 8px;
                    display: flex; align-items: center; justify-content: center;
                    cursor: pointer;
                    overflow: hidden;
                }
                .preview-box:hover { border-color: #a855f7; }
                .preview-box img { width: 100%; height: 100%; object-fit: contain; }
            </style>
            
            <qp-cartridge title="OpenPose Editor" type="tool" icon="person-standing" brick-id="${brickId}">
                <div style="display:flex; flex-direction:column; gap:1rem;">
                    <div class="preview-box" onclick="this.getRootNode().host.openEditor()">
                        ${this.generatedImage
                ? `<img src="${this.generatedImage}">`
                : `<div style="color:#475569; text-align:center;">
                                <sl-icon name="pencil" style="font-size:2rem; margin-bottom:0.5rem;"></sl-icon><br>
                                Click to Edit Pose
                               </div>`
            }
                    </div>
                    
                    <sl-button variant="primary" style="width:100%" onclick="this.getRootNode().host.openEditor()">
                        <sl-icon slot="prefix" name="pencil-square"></sl-icon> Open Editor
                    </sl-button>
                </div>
            </qp-cartridge>
        `;
    }
}
customElements.define('qp-openpose-editor', QpOpenPoseEditor);
