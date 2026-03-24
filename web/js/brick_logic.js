/**
 * Logic_Zone: SAFE | Delta_Initial: 0.15 | Resolution: convergent
 * [WFGY-Metadata]
 * Component: BrickLogic
 * Purpose: Central Registry and Engine for brick dependencies.
 * How to add a brick: Add a new entry in BRICK_MAP with requirements.
 */

const BRICK_MAP = {
    // GENERATORS
    'qp-render-sdxl': {
        required: ['qp-prompt'],
        optional: ['qp-lora-manager', 'qp-controlnet', 'qp-image-blender', 'qp-styles'],
        label: 'SDXL Generator'
    },
    'qp-render-flux': {
        required: ['qp-prompt'],
        optional: ['qp-lora-manager', 'qp-settings'],
        label: 'Flux Generator'
    },
    'qp-narrator': {
        required: ['qp-llm-assistant'],
        optional: ['qp-prompt'],
        label: 'Story Narrator',
        note: 'Needs a generator in the workflow to produce images.'
    },
    'qp-sprite': {
        required: ['qp-prompt'],
        label: 'Sprite Animator'
    },

    // PROCESSORS
    'qp-controlnet': {
        required: ['qp-render-sdxl'], // Currently SDXL only
        optional: ['qp-openpose-editor', 'qp-image-input'],
        label: 'ControlNet Wrapper'
    },
    'qp-inpaint': {
        required: ['qp-image-input', 'qp-render-sdxl'],
        label: 'Inpaint Editor'
    },
    'qp-outpaint': {
        required: ['qp-image-input', 'qp-render-sdxl'],
        label: 'Outpaint Editor'
    },
    'qp-upscaler-v3': {
        optional: ['qp-image-input'],
        label: 'Super Resolution V3'
    },
    'qp-auto-background': {
        required: ['qp-image-input'],
        label: 'Smart Background'
    },
    'qp-rembg': {
        required: ['qp-image-input'],
        label: 'Background Remover'
    },
    'qp-image-blender': {
        required: ['qp-render-sdxl'],
        label: 'IP-Adapter Blender'
    },

    // SPECIALIZED INPUTS
    'qp-openpose-editor': {
        required: ['qp-controlnet'],
        label: 'OpenPose Editor'
    },

    // INPUTS & HELPERS (Implicitly ready)
    'qp-prompt': { label: 'Prompt Input' },
    'qp-settings': { label: 'Generation Settings' },
    'qp-lora-manager': { label: 'LoRA Manager' },
    'qp-styles': { label: 'Styles Gallery' },
    'qp-image-input': { label: 'Image Source' },
    'qp-history-log': { label: 'History Log' }
};

class BrickLogic {
    /**
     * Scans the current workflow and returns status for a specific brick.
     * @param {string} type - Tag name of the brick (e.g. 'qp-openpose-editor')
     * @returns {object} { status: 'ready'|'missing'|'conflict', missing: [], conflicts: [] }
     */
    static validate(type) {
        const rules = BRICK_MAP[type];
        const label = rules?.label || this.getLabel(type);
        
        if (!rules) return { status: 'ready', missing: [], conflicts: [], label };

        const currentBricks = Array.from(document.querySelectorAll('[brick-id]')).map(el => el.tagName.toLowerCase());
        const missing = [];
        const conflicts = [];

        // Check required
        if (rules.required) {
            for (const req of rules.required) {
                if (!currentBricks.includes(req)) {
                    missing.push(req);
                }
            }
        }

        // Special context rules (e.g. ControlNet doesn't work with Flux yet)
        if (type === 'qp-controlnet' && currentBricks.includes('qp-render-flux') && !currentBricks.includes('qp-render-sdxl')) {
            conflicts.push('qp-render-flux');
        }

        let status = 'ready';
        if (conflicts.length > 0) status = 'conflict';
        else if (missing.length > 0) status = 'missing';

        return { status, missing, conflicts, label };
    }

    /**
     * Human readable name for a brick tag
     */
    static getLabel(type) {
        return BRICK_MAP[type]?.label || type.replace('qp-', '').replace(/-/g, ' ');
    }
}

// Export to window for global access
window.BrickLogic = BrickLogic;
