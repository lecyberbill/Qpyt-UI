from typing import List, Dict, Any, Optional
import uuid

class QpytUI:
    """
    Qpyt-UI Framework Core.
    Allows defining a modular UI from Python.
    """
    def __init__(self, title: str = "Qpyt-UI App"):
        self.title = title
        self.workflow: List[Dict[str, Any]] = []
        self.available_library: List[Dict[str, Any]] = [
            {"id": "qp-image-input", "label": "Source Image", "type": "input", "icon": "image"},
            {"id": "qp-prompt", "label": "Prompt", "type": "input", "icon": "chat-left-text"},
            {"id": "qp-settings", "label": "Global Settings", "type": "setting", "icon": "gear"},
            {"id": "qp-translator", "label": "Translator (GPT/Qwen)", "type": "input", "icon": "translate"},
            {"id": "qp-styles", "label": "Styles Selector", "type": "input", "icon": "palette"},
            {"id": "qp-llm-prompter", "label": "LLM Prompt Enhancer", "type": "input", "icon": "stars"},
            {"id": "qp-img2prompt", "label": "Image-to-Prompt", "type": "input", "icon": "camera"},
            {"id": "qp-render-sdxl", "label": "SDXL Generator", "type": "generator", "icon": "lightning-charge"},
            {"id": "qp-render-flux", "label": "FLUX Generator", "type": "generator", "icon": "magic"},
            {"id": "qp-render-flux-klein", "label": "FLUX.2 Klein 4B", "type": "generator", "icon": "lightning-charge"},
            {"id": "qp-render-sd35turbo", "label": "SD3.5 Turbo (Lightning)", "type": "generator", "icon": "lightning-charge"},
            {"id": "qp-sprite", "label": "Sprite Animator", "type": "generator", "icon": "film"},
            {"id": "qp-civitai", "label": "Civitai Explorer", "type": "input", "icon": "search"},

            {"id": "qp-music-gen", "label": "Music Generator (Medium)", "type": "generator", "icon": "music-note-beamed"},
            {"id": "qp-img2img", "label": "Img2Img Refiner", "type": "generator", "icon": "magic"},

            {"id": "q-upscaler-v3", "label": "Tiled Upscaler", "type": "generator", "icon": "aspect-ratio"},
            {"id": "qp-lora-manager", "label": "LoRA Manager", "type": "generator", "icon": "layers-half"},
            {"id": "qp-controlnet", "label": "ControlNet", "type": "tool", "icon": "diagram-3"},
            {"id": "qp-openpose-editor", "label": "OpenPose Editor", "type": "tool", "icon": "person-standing"},
            {"id": "qp-image-out", "label": "Final Output", "type": "output", "icon": "check-circle"},
            {"id": "qp-rembg", "label": "Background Removal", "type": "generator", "icon": "scissors"},
            {"id": "qp-depth-estimator", "label": "Depth Estimator", "type": "generator", "icon": "layers"},
            {"id": "qp-normal-map", "label": "Normal Map", "type": "generator", "icon": "box"},
            {"id": "qp-save-to-disk", "label": "Save to Disk", "type": "output", "icon": "download"},
            {"id": "qp-inpaint", "label": "Inpainting", "type": "generator", "icon": "brush"},
            {"id": "qp-outpaint", "label": "Outpainting", "type": "generator", "icon": "arrows-angle-expand"},
            {"id": "qp-vectorize", "label": "Vectorize (SVG)", "type": "generator", "icon": "vector-pen"},
            {"id": "qp-filter", "label": "Photo Editing", "type": "output", "icon": "sliders"},
            {"id": "qp-canvas", "label": "Sketch Canvas", "type": "input", "icon": "brush"},
            {"id": "qp-queue-monitor", "label": "Job Queue Monitor", "type": "setting", "icon": "collection"},
            {"id": "qp-monitor", "label": "System Monitor", "type": "setting", "icon": "activity"},
            {"id": "qp-batch-runner", "label": "Batch Runner", "type": "tool", "icon": "grid-3x3-gap"}
        ]

    def load_workflow(self, workflow_data: List[Dict[str, Any]]):
        """Replaces the current workflow with new data."""
        if not isinstance(workflow_data, list):
            print("Error: Workflow data must be a list of bricks.")
            return self
        self.workflow = workflow_data
        # self._sort_workflow() # Disabled for manual order
        return self

    def add_brick(self, brick_type: str, **kwargs):
        """Ajoute une brique au workflow avec un ID unique, insérée selon l'ordre logique."""
        brick_id = f"{brick_type}-{uuid.uuid4().hex[:6]}"
        new_brick = {
            "id": brick_id,
            "type": brick_type,
            "props": kwargs
        }

        # Smart Insert: Find the correct position based on priority
        # structure: { "type": 0, ... }
        priority_map = self._get_priority_map()
        new_priority = priority_map.get(brick_type, 50)
        
        insert_index = len(self.workflow) # Default: append
        
        for i, existing_brick in enumerate(self.workflow):
            existing_priority = priority_map.get(existing_brick["type"], 50)
            if new_priority < existing_priority:
                insert_index = i
                break
            # If priority is equal, we keep going to insert after the last same-priority item (stable)
        
        self.workflow.insert(insert_index, new_brick)
        return brick_id

    def _get_priority_map(self):
        return {
            "qp-image-input": 0,
            "qp-styles": 1,
            "qp-llm-prompter": 1,
            "qp-img2prompt": 2,
            "qp-prompt": 2,
            "qp-civitai": 2,
            "qp-canvas": 2,
            "qp-translator": 3,
            "qp-settings": 10,
            "qp-render-sdxl": 100,
            "qp-render-flux": 101,
            "qp-render-flux-klein": 102,
            "qp-render-sd35turbo": 103,
            "qp-sprite": 103,

            "qp-music-gen": 102,
            "qp-img2img": 103,
            "qp-inpaint": 103,
            "qp-outpaint": 103,
            "q-upscaler-v3": 104,
            "qp-rembg": 105,
            "qp-depth-estimator": 105,
            "qp-normal-map": 106,
            "qp-vectorize": 107,
            "qp-controlnet": 120,    # Added explicitly
            "qp-openpose-editor": 121, # Added explicitly
            "qp-filter": 150,
            "qp-filter": 150,
            "qp-queue-monitor": 11,
            "qp-monitor": 12,
            "qp-batch-runner": 50,
            "qp-image-out": 200,
            "qp-save-to-disk": 201,
        }

    def _sort_workflow(self):
        """Trie le workflow (Legacy/Force Reset)."""
        priority = self._get_priority_map()
        self.workflow.sort(key=lambda b: priority.get(b["type"], 50))

    def remove_brick(self, brick_id: str):
        self.workflow = [b for b in self.workflow if isinstance(b, dict) and b.get("id") != brick_id]
        # self._sort_workflow()
        return self

    def get_config(self) -> Dict[str, Any]:
        """Returns the full UI configuration for the frontend."""
        return {
            "title": self.title,
            "workflow": self.workflow,
            "library": self.available_library
        }
