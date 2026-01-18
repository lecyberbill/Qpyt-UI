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
            {"id": "qp-render-sd35turbo", "label": "SD3.5 Turbo (Lightning)", "type": "generator", "icon": "lightning-charge"},
            {"id": "qp-img2img", "label": "Img2Img Refiner", "type": "generator", "icon": "magic"},
            {"id": "qp-upscaler", "label": "Tiled Upscaler", "type": "generator", "icon": "aspect-ratio"},
            {"id": "qp-lora-manager", "label": "LoRA Manager", "type": "generator", "icon": "layers-half"},
            {"id": "qp-image-out", "label": "Final Output", "type": "output", "icon": "check-circle"},
            {"id": "qp-rembg", "label": "Background Removal", "type": "generator", "icon": "scissors"},
            {"id": "qp-save-to-disk", "label": "Save to Disk", "type": "output", "icon": "download"},
            {"id": "qp-inpaint", "label": "Inpainting", "type": "generator", "icon": "brush"},
            {"id": "qp-outpaint", "label": "Outpainting", "type": "generator", "icon": "arrows-angle-expand"},
            {"id": "qp-vectorize", "label": "Vectorize (SVG)", "type": "generator", "icon": "vector-pen"},
            {"id": "qp-filter", "label": "Photo Editing", "type": "output", "icon": "sliders"}
        ]

    def load_workflow(self, workflow_data: List[Dict[str, Any]]):
        """Replaces the current workflow with new data."""
        self.workflow = workflow_data
        self._sort_workflow()
        return self

    def add_brick(self, brick_type: str, **kwargs):
        """Ajoute une brique au workflow avec un ID unique."""
        brick_id = f"{brick_type}-{uuid.uuid4().hex[:6]}"
        self.workflow.append({
            "id": brick_id,
            "type": brick_type,
            "props": kwargs
        })
        self._sort_workflow()
        return brick_id

    def _sort_workflow(self):
        """Trie le workflow selon un ordre logique : Input -> Logic -> Output."""
        # Définition de l'ordre prioritaire par type de brique
        priority = {
            "qp-image-input": 0,
            "qp-styles": 1,
            "qp-llm-prompter": 1,
            "qp-img2prompt": 2,
            "qp-prompt": 2,
            "qp-translator": 3,
            "qp-settings": 10,
            "qp-render-sdxl": 100,
            "qp-render-flux": 101,
            "qp-render-sd35turbo": 102,
            "qp-img2img": 103,
            "qp-inpaint": 103,
            "qp-outpaint": 103,
            "qp-upscaler": 104,
            "qp-rembg": 105,
            "qp-vectorize": 106,
            "qp-filter": 150,
            "qp-image-out": 200,
            "qp-save-to-disk": 201,
        }
        # Tri stable pour garder l'ordre d'insertion pour les briques de même priorité
        # On utilise une priorité par défaut de 50 pour les types non listés
        self.workflow.sort(key=lambda b: priority.get(b["type"], 50))

    def remove_brick(self, brick_id: str):
        self.workflow = [b for b in self.workflow if b.get("id") != brick_id]
        self._sort_workflow()
        return self

    def get_config(self) -> Dict[str, Any]:
        """Returns the full UI configuration for the frontend."""
        return {
            "title": self.title,
            "workflow": self.workflow,
            "library": self.available_library
        }
