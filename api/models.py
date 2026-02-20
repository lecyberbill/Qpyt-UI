from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import uuid

# --- LE CONTRAT D'ENTRÉE ---
class ImageGenerationRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Description de l'image à générer")
    negative_prompt: Optional[str] = Field(None, description="Éléments à exclure de l'image. Ignoré pour les modèles Flux.")
    model_type: str = Field("sdxl", description="Type de modèle (sdxl, flux, sd3)")
    model_name: Optional[str] = Field(None, description="Nom du fichier .safetensors spécifique")
    width: int = Field(1024, ge=256, le=2048)
    height: int = Field(1024, ge=256, le=2048)
    guidance_scale: float = Field(7.0, ge=0.0, le=30.0)
    num_inference_steps: int = Field(30, ge=1, le=100)
    seed: Optional[int] = None
    vae_name: Optional[str] = Field(None, description="Nom du fichier VAE spécifique")
    sampler_name: Optional[str] = Field(None, description="Nom du sampler (scheduler)")
    image: Optional[str] = Field(None, description="Image source en Base64 pour Img2Img")
    image_a: Optional[str] = Field(None, description="Image A pour Blender (IP-Adapter)")
    image_b: Optional[str] = Field(None, description="Image B pour Blender (IP-Adapter)")
    weight_a: float = Field(0.5, ge=0.0, le=2.0, description="Poids Image A")
    weight_b: float = Field(0.5, ge=0.0, le=2.0, description="Poids Image B")
    ip_adapter_scale: float = Field(0.5, ge=0.0, le=2.0, description="Force de l'influence IP-Adapter (Legacy/Global)")
    denoising_strength: float = Field(0.5, ge=0.0, le=1.0, description="Force de transformation (0=identique, 1=nouveau)")
    batch_count: Optional[int] = Field(None, description="Nombre d'images à générer")
    output_format: str = Field("png", description="Format de sortie (png, jpeg, webp)")
    loras: Optional[List[Dict[str, Any]]] = Field(None, description="Liste des LoRAs: [{'path': '...', 'weight': 1.0, 'enabled': True}]")
    controlnet_image: Optional[str] = Field(None, description="Image de contrôle en Base64 (Depth map, Canny, etc.)")
    controlnet_conditioning_scale: float = Field(0.7, ge=0.0, le=2.0, description="Force de l'influence ControlNet")
    controlnet_model: Optional[str] = Field("diffusers/controlnet-depth-sdxl-1.0", description="Nom du modèle ControlNet sur HF ou chemin local")
    low_vram: bool = Field(False, description="Activer le mode faible VRAM (quantization) pour Flux")
    workflow: Optional[List[Any]] = Field(None, description="Données du workflow JSON pour injection")
# --- LE CONTRAT DE SORTIE ---
class ImageGenerationResponse(BaseModel):
    request_id: uuid.UUID
    image_url: str  # L'URL servie par Starlette (ex: /view/abc-123.png)
    execution_time: float = Field(..., description="Temps de génération en secondes")
    metadata: dict = Field(default_factory=dict, description="Paramètres réellement utilisés")

class ImageAnalysisResponse(BaseModel):
    status: str
    prompt: Optional[str] = None
    message: Optional[str] = None

class FilterRequest(BaseModel):
    image: str # Base64 source
    settings: Dict[str, Any] # Dictionary of filter values

class UpscaleRequest(BaseModel):
    image: str = Field(..., description="Image source en Base64")
    scale_factor: float = Field(2.0, ge=1.0, le=4.0)
    denoising_strength: float = Field(0.2, ge=0.0, le=1.0)
    tile_size: int = Field(768, ge=256, le=2048)
    prompt: Optional[str] = None
    negative_prompt: Optional[str] = None
    model_name: Optional[str] = None
    model_type: Optional[str] = "upscale"
    output_format: str = Field("png", description="Format de sortie (png, jpeg, webp)")
    seed: Optional[int] = None
    loras: Optional[List[Dict[str, Any]]] = Field(None, description="LoRAs pour l'upscale")

class RembgRequest(BaseModel):
    image: str = Field(..., description="Image source en Base64 ou chemin /view/")

class SaveToDiskRequest(BaseModel):
    image_url: str = Field(..., description="URL de l'image (/view/...)")
    custom_path: str = Field(..., description="Chemin dossier sur le disque")
    pattern: str = Field("result_{seed}", description="Paterne de nom de fichier")
    output_format: str = "png"
    params: Dict[str, Any] = {}

class InpaintRequest(ImageGenerationRequest):
    mask: str = Field(..., description="Masque en Base64")
    # Inherits prompt, model_type, width, height, etc. from ImageGenerationRequest

class OutpaintRequest(InpaintRequest):
    # Same as inpaint but often handled with different padding logic
    pass

class DepthRequest(BaseModel):
    image: str = Field(..., description="Image source en Base64 ou chemin /view/")

class NormalRequest(BaseModel):
    image: str = Field(..., description="Image source en Base64 ou chemin /view/ (Depth Map)")
    strength: float = Field(2.0, ge=0.1, le=10.0, description="Force du relief")

class AudioGenerationRequest(BaseModel):
    prompt: str = Field(..., description="Description de la musique")
    duration: int = Field(10, ge=1, le=300, description="Durée en secondes")

class SpriteRequest(BaseModel):
    prompt: str = Field(..., description="Description du sprite à animer")
    negative_prompt: Optional[str] = Field("blur, fuzzy, extra limbs, malformed", description="Éléments à exclure")
    width: int = Field(256, ge=64, le=1024, description="Largeur du sprite (256 recommandé)")
    height: int = Field(256, ge=64, le=1024, description="Hauteur du sprite (256 recommandé)")
    frames: int = Field(16, ge=8, le=64, description="Nombre de frames (16 standard)")
    steps: int = Field(8, ge=1, le=50, description="Étapes d'inférence (8 pour rapide)")
    guidance: float = Field(7.5, ge=1.0, le=20.0, description="Guidance Scale")
    seed: Optional[int] = None
    model_name: Optional[str] = Field(None, description="Nom du modèle SD1.5 à utiliser")
    loras: Optional[List[Dict[str, Any]]] = Field(None, description="Liste des LoRAs")

class VideoGenerationRequest(BaseModel):
    prompt: str = Field(..., description="Description de la vidéo")
    model_name: Optional[str] = Field("THUDM/CogVideoX-2b", description="Nom du modèle CogVideo")
    num_frames: int = Field(49, ge=8, le=128, description="Nombre de frames")
    fps: int = Field(8, ge=4, le=30, description="Frames par seconde")
    num_inference_steps: int = Field(50, ge=1, le=200, description="Étapes d'inférence")
    guidance_scale: float = Field(6.0, ge=1.0, le=20.0, description="Guidance Scale")
    seed: Optional[int] = None
    low_vram: bool = Field(True, description="Mode faible VRAM")
