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
    guidance_scale: float = Field(7.0, ge=1.0, le=20.0)
    num_inference_steps: int = Field(30, ge=1, le=100)
    seed: Optional[int] = None
    vae_name: Optional[str] = Field(None, description="Nom du fichier VAE spécifique")
    sampler_name: Optional[str] = Field(None, description="Nom du sampler (scheduler)")
    image: Optional[str] = Field(None, description="Image source en Base64 pour Img2Img")
    denoising_strength: float = Field(0.5, ge=0.0, le=1.0, description="Force de transformation (0=identique, 1=nouveau)")
    batch_count: Optional[int] = Field(None, description="Nombre d'images à générer")
    output_format: str = Field("png", description="Format de sortie (png, jpeg, webp)")
    loras: Optional[List[Dict[str, Any]]] = Field(None, description="Liste des LoRAs: [{'path': '...', 'weight': 1.0, 'enabled': True}]")

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
