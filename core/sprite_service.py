import torch
import gc
from diffusers import AnimateDiffPipeline, MotionAdapter, EulerAncestralDiscreteScheduler
from diffusers.utils import export_to_gif
from datetime import datetime
import os
import random
import logging

logger = logging.getLogger("qpyt-ui")
from core.filters import ImageEditor

class SpriteService:
    _instance = None
    _pipeline = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = SpriteService()
        return cls._instance

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = "emilianJR/epiCRealism" # Base SD1.5 model (good for general purpose)
        # We can switch to a pixel art specific model later if needed, e.g. "on1ycat/pixel-art-v1"
        self.adapter_id = "guoyww/animatediff-motion-adapter-v1-5-2"

    def load_model(self, model_name=None):
        target_model = model_name if model_name else self.model_id
        
        # Check if we need to reload
        if self._pipeline is not None:
             if hasattr(self, 'current_model') and self.current_model == target_model:
                 return
             else:
                 logger.info(f"[SpriteService] Switching model from {getattr(self, 'current_model', 'None')} to {target_model}")
                 self.unload_model()

        logger.info(f"[SpriteService] Loading AnimateDiff pipeline with base: {target_model}")
        try:
            adapter = MotionAdapter.from_pretrained(self.adapter_id, torch_dtype=torch.float16)
            
            # Check if model is a local file
            from core.config import config
            models_dir = config.settings.get('MODELS_DIR', 'models')
            possible_path = os.path.join(models_dir, target_model)
            
            if os.path.exists(possible_path) and os.path.isfile(possible_path):
                logger.info(f"[SpriteService] Loading local single file: {possible_path}")
                self._pipeline = AnimateDiffPipeline.from_single_file(
                    possible_path,
                    motion_adapter=adapter,
                    torch_dtype=torch.float16
                ).to(self.device)
            elif os.path.exists(target_model) and os.path.isfile(target_model):
                # Absolute path provided
                 logger.info(f"[SpriteService] Loading absolute path: {target_model}")
                 self._pipeline = AnimateDiffPipeline.from_single_file(
                    target_model,
                    motion_adapter=adapter,
                    torch_dtype=torch.float16
                ).to(self.device)
            else:
                # Assume HF repo or diffusers directory
                self._pipeline = AnimateDiffPipeline.from_pretrained(
                    target_model,
                    motion_adapter=adapter,
                    torch_dtype=torch.float16
                ).to(self.device)
            
            self.current_model = target_model
            
            # Use EulerAncestralDiscreteScheduler (excellent for low-step quality)
            self._pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
                self._pipeline.scheduler.config, 
                beta_schedule="scaled_linear", # Standard for SD1.5
                timestep_spacing="linspace",
                steps_offset=1
            )
            
            # Optimization
            self._pipeline.enable_vae_slicing()
            logger.info(f"[SpriteService] Model loaded successfully.")
        except Exception as e:
            logger.error(f"[SpriteService] Error loading model: {e}")
            raise e

    def unload_model(self):
        if self._pipeline is not None:
            logger.info("[SpriteService] Unloading model...")
            del self._pipeline
            self._pipeline = None
            if hasattr(self, 'current_model'):
                del self.current_model
            gc.collect()
            torch.cuda.empty_cache()

    def generate_sprite(self, prompt, negative_prompt="", width=256, height=256, frames=16, steps=8, guidance=7.5, seed=None, model_name=None, loras=None):
        self.load_model(model_name)

        if seed is None:
            seed = random.randint(0, 2**32 - 1)

        # Logic to improve Pixel Art quality
        is_pixel_art = "pixel" in prompt.lower() or "sprite" in prompt.lower() or "8-bit" in prompt.lower()
        
        if is_pixel_art:
            logger.info("[SpriteService] Pixel Art detected! Optimizing parameters...")
            # 1. Enforce resolution for coherence (512x512 is better for SD1.5 than 256)
            width = max(width, 512)
            height = max(height, 512)
            
            # 2. Reinforce Prompt
            if "pixel art" not in prompt.lower():
                prompt += ", pixel art, 16-bit, sharp, retro style, perfect alignment, clean lines"
            
            # 3. Reinforce Negative Prompt
            negative_prompt += ", blur, smooth, realistic, antialiasing, fuzz, noise, artifacts, messy, photography, 3d render"
        
        generator = torch.Generator(device=self.device).manual_seed(seed)
        
        # Handle LoRAs
        active_adapters = []
        if loras:
            logger.info(f"[SpriteService] Loading {len(loras)} LoRAs...")
            try:
                self._pipeline.unload_lora_weights() # Clear previous
                
                for lora in loras:
                    if not lora.get('enabled', True): continue
                    path = lora['path']
                    weight = float(lora.get('weight', 1.0))
                    name = os.path.splitext(os.path.basename(path))[0]
                    
                    self._pipeline.load_lora_weights(path, adapter_name=name)
                    active_adapters.append(name)
                    
                # Set weights after loading all adapters
                weights = [float(l.get('weight', 1.0)) for l in loras if l.get('enabled', True)]
                if active_adapters:
                    self._pipeline.set_adapters(active_adapters, adapter_weights=weights)
                    
            except Exception as e:
                 logger.error(f"[SpriteService] Failed to load LoRAs: {e}")

        logger.info(f"[SpriteService] Generating sprite: '{prompt}' ({width}x{height}, {frames} frames)")
        
        output = self._pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=frames,
            guidance_scale=guidance,
            num_inference_steps=steps,
            width=width,
            height=height,
            generator=generator
        )

        frames_output = output.frames[0]
        
        # Post-Processing for Pixel Art
        if is_pixel_art:
            logger.info("[SpriteService] Applying pixelation filter...")
            pixelated_frames = []
            for frame in frames_output:
                editor = ImageEditor(frame)
                # Pixel size 8 on 512x512 -> 64x64 grid
                editor.apply_pixelize(pixel_size=8)
                pixelated_frames.append(editor.img)
            frames_output = pixelated_frames
        
        # Save output
        from core.config import config
        now = datetime.now()
        day_folder = now.strftime("%Y_%m_%d")
        timestamp = now.strftime("%H%M%S")
        
        output_dir = os.path.join(config.OUTPUT_DIR, day_folder)
        os.makedirs(output_dir, exist_ok=True)
        
        filename = f"sprite_{timestamp}_{seed}.gif"
        save_path = os.path.join(output_dir, filename)
        
        export_to_gif(frames_output, save_path)
        logger.info(f"[SpriteService] Saved to {save_path}")
        
        return {
            "status": "success",
            "path": save_path,
            "url": f"/view/{day_folder}/{filename}",
            "seed": seed
        }
