import torch
import imageio
import numpy as np
from diffusers import CogVideoXPipeline
import os
import uuid
import logging
import cv2
from pathlib import Path
from core.config import config

logger = logging.getLogger("Qpyt-Video")

class CogVideoManager:
    _pipeline = None
    _current_model_name = None

    @classmethod
    def get_pipeline(cls, model_name="THUDM/CogVideoX-2b", device="cuda", low_vram=True):
        if cls._pipeline is not None and cls._current_model_name == model_name:
            return cls._pipeline

        logger.info(f"Loading CogVideo model: {model_name}")
        
        # Determine paths
        model_path = model_name
        if not os.path.exists(model_path):
             # Try to find in MODELS_DIR/cogvideo if it's just a folder name
             potential_local = Path(config.get('MODELS_DIR', '.')) / "cogvideo" / model_name
             if potential_local.exists():
                 model_path = str(potential_local)

        dtype = torch.bfloat16 # CogVideo prefers bfloat16
        
        try:
            pipe = CogVideoXPipeline.from_pretrained(
                model_path,
                torch_dtype=dtype
            )
            
            if low_vram:
                # Optimized for lower VRAM
                pipe.enable_model_cpu_offload()
                # pipe.enable_sequential_cpu_offload() # Extreme case
                pipe.vae.enable_tiling()
            else:
                pipe.to(device)

            cls._pipeline = pipe
            cls._current_model_name = model_name
            return cls._pipeline
        except Exception as e:
            logger.error(f"Failed to load CogVideo: {e}")
            raise e

    @staticmethod
    def _extract_thumbnail(video_path, thumbnail_path):
        """Extracts the first frame of the video as a thumbnail."""
        try:
            cap = cv2.VideoCapture(str(video_path))
            success, frame = cap.read()
            if success:
                cv2.imwrite(str(thumbnail_path), frame)
                logger.info(f"Thumbnail saved to {thumbnail_path}")
            cap.release()
            return success
        except Exception as e:
            logger.error(f"Failed to extract thumbnail: {e}")
        return False

    @classmethod
    def generate(cls, prompt, model_name="THUDM/CogVideoX-2b", num_frames=49, fps=8, num_inference_steps=50, guidance_scale=6.0, seed=None, low_vram=True):
        pipe = cls.get_pipeline(model_name, low_vram=low_vram)
        
        generator = torch.Generator(device="cuda")
        if seed is not None:
            generator.manual_seed(seed)
        else:
            seed = generator.seed()

        logger.info(f"Generating video: '{prompt}' | Steps: {num_inference_steps} | Seed: {seed}")
        
        video_frames = pipe(
            prompt=prompt,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).frames[0]

        # Save to output (align with history log date-based folders)
        from datetime import datetime
        day_str = datetime.now().strftime("%Y_%m_%d")
        output_dir = Path(config.get('OUTPUT_DIR', 'outputs')) / day_str
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"video_{uuid.uuid4().hex[:8]}_{seed}.mp4"
        output_path = output_dir / filename
        
        import numpy as np
        # Convert to uint8 if necessary and ensure healthy array format for imageio
        video_np = [np.array(frame).astype(np.uint8) for frame in video_frames]
        
        imageio.mimsave(str(output_path), video_np, fps=fps, codec='libx264', pixelformat='yuv420p')
        
        if output_path.exists():
            thumb_filename = f"{Path(filename).stem}_thumb.jpg"
            thumb_path = output_dir / thumb_filename
            cls._extract_thumbnail(output_path, thumb_path)
            
        return {
            "status": "success",
            "video_url": filename,
            "thumbnail_url": thumb_filename if (output_dir / thumb_filename).exists() else None,
            "metadata": {
                "seed": seed,
                "prompt": prompt,
                "model": model_name,
                "num_frames": num_frames,
                "fps": fps
            }
        }
