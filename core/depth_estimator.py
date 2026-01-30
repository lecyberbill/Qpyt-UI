import gc
import torch
import cv2
import numpy as np
from PIL import Image
import io
import base64
from pathlib import Path
from core.config import config
import uuid
from datetime import datetime
from models.depth_anything_v2.dpt import DepthAnythingV2

_depth_model = None

def get_model():
    global _depth_model
    if _depth_model is None:
        print("[DeepEstimator] Loading Depth-Anything-V2-Large (ViT-L)...")
        _depth_model = DepthAnythingV2(encoder='vitl', features=256, out_channels=[256, 512, 1024, 1024])
        
        pth_path = config.get("DEPTH_MODEL_PATH")
        if not pth_path or not Path(pth_path).exists():
            # Check if directory is configured but not file
             raise FileNotFoundError(f"Depth model not found at {pth_path}")

        _depth_model.load_state_dict(torch.load(pth_path, map_location='cpu'))
        _depth_model.eval()
        
        if torch.cuda.is_available():
            _depth_model = _depth_model.cuda()
            
    return _depth_model

def unload_model():
    global _depth_model
    if _depth_model is not None:
        print("[DeepEstimator] Unloading model...")
        del _depth_model
        _depth_model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def infer_depth(image_input: str) -> str:
    """
    Generates a depth map from an image.
    Returns relative path to saved PNG.
    """
    try:
        # 1. Load Image
        if image_input.startswith("/view/"):
            rel_path = image_input.replace("/view/", "").lstrip("/")
            full_path = Path(config.OUTPUT_DIR) / rel_path
            pil_img = Image.open(full_path).convert("RGB")
        else:
            header, encoded = image_input.split(",", 1) if "," in image_input else (None, image_input)
            image_data = base64.b64decode(encoded)
            pil_img = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        # Convert to CV2/Numpy (RGB -> BGR for cv2 usually, but DPT implementation handles it? 
        # Looking at dpt.py: image2tensor does cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB)
        # So it expects BGR inputs if using cv2.imread.
        # But we act on PIL RGB. 
        # Let's convert PIL RGB to Numpy RGB, then to BGR for 'raw_img' expectation if needed, or just adapt.
        # infer_image calls image2tensor. 
        # image2tensor calls cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB). 
        # So infer_image expects BGR.
        
        np_img = np.array(pil_img)
        # PIL is RGB. CV2 expects BGR input for that specific method logic?
        # If we pass RGB to image2tensor, and it does BGR2RGB, it will scramble colors.
        # So we should convert RGB(PIL) -> BGR(Numpy) before passing to infer_image.
        raw_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)

    except Exception as e:
        print(f"[DeepEstimator] Error loading image: {e}")
        raise e

    try:
        # 2. Process
        print("[DeepEstimator] Running inference...")
        model = get_model()
        
        with torch.no_grad():
            depth = model.infer_image(raw_img) # Returns numpy array HxW

        # 3. Normalize & Colorize (Grayscale)
        depth_min = depth.min()
        depth_max = depth.max()
        
        # Avoid division by zero
        if depth_max - depth_min > 1e-6:
            depth_norm = (depth - depth_min) / (depth_max - depth_min) * 255.0
        else:
            depth_norm = np.zeros_like(depth)

        depth_norm = depth_norm.astype(np.uint8)
        
        depth_pil = Image.fromarray(depth_norm, mode='L')

        # 4. Save
        now = datetime.now()
        day_folder = now.strftime("%Y_%m_%d")
        file_name = f"depth_{uuid.uuid4().hex[:8]}.png"
        
        final_output_dir = Path(config.OUTPUT_DIR) / day_folder
        final_output_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = final_output_dir / file_name
        depth_pil.save(file_path, "PNG")
        
        print(f"[DeepEstimator] Success: {file_path}")
        return f"{day_folder}/{file_name}"

    except Exception as e:
        print(f"[DeepEstimator] Inference failed: {e}")
        raise e
    finally:
        # Always unload model to free VRAM for SDXL
        unload_model()
