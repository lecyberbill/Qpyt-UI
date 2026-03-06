import gc
import torch
from rembg import remove, new_session
from PIL import Image
import io
import base64
from pathlib import Path
from core.config import config
import uuid
from datetime import datetime
from core.utils import load_image_from_input

# Global session to allow reuse but also explicit unloading
_rembg_session = None

def get_session():
    global _rembg_session
    if _rembg_session is None:
        print("[REMBG] Loading model (u2net)...")
        # u2net is the default, but we could make it configurable
        _rembg_session = new_session("u2net")
    return _rembg_session

def unload_model():
    global _rembg_session
    if _rembg_session is not None:
        print("[REMBG] Unloading model...")
        del _rembg_session
        _rembg_session = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def remove_background(image_input: str) -> str:
    """
    Removes background from an image.
    Supports base64 or local paths (/view/).
    Returns the relative path to the processed PNG.
    """
    # 1. Load Image
    try:
        img = load_image_from_input(image_input, mode="RGB")
    except Exception as e:
        print(f"[REMBG] Error loading image: {e}")
        raise e

    # 2. Process
    print("[REMBG] Processing background removal...")
    session = get_session()
    output_image = remove(img, session=session)

    # 3. Save
    now = datetime.now()
    day_folder = now.strftime("%Y_%m_%d")
    file_name = f"rembg_{uuid.uuid4().hex[:8]}.png"
    
    final_output_dir = Path(config.OUTPUT_DIR) / day_folder
    final_output_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = final_output_dir / file_name
    output_image.save(file_path, "PNG")
    
    print(f"[REMBG] Success: {file_path}")
    return f"{day_folder}/{file_name}"
