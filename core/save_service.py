import os
import shutil
from pathlib import Path
from PIL import Image
from core.config import config
import uuid
from datetime import datetime
import base64
import io

def save_to_disk(image_url: str, custom_path: str, pattern: str, output_format: str = "png", params: dict = None) -> str:
    """
    Saves or converts an existing image from the output directory to a custom disk location
    with a specific filename pattern and format. 
    Supports local paths and Base64 Data URIs.
    """
    if params is None:
        params = {}

    img = None
    
    # 1. Resolve source image (Base64 or File Path)
    if image_url.startswith("data:image"):
        try:
            # Decode Base64
            _, encoded = image_url.split(",", 1)
            data = base64.b64decode(encoded)
            img = Image.open(io.BytesIO(data))
        except Exception as e:
            raise ValueError(f"Invalid Base64 image data: {e}")
    else:
        # Resolve from output dir
        rel_path = image_url.replace("/view/", "").lstrip("/")
        source_path = Path(config.OUTPUT_DIR) / rel_path
        
        if not source_path.exists():
            raise FileNotFoundError(f"Source image not found: {source_path}")
        
        img = Image.open(source_path)

    # 2. Prepare target directory
    target_dir = Path(custom_path)
    target_dir.mkdir(parents=True, exist_ok=True)

    # 3. Handle pattern tokens
    # Basic tokens for now: {date}, {time}, {uuid}, {ext}
    now = datetime.now()
    tokens = {
        "{date}": now.strftime("%Y-%m-%d"),
        "{time}": now.strftime("%H%M%S"),
        "{uuid}": uuid.uuid4().hex[:8],
        "{ext}": output_format.lower().replace("jpeg", "jpg")
    }
    
    # Inject params as tokens (e.g. {seed} -> params["seed"])
    for key, value in params.items():
        tokens[f"{{{key}}}"] = str(value)
    
    filename = pattern
    for token, value in tokens.items():
        filename = filename.replace(token, value)
    
    # Ensure extension matches output_format
    ext = f".{tokens['{ext}']}"
    if not filename.lower().endswith(tuple(['.png', '.jpg', '.jpeg', '.webp'])):
        filename += ext
    else:
        # If user put an extension in pattern, force it to match output_format
        filename = os.path.splitext(filename)[0] + ext

    target_path = target_dir / filename
    
    # 4. Save/Convert
    # img is already loaded from Step 1
    
    # Handle RGB conversion for JPEG if source has Alpha
    if output_format.lower() in ["jpg", "jpeg"] and img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
        
    save_kwargs = {}
    if output_format.lower() in ["jpg", "jpeg", "webp"]:
        save_kwargs["quality"] = 90
        save_kwargs["optimize"] = True

    img.save(target_path, **save_kwargs)
    
    print(f"[SaveToDisk] Image saved to {target_path} (format: {output_format})")
    return str(target_path)
