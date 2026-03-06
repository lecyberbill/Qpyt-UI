import base64
import io
from pathlib import Path
from PIL import Image
from core.config import config

def load_image_from_input(image_input: str, mode: str = 'RGB') -> Image.Image:
    """
    Robustly loads a PIL image from:
    - /view/ or /outputs/ internal paths
    - Local file paths
    - Base64 strings (with/without headers, handling padding)
    """
    if not image_input:
        raise ValueError("Empty image input")

    image_input = image_input.strip()

    # 1. Handle internal paths or local files
    is_path = False
    clean_path = image_input
    
    if image_input.startswith("/view/"):
        clean_path = image_input.replace("/view/", "", 1).lstrip("/")
        is_path = True
    elif image_input.startswith("/outputs/"):
        clean_path = image_input.replace("/outputs/", "", 1).lstrip("/")
        is_path = True
    elif not image_input.startswith("data:") and ("/" in image_input or "\\" in image_input) and "." in image_input:
        # Probable local path or partial path
        is_path = True

    if is_path:
        full_path = Path(config.OUTPUT_DIR) / clean_path
        if full_path.exists():
            return Image.open(full_path).convert(mode)
        # If it doesn't exist but has extensions, maybe it's an absolute path
        if Path(image_input).exists():
             return Image.open(image_input).convert(mode)
        
    # 2. Handle Base64
    # If it's a URL (http), and we couldn't load it as a local path, we have an issue 
    # but for now let's assume it should have been a path.
    if image_input.startswith("http"):
         raise ValueError(f"Direct URL input not supported for local processing: {image_input}")

    # Strip data-uri header if present
    if "," in image_input:
        _, encoded = image_input.split(",", 1)
    else:
        encoded = image_input

    # Remove any whitespaces/newlines from base64
    encoded = encoded.replace(" ", "").replace("\n", "").replace("\r", "")

    # Robust padding fix
    missing_padding = len(encoded) % 4
    if missing_padding:
        if missing_padding == 1:
            # Mathematical impossibility for valid B64, usually truncation
            # Strip the orphan char to try and recover
            encoded = encoded[:-1]
        else:
            encoded += "=" * (4 - missing_padding)

    try:
        image_data = base64.b64decode(encoded)
        return Image.open(io.BytesIO(image_data)).convert(mode)
    except Exception as e:
        raise ValueError(f"Failed to decode base64 image: {str(e)}")
