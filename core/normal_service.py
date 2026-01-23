import cv2
import numpy as np
from PIL import Image
import io
import base64
from pathlib import Path
from core.config import config
import uuid
from datetime import datetime

def compute_normal_map(image_input: str, strength: float = 2.0) -> str:
    """
    Generates a Normal Map from a Depth Map (Grayscale image).
    Uses Sobel filters to compute gradients.
    
    Args:
        image_input: Standard image input (base64 or /view/ path)
        strength: Multiplier for the gradient (higher = more pronounced relief)
    """
    try:
        # 1. Load Image
        if image_input.startswith("/view/"):
            rel_path = image_input.replace("/view/", "").lstrip("/")
            full_path = Path(config.OUTPUT_DIR) / rel_path
            pil_img = Image.open(full_path).convert("L") # Convert to Grayscale
        else:
            header, encoded = image_input.split(",", 1) if "," in image_input else (None, image_input)
            image_data = base64.b64decode(encoded)
            pil_img = Image.open(io.BytesIO(image_data)).convert("L")
        
        # Convert to float for math
        img_np = np.array(pil_img).astype(np.float32) / 255.0
        
        # 2. Compute Gradients (Sobel)
        # Gradient X (horizontal changes)
        dzdx = cv2.Sobel(img_np, cv2.CV_32F, 1, 0, ksize=3)
        # Gradient Y (vertical changes)
        dzdy = cv2.Sobel(img_np, cv2.CV_32F, 0, 1, ksize=3)
        
        # 3. Construct Normal Vector ( n = (-dz/dx, -dz/dy, 1) )
        # We negate the gradients because typically 'white' is 'close/high' in depth maps.
        # If white is high, the slope 'up' to the right means dx is positive, 
        # so the normal should point 'back' to the left.
        
        # Apply strength factor
        dzdx *= strength
        dzdy *= strength
        
        # Initialize normal map array (H, W, 3)
        h, w = img_np.shape
        normal_map = np.zeros((h, w, 3), dtype=np.float32)
        
        # -- X component: Map [-1, 1] to [0, 1]
        # Normal X is proportional to -dzdx
        # But for normal maps: R = 0.5 * (nx + 1)
        # Let's normalize vector first? 
        # Actually simpler approximation:
        # Normal vector N = (-dzdx, -dzdy, 1.0) normalized.
        
        ones = np.ones_like(img_np)
        
        # Stack components
        # Note: OpenGL standard often uses Y+ up (green up). DirectX uses Y- down.
        # Let's target standard "Blue = Flat" (0, 0, 1) maps.
        # R = X, G = Y, B = Z
        
        # We need to normalize per pixel: length = sqrt(x^2 + y^2 + z^2)
        # Vector = (-dzdx, -dzdy, 1.0)
        # White is near/high.
        norm = np.sqrt(dzdx**2 + dzdy**2 + 1.0)
        
        n_x = -dzdx / norm
        n_y = -dzdy / norm
        n_z = 1.0 / norm
        
        # Map to 0-255 range
        # [-1, 1] -> [0, 255] : (val + 1) * 0.5 * 255
        
        normal_map[:, :, 0] = (n_x + 1.0) * 0.5 * 255.0 # R
        normal_map[:, :, 1] = (n_y + 1.0) * 0.5 * 255.0 # G
        normal_map[:, :, 2] = n_z * 255.0               # B (Z is always 0..1 positive)
        
        # Clip just in case
        normal_map = np.clip(normal_map, 0, 255).astype(np.uint8)
        
        # 4. Save
        res_img = Image.fromarray(normal_map, mode='RGB')
        
        now = datetime.now()
        day_folder = now.strftime("%Y-%m-%d")
        file_name = f"normal_{uuid.uuid4().hex[:8]}.png"
        
        final_output_dir = Path(config.OUTPUT_DIR) / day_folder
        final_output_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = final_output_dir / file_name
        res_img.save(file_path, "PNG")
        
        print(f"[NormalService] Success: {file_path}")
        return f"{day_folder}/{file_name}"
        
    except Exception as e:
        print(f"[NormalService] Error: {e}")
        raise e
