from PIL import Image, ImageEnhance, ImageFilter, ImageChops
import numpy as np
from typing import Optional, Tuple

class ImageEditor:
    """
    Image processing engine adapted for Qpyt-UI.
    Uses PIL for high-performance CPU-based editing.
    """

    def __init__(self, image: Image.Image):
        self.img = image.convert("RGB") # Ensure RGB base

    def to_grayscale(self):
        self.img = self.img.convert("L").convert("RGB") # Keep RGB format even if gray
        return self

    def rotate(self, angle: int):
        if angle != 0:
            self.img = self.img.rotate(angle, expand=True)
        return self

    def mirror(self, mirror_type: str):
        if mirror_type == "horizontal":
            self.img = self.img.transpose(Image.FLIP_LEFT_RIGHT)
        elif mirror_type == "vertical":
            self.img = self.img.transpose(Image.FLIP_TOP_BOTTOM)
        return self

    def blur(self, radius: int):
        if radius > 0:
            self.img = self.img.filter(ImageFilter.GaussianBlur(radius=radius))
        return self

    def sharpen(self, factor: float):
        if factor > 0:
            enhancer = ImageEnhance.Sharpness(self.img)
            self.img = enhancer.enhance(factor)
        return self

    def adjust_contrast(self, factor: float):
        enhancer = ImageEnhance.Contrast(self.img)
        self.img = enhancer.enhance(factor)
        return self

    def adjust_saturation(self, factor: float):
        enhancer = ImageEnhance.Color(self.img)
        self.img = enhancer.enhance(factor)
        return self

    def adjust_color_boost(self, factor: float):
        enhancer = ImageEnhance.Brightness(self.img)
        self.img = enhancer.enhance(factor)
        return self

    # --- Special Filters ---
    def apply_sepia(self):
        sepia_matrix = (0.393, 0.769, 0.189, 0,
                        0.349, 0.686, 0.168, 0,
                        0.272, 0.534, 0.131, 0)
        self.img = self.img.convert("RGB", matrix=sepia_matrix)
        return self

    def apply_contour(self):
        self.img = self.img.filter(ImageFilter.CONTOUR)
        return self

    def apply_negative(self):
        self.img = ImageChops.invert(self.img)
        return self

    def apply_emboss(self):
        self.img = self.img.filter(ImageFilter.EMBOSS)
        return self
    
    def apply_pixelize(self, pixel_size: int = 8):
        if pixel_size <= 1: return self
        width, height = self.img.size
        # Resize small then back up
        small = self.img.resize((width // pixel_size, height // pixel_size), Image.NEAREST)
        self.img = small.resize((width, height), Image.NEAREST)
        return self

    def apply_vignette(self, radius: int = 100):
        # Optimized vignette using radial gradient
        width, height = self.img.size
        # Create gradient mask
        # Simple approximation for speed
        import math
        cx, cy = width / 2, height / 2
        # Use numpy for fast mask generation
        x = np.arange(width) - cx
        y = np.arange(height) - cy
        X, Y = np.meshgrid(x, y)
        dist = np.sqrt(X**2 + Y**2)
        
        # Soft vignette curve
        mask_arr = 255 - np.clip((dist - radius) * 0.5, 0, 255)
        mask = Image.fromarray(mask_arr.astype('uint8'), mode='L')
        
        black = Image.new("RGB", (width, height), "black")
        self.img = Image.composite(self.img, black, mask)
        return self

    # --- Advanced Colors ---
    def adjust_vibrance(self, factor: float):
        # Simplified vibrance: boost saturation on low-sat pixels logic is complex in pure PIL
        # Using simple blending for now as per reference code
        if factor == 0: return self
        # Reference logic uses simple blend with avg
        r, g, b = self.img.split()
        avg = ImageChops.add(r, ImageChops.add(g, b)).point(lambda x: x / 3)
        # Factor 0 in logic meant "no change"? The user code slider was 0-2, default 0
        # If default is 0, we assume 0 means "neutral" or "off"? 
        # Actually in user code `vibrance` slider default 0. Let's assume factor 0 = no effect.
        if factor > 0:
            # We map 0..2 to blend factor? 
            # Logic: r = ImageChops.blend(r, avg, factor) -> this actually desaturates towards grey if factor > 0?
            # Wait, blend(img1, img2, alpha): out = img1 * (1.0 - alpha) + img2 * alpha
            # If img2 is Greyscale (avg), then increasing alpha (factor) makes it MORE GREY (less vibrant).
            # The User code logic seems to INVERT Vibrance (making it B&W). 
            # I will fix this: Vibrance usually means INCREASE saturation.
            # I will use Color enhancement instead for simplicity or invert the logic.
            # Let's stick to simple Color Enhancement for now to be safe, labeled "Vibrance".
            enhancer = ImageEnhance.Color(self.img)
            self.img = enhancer.enhance(1.0 + factor) 
        return self

    def adjust_hue(self, angle: int):
        if angle == 0: return self
        # Convert to HSV, shift Hue channel
        img_hsv = self.img.convert("HSV")
        h, s, v = img_hsv.split()
        h = h.point(lambda x: (x + angle) % 256)
        self.img = Image.merge("HSV", (h, s, v)).convert("RGB")
        return self

    def apply_color_shift(self, r: int, g: int, b: int):
        if r == 0 and g == 0 and b == 0: return self
        R, G, B = self.img.split()
        R = R.point(lambda x: (x + r) % 256)
        G = G.point(lambda x: (x + g) % 256)
        B = B.point(lambda x: (x + b) % 256)
        self.img = Image.merge("RGB", (R, G, B))
        return self

    # --- Master Apply ---
    def apply_filters(self, settings: dict):
        """
        Applies a dictionary of settings sequentially.
        """
        # 1. Structure / Basic
        if settings.get('grayscale', False): self.to_grayscale()
        self.rotate(settings.get('rotation_angle', 0))
        self.mirror(settings.get('mirror_type', 'none'))
        
        # 2. Detail
        self.blur(settings.get('blur_radius', 0))
        self.sharpen(settings.get('sharpness_factor', 1.0)) # Default 1.0 = normal
        
        # 3. Tone / Color
        self.adjust_contrast(settings.get('contrast', 1.0))
        self.adjust_saturation(settings.get('saturation', 1.0))
        self.adjust_color_boost(settings.get('color_boost', 1.0))
        
        # 4. Creative
        special = settings.get('special_filter', 'none')
        if special != 'none' and special != 'aucun':
            method = getattr(self, f"apply_{special}", None)
            if method: method()
            elif special == "vignette": self.apply_vignette() 
            # Note: getattr handles explicit methods defined above
        
        # 5. Advanced
        self.adjust_vibrance(settings.get('vibrance', 0.0))
        self.adjust_hue(settings.get('hue_angle', 0))
        
        # Shift
        self.apply_color_shift(
            settings.get('color_shift_r', 0),
            settings.get('color_shift_g', 0),
            settings.get('color_shift_b', 0)
        )
        
        return self.img
