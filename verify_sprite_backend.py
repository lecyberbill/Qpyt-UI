
import sys
import os

# Add current directory to path
sys.path.append(os.getcwd())

try:
    print("Testing Sprite Service imports...")
    from core.sprite_service import SpriteService
    print("SpriteService imported successfully.")
    
    import diffusers
    print(f"Diffusers version: {diffusers.__version__}")
    
    from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler
    print("AnimateDiff classes imported successfully.")
    
    print("Verification PASSED.")
except Exception as e:
    print(f"Verification FAILED: {e}")
    sys.exit(1)
