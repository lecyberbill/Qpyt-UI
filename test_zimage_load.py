import sys
import os
import torch
from pathlib import Path

# Add current dir to path so we can import core
sys.path.append(os.getcwd())

# Mock Config if needed or ensure it reads from file
from core.config import config
from core.generator import ModelManager

def test_load():
    print("Testing Z-Image-Turbo Loading...")
    
    # Path from config.local.json (simulated)
    # The user said: "G:\models\zit" is where files are.
    # We added ZIMAGE_MODELS_DIR to config, but for this test we'll pass the path directly 
    # to mimic what the frontend/preset sends.
    model_path = r"G:\models\zit"
    
    try:
        pipe, compel = ModelManager.get_pipeline(
            model_type="zimage", 
            model_path=model_path,
            low_vram=False
        )
        print("SUCCESS: Pipeline loaded!")
        
        # Basic check
        print(f"Pipeline class: {type(pipe)}")
        print(f"Transformer type: {type(pipe.transformer)}")
        print(f"VAE type: {type(pipe.vae)}")
        
    except Exception as e:
        print(f"FAILURE: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_load()
