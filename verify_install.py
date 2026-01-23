
import sys
import os
import json
from pathlib import Path

def print_status(msg, status="INFO"):
    colors = {
        "INFO": "\033[94m[INFO]\033[0m",
        "OK": "\033[92m[OK]\033[0m",
        "WARN": "\033[93m[WARN]\033[0m",
        "FAIL": "\033[91m[FAIL]\033[0m"
    }
    # Fallback for colorless terminals (Window CMD default sometimes)
    if os.name == 'nt':
        os.system('color')
    
    prefix = colors.get(status, "[?]")
    print(f"{prefix} {msg}")

def verify_install():
    print("="*60)
    print(" Qpyt-UI Installation Validator")
    print("="*60)

    # 0. Check Environment (Venv)
    is_venv = (sys.prefix != sys.base_prefix)
    if not is_venv:
        print_status("Running outside of a virtual environment (venv)!", "WARN")
        print_status("It is highly recommended to run this script within an activated venv.", "WARN")

    # 1. Python Version
    py_ver = sys.version_info
    if py_ver.major == 3 and py_ver.minor >= 10:
        print_status(f"Python Version: {sys.version.split()[0]}", "OK")
    else:
        print_status(f"Python Version: {sys.version.split()[0]} (Recommended: 3.10+)", "WARN")

    # 2. Dependencies (Torch)
    try:
        import torch
        print_status(f"Torch Installed: {torch.__version__}", "OK")
        
        if torch.cuda.is_available():
            vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print_status(f"CUDA Available: {torch.cuda.get_device_name(0)} ({vram:.1f} GB VRAM)", "OK")
        else:
            print_status("CUDA NOT DETECTED! Generation will be extremely slow (CPU mode).", "FAIL")
    except ImportError:
        print_status("Torch not installed. Run 'pip install -r requirements.txt'", "FAIL")

    # 3. Config File
    config_path = Path("config.json")
    local_config_path = Path("config.local.json")
    active_config = None

    if local_config_path.exists():
        print_status("Found 'config.local.json' (Using as override)", "OK")
        active_config = local_config_path
    elif config_path.exists():
        print_status("Found 'config.json'", "OK")
        active_config = config_path
    else:
        print_status("No config file found! Copy 'config.example.json' to 'config.json'", "FAIL")
        return

    # 4. Check Config Paths
    try:
        with open(active_config, 'r', encoding='utf-8') as f:
            cfg = json.load(f)
        
        path_keys = ["MODELS_DIR", "VAE_DIR", "LORAS_DIR", "CONTROLNET_DIR", "OUTPUT_DIR"]
        for key in path_keys:
            path_str = cfg.get(key)
            if path_str:
                p = Path(path_str)
                if p.exists():
                    print_status(f"Directory verified: {key} -> {path_str}", "OK")
                else:
                    try:
                        p.mkdir(parents=True, exist_ok=True)
                        print_status(f"Created missing directory: {key} -> {path_str}", "WARN")
                    except:
                        print_status(f"Directory MISSING and uncreateable: {key} -> {path_str}", "FAIL")
    except Exception as e:
        print_status(f"Failed to read/parse config: {e}", "FAIL")

    # 5. Check Requirements
    print_status("Verifying other dependencies...", "INFO")
    try:
        import fastapi
        import uvicorn
        import diffusers
        import transformers
        print_status("Core libraries (FastAPI, Diffusers, Transformers) found.", "OK")
    except ImportError as e:
        print_status(f"Missing dependency: {e.name}", "FAIL")

    print("="*60)
    print("Verification Complete.")
    print("If all checks are [OK] or [WARN], you can run 'start.bat'.")
    print("="*60)

if __name__ == "__main__":
    verify_install()
