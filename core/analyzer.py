import os
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
import traceback

# Configuration
# Default model ID from user provided template
MODEL_ID_FLORENCE2 = os.environ.get("FLORENCE2_MODEL_ID", "MiaoshouAI/Florence-2-base-PromptGen-v2.0")

# Global variables for the model (caching)
_caption_model = None
_caption_processor = None
_caption_device = None
_current_model_id = None

# Florence-2 tasks
FLORENCE2_TASKS = [
    "<DETAILED_CAPTION>",
    "<MORE_DETAILED_CAPTION>",
    "<CAPTION>",
    "<GENERATE_TAGS>",
    "<ANALYZE>",
    "<MIXED_CAPTION>",
    "<MIXED_CAPTION_PLUS>"
]
DEFAULT_FLORENCE2_TASK = "<DETAILED_CAPTION>"

def _load_model_if_needed(model_id, device):
    global _caption_model, _caption_processor, _current_model_id, _caption_device
    
    if _caption_model is not None and _current_model_id == model_id and _caption_device == device:
        return True
    
    if _caption_model is not None:
        print(f"[INFO] Unloading previous analyzer model: {_current_model_id}")
        unload_model()
        
    print(f"[INFO] Loading Florence-2 model: {model_id} on {device}...")
    try:
        # Use float16 on GPU, float32 on CPU
        dtype = torch.float16 if "cuda" in device else torch.float32
        _caption_processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        _caption_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            trust_remote_code=True,
            attn_implementation="eager"
        ).eval().to(device)
        
        # Patch for transformers >= 4.45 compatibility
        # Recent transformers versions removed GenerationMixin from PreTrainedModel, 
        # breaking some remote code models like Florence-2.
        if hasattr(_caption_model, "language_model") and not hasattr(_caption_model.language_model, "generate"):
            try:
                from transformers.generation import GenerationMixin
                from transformers import GenerationConfig
                print("[INFO] Patching Florence-2 language_model with GenerationMixin...")
                cls = _caption_model.language_model.__class__
                if GenerationMixin not in cls.__bases__:
                    cls.__bases__ = (GenerationMixin,) + cls.__bases__
                
                # Ensure generation_config is not None to avoid 'NoneType' object has no attribute '_from_model_config'
                if _caption_model.language_model.generation_config is None:
                    print("[INFO] Initializing missing generation_config for language_model...")
                    _caption_model.language_model.generation_config = GenerationConfig.from_model_config(_caption_model.language_model.config)
            except Exception as e:
                print(f"[WARN] Failed to patch language_model: {e}")

        _current_model_id = model_id
        _caption_device = device
        print(f"[OK] Florence-2 model loaded successfully.")
        return True
    except Exception as e:
        print(f"[ERROR] Error loading Florence-2 model: {str(e)}")
        traceback.print_exc()
        return False

def unload_model():
    """Explicitly unload the model to free VRAM."""
    global _caption_model, _caption_processor, _current_model_id, _caption_device
    if _caption_model is not None:
        print(f"[INFO] Unloading Florence-2 model ({_current_model_id})...")
        _caption_model = None
        _caption_processor = None
        _current_model_id = None
        _caption_device = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        print(f"[OK] Analyzer model unloaded.")

def analyze_image(image: Image.Image, task: str = DEFAULT_FLORENCE2_TASK, device: str = "cuda"):
    """
    Generates a prompt from an image using Florence-2.
    """
    if not _load_model_if_needed(MODEL_ID_FLORENCE2, device):
        return f"[ERROR] Could not load analyzer model."
    
    if task not in FLORENCE2_TASKS:
        print(f"[WARN] Invalid task {task}, using default {DEFAULT_FLORENCE2_TASK}")
        task = DEFAULT_FLORENCE2_TASK
        
    print(f"[INFO] Analyzing image with task: {task}")
    
    try:
        # Ensure image is RGB
        if image.mode != "RGB":
            image = image.convert("RGB")
            
        inputs = _caption_processor(text=task, images=image, return_tensors="pt").to(_caption_model.device)
        
        # Match model dtype (especially for FP16)
        inputs = inputs.to(_caption_model.dtype)
        
        with torch.no_grad():
            generated_ids = _caption_model.generate(
                input_ids=inputs["input_ids"], 
                pixel_values=inputs["pixel_values"], 
                max_new_tokens=1024, 
                do_sample=False, 
                num_beams=3,
                use_cache=False
            )
            
        generated_text = _caption_processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
        parsed_answer = _caption_processor.post_process_generation(generated_text, task=task, image_size=(image.width, image.height))
        
        # Extract prompt from results
        prompt = parsed_answer.get(task, "")
        if isinstance(prompt, str):
            # Cleanup
            prompt = prompt.strip('{}').strip('"').strip()
            
        print(f"[INFO] Analysis complete.")
        return prompt
    except Exception as e:
        print(f"[ERROR] Analysis failed: {str(e)}")
        traceback.print_exc()
        return f"[ERROR] {str(e)}"
