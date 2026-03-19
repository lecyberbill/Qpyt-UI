import os
import sys
import logging
from typing import List, Optional
from fastmcp import FastMCP
from pathlib import Path

# Add project root to path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.config import config
from core.generator import list_models
import core.generator as generator_lib
import core.analyzer as analyzer_lib

# [WFGY-Metadata] Header
# Logic_Zone: TRANSIT | Delta_Initial: 0.8 | Resolution: convergent

# Initialize MCP Server
mcp = FastMCP("Qpyt-UI")

@mcp.tool()
def list_available_models() -> List[str]:
    """
    Returns a list of all locally available Stable Diffusion / Flux models (.safetensors).
    """
    all_models = []
    # Collect models from all supported architectures
    for m_type in ["sdxl", "flux", "sd3", "zimage"]:
        try:
            models = list_models(m_type)
            if models:
                all_models.extend(models)
        except Exception as e:
            logging.error(f"Error listing {m_type} models: {e}")
            
    return sorted(list(set(all_models)))

@mcp.tool()
async def generate_image(
    prompt: str, 
    model_name: Optional[str] = None, 
    width: int = 1024, 
    height: int = 1024, 
    steps: int = 30, 
    guidance: float = 7.0,
    seed: int = -1
) -> str:
    """
    Generates an image using the specified prompt and parameters.
    Returns the absolute path to the generated image.
    """
    from core.generator import ModelManager
    
    # Validate model_name against available models to avoid crashes
    available_models = list_available_models()
    if model_name and model_name not in available_models:
        logging.warning(f"Requested model '{model_name}' not found. Falling back to default.")
        model_name = None
    
    # Run generation
    try:
        # We call the ModelManager.generate directly.
        # It handles model loading, device placement, and saving to disk.
        result = ModelManager.generate(
            prompt=prompt,
            model_name=model_name,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance,
            seed=None if seed == -1 else seed
        )
        
        if result and isinstance(result, tuple):
            rel_path, exec_time, params = result
            # We return a local web URL (hosted by the Qpyt-UI FastAPI server on port 8000)
            # This is much more reliable for an AI to present in a chat.
            web_url = f"http://localhost:8000/outputs/{rel_path}"
            return f"Image successfully generated. Local path: {rel_path}. Access it via: {web_url}"
        else:
            return "Error: Generation failed to return a valid result."
            
    except Exception as e:
        logging.error(f"Error during MCP image generation: {e}")
        return f"Error during generation: {str(e)}"

@mcp.tool()
async def analyze_image_tags(image_path: str) -> str:
    """
    Analyzes an existing image and returns suggested prompts/tags.
    Useful for 'Image-to-Prompt' workflows.
    """
    if not os.path.exists(image_path):
        return "Error: Image file not found."
    
    try:
        from PIL import Image
        img = Image.open(image_path)
        tags = analyzer_lib.analyze(img)
        return tags
    except Exception as e:
        return f"Error during analysis: {str(e)}"

@mcp.tool()
def get_current_config() -> str:
    """
    Returns the current application configuration as a JSON string.
    """
    import json
    return json.dumps(config.settings, indent=4)

if __name__ == "__main__":
    # Pre-warm the server by listing models (ensures config is loaded)
    try:
        logging.info("Pre-warming MCP server...")
        list_available_models()
    except:
        pass
        
    # Start the MCP server using stdio transport
    mcp.run()
