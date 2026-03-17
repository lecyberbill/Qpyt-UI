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
    return list_models()

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
    Returns the path to the generated image.
    """
    import uuid
    from core.generator import ModelManager
    
    # Load model if not already loaded or different
    if model_name:
        ModelManager.get_instance().load_model(model_name)
    
    # Generate unique filename
    output_filename = f"mcp_{uuid.uuid4().hex[:8]}.png"
    output_path = Path(config.OUTPUT_DIR) / output_filename
    
    # Run generation
    try:
        # Note: In a real scenario, we might want to use the queue manager, 
        # but for direct MCP control, we call the generator directly.
        image = generator_lib.generate(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance,
            seed=seed
        )
        image.save(output_path)
        return str(output_path.absolute())
    except Exception as e:
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
    # Start the MCP server using stdio transport
    mcp.run()
