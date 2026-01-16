import logging
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from api.models import (
    ImageGenerationRequest, ImageGenerationResponse, UpscaleRequest, 
    RembgRequest, SaveToDiskRequest, InpaintRequest, OutpaintRequest, FilterRequest
)
from api.framework import QpytUI
from core.config import config
import core.generator as generator_lib
from core.generator import list_models, ModelManager, list_vaes, list_samplers
import core.analyzer as analyzer_lib
from core.translator import TranslationManager
from core.llm_prompter import LlmPrompterManager
from core.filters import ImageEditor
from starlette.concurrency import run_in_threadpool
from fastapi import UploadFile, File, Form
from PIL import Image
import io

# Log configuration
LOG_DIR = Path(config.base_dir) / "logs"
LOG_DIR.mkdir(exist_ok=True)
logging.basicConfig(
    filename=LOG_DIR / "app.log",
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("qpyt-ui")

# Filter out uvicorn access logs for /generate/preview
class EndpointFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "/generate/preview" not in record.getMessage()

logging.getLogger("uvicorn.access").addFilter(EndpointFilter())
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

app = FastAPI(title="Qpyt-UI API")

# Default Qpyt-UI configuration
app_ui = QpytUI(title="Qpyt-UI V0.9 TURBO")
app_ui.add_brick("qp-prompt")
app_ui.add_brick("qp-settings")
app_ui.add_brick("qp-render-sdxl")

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static folders mount
# /view -> config.OUTPUT_DIR
app.mount("/view", StaticFiles(directory=config.OUTPUT_DIR), name="outputs")
# /static -> web/
app.mount("/static", StaticFiles(directory="web"), name="web")

# Workflow Persistence Endpoints
WORKFLOWS_DIR = Path(config.base_dir) / "workflows"
WORKFLOWS_DIR.mkdir(exist_ok=True)

@app.get("/workflows")
async def list_workflows():
    files = list(WORKFLOWS_DIR.glob("*.json"))
    return sorted([f.stem for f in files])

@app.post("/workflows/save")
async def save_workflow(request: Request):
    try:
        data = await request.json()
        name = data.get("name", "untitled")
        # Sanitize name
        safe_name = "".join([c for c in name if c.isalnum() or c in (' ', '-', '_')]).strip()
        if not safe_name: safe_name = "workflow"
        
        file_path = WORKFLOWS_DIR / f"{safe_name}.json"
        
        # Save current workflow from frontend or fallback to app_ui
        workflow_data = data.get("workflow", app_ui.workflow)
        
        # Sync app_ui with this data if provided
        if "workflow" in data:
            app_ui.load_workflow(workflow_data)

        import json
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(workflow_data, f, indent=4)
        
        return {"status": "success", "message": f"Workflow '{safe_name}' saved."}
    except Exception as e:
        logger.error(f"Error saving workflow: {e}")
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@app.post("/workflows/load")
async def load_workflow(request: Request):
    try:
        data = await request.json()
        name = data.get("name")
        file_path = WORKFLOWS_DIR / f"{name}.json"
        
        if not file_path.exists():
            return JSONResponse(status_code=404, content={"status": "error", "message": "Workflow not found."})
            
        import json
        with open(file_path, "r", encoding="utf-8") as f:
            workflow_data = json.load(f)
        
        app_ui.load_workflow(workflow_data)
        return {"status": "success", "workflow": app_ui.workflow}
    except Exception as e:
        logger.error(f"Error loading workflow: {e}")
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@app.post("/workflows/delete")
async def delete_workflow(request: Request):
    try:
        data = await request.json()
        name = data.get("name")
        file_path = WORKFLOWS_DIR / f"{name}.json"
        
        if not file_path.exists():
            return JSONResponse(status_code=404, content={"status": "error", "message": "Workflow not found."})
            
        file_path.unlink()
        return {"status": "success", "message": f"Workflow '{name}' deleted."}
    except Exception as e:
        logger.error(f"Error deleting workflow: {e}")
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@app.get("/styles")
async def get_styles():
    try:
        styles_path = Path(config.base_dir) / "styles.json"
        if styles_path.exists():
            import json
            with open(styles_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return []
    except Exception as e:
        logger.error(f"Error loading styles: {e}")
        return []

@app.get("/config")
async def get_config():
    # Return current settings for the config drawer
    cfg = app_ui.get_config()
    cfg["settings"] = config.settings
    return cfg

@app.post("/config/save")
async def save_config(request: Request):
    try:
        data = await request.json()
        logger.info(f"Saving configuration: {data}")
        
        # Update config instance and save to config.local.json
        config_path = Path(config.base_dir) / "config.local.json"
        
        import json
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        
        # Hot reload
        config.load()
        
        return {"status": "success", "message": "Configuration saved and reloaded"}
    except Exception as e:
        logger.error(f"Error during save: {str(e)}")
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@app.delete("/brick/{brick_id}")
async def remove_brick(brick_id: str):
    logger.info(f"Removing brick: {brick_id}")
    app_ui.remove_brick(brick_id)
    return {"status": "success", "workflow": app_ui.workflow}

@app.post("/brick")
async def add_brick(request: Request):
    data = await request.json()
    brick_type = data.get("type", "qp-prompt")
    logger.info(f"Adding brick: {brick_type}")
    brick_id = app_ui.add_brick(brick_type)
    return {"status": "success", "brick_id": brick_id, "workflow": app_ui.workflow}

@app.get("/models/{model_type}")
async def get_models(model_type: str):
    models = list_models(model_type)
    return {"status": "success", "models": models}

@app.post("/generate/stop")
async def stop_generation():
    ModelManager.interrupt()
    return {"status": "success", "message": "Interruption request sent"}

@app.get("/vaes")
async def get_vaes():
    return {"status": "success", "vaes": list_vaes()}

@app.get("/samplers")
async def get_samplers():
    return {"status": "success", "samplers": list_samplers()}

@app.get("/generate/preview")
async def get_preview():
    return {
        "status": "success", 
        "preview": ModelManager._current_preview,
        "current_step": ModelManager._current_step,
        "total_steps": ModelManager._total_steps
    }

@app.post("/generate")
async def generate(request: ImageGenerationRequest):
    logger.info(f"Request received ({request.model_type}): {request.prompt}")
    start_time = time.time()
    
    try:
        # Async generation (via threadpool because generate is blocking)
        image_url, exec_time, used_params = await run_in_threadpool(
            ModelManager.generate,
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            model_type=request.model_type,
            model_name=request.model_name,
            width=request.width,
            height=request.height,
            guidance_scale=request.guidance_scale,
            num_inference_steps=request.num_inference_steps,
            seed=request.seed,
            vae_name=request.vae_name,
            sampler_name=request.sampler_name,
            image=request.image,
            denoising_strength=request.denoising_strength,
            output_format=request.output_format
        )
        
        execution_time = time.time() - start_time
        full_image_url = f"/view/{image_url}"
        
        # Return simple object for frontend mapping
        return {
            "status": "success",
            "data": {
                "request_id": str(uuid.uuid4()),
                "image_url": full_image_url,
                "execution_time": execution_time,
                "metadata": used_params,
                "status": "success"
            }
        }
    except Exception as e:
        logger.error(f"Error during generation: {str(e)}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

@app.post("/inpaint")
async def inpaint(req: InpaintRequest):
    logger.info(f"Inpaint request received ({req.model_type}): {req.prompt}")
    logger.info(f"Mask length: {len(req.mask) if req.mask else 0}, Image length: {len(req.image) if req.image else 0}")
    request_id = uuid.uuid4()
    start_time = time.time()
    try:
        url, exec_time, params = await run_in_threadpool(
            ModelManager.generate,
            prompt=req.prompt,
            negative_prompt=req.negative_prompt,
            model_type=req.model_type,
            model_name=req.model_name,
            width=req.width,
            height=req.height,
            guidance_scale=req.guidance_scale,
            num_inference_steps=req.num_inference_steps,
            seed=req.seed,
            vae_name=req.vae_name,
            sampler_name=req.sampler_name,
            image=req.image,
            mask=req.mask,
            denoising_strength=req.denoising_strength,
            output_format=req.output_format
        )
        if url is None:
            return JSONResponse({"status": "error", "message": "Generation interrupted or failed"}, status_code=400)
            
        execution_time = time.time() - start_time
        return {
            "status": "success",
            "data": {
                "request_id": str(request_id),
                "image_url": f"/view/{url}",
                "execution_time": execution_time,
                "metadata": params,
                "status": "success"
            }
        }
    except Exception as e:
        logger.error(f"Inpaint error: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

@app.post("/outpaint")
async def outpaint(req: OutpaintRequest):
    logger.info(f"Outpaint request received: {req.prompt}")
    return await inpaint(req)
async def upscale(request: UpscaleRequest):
    logger.info(f"Upscale request received: {request.scale_factor}x")
    start_time = time.time()
    try:
        image_url, exec_time, used_params = await run_in_threadpool(
            ModelManager.upscale,
            image_base64=request.image,
            scale_factor=request.scale_factor,
            denoising_strength=request.denoising_strength,
            tile_size=request.tile_size,
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            model_name=request.model_name,
            output_format=request.output_format,
            seed=request.seed
        )
        
        execution_time = time.time() - start_time
        full_image_url = f"/view/{image_url}"
        
        return {
            "status": "success",
            "data": {
                "request_id": str(uuid.uuid4()),
                "image_url": full_image_url,
                "execution_time": execution_time,
                "metadata": used_params,
                "status": "success"
            }
        }
    except Exception as e:
        logger.error(f"Error during upscale: {str(e)}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

@app.post("/rembg")
async def rembg(request: RembgRequest):
    logger.info("Background removal request received")
    start_time = time.time()
    try:
        from core.rembg_service import remove_background
        image_url = await run_in_threadpool(
            remove_background,
            image_input=request.image
        )
        
        execution_time = time.time() - start_time
        full_image_url = f"/view/{image_url}"
        
        return {
            "status": "success",
            "data": {
                "request_id": str(uuid.uuid4()),
                "image_url": full_image_url,
                "execution_time": execution_time,
                "status": "success"
            }
        }
    except Exception as e:
        logger.error(f"Error during rembg: {str(e)}")
        import traceback
        traceback.print_exc()
        return JSONResponse(
            status_code=500,
            content={"status": "error", "message": str(e)}
        )

@app.post("/save-to-disk")
async def save_to_disk_endpoint(request: SaveToDiskRequest):
    try:
        # Save operation (might involve PIL conversion, so run in threadpool)
        from core.save_service import save_to_disk
        dest_path = await run_in_threadpool(
            save_to_disk, 
            request.image_url, 
            request.custom_path, 
            request.pattern,
            request.output_format,
            request.params
        )
        return {"status": "success", "message": f"Saved to {dest_path}"}
    except Exception as e:
        logger.error(f"Save to Disk error: {e}")
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@app.post("/analyze")
async def analyze(file: UploadFile = File(...), task: str = Form("<DETAILED_CAPTION>")):
    logger.info(f"Analysis request received: {task} for file {file.filename}")
    try:
        # Load image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Call analyzer in threadpool
        prompt = await run_in_threadpool(
            analyzer_lib.analyze_image,
            image=image,
            task=task,
            device=config.DEVICE
        )
        
        if "[ERROR]" in prompt:
            return JSONResponse(status_code=500, content={"status": "error", "message": prompt})
            
        return {"status": "success", "prompt": prompt}
    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@app.post("/translate")
async def translate_text(request: Request):
    try:
        data = await request.json()
        text = data.get("text", "")
        if not text:
            return {"status": "success", "translated_text": ""}
        
        result = await run_in_threadpool(TranslationManager.translate, text)
        return {"status": "success", "translated_text": result}
    except Exception as e:
        logger.error(f"Translation error: {e}")
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@app.post("/enhance")
async def enhance_prompt(request: Request):
    try:
        data = await request.json()
        text = data.get("text", "")
        if not text:
            return {"status": "success", "prompt": ""}
        
        result = await run_in_threadpool(LlmPrompterManager.enhance_prompt, text)
        return {"status": "success", "prompt": result}
    except Exception as e:
        logger.error(f"Enhancement error: {e}")
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@app.get("/")
async def read_index():
    from fastapi.responses import FileResponse
    return FileResponse('web/index.html')

@app.get("/logs")
async def get_logs():
    if Path("logs/app.log").exists():
        with open("logs/app.log", "r") as f:
            return {"logs": f.readlines()}
    return {"logs": []}

@app.post("/filter")
async def apply_filters(req: FilterRequest):
    try:
        # Decode input
        if "," in req.image:
            header, encoded = req.image.split(",", 1)
        else:
            header, encoded = "data:image/png;base64", req.image
            
        import base64
        # Validate base64
        try:
            image_data = base64.b64decode(encoded)
        except Exception:
             return JSONResponse({"status": "error", "message": "Invalid Base64 string"}, status_code=400)

        pil_image = Image.open(io.BytesIO(image_data))
        
        def processing_task(img, settings):
            editor = ImageEditor(img)
            res_img = editor.apply_filters(settings)
            
            # Encode back to base64
            buffered = io.BytesIO()
            res_img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            return f"data:image/png;base64,{img_str}"

        # Run in thread to not block async loop
        result_b64 = await run_in_threadpool(processing_task, pil_image, req.settings)
        
        return {"status": "success", "image": result_b64}

    except Exception as e:
        logger.error(f"Filter error: {e}")
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

