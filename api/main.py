import logging
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import asyncio
from watchfiles import awatch

from api.models import (
    ImageGenerationRequest, ImageGenerationResponse, UpscaleRequest, 
    RembgRequest, SaveToDiskRequest, InpaintRequest, OutpaintRequest, FilterRequest, DepthRequest, NormalRequest,
    AudioGenerationRequest, SpriteRequest, VideoGenerationRequest
)
from api.framework import QpytUI
from core.config import config
import core.generator as generator_lib
from core.generator import list_models, ModelManager, list_vaes, list_samplers
import core.analyzer as analyzer_lib
from core.translator import TranslationManager
from core.llm_prompter import LlmPrompterManager
from core.filters import ImageEditor
from api.monitor import router as monitor_router
from starlette.concurrency import run_in_threadpool
from fastapi import UploadFile, File, Form
from PIL import Image
import io
from api.history_log import HistoryLogManager
from api.queue_manager import QueueManager, TaskStatus

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

import webbrowser

_browser_opened = False

@app.on_event("startup")
async def startup_event():
    queue_mgr = QueueManager.get_instance()
    await queue_mgr.start_worker()
    # Start file watcher for hot reload
    asyncio.create_task(hot_reload_watcher())
    
    # Auto-open browser
    global _browser_opened
    if not _browser_opened:
        port = 8000 # Default port
        # In a real scenario we'd parse sys.argv or config, but 8000 is the project standard
        url = f"http://127.0.0.1:{port}/"
        logger.info(f"Opening browser at {url}")
        
        # Give uvicorn a moment to start the socket
        async def open_delayed():
            await asyncio.sleep(1.5)
            webbrowser.open(url)
            
        asyncio.create_task(open_delayed())
        _browser_opened = True

# Hot Reload logic
connected_clients = set()

@app.websocket("/ws/hot-reload")
async def hot_reload_endpoint(websocket: WebSocket):
    await websocket.accept()
    connected_clients.add(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        connected_clients.remove(websocket)

async def hot_reload_watcher():
    logger.info("Hot reload watcher started for 'web/' directory.")
    async for changes in awatch("web/"):
        logger.info(f"File changes detected: {changes}. Signaling reload...")
        if connected_clients:
            # We use a copy of the set to avoid issues if clients disconnect during broadcast
            for client in list(connected_clients):
                try:
                    await client.send_text("reload")
                except Exception as e:
                    logger.error(f"Failed to send reload signal: {e}")
                    connected_clients.remove(client)

# Default Qpyt-UI configuration
app_ui = QpytUI(title="Qpyt - UI V.09")
app_ui.add_brick("qp-prompt")
app_ui.add_brick("qp-settings")
app_ui.add_brick("qp-render-sdxl")

# ... (middle code unchanged)

@app.post("/generate/video", response_model=ImageGenerationResponse)
async def generate_video(request: VideoGenerationRequest):
    logger.info(f"Adding video generation to queue: {request.prompt}")
    queue_mgr = QueueManager.get_instance()
    
    task_id = await queue_mgr.add_task(
        "video",
        task_video_wrapper,
        prompt=request.prompt,
        model_name=request.model_name,
        num_frames=request.num_frames,
        fps=request.fps,
        num_inference_steps=request.num_inference_steps,
        guidance_scale=request.guidance_scale,
        seed=request.seed,
        low_vram=request.low_vram
    )
    
    return JSONResponse(status_code=202, content={"status": "queued", "task_id": task_id})

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# /outputs -> config.OUTPUT_DIR
app.mount("/outputs", StaticFiles(directory=config.OUTPUT_DIR), name="outputs")
# /static -> web/
app.mount("/static", StaticFiles(directory="web"), name="web")

app.include_router(monitor_router)

# Workflow & Presets Persistence Endpoints
WORKFLOWS_DIR = Path(config.base_dir) / "workflows"
PRESETS_DIR = Path(config.base_dir) / "presets"
WORKFLOWS_DIR.mkdir(exist_ok=True)
PRESETS_DIR.mkdir(exist_ok=True)

@app.get("/workflows")
async def list_workflows():
    user_files = list(WORKFLOWS_DIR.glob("*.json"))
    system_files = list(PRESETS_DIR.glob("*.json"))
    return {
        "user": sorted([f.stem for f in user_files]),
        "system": sorted([f.stem for f in system_files])
    }

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
        
        # Check in user workflows first, then in presets
        if not file_path.exists():
            file_path = PRESETS_DIR / f"{name}.json"
            
        if not file_path.exists():
            return JSONResponse(status_code=404, content={"status": "error", "message": "Workflow or Preset not found."})
            
        import json
        with open(file_path, "r", encoding="utf-8") as f:
            workflow_data = json.load(f)
        
        app_ui.load_workflow(workflow_data)
        return {"status": "success", "workflow": app_ui.workflow}
    except Exception as e:
        logger.error(f"Error loading workflow: {e}")
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@app.post("/workflows/extract")
async def extract_workflow(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))
        
        workflow_data = None
        ext = Path(file.filename).suffix.lower()
        
        # 1. Try PNG tEXt chunk
        if ext == '.png':
            workflow_data = img.info.get("qpyt_workflow")
            
        # 2. Try EXIF (JPEG/WebP) - UserComment (0x9286)
        if not workflow_data:
            exif = img.getexif()
            if exif:
                workflow_data = exif.get(0x9286)
        
        if not workflow_data:
            return JSONResponse(status_code=404, content={"status": "error", "message": "No workflow metadata found in this image."})
            
        try:
            import json
            data = json.loads(workflow_data)
            return {"status": "success", "workflow": data}
        except Exception as e:
            return JSONResponse(status_code=422, content={"status": "error", "message": f"Malformed workflow data: {str(e)}"})
            
    except Exception as e:
        logger.error(f"Error extracting workflow: {e}")
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@app.post("/workflows/delete")
async def delete_workflow(request: Request):
    try:
        data = await request.json()
        name = data.get("name")
        file_path = WORKFLOWS_DIR / f"{name}.json"
        preset_path = PRESETS_DIR / f"{name}.json"

        if preset_path.exists():
             return JSONResponse(status_code=403, content={"status": "error", "message": "Factory presets cannot be deleted."})
        
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

@app.get("/config/loras")
async def get_loras():
    """Returns the list of available LoRAs."""
    from core.generator import list_loras
    return {"status": "success", "loras": list_loras()}

@app.get("/config/controlnets")
async def get_controlnets():
    """Returns the list of available ControlNet models."""
    from core.generator import list_controlnets
    return {"status": "success", "models": list_controlnets()}

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
async def get_models(model_type: str, path: str = None):
    logger.info(f"[Debug] get_models called for type={model_type}, path={path}")
    if path:
        # scan specific path override
        directory = Path(path)
        if not directory.is_absolute():
            # Try to make it absolute relative to app base
            directory = Path(config.base_dir) / directory
            
        logger.info(f"[Debug] Scanning real path: {directory}")
        if not directory.exists():
            logger.warning(f"[Debug] Path does not exist: {directory}")
            return {"status": "success", "models": [], "message": f"Path '{path}' does not exist"}
        
        models = []
        for ext in ['*.safetensors', '*.ckpt', '*.gguf', '*.sft']:
             models.extend([f.name for f in directory.glob(ext)])
        
        result = sorted(list(set(models)))
        logger.info(f"[Debug] Found {len(result)} models in {directory}")
        return {"status": "success", "models": result}
    else:
        models = list_models(model_type)
        logger.info(f"[Debug] Using default list_models for {model_type}, found {len(models)}")
        return {"status": "success", "models": models}

@app.post("/generate/stop")
async def stop_generation():
    ModelManager.interrupt()
    return {"status": "success", "message": "Interruption request sent"}

@app.post("/config/unload")
async def unload_model():
    """Force unloads the current model from VRAM."""
    try:
        await run_in_threadpool(ModelManager.unload_current_model)
        return {"status": "success", "message": "Model unloaded and VRAM cleared."}
    except Exception as e:
        logger.error(f"Error unloading model: {e}")
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@app.post("/queue/stop_all")
async def stop_all():
    """Emergency stop: interrupts current generation and clears the queue."""
    try:
        # Interrupt current running task in ModelManager
        ModelManager.interrupt()
        
        # Cancel all pending tasks in QueueManager
        queue_mgr = QueueManager.get_instance()
        queue_mgr.cancel_all()
        
        return {"status": "success", "message": "Emergency stop: Current task interrupted and queue cleared."}
    except Exception as e:
        logger.error(f"Error during emergency stop: {e}")
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@app.get("/vaes")
async def get_vaes():
    return {"status": "success", "vaes": list_vaes()}

@app.get("/samplers")
async def get_samplers():
    return {"status": "success", "samplers": list_samplers()}

# Queue Management Endpoints
@app.get("/queue/status/{task_id}")
async def get_task_status(task_id: str):
    queue_mgr = QueueManager.get_instance()
    task = queue_mgr.get_task(task_id)
    if not task:
        return {"status": "error", "message": "Task not found"}
    return task

@app.get("/queue/list")
async def list_queue():
    queue_mgr = QueueManager.get_instance()
    return {"status": "success", "tasks": queue_mgr.list_tasks()}

@app.post("/queue/cancel/{task_id}")
async def cancel_task(task_id: str):
    queue_mgr = QueueManager.get_instance()
    success = queue_mgr.cancel_task(task_id)
    return {"status": "success" if success else "error"}

@app.post("/queue/clear")
async def clear_queue():
    queue_mgr = QueueManager.get_instance()
    queue_mgr.clear_completed()
    return {"status": "success"}

@app.get("/generate/preview")
async def get_preview():
    return {
        "status": "success", 
        "preview": ModelManager._current_preview,
        "current_step": ModelManager._current_step,
        "total_steps": ModelManager._total_steps
    }

# --- Task Wrappers for Queue Worker ---

async def task_generate_wrapper(**kwargs):
    # Unload other heavy models to free VRAM for Image Gen
    from core.audio_generator import MusicGenerator
    MusicGenerator.unload()
    from core.depth_estimator import unload_model as unload_depth
    unload_depth()
    
    # Clean up kwargs to match ModelManager.generate signature if needed
    image_url, exec_time, used_params = await run_in_threadpool(
        ModelManager.generate,
        **kwargs
    )
    try:
        output_dir = Path(config.OUTPUT_DIR) / image_url.split('/')[0]
        HistoryLogManager.add_to_log(output_dir, image_url.split('/')[-1], used_params, exec_time)
    except Exception as log_err:
        logger.error(f"Failed to log generation: {log_err}")
    
    return {
        "image_url": f"/outputs/{image_url}",
        "execution_time": exec_time,
        "metadata": used_params,
        "warnings": used_params.get("lora_warnings", []),
        "status": "success"
    }

async def task_upscale_wrapper(**kwargs):
    # Unload other heavy models
    from core.audio_generator import MusicGenerator
    MusicGenerator.unload()
    from core.depth_estimator import unload_model as unload_depth
    unload_depth()

    image_url, exec_time, used_params = await run_in_threadpool(
        ModelManager.upscale,
        **kwargs
    )
    return {
        "image_url": f"/outputs/{image_url}",
        "execution_time": exec_time,
        "metadata": used_params
    }

async def task_video_wrapper(**kwargs):
    from core.video_manager import CogVideoManager
    start_time = time.time()
    
    # We use run_in_threadpool because CogVideo generation is blocking
    result = await run_in_threadpool(
        CogVideoManager.generate,
        **kwargs
    )
    
    end_time = time.time()
    from datetime import datetime
    day_str = datetime.now().strftime("%Y_%m_%d")
    
    video_filename = result['video_url']
    thumb_filename = result['thumbnail_url']
    
    video_url = f"/outputs/{day_str}/{video_filename}"
    thumb_url = f"/outputs/{day_str}/{thumb_filename}" if thumb_filename else None
    
    try:
        from core.config import config
        output_dir = Path(config.OUTPUT_DIR) / day_str
        HistoryLogManager.add_to_log(output_dir, video_filename, result['metadata'], end_time - start_time)
    except Exception as log_err:
        logger.error(f"Failed to log video generation: {log_err}")

    return {
        "image_url": video_url,
        "thumbnail_url": thumb_url,
        "execution_time": end_time - start_time,
        "metadata": result["metadata"],
        "status": "success"
    }

# --- Endpoints Refactored to use Queue ---

@app.post("/generate")
async def generate(request: ImageGenerationRequest):
    try:
        logger.info(f"Adding generation to queue ({request.model_type}): {request.prompt}")
        queue_mgr = QueueManager.get_instance()
        
        # We pass the wrapper function and its arguments
        task_id = await queue_mgr.add_task(
            "generate",
            task_generate_wrapper,
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
            image_a=request.image_a,
            image_b=request.image_b,
            weight_a=request.weight_a,
            weight_b=request.weight_b,
            ip_adapter_scale=request.ip_adapter_scale,
            denoising_strength=request.denoising_strength,
            output_format=request.output_format,
            loras=request.loras,
            controlnet_image=request.controlnet_image,
            controlnet_conditioning_scale=request.controlnet_conditioning_scale,
            controlnet_model=request.controlnet_model,
            low_vram=request.low_vram,
            workflow=request.workflow
        )
        
        return {
            "status": "queued",
            "task_id": task_id
        }
    except Exception as e:
        logger.error(f"Error submitting /generate task: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

@app.post("/inpaint")
async def inpaint(request: InpaintRequest):
    logger.info(f"Adding inpaint to queue ({request.model_type}): {request.prompt}")
    queue_mgr = QueueManager.get_instance()
    
    task_id = await queue_mgr.add_task(
        "inpaint",
        task_generate_wrapper,
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
        image_a=request.image_a,
        image_b=request.image_b,
        weight_a=request.weight_a,
        weight_b=request.weight_b,
        ip_adapter_scale=request.ip_adapter_scale,
        mask=request.mask,
        denoising_strength=request.denoising_strength,
        output_format=request.output_format,
        loras=request.loras,
        is_inpaint=True,
        workflow=request.workflow
    )
    
    return {"status": "queued", "task_id": task_id}

@app.post("/outpaint")
async def outpaint(request: OutpaintRequest):
    logger.info(f"Adding outpaint to queue: {request.prompt}")
    queue_mgr = QueueManager.get_instance()
    
    task_id = await queue_mgr.add_task(
        "outpaint",
        task_generate_wrapper,
        prompt=request.prompt,
        negative_prompt=request.negative_prompt,
        model_type="sdxl", # Outpaint forced to SDXL for now
        width=request.width,
        height=request.height,
        guidance_scale=request.guidance_scale,
        num_inference_steps=request.num_inference_steps,
        seed=request.seed,
        image=request.image,
        mask=request.mask,
        denoising_strength=request.denoising_strength,
        is_inpaint=True,
        workflow=request.workflow
    )
    
    return {"status": "queued", "task_id": task_id}

@app.post("/upscale")
async def upscale(request: UpscaleRequest):
    logger.info(f"Adding upscale to queue for image data")
    queue_mgr = QueueManager.get_instance()
    
    task_id = await queue_mgr.add_task(
        "upscale",
        task_upscale_wrapper,
        image_base64=request.image,
        scale_factor=request.scale_factor,
        denoising_strength=request.denoising_strength,
        tile_size=request.tile_size,
        prompt=request.prompt,
        negative_prompt=request.negative_prompt,
        model_name=request.model_name,
        output_format=request.output_format,
        seed=request.seed,
        loras=request.loras
    )
    
    return {"status": "queued", "task_id": task_id}
@app.post("/rembg")
async def rembg(request: RembgRequest):
    logger.info("Adding background removal to queue")
    queue_mgr = QueueManager.get_instance()
    
    async def task_rembg_wrapper(image_input):
        from core.rembg_service import remove_background
        start_time = time.time()
        image_url = await run_in_threadpool(remove_background, image_input=image_input)
        return {
            "image_url": f"/outputs/{image_url}",
            "execution_time": time.time() - start_time
        }

    task_id = await queue_mgr.add_task("rembg", task_rembg_wrapper, request.image)
    return {"status": "queued", "task_id": task_id}

@app.post("/depth")
async def depth(request: DepthRequest):
    logger.info("Adding depth estimation to queue")
    queue_mgr = QueueManager.get_instance()
    
    async def task_depth_wrapper(image_input):
        # Unload Image Gen & Audio Models
        from core.generator import ModelManager
        ModelManager.unload_current_model()
        from core.audio_generator import MusicGenerator
        MusicGenerator.unload()

        from core.depth_estimator import infer_depth
        start_time = time.time()
        image_url = await run_in_threadpool(infer_depth, image_input=image_input)
        return {
            "image_url": f"/outputs/{image_url}",
            "execution_time": time.time() - start_time
        }

    task_id = await queue_mgr.add_task("depth", task_depth_wrapper, request.image)
    return {"status": "queued", "task_id": task_id}

@app.post("/normal")
async def normal_map(request: NormalRequest):
    logger.info(f"Adding normal map generation to queue (Strength: {request.strength})")
    queue_mgr = QueueManager.get_instance()
    
    async def task_normal_wrapper(image_input, strength):
        from core.normal_service import compute_normal_map
        start_time = time.time()
        image_url = await run_in_threadpool(compute_normal_map, image_input, strength)
        return {
            "image_url": f"/outputs/{image_url}",
            "execution_time": time.time() - start_time
        }

    task_id = await queue_mgr.add_task("normal", task_normal_wrapper, request.image, request.strength)
    return {"status": "queued", "task_id": task_id}

@app.post("/generate/audio")
async def generate_audio(request: AudioGenerationRequest):
    logger.info(f"Adding audio generation to queue: {request.prompt}")
    queue_mgr = QueueManager.get_instance()
    
    async def task_audio_wrapper(prompt, duration, guidance_scale):
        # Unload Image Gen & Depth Models
        from core.generator import ModelManager
        ModelManager.unload_current_model()
        from core.depth_estimator import unload_model as unload_depth
        unload_depth()

        from core.audio_generator import MusicGenerator
        start_time = time.time()
        audio_url = await run_in_threadpool(MusicGenerator.generate, prompt, duration, guidance_scale)
        return {
            "image_url": f"/outputs/{audio_url}", # Reusing 'image_url' output field for generic file path to avoid breaking frontend parsers
            "execution_time": time.time() - start_time,
            "metadata": {"prompt": prompt, "duration": duration}
        }

    task_id = await queue_mgr.add_task("audio", task_audio_wrapper, request.prompt, request.duration, request.guidance_scale)
    return {"status": "queued", "task_id": task_id}

@app.post("/generate/sprite")
async def generate_sprite(request: SpriteRequest):
    logger.info(f"Adding sprite generation to queue: {request.prompt}")
    queue_mgr = QueueManager.get_instance()
    
    async def task_sprite_wrapper(prompt, negative_prompt, width, height, frames, steps, guidance, seed, model_name, loras):
        # Unload heavy models
        from core.generator import ModelManager
        ModelManager.unload_current_model()
        from core.audio_generator import MusicGenerator
        MusicGenerator.unload()
        from core.depth_estimator import unload_model as unload_depth
        unload_depth()
        
        from core.sprite_service import SpriteService
        service = SpriteService.get_instance()
        
        start_time = time.time()
        result = await run_in_threadpool(
            service.generate_sprite, 
            prompt=prompt, 
            negative_prompt=negative_prompt, 
            width=width, 
            height=height, 
            frames=frames, 
            steps=steps, 
            guidance=guidance, 
            seed=seed,
            model_name=model_name,
            loras=loras
        )
        
        # Log to history
        try:
            from api.history_log import HistoryLogManager
            from pathlib import Path
            
            path_obj = Path(result["path"])
            output_dir = path_obj.parent
            image_name = path_obj.name
            
            metadata = {
                "prompt": prompt,
                "seed": seed,
                "model_name": model_name or "AnimateDiff-v1.5",
                "num_inference_steps": steps,
                "guidance_scale": guidance,
                "width": width,
                "height": height,
                "loras": loras or []
            }
            
            HistoryLogManager.add_to_log(output_dir, image_name, metadata, time.time() - start_time)
        except Exception as e:
            logger.error(f"Failed to log sprite generation: {e}")

        return {
            "image_url": result["url"], # Use generic field for frontend compatibility
            "execution_time": time.time() - start_time,
            "metadata": result
        }

    task_id = await queue_mgr.add_task(
        "sprite", 
        task_sprite_wrapper, 
        request.prompt, 
        request.negative_prompt, 
        request.width, 
        request.height, 
        request.frames, 
        request.steps, 
        request.guidance, 
        request.seed,
        request.model_name,
        request.loras
    )
    return {"status": "queued", "task_id": task_id}

@app.post("/vectorize")
async def vectorize(request: Request):
    # Vectorize usually comes from direct prompt or result, let's assume it has basic params
    data = await request.json()
    image_base64 = data.get("image")
    logger.info("Adding vectorization to queue")
    queue_mgr = QueueManager.get_instance()

    async def task_vectorize_wrapper(image_data):
        from core.analyzer import vectorize_image
        start_time = time.time()
        image_url = await run_in_threadpool(vectorize_image, image_data=image_data)
        return {
            "image_url": f"/view/{image_url}",
            "execution_time": time.time() - start_time
        }

    task_id = await queue_mgr.add_task("vectorize", task_vectorize_wrapper, image_base64)
    return {"status": "queued", "task_id": task_id}

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
        # Decode input
        if req.image.startswith("/view/") or req.image.startswith("http"):
            # It's a URL path, likely /view/filename.png
            filename = req.image.split("/")[-1]
            file_path = config.OUTPUT_DIR / filename
            if not file_path.exists():
                return JSONResponse({"status": "error", "message": "File not found"}, status_code=404)
            pil_image = Image.open(file_path)
            
        elif "," in req.image:
            header, encoded = req.image.split(",", 1)
            import base64
            image_data = base64.b64decode(encoded)
            pil_image = Image.open(io.BytesIO(image_data))
        else:
             # Assume Raw Base64 or local path if absolute
            if ":" in req.image and "\\" in req.image: # Quick check for windows path
                 pil_image = Image.open(req.image)
            else:
                import base64
                header, encoded = "data:image/png;base64", req.image
                try:
                    image_data = base64.b64decode(encoded)
                    pil_image = Image.open(io.BytesIO(image_data))
                except:
                     return JSONResponse({"status": "error", "message": "Invalid Image Source"}, status_code=400)
        
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

@app.get("/prompt-helper")
async def get_prompt_helper():
    try:
        path = Path("web/prompt_helper.json")
        if path.exists():
            import json
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"categories": []}
    except Exception as e:
        logger.error(f"Error loading prompt helper data: {e}")
        return {"categories": []}

@app.post("/prompt-helper/add")
async def add_prompt_keyword(request: Request):
    try:
        data = await request.json()
        category_id = data.get("category_id")
        keyword = data.get("keyword")
        
        path = Path("web/prompt_helper.json")
        if not path.exists():
            return JSONResponse(status_code=404, content={"status": "error", "message": "Prompt helper data file not found."})
            
        import json
        with open(path, "r", encoding="utf-8") as f:
            ph_data = json.load(f)
            
        # Find category and add keyword
        found = False
        for cat in ph_data.get("categories", []):
            if cat["id"] == category_id:
                if keyword not in cat["keywords"]:
                    cat["keywords"].append(keyword)
                    found = True
                    break
                else:
                    return {"status": "success", "message": "Keyword already exists."}
                    
        if not found:
            return JSONResponse(status_code=404, content={"status": "error", "message": f"Category '{category_id}' not found."})
            
        with open(path, "w", encoding="utf-8") as f:
            json.dump(ph_data, f, indent=4)
            
        return {"status": "success", "message": f"Keyword '{keyword}' added to '{category_id}'."}
    except Exception as e:
        logger.error(f"Error adding prompt keyword: {e}")
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})

