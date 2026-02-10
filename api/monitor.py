import time
import psutil
from fastapi import APIRouter
from starlette.concurrency import run_in_threadpool  # Fix import
import logging

# Try importing pynvml for GPU stats
try:
    import pynvml
    HAS_GPU_STATS = True
except ImportError:
    HAS_GPU_STATS = False

import shutil
from core.config import config

router = APIRouter()
logger = logging.getLogger("qpyt-monitor")

def get_gpu_stats():
    """
    Fetches GPU usage and VRAM usage using pynvml.
    Returns a list of dicts (one per GPU) or empty list if failed.
    """
    if not HAS_GPU_STATS:
        return []

    gpus = []
    try:
        pynvml.nvmlInit()
        device_count = pynvml.nvmlDeviceGetCount()
        
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            # Encode/decode might be needed depending on python version/pynvml version, 
            # usually it returns str or bytes. Safe check:
            if isinstance(name, bytes):
                name = name.decode("utf-8")
                
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util_rate = pynvml.nvmlDeviceGetUtilizationRates(handle)
            
            # Convert bytes to MB for easier reading, or keep bytes for frontend formatting
            # Let's send MB to be consistent with common tools
            total_mb = mem_info.total / 1024 / 1024
            used_mb = mem_info.used / 1024 / 1024
            
            gpus.append({
                "index": i,
                "name": name,
                "vram_total_mb": round(total_mb),
                "vram_used_mb": round(used_mb),
                "vram_percent": round((used_mb / total_mb) * 100, 1),
                "gpu_util_percent": util_rate.gpu
            })
            
    except Exception as e:
        logger.warning(f"Failed to get GPU stats: {e}")
    finally:
        try:
            pynvml.nvmlShutdown()
        except:
            pass
            
    return gpus

def get_disk_usage():
    try:
        # Check output directory
        path = config.OUTPUT_DIR
        total, used, free = shutil.disk_usage(path)
        
        return {
            "total_gb": round(total / (1024**3), 2),
            "used_gb": round(used / (1024**3), 2),
            "free_gb": round(free / (1024**3), 2),
            "percent": round((used / total) * 100, 1)
        }
    except Exception as e:
        logger.warning(f"Disk usage check failed: {e}")
        return {"total_gb": 0, "free_gb": 0, "percent": 0}

def get_system_stats():
    # CPU
    cpu_percent = psutil.cpu_percent(interval=None) # Non-blocking
    
    # RAM
    ram = psutil.virtual_memory()
    ram_total_mb = ram.total / 1024 / 1024
    ram_used_mb = ram.used / 1024 / 1024
    
    # GPU
    gpus = get_gpu_stats()
    
    # Disk
    disk = get_disk_usage()
    
    return {
        "status": "success",
        "cpu": {
            "percent": cpu_percent
        },
        "ram": {
            "total_mb": round(ram_total_mb),
            "used_mb": round(ram_used_mb),
            "percent": ram.percent
        },
        "gpus": gpus,
        "disk": disk
    }

@router.get("/monitor/stats")
async def monitor_stats():
    # Run synchronous psutil/pynvml calls in threadpool to avoid blocking event loop
    return await run_in_threadpool(get_system_stats)
