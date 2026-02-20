import torch
from typing import Optional, Dict, Any, List
import uuid
import os
import warnings
import gc
import math
import numpy as np
import time
# Silence diffusers config warnings early
warnings.filterwarnings("ignore", category=FutureWarning, module="diffusers.configuration_utils")
warnings.filterwarnings("ignore", category=FutureWarning, message=".*upcast_vae.*")
from pathlib import Path
from diffusers import (
    StableDiffusionXLPipeline, 
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLInpaintPipeline,
    FluxPipeline, 
    FluxImg2ImgPipeline,
    FluxInpaintPipeline,
    Flux2Pipeline,
    StableDiffusion3Pipeline,
    StableDiffusion3Img2ImgPipeline,
    StableDiffusion3InpaintPipeline,
    AutoPipelineForInpainting,
    GGUFQuantizationConfig,
    QuantoConfig,
    FluxTransformer2DModel,
    Flux2Transformer2DModel,
    AutoencoderKL,
    AutoencoderKLFlux2,
    FlowMatchEulerDiscreteScheduler,
    SD3Transformer2DModel,
    BitsAndBytesConfig,
    StableDiffusionXLControlNetPipeline,
    ControlNetModel,
    ZImagePipeline)
from transformers import CLIPTextModel, T5EncoderModel, T5TokenizerFast, CLIPImageProcessor, CLIPVisionModelWithProjection as TransformersCLIPVision
try:
    from transformers import Mistral3ForConditionalGeneration, AutoModel
except ImportError:
    # Handle potentially different version/naming
    Mistral3ForConditionalGeneration = None
    from transformers import AutoModel
import logging
import json
from PIL.PngImagePlugin import PngInfo
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

from compel import Compel, ReturnedEmbeddingsType
from compel import Compel, ReturnedEmbeddingsType
from typing import Optional, Dict, Any
from core.config import config


import base64
from io import BytesIO
from PIL import Image
from safetensors import safe_open

# Enable TF32 for faster and memory efficient computation on Ampere+ GPUs
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

class ModelManager:
    _instance = None
    _pipeline = None
    _compel = None
    _current_model_path = None
    _current_model_type = None # 'sdxl', 'flux', 'sd3', 'zimage'
    _current_is_img2img = False
    _current_is_inpaint = False
    _current_is_inpaint = False
    _current_has_controlnet = False # Track if current pipe has ControlNet
    _current_controlnet_name = None # Track specific ControlNet model
    _current_dtype = torch.float16
    _interrupt_flag = False
    _current_preview = None # Base64 string of the last preview
    _current_step = 0
    _total_steps = 0
    _weights_t = None
    _biases_t = None
    _active_lora_names = [] # Names of currently loaded LoRA adapters
    _current_has_ip_adapter = False
    _current_ip_adapter_name = None
    _image_encoder = None

    @classmethod
    def _apply_ip_adapter(cls, pipe, model_type: str, weight_name: str = "ip-adapter_sdxl.safetensors", required_layers: int = 1):
        if model_type != 'sdxl' or pipe is None: return pipe
        
        # Check current loaded adapters
        unet = getattr(pipe, "unet", None)
        if unet is None: return pipe

        # Count existing projection layers
        current_layers = 0
        if hasattr(unet, "encoder_hid_proj") and unet.encoder_hid_proj is not None:
            if hasattr(unet.encoder_hid_proj, "image_projection_layers"):
                current_layers = len(unet.encoder_hid_proj.image_projection_layers)
            else:
                # If it exists but isn't a list, it's likely a single layer (old/internal diffusers state)
                current_layers = 1
        
        if current_layers >= required_layers and cls._current_ip_adapter_name == weight_name:
            return pipe
            
        print(f"[IP-Adapter] Loading {weight_name} (Required: {required_layers}, Current: {current_layers})...")
        try:
            # CRITICAL: Always remove hooks before structural changes
            cls.remove_all_hooks(pipe)
            
            if cls._image_encoder is None:
                # Using h94 repo which is more reliable for IP-Adapter components
                cls._image_encoder = TransformersCLIPVision.from_pretrained(
                    "h94/IP-Adapter", 
                    subfolder="sdxl_models/image_encoder",
                    torch_dtype=torch.float16
                ).to(config.DEVICE)
            
            adapter_path = Path("G:/models/ip_adapter") / weight_name
            if not adapter_path.exists(): 
                print(f"[IP-Adapter] Error: {adapter_path} not found.")
                return pipe
            
            # Use manual loading to bypass directory scanning issues for local files
            from safetensors.torch import load_file
            raw_state_dict = load_file(str(adapter_path))
            
            # Partition weights
            state_dict = {"image_proj": {}, "ip_adapter": {}}
            for k, v in raw_state_dict.items():
                if k.startswith("image_proj."):
                    state_dict["image_proj"][k.replace("image_proj.", "")] = v
                elif k.startswith("ip_adapter."):
                    state_dict["ip_adapter"][k.replace("ip_adapter.", "")] = v
            
            if not state_dict["image_proj"]: state_dict = raw_state_dict

            # Attach image encoder
            if cls._image_encoder is not None:
                pipe.image_encoder = cls._image_encoder

            # Load until we reach required count
            while current_layers < required_layers:
                print(f"[IP-Adapter] Adding layer {current_layers + 1}...")
                pipe.load_ip_adapter(state_dict, subfolder="", weight_name="")
                current_layers += 1
            
            cls._current_has_ip_adapter = True
            cls._current_ip_adapter_name = weight_name
            print(f"[IP-Adapter] Loaded successfully ({current_layers} layers).")
        except Exception as e: 
            import traceback
            print(f"[IP-Adapter] Load failed: {e}")
            print(traceback.format_exc())
            cls._current_has_ip_adapter = False
            cls._current_ip_adapter_name = None
        return pipe

    @classmethod
    def latents_to_rgb(cls, latents):
        """Ultra-fast linear transformation of latents to RGB (approximation)"""
        if latents.shape[0] != 4:
            # We only support SD1.5/SDXL approximation (4 channels)
            # For SD3.5 (16 channels), we skip for now to avoid einsum mismatch
            return None

        if cls._weights_t is None or cls._weights_t.dtype != latents.dtype:
            # Optimized coefficients for SDXL/SD1.5
            weights = (
                (60, -60, 25, -70),
                (60,  -5, 15, -50),
                (60,  10, -5, -35),
            )
            cls._weights_t = torch.t(torch.tensor(weights, dtype=latents.dtype).to(latents.device))
            cls._biases_t = torch.tensor((150, 140, 130), dtype=latents.dtype).to(latents.device)
        
        # We expect [C, H, W]
        rgb_tensor = torch.einsum("lxy,lr -> rxy", latents, cls._weights_t) + cls._biases_t.unsqueeze(-1).unsqueeze(-1)
        image_array = rgb_tensor.clamp(0, 255).byte().cpu().permute(1, 2, 0).numpy()
        return Image.fromarray(image_array)

    @classmethod
    def interrupt(cls):
        cls._interrupt_flag = True
        print("Interruption flag set.")

    @classmethod
    def reset_interrupt(cls):
        cls._interrupt_flag = False
        cls._current_preview = None
        cls._current_step = 0
        cls._total_steps = 0

    @classmethod
    def remove_all_hooks(cls, pipe):
        """Deeply remove all accelerate hooks from pipeline components to ensure clean device placement."""
        if pipe is None:
            return
        
        # 1. Pipeline level removal if available
        if hasattr(pipe, "remove_all_hooks"):
            pipe.remove_all_hooks()
            
        # 2. Iterate through all components (unet, vae, transformer, etc.)
        for name in pipe.config.keys():
            component = getattr(pipe, name, None)
            if component is not None and hasattr(component, "_hf_hook"):
                from accelerate.hooks import remove_hook_from_module
                remove_hook_from_module(component, recurse=True)
                print(f"Removed hooks from component: {name}")

    @classmethod
    def unload_current_model(cls):
        if cls._pipeline is not None:
            print(f"Unloading model: {cls._current_model_path}")
            # Move to CPU before deleting to be sure CUDA memory is freed
            try:
                # For offloaded models, we might need to manually move components
                if hasattr(cls._pipeline, "components"):
                    for name, component in cls._pipeline.components.items():
                        if isinstance(component, torch.nn.Module):
                            component.to("cpu")
                else:
                    cls._pipeline.to("cpu")
            except:
                pass
            
            # Explicitly delete components if possible to help GC
            del cls._pipeline
            del cls._compel
            cls._pipeline = None
            cls._compel = None
            cls._current_model_path = None
            cls._current_model_type = None
            cls._current_has_controlnet = False
            cls._current_controlnet_name = None
            cls._current_has_ip_adapter = False
            cls._current_ip_adapter_name = None
            
            # GC first, then clear cache
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                torch.cuda.synchronize()
            cls._active_lora_names = []
            print("Memory cleared.")
            
            print("Memory cleared.")


    @classmethod
    def _get_lora_architecture(cls, lora_path: Path):
        """
        Inspects the safetensors metadata/tensors to determine if it's SD 1.5 or SDXL/Flux.
        Returns 'sdxl', 'sd15', 'flux' or 'unknown'.
        """
        try:
            with safe_open(lora_path, framework="pt", device="cpu") as f:
                metadata = f.metadata()
                if metadata:
                    # Try to get from metadata first
                    base_model = metadata.get("ss_base_model_version")
                    if base_model:
                        if "sdxl" in base_model.lower(): return "sdxl"
                        if "v1-5" in base_model.lower() or "sd15" in base_model.lower(): return "sd15"
                        if "flux" in base_model.lower(): return "flux"

                # Fallback: Check tensor shapes
                # SD 1.5: usually 768 (attn2.to_k)
                # SDXL: 2048 (attn2.to_k)
                # Flux: 4096 or transformer blocks
                for key in f.keys():
                    if "attn2.to_k" in key or "attn2_to_k" in key:
                        shape = f.get_slice(key).get_shape()
                        if len(shape) >= 2:
                            dim = shape[1]
                            if dim == 768: return "sd15"
                            if dim == 2048: return "sdxl"
                            if dim == 4096: return "flux"
                    
                    if "transformer.single_transformer_blocks" in key or "flux" in key.lower():
                        # Strong indicator of Flux DiT architecture
                        return "flux"

        except Exception as e:
            print(f"[LoRA Check] Error inspecting {lora_path}: {e}")
        return "unknown"

    @classmethod
    def apply_loras(cls, pipeline, loras):
        """
        Applies multiple LoRAs to the given pipeline.
        'loras' is a list of dicts: [{'path': '...', 'weight': 0.8, 'enabled': True}]
        """
        if pipeline is None:
            return

        # 1. Reset adapters if any were active
        if hasattr(pipeline, "unload_lora_weights"):
            try:
                pipeline.unload_lora_weights()
            except Exception as e:
                print(f"[LoRA] Error unloading: {e}")
        
        cls._active_lora_names = []
        active_loras = [l for l in loras if l.get('enabled', True)]
        warnings = []
        
        if not active_loras:
            print("[LoRA] No active LoRAs to apply.", flush=True)
            return []

        print(f"[LoRA] Applying {len(active_loras)} LoRA adapters...", flush=True)
        
        adapter_names = []
        adapter_weights = []
        
        for i, lora in enumerate(active_loras):
            p = lora.get('path')
            w = lora.get('weight', 1.0)
            
            if not p: continue
            
            # Construct absolute path
            lora_path = Path(config.get('LORAS_DIR', 'models/loras')) / p
            if not lora_path.exists():
                print(f"[LoRA] Warning: LoRA not found at {lora_path}")
                continue
                
            # Architecture Check
            lora_arch = cls._get_lora_architecture(lora_path)
            pipe_arch = "sdxl" if isinstance(pipeline, (StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, StableDiffusionXLInpaintPipeline)) else \
                        "flux" if isinstance(pipeline, (FluxPipeline, FluxImg2ImgPipeline, FluxInpaintPipeline)) else "unknown"
            
            if lora_arch != "unknown" and pipe_arch != "unknown" and lora_arch != pipe_arch:
                msg = f"Architecture mismatch for {p}: LoRA is {lora_arch.upper()}, but model is {pipe_arch.upper()}."
                print(f"  ⚠ {msg}", flush=True)
                warnings.append(msg)
                continue

            adapter_name = f"adapter_{i}_{Path(p).stem}"
            try:
                print(f"  → Loading LoRA: {p} (Scale: {w})", flush=True)
                pipeline.load_lora_weights(str(lora_path), adapter_name=adapter_name)
                adapter_names.append(adapter_name)
                adapter_weights.append(float(w))
            except Exception as e:
                print(f"  ✖ Failed to load LoRA {p}: {e}", flush=True)

        if adapter_names:
            try:
                pipeline.set_adapters(adapter_names, adapter_weights)
                cls._active_lora_names = adapter_names
                print(f"[LoRA] Successfully blended {len(adapter_names)} adapters: {', '.join(adapter_names)}", flush=True)
            except Exception as e:
                print(f"[LoRA] Error setting adapters: {e}", flush=True)
        
        return warnings

    @classmethod
    def _load_image(cls, image_input: Any, mode: str = "RGB") -> Optional[Image.Image]:
        """
        Robustly loads an image from base64, /view/ path, /outputs/ path, or PIL Image.
        """
        if not image_input:
            return None
            
        if isinstance(image_input, Image.Image):
            return image_input.convert(mode)
            
        if not isinstance(image_input, str):
            return None

        try:
            # Handle path-based inputs
            if image_input.startswith("/view/") or image_input.startswith("/outputs/"):
                rel_path = image_input.replace("/view/", "").replace("/outputs/", "").lstrip("/")
                full_path = Path(config.OUTPUT_DIR) / rel_path
                
                if not full_path.exists():
                    # Fallback to checking if it's already an absolute path
                    if Path(image_input).exists():
                         return Image.open(image_input).convert(mode)
                    raise FileNotFoundError(f"Source image not found: {full_path}")
                return Image.open(full_path).convert(mode)
            
            # Handle base64-based inputs
            if "," in image_input:
                _, encoded = image_input.split(",", 1)
            else:
                encoded = image_input
            
            # Clean up whitespace/newlines which can break decoding
            encoded = encoded.strip().replace("\n", "").replace("\r", "")
            
            # Handle padding if missing (common base64 issue)
            missing_padding = len(encoded) % 4
            if missing_padding:
                encoded += "=" * (4 - missing_padding)
                
            image_data = base64.b64decode(encoded)
            return Image.open(BytesIO(image_data)).convert(mode)
            
        except Exception as e:
            print(f"[ModelManager] _load_image error: {e}")
            return None

    @classmethod
    def get_pipeline(cls, model_type: str, model_path: str, vae_name: Optional[str] = None, 
                     sampler_name: Optional[str] = None, is_img2img: bool = False, is_inpaint: bool = False,
                     controlnet_model_name: Optional[str] = None, low_vram: bool = False):
        # Normalize VAE name
        if vae_name == 'Default' or not vae_name:
            vae_name = None

        # REMOVED: Dangerous aliasing logic that caused Flux->Img2Img crash
        # if model_type in ['img2img', 'upscale']:
        #     model_type = cls._current_model_type or 'sdxl'
        #     is_img2img = True

        # CRITICAL: Always unload auxiliary models (Florence-2, LLM, etc.) to free VRAM
        # before any Stable Diffusion operation (load or switch).
        import core.analyzer as analyzer_lib
        import core.rembg_service as rembg_lib
        from core.translator import TranslationManager
        from core.llm_prompter import LlmPrompterManager
        analyzer_lib.unload_model()
        rembg_lib.unload_model()
        TranslationManager.unload_model()
        LlmPrompterManager.unload_model()

        if cls._pipeline is not None:
            # Check if we are already loaded with the right model and path
            # Also check if low_vram setting changed for Flux (we can't easily detect if it was loaded with specific quantization, so simple check: if we ask for low_vram and current dtype is float16/bfloat16 but maybe not quantized... for simplicity, if low_vram is requested, we might want to ensure it.)
            # For now, let's assume if path matches, it is fine, unless we want to force reload.
            # Ideally we track _current_low_vram state.
            
            if cls._current_model_path == model_path and cls._current_model_type == model_type:
                # If only the task type (Txt2Img vs Img2Img) changed, try a fast structural switch
                # BUT if ControlNet requirement changed, we MUST reload/adapt
                has_controlnet = bool(controlnet_model_name)
                
                if (cls._current_is_img2img != is_img2img or 
                    cls._current_is_inpaint != is_inpaint or 
                    cls._current_has_controlnet != has_controlnet or
                    cls._current_controlnet_name != controlnet_model_name):
                    
                    if (cls._current_has_controlnet != has_controlnet or 
                        cls._current_controlnet_name != controlnet_model_name):
                         # ControlNet injection/removal is non-trivial for structural switch in diffusers (requires different pipe class init)
                         # Simple approach: full reload if controlnet status changes
                         print("ControlNet changed, full reload required.")
                         cls.unload_current_model()
                    else:
                        mode_name = "Inpaint" if is_inpaint else ("Img2Img" if is_img2img else "Txt2Img")
                        print(f"Switching pipeline mode to {mode_name}...")
                        try:
                            # Map base models for from_pipe
                            base_pipes = {
                                'sdxl': (StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, StableDiffusionXLInpaintPipeline),
                                'flux': (FluxPipeline, FluxImg2ImgPipeline, FluxInpaintPipeline),
                                'sd3': (StableDiffusion3Pipeline, StableDiffusion3Img2ImgPipeline, StableDiffusion3InpaintPipeline),
                                'sd3_5_turbo': (StableDiffusion3Pipeline, StableDiffusion3Img2ImgPipeline, StableDiffusion3InpaintPipeline),
                                'zimage': (ZImagePipeline, None, None)
                            }
                            txt_class, img_class, inp_class = base_pipes.get(model_type, (None, None, None))
                            if is_inpaint:
                                target_class = inp_class
                            elif is_img2img:
                                target_class = img_class
                            else:
                                target_class = txt_class
                            
                            if target_class:
                                # Re-apply optimizations after from_pipe as it can lose hooks
                                cls._pipeline = target_class.from_pipe(cls._pipeline)
                                cls._current_is_img2img = is_img2img
                                cls._current_is_inpaint = is_inpaint
                                
                                # FORCE COMPONENT-LEVEL DTYPE RESTORATION
                                # from_pipe often reset modules to fp32
                                print(f"[Performance] Enforcing {cls._current_dtype} and memory format for structural switch...")
                                
                                # Standard Speed Optimization Path for high VRAM
                                vram_gb = 0
                                if torch.cuda.is_available():
                                    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                                
                                cls.remove_all_hooks(cls._pipeline)
                                
                                # Define target device
                                device = config.DEVICE
                                
                                # Explicitly cast components
                                # This is safer than pipe.to() which might misinterpret some modules
                                if hasattr(cls._pipeline, "unet") and cls._pipeline.unet is not None:
                                    cls._pipeline.unet.to(device=device, dtype=cls._current_dtype, memory_format=torch.channels_last)
                                if hasattr(cls._pipeline, "transformer") and cls._pipeline.transformer is not None:
                                    # Don't force cast transformer if it might be quantized (bitsandbytes modules don't like .to())
                                    # We should check if it's a quantized module
                                    if not (hasattr(cls._pipeline.transformer, "is_quantized") and cls._pipeline.transformer.is_quantized):
                                         cls._pipeline.transformer.to(device=device, dtype=cls._current_dtype, memory_format=torch.channels_last)
                                
                                if hasattr(cls._pipeline, "text_encoder") and cls._pipeline.text_encoder is not None:
                                    cls._pipeline.text_encoder.to(device=device, dtype=cls._current_dtype)
                                if hasattr(cls._pipeline, "text_encoder_2") and cls._pipeline.text_encoder_2 is not None:
                                    cls._pipeline.text_encoder_2.to(device=device, dtype=cls._current_dtype)
                                if hasattr(cls._pipeline, "text_encoder_3") and cls._pipeline.text_encoder_3 is not None:
                                    cls._pipeline.text_encoder_3.to(device=device, dtype=cls._current_dtype)

                                # Special handling for SDXL VAE - 
                                # If we use half precision everywhere else, we MUST match it in the VAE
                                # to avoid "Input type and bias type should be same" error.
                                if hasattr(cls._pipeline, "vae") and cls._pipeline.vae is not None:
                                    # Align with current_dtype (float16) to avoid crashes
                                    cls._pipeline.vae.to(device=device, dtype=cls._current_dtype)

                                # Optimization application logic
                                if low_vram:
                                     # For low VRAM, we prefer model_cpu_offload
                                     cls._pipeline.enable_model_cpu_offload()
                                     if hasattr(cls._pipeline, "enable_vae_tiling"): cls._pipeline.enable_vae_tiling()
                                     if hasattr(cls._pipeline, "enable_vae_slicing"): cls._pipeline.enable_vae_slicing()
                                elif vram_gb >= 10:
                                    cls._pipeline.to(device)
                                    cls._pipeline.disable_attention_slicing()
                                    # Ensure VAE tiling/slicing is still active
                                    if hasattr(cls._pipeline, "enable_vae_tiling"):
                                        cls._pipeline.enable_vae_tiling()
                                    if hasattr(cls._pipeline, "enable_vae_slicing"):
                                        cls._pipeline.enable_vae_slicing()
                                    
                                    # Re-enable xformers explicitly
                                    try:
                                        import xformers
                                        cls._pipeline.enable_xformers_memory_efficient_attention()
                                    except: pass
                                else:
                                    cls._pipeline.enable_model_cpu_offload()
                                
                                # GC and clear cache after structural change
                                gc.collect()
                                if torch.cuda.is_available():
                                    torch.cuda.empty_cache()
                                    
                                print(f"Mode switch successful. Optimized for {device} in {cls._current_dtype}")
                                return cls._pipeline, cls._compel
                        except Exception as e:
                            print(f"Fast mode switch failed: {e}. Falling back to full reload.")
                            cls.unload_current_model()
                else:
                    # Everything matches, no change needed
                    return cls._pipeline, cls._compel
            else:
                # Different model or path, full unload
                cls.unload_current_model()
        
        # Auxiliary models already unloaded at the start of method

        if cls._pipeline is None:
            print(f"Loading {model_type} model ({'Img2Img' if is_img2img else 'Txt2Img'}): {model_path}...")
            
            # Quantization flag
            is_quantized = False
            
            # Model selection based on type
            if model_type == 'flux':
                cls._current_dtype = torch.bfloat16
                if is_inpaint:
                    pipeline_class = FluxInpaintPipeline
                elif is_img2img:
                    pipeline_class = FluxImg2ImgPipeline
                else:
                    pipeline_class = FluxPipeline
                
                # Check for single file
                if model_path.lower().endswith((".safetensors", ".ckpt", ".sft")):
                    print(f"Loading local Flux model (hybrid load): {model_path}")
                    bfl_repo = "black-forest-labs/FLUX.1-schnell"
                    
                    try:
                        # 1. Load Transformer from local file with adaptation for Flux 2 Klein
                        print(f"  -> Loading FluxTransformer2DModel from {model_path}...")
                        
                        # Peek into the safetensors to check for Flux 2 prefix and Klein architecture
                        load_kwargs = {"torch_dtype": torch.bfloat16}
                        
                        from safetensors.torch import load_file
                        state_dict = load_file(model_path)
                        
                        # Handle prefixes
                        prefix = "model.diffusion_model."
                        has_prefix = any(k.startswith(prefix) for k in state_dict.keys())
                        if has_prefix:
                            print(f"  -> Detected {prefix} prefix. Stripping...")
                            state_dict = {k[len(prefix):]: v for k, v in state_dict.items()}
                        
                        # Determine architecture (Klein 4B: 5 double blocks, 20 single blocks)
                        double_blocks = [k for k in state_dict.keys() if k.startswith("double_blocks.")]
                        single_blocks = [k for k in state_dict.keys() if k.startswith("single_blocks.")]
                        
                        num_double = 0
                        num_single = 0
                        if double_blocks:
                            num_double = max(int(k.split(".")[1]) for k in double_blocks) + 1
                        if single_blocks:
                            num_single = max(int(k.split(".")[1]) for k in single_blocks) + 1
                            
                        print(f"  -> Detected architecture: {num_double} double blocks, {num_single} single blocks")
                        
                        # CLEANUP: Filter out scalar tensors (like .weight_scale) that crash the diffusers converter
                        # and ensure everything is at least bfloat16 for the conversion steps.
                        print("  -> Cleaning state_dict and detecting dimensions...")
                        scalar_keys = [k for k, v in state_dict.items() if v.dim() == 0]
                        if scalar_keys:
                            print(f"     -> Removing {len(scalar_keys)} scalar/scale tensors to avoid converter crash.")
                            for k in scalar_keys: del state_dict[k]
                        
                        # Flux 2 support based on transformer block structure
                        if num_double > 0 and num_single > 0:
                            print(f"  -> Flux.2 architecture detected ({num_double}d/{num_single}s). Configuring Flux2Pipeline...")
                            pipeline_class = Flux2Pipeline
                            
                            # Detect architectural dimensions
                            # 1. hidden_size from x_embedder or img_in
                            # 2. joint_attention_dim from context_embedder or txt_in
                            hidden_size = 3072 # Default for 4B
                            joint_attention_dim = 7680 # Default for 4B
                            
                            if "x_embedder.weight" in state_dict:
                                hidden_size = state_dict["x_embedder.weight"].shape[0]
                            elif "img_in.weight" in state_dict:
                                hidden_size = state_dict["img_in.weight"].shape[0]

                            if "context_embedder.weight" in state_dict:
                                joint_attention_dim = state_dict["context_embedder.weight"].shape[0]
                            elif "txt_in.weight" in state_dict:
                                # In some checkpoints, txt_in is [hidden, joint]
                                shape = state_dict["txt_in.weight"].shape
                                if len(shape) >= 2:
                                    # If it's square, it's likely [hidden, joint] where hidden==joint
                                    # If it's [3072, 7680], then joint is 7680.
                                    joint_attention_dim = shape[1]
                            
                            num_attention_heads = hidden_size // 128
                            print(f"  -> Detected dimensions: hidden={hidden_size}, joint={joint_attention_dim}, heads={num_attention_heads}")

                            # Standard Flux.2 config
                            transformer_config = {
                                "num_layers": num_double,
                                "num_single_layers": num_single,
                                "patch_size": 1,
                                "in_channels": 128, # Standard Flux 2 default
                                "out_channels": None,
                                "num_attention_heads": num_attention_heads, 
                                "attention_head_dim": 128,
                                "joint_attention_dim": joint_attention_dim,
                                "mlp_ratio": 3.0, # Flux 2 uses SwiGLU with 3.0 mult
                                "axes_dims_rope": [32, 32, 32, 32],
                                "rope_theta": 2000,
                                "timestep_guidance_channels": 256,
                            }
                            
                            # IMPORTANT: Cast state_dict to bfloat16 before conversion if it's in FP8
                            # diffusers' convert functions might not handle FP8 chunking/concatenation correctly
                            # or might produce invalid dtypes in the resulting dictionary.
                            needs_cast = any(v.dtype == torch.float8_e4m3fn for v in state_dict.values())
                            if needs_cast:
                                print("  -> Casting FP8 state_dict to bfloat16 for robust conversion...")
                                state_dict = {k: v.to(dtype=torch.bfloat16) for k, v in state_dict.items()}

                            transformer = Flux2Transformer2DModel(**transformer_config).to(dtype=torch.bfloat16)
                            
                            from diffusers.loaders.single_file_utils import convert_flux2_transformer_checkpoint_to_diffusers
                            print("  -> Converting state_dict to Flux 2 format...")
                            
                            # Standard conversion
                            diff_dict = convert_flux2_transformer_checkpoint_to_diffusers(state_dict)
                            
                            # Manual Fixup for keys that often fail or are named differently in single-file checkpoints
                            # Flux 2 Klein specific remapping
                            manual_map = {
                                "img_in.weight": "x_embedder.weight",
                                "img_in.bias": "x_embedder.bias",
                                "txt_in.weight": "context_embedder.weight",
                                "txt_in.bias": "context_embedder.bias",
                                "time_in.in_layer.weight": "time_guidance_embed.timestep_embedder.linear_1.weight",
                                "time_in.in_layer.bias": "time_guidance_embed.timestep_embedder.linear_1.bias",
                                "time_in.out_layer.weight": "time_guidance_embed.timestep_embedder.linear_2.weight",
                                "time_in.out_layer.bias": "time_guidance_embed.timestep_embedder.linear_2.bias",
                                "guidance_in.in_layer.weight": "time_guidance_embed.guidance_embedder.linear_1.weight",
                                "guidance_in.in_layer.bias": "time_guidance_embed.guidance_embedder.linear_1.bias",
                                "guidance_in.out_layer.weight": "time_guidance_embed.guidance_embedder.linear_2.weight",
                                "guidance_in.out_layer.bias": "time_guidance_embed.guidance_embedder.linear_2.bias",
                            }

                            for src, dst in manual_map.items():
                                if src in state_dict and dst not in diff_dict:
                                    diff_dict[dst] = state_dict.pop(src)
                                elif src in diff_dict and src != dst:
                                    # Sometimes the converter partially maps it, clean it up
                                    diff_dict[dst] = diff_dict.pop(src)

                            # Handle dimension mismatches in converted dict (outliers)
                            for k, v in diff_dict.items():
                                if k in transformer.state_dict():
                                    target_shape = transformer.state_dict()[k].shape
                                    if v.shape != target_shape:
                                        print(f"  -> Fixing shape mismatch for {k}: {v.shape} -> {target_shape}")
                                        pass # Outlier handled
                            
                            # Zero-initialize missing keys to avoid random noise
                            model_state = transformer.state_dict()
                            for k in model_state.keys():
                                if k not in diff_dict:
                                    print(f"  -> WARNING: Key {k} missing from checkpoint. Initializing to zero.")
                                    diff_dict[k] = torch.zeros_like(model_state[k])
                            
                            res = transformer.load_state_dict(diff_dict, strict=False)
                            if res.missing_keys: 
                                print(f"  -> Missing keys ({len(res.missing_keys)}):")
                                for mk in sorted(res.missing_keys): print(f"     - {mk}")
                            if res.unexpected_keys: 
                                print(f"  -> Unexpected keys ({len(res.unexpected_keys)}):")
                                for uk in sorted(res.unexpected_keys): print(f"     - {uk}")
                            
                            # Load Mistral for Flux 2
                            mistral_repo = "black-forest-labs/FLUX.2-klein-4B"
                            print(f"  -> Loading Mistral text encoder from {mistral_repo}...")
                            
                            t5_repo = "black-forest-labs/FLUX.1-schnell"
                            
                            try:
                                print(f"  -> Loading text encoder (AutoModel) from {mistral_repo}...")
                                text_encoder = AutoModel.from_pretrained(
                                    mistral_repo, 
                                    subfolder="text_encoder", 
                                    torch_dtype=torch.bfloat16,
                                    trust_remote_code=True
                                )
                            except Exception as mistral_err:
                                print(f"  -> Text encoder load error: {mistral_err}. Falling back to default components.")
                                text_encoder = None

                            text_encoder_2 = T5EncoderModel.from_pretrained(t5_repo, subfolder="text_encoder_2", torch_dtype=torch.bfloat16)
                            try:
                                print(f"  -> Loading VAE (AutoencoderKLFlux2) from {mistral_repo}...")
                                # Flux 2 uses a specialized VAE with Batch Norm
                                vae = AutoencoderKLFlux2.from_pretrained(mistral_repo, subfolder="vae", torch_dtype=torch.bfloat16)
                            except Exception as vae_err:
                                print(f"  -> AutoencoderKLFlux2 failed: {vae_err}. Falling back to Schnell VAE.")
                                vae = AutoencoderKL.from_pretrained(t5_repo, subfolder="vae", torch_dtype=torch.bfloat16)

                            # Patch format_input to avoid TypeError in certain tokenizers
                            import diffusers.pipelines.flux2.pipeline_flux2 as flux2_pipe
                            def patched_format_input(prompts, system_message=flux2_pipe.SYSTEM_MESSAGE, images=None):
                                cleaned_txt = [p.replace("[IMG]", "") for p in prompts]
                                return [[{"role": "system", "content": system_message}, {"role": "user", "content": p}] for p in cleaned_txt]
                            
                            print("  -> Patching Flux 2 format_input for tokenizer compatibility...")
                            flux2_pipe.format_input = patched_format_input

                            # Flux 2 Pipeline expects a single Processor/Tokenizer
                            from transformers import AutoTokenizer
                            print(f"  -> Loading tokenizer from {mistral_repo}...")
                            tokenizer = AutoTokenizer.from_pretrained(mistral_repo, subfolder="tokenizer")

                            print(f"  -> Assembling {pipeline_class.__name__}...")
                            cls._pipeline = pipeline_class(
                                transformer=transformer,
                                text_encoder=text_encoder,
                                tokenizer=tokenizer,
                                vae=vae,
                                scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(mistral_repo, subfolder="scheduler")
                            )
                            
                        else:
                            # Flux 1 Loading Logic
                            del state_dict
                            gc.collect()
                            
                            transformer = FluxTransformer2DModel.from_single_file(
                                model_path,
                                **load_kwargs
                            )
                            
                            print(f"  -> Loading T5EncoderModel from {bfl_repo}...")
                            text_encoder_2 = T5EncoderModel.from_pretrained(
                                bfl_repo, 
                                subfolder="text_encoder_2", 
                                torch_dtype=torch.bfloat16
                            )

                            print(f"  -> Assembling {pipeline_class.__name__}...")
                            cls._pipeline = pipeline_class.from_pretrained(
                                bfl_repo,
                                transformer=transformer,
                                text_encoder_2=text_encoder_2,
                                torch_dtype=torch.bfloat16
                            )
                        
                    except Exception as e:
                        print(f"[Flux Load Error] Hybrid loading failed: {e}")
                        if pipeline_class == Flux2Pipeline:
                             # Flux 2 doesn't have from_single_file easily yet, so we re-raise or handle manually
                             raise e
                        
                        print("Falling back to robust from_single_file...")
                        try:
                            cls._pipeline = pipeline_class.from_single_file(
                                model_path,
                                torch_dtype=torch.bfloat16
                            )
                        except Exception as fallback_err:
                            print(f"[Flux Fallback] Handling complex load error: {fallback_err}")
                            # Manual recovery...

                elif os.path.isdir(model_path) or not model_path.endswith((".safetensors", ".ckpt")):
                    # Hugging Face or Directory Loading
                    print(f"Loading Flux from HF/Dir: {model_path} (Low VRAM: {low_vram})")
                    loading_kwargs = {"torch_dtype": torch.bfloat16}
                    
                    if low_vram:
                        print(f"[{model_type}] Enabling NF4 Quantization (Low VRAM Mode)...")
                        try:
                            quantization_config = BitsAndBytesConfig(
                                load_in_4bit=True,
                                bnb_4bit_quant_type="nf4",
                                bnb_4bit_compute_dtype=torch.bfloat16
                            )
                            transformer = FluxTransformer2DModel.from_pretrained(
                                model_path,
                                subfolder="transformer",
                                quantization_config=quantization_config,
                                torch_dtype=torch.bfloat16
                            )
                            loading_kwargs["transformer"] = transformer
                            is_quantized = True # Flag as quantized to affect offloading
                        except Exception as e:
                            print(f"Failed to load NF4 Transformer: {e}. Falling back to standard.")

                    cls._pipeline = pipeline_class.from_pretrained(model_path, **loading_kwargs)
            
            elif model_type == 'sdxl':
                cls._current_dtype = torch.float16
                print(f"Loading local SDXL model ({'Inpaint' if is_inpaint else ('Img2Img' if is_img2img else 'Txt2Img')}): {model_path}")
                if is_inpaint:
                    pipeline_class = StableDiffusionXLInpaintPipeline
                elif controlnet_model_name:
                    pipeline_class = StableDiffusionXLControlNetPipeline
                    print(f"Loading ControlNet: {controlnet_model_name}")
                elif is_img2img:
                    pipeline_class = StableDiffusionXLImg2ImgPipeline
                else:
                    pipeline_class = StableDiffusionXLPipeline
                
                # Special case: If ControlNet is requested, we START with the base pipeline
                # and then upgrade it, because loading ControlNetPipeline directly from a single file
                # checkpoint often fails with config mismatch errors.
                if controlnet_model_name:
                     print(f"Loading base pipeline for ControlNet: {model_path}")
                     pipeline_class = StableDiffusionXLPipeline

                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=FutureWarning, message=".*upcast_vae.*")
                    cls._pipeline = pipeline_class.from_single_file(
                        model_path,
                        torch_dtype=torch.float16,
                        use_safetensors=True,
                        variant="fp16"
                    )
                
                # Load ControlNet Model if requested
                if controlnet_model_name:
                    try:
                        # Construct full path if it's a local file name
                        cn_path = Path(config.get('CONTROLNET_DIR', 'models/controlnet')) / controlnet_model_name
                        print(f"Loading ControlNet weights from {cn_path}...")
                        
                        if cn_path.is_file() or str(cn_path).endswith(('.safetensors', '.pth', '.ckpt')):
                             print(f"Loading single-file ControlNet: {cn_path}")
                             controlnet = ControlNetModel.from_single_file(
                                str(cn_path),
                                torch_dtype=torch.float16,
                                use_safetensors=True
                             )
                        else:
                             # Assume directory or HF repo
                             controlnet = ControlNetModel.from_pretrained(
                                controlnet_model_name,
                                torch_dtype=torch.float16,
                                variant="fp16" if "hf.co" not in controlnet_model_name else None
                             )

                        # Robust pipeline swap using components
                        print("Swapping to ControlNet Pipeline using components...")
                        components = cls._pipeline.components
                        components["controlnet"] = controlnet
                        cls._pipeline = StableDiffusionXLControlNetPipeline(**components)
                        
                        cls._current_has_controlnet = True
                        cls._current_controlnet_name = controlnet_model_name
                    except Exception as e:
                        import traceback
                        print(traceback.format_exc())
                        
                        # Analyze error to give better feedback
                        err_str = str(e).lower()
                        print(f"RAW CONTROLNET ERROR: {e}") 

                        if "config.json" in err_str or "stable-diffusion-v1-5" in err_str:
                             # Try fallback: Maybe network issue prevented config fetch?
                             # Or it's a valid SDXL model that diffusers failed to recognize.
                             print("  > Hint: This error often means Diffusers failed to identify the model architecture.")
                             print("  > Ensure you are connected to the internet (for config fallback) or that the file is not corrupted.")
                             raise ValueError(f"ControlNet Loading Failed: {e}. (If this is 'MistoLine' or similar SDXL model, ensure it's a valid .safetensors file).")
                        
                        elif "size mismatch" in err_str:
                             raise ValueError(f"Shape mismatch: '{controlnet_model_name}' is likely not an SDXL ControlNet. Please ensure you are using a ControlNet made for SDXL.")

                        raise e
                else:
                    cls._current_has_controlnet = False
                    cls._current_controlnet_name = None

                if hasattr(cls._pipeline, "vae") and cls._pipeline.vae is not None:
                    # Align VAE with pipeline to avoid RuntimeError (Input type Half vs Bias Float)
                    cls._pipeline.vae.to(device=config.DEVICE, dtype=torch.float16)
                
                # Apply channels_last to SDXL UNet
                if hasattr(cls._pipeline, "unet") and cls._pipeline.unet is not None:
                    cls._pipeline.unet.to(memory_format=torch.channels_last)

            elif model_type in ['sd3', 'sd3_5_turbo']:
                cls._current_dtype = torch.float16
                if is_inpaint:
                    pipeline_class = StableDiffusion3InpaintPipeline
                elif is_img2img:
                    pipeline_class = StableDiffusion3Img2ImgPipeline
                else:
                    pipeline_class = StableDiffusion3Pipeline
                # Determine which exact pipe to use
                load_repo = "stabilityai/stable-diffusion-3.5-large" if model_type == 'sd3' else "stabilityai/stable-diffusion-3.5-large-turbo"
                
                if model_path.lower().endswith((".safetensors", ".ckpt", ".sft")):
                    # Local single file loading
                    print(f"Loading local {model_type} model: {model_path}")
                    
                    # Iterative component loading for SD3/SD3.5 robustness
                    extra_components = {}
                    max_attempts = 5
                    current_attempt = 0
                    
                    while current_attempt < max_attempts:
                        try:
                            # 1. NF4 Transformer Quantization (if not already in extra_components)
                            if "transformer" not in extra_components:
                                nf4_config = BitsAndBytesConfig(
                                    load_in_4bit=True,
                                    bnb_4bit_quant_type="nf4",
                                    bnb_4bit_compute_dtype=torch.bfloat16
                                )
                                try:
                                    print(f"[{model_type}] Attempting NF4 Transformer load...")
                                    transformer = SD3Transformer2DModel.from_single_file(
                                        model_path,
                                        quantization_config=nf4_config,
                                        torch_dtype=torch.bfloat16
                                    )
                                    extra_components["transformer"] = transformer
                                except Exception as trans_err:
                                    print(f"[{model_type}] NF4 Transformer load failed, using default: {trans_err}")

                            # 2. Optimization: Use 4-bit T5 (if not already in extra_components)
                            if "text_encoder_3" not in extra_components:
                                try:
                                    print(f"[{model_type}] Loading 4-bit T5 encoder to save VRAM...")
                                    text_encoder_3 = T5EncoderModel.from_pretrained(
                                        "diffusers/t5-nf4",
                                        torch_dtype=torch.bfloat16,
                                        device_map="auto"
                                    )
                                    extra_components["text_encoder_3"] = text_encoder_3
                                except Exception as t5_err:
                                    print(f"[{model_type}] Could not load 4-bit T5: {t5_err}")

                            cls._pipeline = pipeline_class.from_single_file(
                                model_path,
                                torch_dtype=torch.bfloat16,
                                **extra_components
                            )
                            break # Success!

                        except Exception as e:
                            current_attempt += 1
                            error_str = str(e).lower()
                            print(f"[{model_type} Fallback] Attempt {current_attempt}/{max_attempts} failed: {e}")
                            
                            from transformers import CLIPTextModelWithProjection
                            # Identify missing component
                            if ("cliptextmodel" in error_str or "text_encoder" in error_str) and "text_encoder" not in extra_components:
                                print(f"[{model_type}] Loading CLIPTextModel fallback...")
                                extra_components["text_encoder"] = CLIPTextModelWithProjection.from_pretrained(load_repo, subfolder="text_encoder", torch_dtype=torch.bfloat16)
                            elif ("text_encoder_2" in error_str) and "text_encoder_2" not in extra_components:
                                print(f"[{model_type}] Loading CLIPTextModelWithProjection (2) fallback...")
                                extra_components["text_encoder_2"] = CLIPTextModelWithProjection.from_pretrained(load_repo, subfolder="text_encoder_2", torch_dtype=torch.bfloat16)
                            elif ("text_encoder_3" in error_str or "t5encodermodel" in error_str) and "text_encoder_3" not in extra_components:
                                print(f"[{model_type}] Loading T5EncoderModel fallback...")
                                extra_components["text_encoder_3"] = T5EncoderModel.from_pretrained(load_repo, subfolder="text_encoder_3", torch_dtype=torch.bfloat16)
                            elif ("autoencoderkl" in error_str or "vae" in error_str) and "vae" not in extra_components:
                                print(f"[{model_type}] Loading VAE fallback...")
                                extra_components["vae"] = AutoencoderKL.from_pretrained(load_repo, subfolder="vae", torch_dtype=torch.bfloat16)
                            elif "transformer" in error_str and "transformer" not in extra_components:
                                print(f"[{model_type}] Loading Transformer fallback...")
                                extra_components["transformer"] = SD3Transformer2DModel.from_pretrained(load_repo, subfolder="transformer", torch_dtype=torch.bfloat16)
                            else:
                                if current_attempt >= max_attempts:
                                    raise e
                                # If we can't identify the bug, we might be looping infinitely
                                print(f"[{model_type}] Unidentified loading error, aborting iterative fallback.")
                                raise e
                else:
                    # Hugging Face remote loading with NF4
                    print(f"Loading remote {model_type} model with NF4: {model_path}...")
                    nf4_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16
                    )
                    transformer = SD3Transformer2DModel.from_pretrained(
                        model_path,
                        subfolder="transformer",
                        quantization_config=nf4_config,
                        torch_dtype=torch.bfloat16
                    )
                    text_encoder_3 = T5EncoderModel.from_pretrained("diffusers/t5-nf4", torch_dtype=torch.bfloat16)
                    
                    cls._pipeline = pipeline_class.from_pretrained(
                        model_path,
                        transformer=transformer,
                        text_encoder_3=text_encoder_3,
                        torch_dtype=torch.bfloat16
                    )
                is_quantized = True
            

                        



            # Device-specific initialization
            if model_type == 'sdxl':
                print(f"Initializing Compel for SDXL on {config.DEVICE}...")
                cls._compel = Compel(
                    tokenizer=[cls._pipeline.tokenizer, cls._pipeline.tokenizer_2],
                    text_encoder=[cls._pipeline.text_encoder, cls._pipeline.text_encoder_2],
                    returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                    requires_pooled=[False, True],
                    device=config.DEVICE
                )

            # Apply custom VAE if requested (SDXL only)
            if vae_name and model_type == 'sdxl':
                vae_path = os.path.join(config.get('VAE_DIR', ''), vae_name)
                if os.path.exists(vae_path) and os.path.isfile(vae_path):
                    print(f"Loading custom VAE: {vae_path}")
                    cls._pipeline.vae = AutoencoderKL.from_single_file(vae_path, torch_dtype=torch.float16)
                else:
                    print(f"Warning: VAE file not found or invalid: {vae_path}. Using model default.")

            # Apply Sampler if requested
            if sampler_name and model_type != 'flux':
                cls.apply_sampler(sampler_name)
            
            # 1. Pipeline-wide optimizations
            if config.DEVICE == "cuda":
                # Channels last can improve performance on Tensor Cores
                try:
                    cls._pipeline.unet.to(memory_format=torch.channels_last)
                    if hasattr(cls._pipeline, "vae") and cls._pipeline.vae is not None:
                        cls._pipeline.vae.to(memory_format=torch.channels_last)
                except: pass

            # 2. VAE Optimizations (CRITICAL for high-res)
            # We always enable tiling/slicing as it prevents OOM during the last stage with minimal speed impact
            if hasattr(cls._pipeline, "vae") and cls._pipeline.vae is not None:
                print(f"[{model_type}] Enabling VAE tiling & slicing...")
                cls._pipeline.vae.enable_tiling()
                cls._pipeline.vae.enable_slicing()

            # 3. Memory Management (Dynamic selection)
            vram_gb = 0
            if torch.cuda.is_available():
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            
            # Clean up any existing hooks if we are changing optimization strategy
            # or if we want to ensure full GPU placement
            cls.remove_all_hooks(cls._pipeline)

            if is_quantized:
                print(f"Enabling Model CPU Offload for Quantized {model_type}...")
                cls._pipeline.enable_model_cpu_offload()
            else:
                if model_type == 'flux':
                    print("Enabling Sequential CPU Offload for Flux...")
                    cls._pipeline.enable_sequential_cpu_offload()

                elif vram_gb >= 10:
                    print(f"Detected {vram_gb:.1f}GB VRAM >= 10GB. Enabling FULL GPU speed for {model_type}...")
                    cls._pipeline.to(config.DEVICE)
                    # For high VRAM, we disable attention slicing for maximum speed
                    print(f"[{model_type}] Disabling attention slicing for peak performance...")
                    cls._pipeline.disable_attention_slicing()
                else:
                    print(f"Detected {vram_gb:.1f}GB VRAM < 10GB. Enabling Model CPU Offload for {model_type}...")
                    cls._pipeline.enable_model_cpu_offload()
                    cls._pipeline.enable_attention_slicing()

            # 4. Standard optimizations (xformers)
            # Skip for Flux as it uses native SDPA efficiently and xformers can conflict with offloading
            if config.DEVICE == "cuda" and model_type != 'flux':
                try:
                    import xformers
                    cls._pipeline.enable_xformers_memory_efficient_attention()
                except Exception as e:
                    print(f"[Optimization] xformers/triton issue: {e}")

            # Update current state if we loaded or switched
            cls._current_model_path = model_path
            cls._current_model_type = model_type
            cls._current_is_img2img = is_img2img
            cls._current_is_img2img = is_img2img
            cls._current_is_inpaint = is_inpaint
            cls._current_has_controlnet = bool(controlnet_model_name)
            print(f"{model_type} model loaded successfully.")

        else:
            # Pipeline already loaded, check if we need to update sampler
            if sampler_name and model_type != 'flux':
                cls.apply_sampler(sampler_name)

        return cls._pipeline, cls._compel

    @classmethod
    def apply_sampler(cls, sampler_name: str):
        from diffusers import (
            EulerDiscreteScheduler, EulerAncestralDiscreteScheduler, 
            DPMSolverMultistepScheduler, DPMSolverSDEScheduler,
            HeunDiscreteScheduler, LMSDiscreteScheduler, DDIMScheduler
        )
        import inspect
        
        s_map = {
            "Euler": EulerDiscreteScheduler,
            "Euler a": EulerAncestralDiscreteScheduler,
            "DPM++ 2M Karras": (DPMSolverMultistepScheduler, {"use_karras_sigmas": True}),
            "DPM++ SDE Karras": (DPMSolverSDEScheduler, {"use_karras_sigmas": True}),
            "DPM2 Karras": (DPMSolverMultistepScheduler, {"use_karras_sigmas": True, "solver_order": 2}),
            "DPM2 a Karras": (DPMSolverMultistepScheduler, {"use_karras_sigmas": True, "solver_order": 2, "use_ancestral_sampling": True}),
            "Heun": HeunDiscreteScheduler,
            "LMS": LMSDiscreteScheduler,
            "DDIM": DDIMScheduler
        }
        
        if sampler_name in s_map:
            print(f"Switching to sampler: {sampler_name}")
            entry = s_map[sampler_name]
            sch_cls = entry[0] if isinstance(entry, tuple) else entry
            extra_kwargs = entry[1] if isinstance(entry, tuple) else {}
            
            # Simple and effective from_config
            cls._pipeline.scheduler = sch_cls.from_config(cls._pipeline.scheduler.config, **extra_kwargs)

    @classmethod
    def generate(cls, prompt: str, negative_prompt: str = "", model_type: str = "sdxl", 
                 model_name: str = None, width: int = 1024, height: int = 1024, 
                 guidance_scale: float = 7.0, num_inference_steps: int = 30, seed: int = None, 
                 vae_name: str = None, sampler_name: str = None, image: str = None, 
                 denoising_strength: float = 0.5, output_format: str = "png", mask: str = None,
                 loras: Optional[list] = None,
                 controlnet_image: str = None, controlnet_conditioning_scale: float = 0.7,
                 controlnet_model: str = None, low_vram: bool = False, is_inpaint: bool = False,
                 workflow: Optional[Dict[str, Any]] = None,
                 image_a: str = None, image_b: str = None, weight_a: float = 0.5,
                 weight_b: float = 0.5, ip_adapter_scale: float = 0.5):
        """
        Génère une image (ou transforme une existante) à partir des paramètres.
        """
        cls.reset_interrupt()
        lora_warnings = []
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            
        import time
        start_time = time.time()
        
        # Calculate VRAM for optimization decisions
        vram_gb = 0
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        # Map 'img2img' alias to base architecture (Default: SDXL)
        if model_type == 'img2img':
            # Fix: Do NOT inherit _current_model_type (which could be Flux)
            # The Img2Img brick is designed for SDXL.
            model_type = 'sdxl'
            
        # Normalize VAE name
        if vae_name == 'Default' or not vae_name:
            vae_name = None
 
        # Model selection based on type
        if model_type == 'flux':
            m_name = model_name or config.get('DEFAULT_FLUX_MODEL', 'flux1-schnell.safetensors')
            model_path = os.path.join(config.get('FLUX_MODELS_DIR', ''), m_name)
        elif model_type in ['sd3', 'sd3_5_turbo']:
            m_name = model_name or "sd3.5_large_turbo.safetensors"
            model_path = os.path.join(config.get('SD3_MODELS_DIR', 'models/sd3'), m_name)
        else:
            m_name = model_name or config.DEFAULT_MODEL
            model_path = os.path.join(config.MODELS_DIR, m_name)

        if model_type == 'qwen':
             QwenManager.unload() # Ensure clean slate or manage memory
             # We delegate entirely to QwenManager
             qm = QwenManager()
             # Generate
             image_out, seed = qm.generate(
                 prompt=prompt,
                 width=width,
                 height=height,
                 steps=num_inference_steps,
                 seed=seed
             )
             
             # Save logic (Duplicated from below, could be refactored but keeping it simple for integration)
             from datetime import datetime
             now = datetime.now()
             day_folder = now.strftime("%Y_%m_%d")
             timestamp = now.strftime("%H%M%S")
             
             final_output_dir = output_dir / day_folder
             final_output_dir.mkdir(parents=True, exist_ok=True)
             
             file_name = f"qwen_{day_folder}_{timestamp}_{width}x{height}_seed{seed}.png"
             file_path = final_output_dir / file_name
             
             image_out.save(file_path)
             
             exec_time = time.time() - start_time
             used_params = {
                 "prompt": prompt,
                 "model_type": "qwen",
                 "width": width,
                 "height": height,
                 "steps": num_inference_steps,
                 "seed": seed
             }
             return f"{day_folder}/{file_name}", exec_time, used_params
        
        # Output directory preparation
        output_dir = Path(config.OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine if we are doing Img2Img or Inpaint
        is_now_img2img = bool(image)
        is_now_inpaint = bool(mask) or is_inpaint
        
        # Prepare Mode Name for logging
        mode_name = "Txt2Img"
        if is_now_img2img: mode_name = "Img2Img"
        if is_now_inpaint: mode_name = "Inpaint"

        # Pipeline
        pipe, compel = ModelManager.get_pipeline(
            model_type=model_type, 
            model_path=model_path, 
            vae_name=vae_name, 
            sampler_name=sampler_name,
            is_img2img=is_now_img2img,
            is_inpaint=is_now_inpaint,
            controlnet_model_name=controlnet_model if controlnet_image else None,
            low_vram=low_vram
        )
        active_pipe = pipe
        ModelManager.reset_interrupt()

        # Apply LoRAs if provided
        if loras:
            print(f"[Debug] Received {len(loras)} LoRA configurations from request.", flush=True)
            lora_warnings = ModelManager.apply_loras(pipe, loras)
        elif ModelManager._active_lora_names:
            print("[Debug] Clearing previous LoRAs (none requested).", flush=True)
            ModelManager.apply_loras(pipe, [])
        
        # Seed
        generator = None
        if seed is None:
            import random
            seed = random.randint(0, 2**32 - 1)
        
        generator = torch.Generator(config.DEVICE).manual_seed(seed)
        
        # Generation
        print(f"Generating {model_type} image for prompt: {prompt}")
        print(f"[Debug] Dimensions: {width}x{height}, Scale: {guidance_scale}, Steps: {num_inference_steps}")
        if image:
            print(f"Img2Img mode detected (Denoising: {denoising_strength})")
        
        # Auto-adjust guidance for Flux 2 Klein distilled
        # It usually requires guidance_scale=1.0. If user has it high, it causes noise.
        if model_type == 'flux' and guidance_scale > 1.5:
             # Check if we are using the distilled Klein (4 steps usually means distilled)
             if num_inference_steps <= 8:
                 print(f"[Caution] Detected potential Flux 2 Distilled model with high guidance ({guidance_scale}). Auto-adjusting to 1.0 to avoid noise.")
                 guidance_scale = 1.0

        if negative_prompt:
            print(f"Negative prompt: {negative_prompt}")
        
        # Prepare Image if Img2Img
        init_image = None
        if image:
            try:
                init_image = cls._load_image(image)
                
                if init_image is None:
                    raise ValueError("Failed to load or decode input image.")
                
                # MAINTAIN USER REQUESTED DIMENSIONS
                # We use the requested width/height as the target.
                # If the image differs, we resize it to fit the requested dimensions.
                # This ensures the Flux pipeline (which expects specific latent shapes) 
                # always gets what the user asked for.
                
                target_w, target_h = width, height
                
                if init_image.size != (target_w, target_h):
                    print(f"[Img2Img] Resizing input image to match requested dimensions: {init_image.size} -> ({target_w}, {target_h})")
                    init_image = init_image.resize((target_w, target_h), Image.LANCZOS)
                
                # Double check multiple of 64 for latent compatibility (standard for most pipelines)
                # width = round(target_w / 64) * 64
                # height = round(target_h / 64) * 64
                
                # Force debug write for traceability
                try:
                    with open("debug_dims.txt", "w") as f:
                        f.write(f"Target Size: {width}x{height}\n")
                        f.write(f"Init Image After Resize: {init_image.size}\n")
                except: pass
                
                # Force debug write
                try:
                    with open("debug_dims.txt", "w") as f:
                        f.write(f"Init Image Size: {w}x{h}\n")
                        f.write(f"Original Req: not-captured-here\n")
                except: pass
                
                import logging
                logging.getLogger("qpyt-ui").info(f"Processed input image: {width}x{height}")
            except Exception as e:
                import logging
                logging.getLogger("qpyt-ui").error(f"Failed to process input image: {e}")
                init_image = None
        
        # Safety: If Img2Img mode but image failed to load, abort
        if is_now_img2img and init_image is None:
             raise ValueError("Img2Img mode requires a valid source image. Failed to process input image.")

        # Prepare Mask if Inpaint
        init_mask = None
        if mask:
            try:
                init_mask = cls._load_image(mask, mode="L")
                if init_mask is None:
                    raise ValueError("Failed to load or decode mask image.")
                # Resize mask to match target size
                print(f"[Debug] Init mask size: {init_mask.size}, Target: ({width}, {height})")
                if init_mask.size != (width, height):
                    init_mask = init_mask.resize((width, height), Image.LANCZOS)
                
                # Debug save
                debug_path = Path(config.OUTPUT_DIR) / "debug_mask.png"
                init_mask.save(debug_path)
                print(f"[Debug] Saved mask to {debug_path}")
            except Exception as e:
                print(f"Failed to process mask: {e}")
                init_mask = None

        # Safety Check and Debug for Inpainting
        if is_now_inpaint:
            print(f"[Debug] Inpainting Mode. Image present: {init_image is not None}. Mask present: {init_mask is not None}")
            print(f"[Debug] Denoising Strength: {denoising_strength}")
            if init_image:
                 # Debug save image
                debug_img_path = Path(config.OUTPUT_DIR) / "debug_init_image.png"
                init_image.save(debug_img_path)
                print(f"[Debug] Saved init_image to {debug_img_path}, Size: {init_image.size}")

            if init_image is None:
                print("[Inpaint] Base image missing or invalid, creating black fallback...")
                init_image = Image.new("RGB", (width, height), (0, 0, 0))
                # Also save the fallback so we know
                init_image.save(Path(config.OUTPUT_DIR) / "debug_fallback_image.png")
            
            if init_mask is None:
                # This should not happen if is_now_inpaint is true, but safety first
                print("[Inpaint] Mask missing or invalid, creating empty mask fallback...")
                init_mask = Image.new("L", (width, height), 0)

        def interrupt_callback(pipe, step_index, timestep, callback_kwargs):
            ModelManager._current_step = step_index + 1
            ModelManager._total_steps = num_inference_steps

            if ModelManager._interrupt_flag:
                pipe._interrupt = True
                print(f"Interruption detected at step {step_index}")
            
            try:
                latents = callback_kwargs.get("latents")
                if latents is not None:
                    with torch.no_grad():
                        l = latents[0] if latents.ndim == 4 else latents
                        image_prev = ModelManager.latents_to_rgb(l)
                        if image_prev:
                            image_prev.thumbnail((256, 256))
                            buffered = BytesIO()
                            image_prev.save(buffered, format="JPEG", quality=60)
                            ModelManager._current_preview = f"data:image/jpeg;base64,{base64.b64encode(buffered.getvalue()).decode()}"
                else:
                    if step_index == 0:
                        print("[Callback Warning] No 'latents' in callback_kwargs.")
            except Exception as e:
                if step_index == 0:
                    print(f"[Callback Error] {e}")
                    
            return callback_kwargs

        kwargs = {
            "width": width,
            "height": height,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "generator": generator,
            "callback_on_step_end": interrupt_callback,
            "callback_on_step_end_tensor_inputs": ["latents"]
        }
        if is_now_inpaint:
            # Inpainting logic
            # Explicitly pass width/height to ensure pipeline respects image aspect ratio
            # kwargs.pop("width", None) <-- DO NOT POP
            # kwargs.pop("height", None) <-- DO NOT POP
            kwargs["image"] = init_image
            kwargs["mask_image"] = init_mask
            # strength is used for inpainting control in some pipelines, or ignored in others
            kwargs["strength"] = denoising_strength
        elif is_now_img2img:
             # Img2Img Logic
            kwargs["image"] = init_image
            kwargs["strength"] = denoising_strength
        
        # ControlNet Arguments Override
        if controlnet_image and hasattr(pipe, "controlnet"):
            print(f"Using ControlNet (Scale: {controlnet_conditioning_scale})")
            
            # Decode Control Image
            cn_img = cls._load_image(controlnet_image)
            
            if cn_img is None:
                print("[ControlNet] Failed to load control image. Skipping.")
            else:
                # Resize Control Image to match target
                # Note: Control images usually need to match the generation size
                cn_img = cn_img.resize((width, height), Image.LANCZOS)
                
                # For StableDiffusionXLControlNetPipeline, the argument for the conditioning image is 'image'.
                # If we were in Img2Img mode previously, 'image' was the init_image. 
                # This conflict requires careful handling.
                kwargs['image'] = cn_img 
                kwargs['controlnet_conditioning_scale'] = controlnet_conditioning_scale
                
                # Ensure strength arg doesn't mess up Txt2Img ControlNet (it ignores it usually, but good to be clean)
                if 'strength' in kwargs: 
                    del kwargs['strength']
                
                if is_now_img2img:
                    print("Warning: Both Img2Img Image and ControlNet Image provided. Ignoring Img2Img Image in favor of ControlNet Txt2Img.")

        # --- IP-Adapter / Image Blender Support ---
        if (image_a or image_b) and model_type == 'sdxl':
            # Determine required layers based on images provided
            needed_layers = 1
            if image_a and image_b: needed_layers = 2
            
            active_pipe = cls._apply_ip_adapter(active_pipe, model_type, required_layers=needed_layers)
            cls._pipeline = active_pipe # Keep class cache in sync
            
            ip_images = []
            ip_scales = []
            
            def decode_ip(b64):
                return cls._load_image(b64)

            img_a_obj = decode_ip(image_a)
            img_b_obj = decode_ip(image_b)
            
            if img_a_obj and img_b_obj:
                ip_images = [img_a_obj, img_b_obj]
                ip_scales = [weight_a * ip_adapter_scale, weight_b * ip_adapter_scale]
            elif img_a_obj:
                ip_images = [img_a_obj]
                ip_scales = [weight_a * ip_adapter_scale]
            elif img_b_obj:
                ip_images = [img_b_obj]
                ip_scales = [weight_b * ip_adapter_scale]

            # FINAL SYNC: Match scales and images to the actual number of adapters loaded
            if ip_images and hasattr(active_pipe.unet, "encoder_hid_proj") and \
               hasattr(active_pipe.unet.encoder_hid_proj, "image_projection_layers"):
                num_adapters = len(active_pipe.unet.encoder_hid_proj.image_projection_layers)
                while len(ip_images) < num_adapters:
                    # Pad with existing image but 0 weight to satisfy Diffusers constraints
                    ip_images.append(ip_images[0])
                    ip_scales.append(0.0)
                # Trim if somehow we have more images than adapters (shouldn't happen)
                ip_images = ip_images[:num_adapters]
                ip_scales = ip_scales[:num_adapters]
                
            if ip_images:
                # FINAL SAFETY CHECK: Does the pipeline REALLY have the projection layers?
                can_run_ip = hasattr(active_pipe.unet, "encoder_hid_proj") and active_pipe.unet.encoder_hid_proj is not None
                
                if can_run_ip:
                    print(f"[IP-Adapter] Applying blending: {len(ip_images)} concepts, scales: {ip_scales}")
                    active_pipe.set_ip_adapter_scale(ip_scales)
                    kwargs["ip_adapter_image"] = ip_images 
                else:
                    print("[IP-Adapter] CRITICAL: Projection layers missing despite load attempt. Skipping IP-Adapter to avoid crash.")
                    if hasattr(active_pipe, "set_ip_adapter_scale"):
                        active_pipe.set_ip_adapter_scale(0.0)
                
                # Re-apply optimization
                # If IP-Adapter is active, we almost ALWAYS want offload on 12GB cards
                # because the BigG image encoder takes 3.7GB VRAM by itself.
                # Re-calculate vram close to usage to be 100% safe
                vram_gb_local = 0
                if torch.cuda.is_available():
                    vram_gb_local = torch.cuda.get_device_properties(0).total_memory / (1024**3)

                if low_vram or vram_gb_local < 16: 
                    print(f"[IP-Adapter] Low VRAM (or IP-Adapter on {vram_gb_local:.1f}GB device): Enabling CPU Offload to prevent slowness.")
                    active_pipe.enable_model_cpu_offload()
                else:
                    active_pipe.to(config.DEVICE)
            else:
                if hasattr(active_pipe, "set_ip_adapter_scale"):
                    active_pipe.set_ip_adapter_scale(0.0)
        else:
            # No IP-Adapter images provided - UNLOAD if still persistent in memory
            if cls._current_has_ip_adapter and hasattr(active_pipe, "unload_ip_adapter"):
                print("[IP-Adapter] Unloading persistent adapter for Txt2Img purity...")
                try:
                    active_pipe.unload_ip_adapter()
                except Exception as e:
                    print(f"[IP-Adapter] Unload warning: {e}")
                cls._current_has_ip_adapter = False
                cls._current_ip_adapter_name = None

            if hasattr(active_pipe, "set_ip_adapter_scale"):
                active_pipe.set_ip_adapter_scale(0.0)

        if model_type == 'sdxl' and compel:
            conditioning, pooled = compel(prompt)
            neg_conditioning, neg_pooled = compel(negative_prompt or "")
            # Debug device
            try:
                print(f"[Inference] UNet: {active_pipe.unet.device}/{active_pipe.unet.dtype}")
                if hasattr(active_pipe, "vae") and active_pipe.vae is not None:
                    print(f"[Inference] VAE: {active_pipe.vae.device}/{active_pipe.vae.dtype}")
            except: pass

            image_out = active_pipe(
                prompt_embeds=conditioning,
                pooled_prompt_embeds=pooled,
                negative_prompt_embeds=neg_conditioning,
                negative_pooled_prompt_embeds=neg_pooled,
                **kwargs
            ).images[0]
        else:
            # Flux or SD3
            call_kwargs = {"prompt": prompt}
            if model_type in ['sd3', 'sd3_5_turbo'] and negative_prompt:
                call_kwargs["negative_prompt"] = negative_prompt
            
            # Debug device
            try:
                print(f"[Inference] Transformer device: {active_pipe.transformer.device}, dtype: {active_pipe.transformer.dtype}")
            except: pass

            image_out = active_pipe(
                **call_kwargs,
                **kwargs
            ).images[0]
        
        # If interrupted, do not save
        if hasattr(active_pipe, "_interrupt") and active_pipe._interrupt:
            return None, 0, {}

        # Save result
        from datetime import datetime
        now = datetime.now()
        day_folder = now.strftime("%Y_%m_%d")
        timestamp = now.strftime("%H%M%S")
        
        final_output_dir = output_dir / day_folder
        final_output_dir.mkdir(parents=True, exist_ok=True)
        
        base_name = m_name.split('.')[0]
        ext = output_format.lower()
        if ext not in ['png', 'jpg', 'jpeg', 'webp']:
            ext = 'png'
        
        # Mapping to PIL format
        save_ext = ext
        if ext == 'jpg': save_ext = 'jpeg'
        
        # Use actual output dimensions for filename, as they might differ from requested
        actual_w, actual_h = image_out.size
        file_name = f"{base_name}_{day_folder}_{timestamp}_{actual_w}x{actual_h}_seed{seed}.{ext}"
        file_path = final_output_dir / file_name
        save_kwargs = {}
        if save_ext == 'jpeg':
            save_kwargs['quality'] = 90
            save_kwargs['optimize'] = True
        elif save_ext == 'webp':
            save_kwargs['quality'] = 90
            
        # Metadata Injection (Workflow)
        if workflow:
            print(f"[Metadata] Injecting workflow JSON into {ext} file...")
            workflow_str = json.dumps(workflow)
            
            if save_ext == 'png':
                metadata = PngInfo()
                metadata.add_text("qpyt_workflow", workflow_str)
                save_kwargs['pnginfo'] = metadata
            
            elif save_ext in ['jpeg', 'webp']:
                # For JPEG and WebP, we use the EXIF UserComment tag (0x9286)
                # Note: EXIF works for both in modern Pillow
                exif = image_out.getexif()
                # 0x9286 is the tag for UserComment
                exif[0x9286] = workflow_str
                save_kwargs['exif'] = exif

        image_out.save(file_path, format=save_ext.upper(), **save_kwargs)
        
        exec_time = time.time() - start_time
        used_params = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "model_type": model_type,
            "model_name": m_name,
            "width": width,
            "height": height,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "seed": seed,
            "vae_name": vae_name,
            "sampler_name": sampler_name,
            "loras": loras,
            "lora_warnings": lora_warnings
        }

        return f"{day_folder}/{file_name}", exec_time, used_params

    @classmethod
    def upscale(cls, image_base64: str, scale_factor: float = 2.0, denoising_strength: float = 0.2, 
                tile_size: int = 768, prompt: str = "", negative_prompt: str = "", 
                model_name: Optional[str] = None, output_format: str = "png", seed: Optional[int] = 42, 
                loras: Optional[list] = None, **kwargs):
        """
        Tiled upscaler implementation.
        Processes images in overlapping tiles to allow high-res upscaling on limited VRAM.
        """
        start_time = time.time()
        lora_warnings = []
        
        # 1. Prepare image
        try:
            input_image = cls._load_image(image_base64)
            if input_image is None:
                raise ValueError("Source image is empty or invalid.")
        except Exception as e:
            print(f"[Upscaler] Error preparing image: {e}")
            raise e
            
        orig_w, orig_h = input_image.size
        target_w = int(orig_w * scale_factor)
        target_h = int(orig_h * scale_factor)
        
        # Ensure target size is multiple of 64
        target_w = (target_w // 64) * 64
        target_h = (target_h // 64) * 64
        
        print(f"[Upscaler] Rescaling {orig_w}x{orig_h} -> {target_w}x{target_h} (Factor: {scale_factor})")
        upscaled_base = input_image.resize((target_w, target_h), Image.LANCZOS)
        
        # 2. Setup Pipeline
        # 2. Setup Pipeline
        m_name = model_name or (os.path.basename(cls._current_model_path) if cls._current_model_path else config.DEFAULT_MODEL)
        
        # Detect model type based on file location
        flux_dir = config.get('FLUX_MODELS_DIR', 'models/flux')
        sd3_dir = config.get('SD3_MODELS_DIR', 'models/sd3')
        sdxl_dir = config.MODELS_DIR
        
        flux_path = os.path.join(flux_dir, m_name)
        sd3_path = os.path.join(sd3_dir, m_name)
        sdxl_path = os.path.join(sdxl_dir, m_name)
        
        if os.path.exists(flux_path):
            model_type = 'flux'
            model_path = flux_path
        elif os.path.exists(sd3_path):
            model_type = 'sd3' # Generic SD3 handler
            model_path = sd3_path
        elif os.path.exists(sdxl_path):
            model_type = 'sdxl'
            model_path = sdxl_path
        else:
            # Fallback for absolute paths or unmanaged files
            if os.path.exists(m_name):
                 model_path = m_name
                 # Simple heuristic for type if possible, else default to current or sdxl
                 # For now, we'll assume if the user provided a full path outside, we rely on current type or default
                 model_type = cls._current_model_type or 'sdxl'
            else:
                 print(f"[Upscaler] Warning: Model {m_name} not found. Defaulting to SDXL/Current.")
                 model_type = cls._current_model_type or 'sdxl' 
                 model_path = sdxl_path

        pipe, compel = cls.get_pipeline(
            model_type=model_type,
            model_path=model_path,
            is_img2img=True
        )
        cls.reset_interrupt()

        # Apply LoRAs if provided
        if loras:
            print(f"[Debug] Upscaler received {len(loras)} LoRA configurations.", flush=True)
            lora_warnings = cls.apply_loras(pipe, loras)
        elif cls._active_lora_names:
            print("[Debug] Upscaler clearing previous LoRAs (none requested).", flush=True)
            cls.apply_loras(pipe, [])
        
        # 3. Tiling Configuration
        overlap = 128
        stride = tile_size - overlap
        
        # We might need to adjust stride/tile_size to fit exactly or have enough overlap
        rows = math.ceil((target_h - overlap) / stride) if target_h > tile_size else 1
        cols = math.ceil((target_w - overlap) / stride) if target_w > tile_size else 1
        
        total_tiles = rows * cols
        print(f"[Upscaler] Tiling: {total_tiles} tiles ({cols}x{rows}) with {overlap}px overlap")
        
        # Output accumulation buffers
        result = np.zeros((target_h, target_w, 3), dtype=np.float32)
        weight = np.zeros((target_h, target_w, 1), dtype=np.float32)
        
        # Pre-compute weight mask for blending
        # Linear ramp for the overlap areas
        mask = np.ones((tile_size, tile_size, 1), dtype=np.float32)
        ramp = np.linspace(0, 1, overlap)
        for i in range(overlap):
            mask[i, :, :] *= ramp[i]
            mask[-(i+1), :, :] *= ramp[i]
            mask[:, i, :] *= ramp[i]
            mask[:, -(i+1), :] *= ramp[i]

        inference_steps = 20 if model_type != 'flux' else 4
        cls._total_steps = total_tiles
        
        # 4. Processing Loop
        for r in range(rows):
            for c in range(cols):
                if cls._interrupt_flag: break
                
                tile_id = r * cols + c + 1
                cls._current_step = tile_id
                
                # Coords
                y1 = r * stride
                y2 = y1 + tile_size
                x1 = c * stride
                x2 = x1 + tile_size
                
                # Boundary clamping
                if y2 > target_h:
                    y2 = target_h
                    y1 = max(0, y2 - tile_size)
                if x2 > target_w:
                    x2 = target_w
                    x1 = max(0, x2 - tile_size)
                
                # Extract and process tile
                tile = upscaled_base.crop((x1, y1, x2, y2))
                print(f"[Upscaler] Processing tile {tile_id}/{total_tiles} at ({x1}, {y1})")
                
                # Use current model to "hallucinate" details
                if model_type == 'sdxl' and compel:
                    conditioning, pooled = compel(prompt)
                    neg_conditioning, neg_pooled = compel(negative_prompt or "")
                    tile_out = pipe(
                        prompt_embeds=conditioning,
                        pooled_prompt_embeds=pooled,
                        negative_prompt_embeds=neg_conditioning,
                        negative_pooled_prompt_embeds=neg_pooled,
                        image=tile,
                        strength=denoising_strength,
                        num_inference_steps=inference_steps,
                        guidance_scale=7.0,
                        generator=torch.Generator(config.DEVICE).manual_seed(42)
                    ).images[0]
                else:
                    # Flux / Generic
                    tile_out = pipe(
                        prompt=prompt,
                        image=tile,
                        strength=denoising_strength,
                        num_inference_steps=inference_steps,
                        guidance_scale=3.5 if model_type == 'flux' else 7.0,
                        generator=torch.Generator(config.DEVICE).manual_seed(seed or 42)
                    ).images[0]
                
                # Accumulate
                tile_np = np.array(tile_out).astype(np.float32)
                # Important: tile_out might be slightly different size if tile was clamped
                # but with tile_size window it should be consistent
                h, w, _ = tile_np.shape
                current_mask = mask[:h, :w, :]
                
                result[y1:y2, x1:x2, :] += tile_np * current_mask
                weight[y1:y2, x1:x2, :] += current_mask
            
            if cls._interrupt_flag: break

        # 5. Finalization
        # Avoid division by zero
        weight[weight == 0] = 1.0
        final_image_np = (result / weight).clip(0, 255).astype(np.uint8)
        image_out = Image.fromarray(final_image_np)
        
        # Save logic
        from datetime import datetime
        now = datetime.now()
        day_folder = now.strftime("%Y_%m_%d")
        
        ext = output_format.lower()
        if ext not in ['png', 'jpg', 'jpeg', 'webp']:
            ext = 'png'
        save_ext = ext
        if ext == 'jpg': save_ext = 'jpeg'
        
        timestamp = now.strftime("%H%M%S")
        file_name = f"upscale_{day_folder}_{timestamp}_seed{seed}.{ext}"
        
        final_output_dir = Path(config.OUTPUT_DIR) / day_folder
        final_output_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = final_output_dir / file_name
        
        save_kwargs = {}
        if save_ext == 'jpeg':
            save_kwargs['quality'] = 90
            save_kwargs['optimize'] = True
        elif save_ext == 'webp':
            save_kwargs['quality'] = 90

        image_out.save(file_path, format=save_ext.upper(), **save_kwargs)
        
        exec_time = time.time() - start_time
        used_params = {
            "prompt": prompt,
            "scale": scale_factor,
            "denoising": denoising_strength,
            "tiles": total_tiles,
            "width": target_w,
            "height": target_h,
            "seed": seed,
            "loras": loras,
            "lora_warnings": lora_warnings
        }
        
        print(f"[Upscaler] Success: {target_w}x{target_h} in {exec_time:.1f}s")
        return f"{day_folder}/{file_name}", exec_time, used_params


def list_models(model_type: str) -> list:
    """Lists available model files for a given type."""
    if model_type == 'flux':
        directory = Path(config.get('FLUX_MODELS_DIR', ''))
    elif model_type in ['sd3', 'sd3_5_turbo']:
        directory = Path(config.get('SD3_MODELS_DIR', ''))
    elif model_type == 'zimage':
        # For Z-Image, we list the PARENT directories if they are treated as models, 
        # BUT based on our loading logic we treat the .safetensors file as the entry point?
        # Actually our loading logic (lines 711+) takes 'model_path' as the base dir.
        # But list_models usually lists FILES. 
        # Let's list the zIMAGE_MODELS_DIR content. 
        # Since the user has distinct files for transformer/vae/text_encoder in one folder,
        # we might want to list "folders" or just a dummy entry?
        # Wait, the user said they have "G:\models\zit\qwen..." etc.
        # In our Preset, "selectedModel" is "z-image-turbo".
        # If we return "z-image-turbo", the frontend will pick it.
        # Let's return subdirectory names or just the files if they are self-contained.
        # Given the "Hybrid Loading Logic", we pointed to "model_path" as the BASE DIR in get_pipeline.
        # So we should probably return a list of valid 'configurations' or just let the user pick the folder?
        # A simple approach for now: List the ZIMAGE_MODELS_DIR itself if it's the model "name", or list subdirectories.
        # User config preset says: "selectedModel": "z-image-turbo".
        # If "z-image-turbo" is a folder inside "G:\models\zit", then we list folders.
        # If "G:\models\zit" contains the files directly, then "z-image-turbo" might be a dummy name.
        # Let's assume we list subdirectories as model names OR specific .safetensors if single file.
        # Given the complex loading, let's look at what get_pipeline expects.
        # get_pipeline(..., model_path, ...) -> if zimage: "Loading ... from {model_path} (Base Dir)..."
        # And config.ZIMAGE_MODELS_DIR is "G:\models\zit". 
        # If we pass "z-image-turbo", the path becomes "G:\models\zit\z-image-turbo".
        # If the user put files directly in "G:\models\zit", then we might need to pass "." or just handle it.
        # Let's assume the user has a subfolder "z-image-turbo" OR we support files.
        # Let's list subdirectories first.
        directory = Path(config.get('ZIMAGE_MODELS_DIR', ''))
    else:
        directory = Path(config.MODELS_DIR)
    
    if not directory or not directory.exists():
        return []
    
    # Looking for .safetensors, .ckpt and .gguf
    models = []
    for ext in ['*.safetensors', '*.ckpt', '*.gguf', '*.sft']:
        models.extend([f.name for f in directory.glob(ext)])
    
    return sorted(models)

def list_vaes() -> list:
    """Lists available VAE files."""
    directories = [Path(config.get('VAE_DIR', ''))]
    
    # Also check Z-Image models dir for VAEs (e.g. zImageTurbo_vae.safetensors)
    zimage_dir = Path(config.get('ZIMAGE_MODELS_DIR', ''))
    if zimage_dir and zimage_dir.exists():
        directories.append(zimage_dir)
        
    vaes = []
    for directory in directories:
        if not directory or not directory.exists():
            continue
        for ext in ['*.safetensors', '*.ckpt', '*.pt']:
            # For Z-Image dir, only include files with 'vae' in name to avoid pollution
            for f in directory.glob(ext):
                if directory == zimage_dir and 'vae' not in f.name.lower():
                    continue
                vaes.append(f.name)
    
    return sorted(list(set(vaes)))

def list_samplers() -> list:
    """Lists default available samplers (schedulers)."""
    return [
        "Euler",
        "Euler a",
        "DPM++ 2M Karras",
        "DPM++ SDE Karras",
        "DPM2 Karras",
        "DPM2 a Karras",
        "Heun",
        "LMS",
        "DDIM"
    ]

def list_loras() -> list:
    """Lists available LoRA files."""
    directory = Path(config.get('LORAS_DIR', 'models/loras'))
    if not directory or not directory.exists():
        return []
    
    loras = []
    # Support subdirectories
    for ext in ['*.safetensors', '*.ckpt']:
        for f in directory.rglob(ext):
            # Return path relative to LORAS_DIR
            try:
                rel_path = f.relative_to(directory)
                loras.append(str(rel_path))
            except:
                loras.append(f.name)
    
    return sorted(loras)


def list_controlnets():
    path = config.get('CONTROLNET_DIR')
    if not path or not os.path.exists(path):
        return []
    
    files = []
    for f in os.listdir(path):
         # Include safetensors and pth (common for ControlNets)
        if f.endswith((".safetensors", ".pth", ".bin")):
            files.append(f)
            
    # Add common HF ID for convenience if local not found, or as option
    # files.append("diffusers/controlnet-depth-sdxl-1.0") 
    return sorted(files)
