import torch
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
    StableDiffusion3Pipeline,
    StableDiffusion3Img2ImgPipeline,
    StableDiffusion3InpaintPipeline,
    AutoPipelineForInpainting,
    GGUFQuantizationConfig,
    QuantoConfig,
    FluxTransformer2DModel,
    AutoencoderKL
)
from transformers import CLIPTextModel, T5EncoderModel, T5TokenizerFast
from compel import Compel, ReturnedEmbeddingsType
from typing import Optional, Dict, Any
from core.config import config
import base64
from io import BytesIO
from PIL import Image

# Enable TF32 for faster and memory efficient computation on Ampere+ GPUs
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

class ModelManager:
    _instance = None
    _pipeline = None
    _compel = None
    _current_model_path = None
    _current_model_type = None # 'sdxl', 'flux', 'sd3'
    _current_is_img2img = False
    _current_is_inpaint = False
    _current_dtype = torch.float16
    _interrupt_flag = False
    _current_preview = None # Base64 string of the last preview
    _current_step = 0
    _total_steps = 0
    _weights_t = None
    _biases_t = None

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
            
            # GC first, then clear cache
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                torch.cuda.synchronize()
            print("Memory cleared.")

    @classmethod
    def get_pipeline(cls, model_type: str, model_path: str, vae_name: Optional[str] = None, 
                     sampler_name: Optional[str] = None, is_img2img: bool = False, is_inpaint: bool = False):
        # Normalize VAE name
        if vae_name == 'Default' or not vae_name:
            vae_name = None

        if model_type in ['img2img', 'upscale']:
            # Use current model type if available, otherwise default to sdxl
            model_type = cls._current_model_type or 'sdxl'
            is_img2img = True

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
            if cls._current_model_path == model_path and cls._current_model_type == model_type:
                # If only the task type (Txt2Img vs Img2Img) changed, try a fast structural switch
                if cls._current_is_img2img != is_img2img:
                    print(f"Switching pipeline mode: {'Txt2Img -> Img2Img' if is_img2img else 'Img2Img -> Txt2Img'}...")
                    try:
                        # Map base models for from_pipe
                        base_pipes = {
                            'sdxl': (StableDiffusionXLPipeline, StableDiffusionXLImg2ImgPipeline, StableDiffusionXLInpaintPipeline),
                            'flux': (FluxPipeline, FluxImg2ImgPipeline, FluxInpaintPipeline),
                            'sd3': (StableDiffusion3Pipeline, StableDiffusion3Img2ImgPipeline, StableDiffusion3InpaintPipeline),
                            'sd3_5_turbo': (StableDiffusion3Pipeline, StableDiffusion3Img2ImgPipeline, StableDiffusion3InpaintPipeline)
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

                            if vram_gb >= 10:
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
                if model_path.lower().endswith((".safetensors", ".ckpt", ".sft")):
                    print(f"Loading local Flux model (quantized): {model_path}")
                    bfl_repo = "black-forest-labs/FLUX.1-schnell"
                    try:
                        q_config = QuantoConfig(weights_dtype="float8")
                        transformer = FluxTransformer2DModel.from_single_file(
                            model_path, 
                            quantization_config=q_config,
                            torch_dtype=torch.bfloat16
                        )
                        text_encoder_2 = T5EncoderModel.from_pretrained(bfl_repo, subfolder="text_encoder_2", torch_dtype=torch.bfloat16)
                        
                        cls._pipeline = pipeline_class.from_pretrained(
                            bfl_repo, 
                            transformer=transformer, 
                            text_encoder_2=text_encoder_2, 
                            torch_dtype=torch.bfloat16
                        )
                        is_quantized = True
                    except Exception as e:
                        print(f"Premium Flux load failed: {e}. Trying standard fallback...")
                        # Standard single file fallback
                        text_encoder = CLIPTextModel.from_pretrained(bfl_repo, subfolder="text_encoder", torch_dtype=torch.bfloat16)
                        text_encoder_2 = T5EncoderModel.from_pretrained(bfl_repo, subfolder="text_encoder_2", torch_dtype=torch.bfloat16)
                        vae = AutoencoderKL.from_pretrained(bfl_repo, subfolder="vae", torch_dtype=torch.bfloat16)
                        
                        cls._pipeline = pipeline_class.from_single_file(
                            model_path, 
                            text_encoder=text_encoder,
                            text_encoder_2=text_encoder_2,
                            vae=vae,
                            torch_dtype=torch.bfloat16
                        )
                elif os.path.isdir(model_path):
                    cls._pipeline = pipeline_class.from_pretrained(model_path, torch_dtype=torch.bfloat16)

            elif model_type == 'sdxl':
                cls._current_dtype = torch.float16
                print(f"Loading local SDXL model ({'Inpaint' if is_inpaint else ('Img2Img' if is_img2img else 'Txt2Img')}): {model_path}")
                if is_inpaint:
                    pipeline_class = StableDiffusionXLInpaintPipeline
                elif is_img2img:
                    pipeline_class = StableDiffusionXLImg2ImgPipeline
                else:
                    pipeline_class = StableDiffusionXLPipeline
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=FutureWarning, message=".*upcast_vae.*")
                    cls._pipeline = pipeline_class.from_single_file(
                        model_path,
                        torch_dtype=torch.float16,
                        use_safetensors=True,
                        variant="fp16"
                    )
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
                    
                    # Optimization: Use 4-bit T5 even for local models to save ~8GB VRAM
                    # This is CRITICAL for 12GB cards
                    try:
                        print(f"[{model_type}] Loading 4-bit T5 encoder to save VRAM...")
                        text_encoder_3 = T5EncoderModel.from_pretrained(
                            "diffusers/t5-nf4",
                            torch_dtype=torch.float16,
                            device_map="auto"
                        )
                    except Exception as t5_err:
                        print(f"[{model_type}] Could not load 4-bit T5, using default: {t5_err}")
                        text_encoder_3 = None

                    try:
                        cls._pipeline = pipeline_class.from_single_file(
                            model_path, 
                            text_encoder_3=text_encoder_3,
                            torch_dtype=torch.float16
                        )
                    except Exception as e:
                        if "CLIPTextModelWithProjection" in str(e) or "text_encoder" in str(e).lower():
                            print(f"[{model_type}] Missing components, loading from repo fallback...")
                            from transformers import CLIPTextModelWithProjection
                            text_encoder = CLIPTextModelWithProjection.from_pretrained(load_repo, subfolder="text_encoder", torch_dtype=torch.float16)
                            text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(load_repo, subfolder="text_encoder_2", torch_dtype=torch.float16)
                            if text_encoder_3 is None:
                                text_encoder_3 = T5EncoderModel.from_pretrained(load_repo, subfolder="text_encoder_3", torch_dtype=torch.float16)
                            
                            cls._pipeline = pipeline_class.from_single_file(
                                model_path,
                                text_encoder=text_encoder,
                                text_encoder_2=text_encoder_2,
                                text_encoder_3=text_encoder_3,
                                torch_dtype=torch.float16
                            )
                        else:
                            raise e
                else:
                    # Hugging Face remote loading
                    print(f"Loading remote {model_type} model: {model_path}...")
                    cls._pipeline = pipeline_class.from_pretrained(
                        model_path,
                        torch_dtype=torch.float16
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
            if config.DEVICE == "cuda":
                try:
                    import xformers
                    cls._pipeline.enable_xformers_memory_efficient_attention()
                except ImportError:
                    pass

            # Update current state if we loaded or switched
            cls._current_model_path = model_path
            cls._current_model_type = model_type
            cls._current_is_img2img = is_img2img
            cls._current_is_inpaint = is_inpaint
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
                 denoising_strength: float = 0.5, output_format: str = "png", mask: str = None):
        """
        Génère une image (ou transforme une existante) à partir des paramètres.
        """
        cls.reset_interrupt()
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            
        import time
        start_time = time.time()
        
        # Map 'img2img' alias to base architecture
        if model_type == 'img2img':
            model_type = ModelManager._current_model_type or 'sdxl'
            
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
        
        # Output directory preparation
        output_dir = Path(config.OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine if we are doing Img2Img or Inpaint
        is_now_img2img = image is not None
        is_now_inpaint = mask is not None
        
        # Pipeline
        pipe, compel = ModelManager.get_pipeline(
            model_type=model_type, 
            model_path=model_path, 
            vae_name=vae_name, 
            sampler_name=sampler_name,
            is_img2img=is_now_img2img,
            is_inpaint=is_now_inpaint
        )
        active_pipe = pipe
        ModelManager.reset_interrupt()
        
        # Seed
        generator = None
        if seed is None:
            import random
            seed = random.randint(0, 2**32 - 1)
        
        generator = torch.Generator(config.DEVICE).manual_seed(seed)
        
        # Generation
        print(f"Generating {model_type} image for prompt: {prompt}")
        if image:
            print(f"Img2Img mode detected (Denoising: {denoising_strength})")
        if negative_prompt:
            print(f"Negative prompt: {negative_prompt}")
        
        # Prepare Image if Img2Img
        init_image = None
        if image:
            try:
                # Decode base64
                header, encoded = image.split(",", 1) if "," in image else (None, image)
                image_data = base64.b64decode(encoded)
                init_image = Image.open(BytesIO(image_data)).convert("RGB")
                
                # Resize if needed (Maintain aspect ratio, multiple of 64)
                # If we have an image, we use its size or fit to max
                w, h = init_image.size
                if w > 1024 or h > 1024:
                    scale = 1024 / max(w, h)
                    w, h = int(w * scale), int(h * scale)
                
                # Snap to 64
                w = (w // 64) * 64
                h = (h // 64) * 64
                if w != init_image.size[0] or h != init_image.size[1]:
                    print(f"Resizing input image for SDXL: {init_image.size} -> ({w}, {h})")
                    init_image = init_image.resize((w, h), Image.LANCZOS)
                
                # Synchronize width/height for pipe
                width, height = w, h
            except Exception as e:
                print(f"Failed to process input image: {e}")
                init_image = None

        # Prepare Mask if Inpaint
        init_mask = None
        if mask:
            try:
                header, encoded = mask.split(",", 1) if "," in mask else (None, mask)
                mask_data = base64.b64decode(encoded)
                init_mask = Image.open(BytesIO(mask_data)).convert("L") # Mode L for mask
                # Resize mask to match target size
                if init_mask.size != (width, height):
                    init_mask = init_mask.resize((width, height), Image.LANCZOS)
            except Exception as e:
                print(f"Failed to process mask: {e}")
                init_mask = None

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

        # Prepare kwargs for Img2Img/Inpaint
        if is_now_inpaint:
            # Inpainting logic
            kwargs.pop("width", None)
            kwargs.pop("height", None)
            kwargs["image"] = init_image
            kwargs["mask_image"] = init_mask
            kwargs["strength"] = denoising_strength
        elif is_now_img2img:
            # Width/Height are not used in Img2Img as they come from the image
            kwargs.pop("width", None)
            kwargs.pop("height", None)
            kwargs["image"] = init_image
            kwargs["strength"] = denoising_strength
            
            # VAE Tiling/Slicing is already enabled in get_pipeline, no need for reload hooks here

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
        
        file_name = f"{base_name}_{day_folder}_{timestamp}_{width}x{height}_seed{seed}.{ext}"
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
            "negative_prompt": negative_prompt,
            "model_type": model_type,
            "model_name": m_name,
            "width": width,
            "height": height,
            "guidance_scale": guidance_scale,
            "num_inference_steps": num_inference_steps,
            "seed": seed,
            "vae_name": vae_name,
            "sampler_name": sampler_name
        }

        return f"{day_folder}/{file_name}", exec_time, used_params

    @classmethod
    def upscale(cls, image_base64: str, scale_factor: float = 2.0, denoising_strength: float = 0.2, 
                tile_size: int = 768, prompt: str = "", negative_prompt: str = "", 
                model_name: Optional[str] = None, output_format: str = "png", seed: Optional[int] = 42, **kwargs):
        """
        Tiled upscaler implementation.
        Processes images in overlapping tiles to allow high-res upscaling on limited VRAM.
        """
        start_time = time.time()
        
        # 1. Prepare image
        try:
            if image_base64.startswith("/view/"):
                # It's a path to a previously generated image
                rel_path = image_base64.replace("/view/", "")
                # Security: ensure no .. or absolute weirdness
                rel_path = rel_path.lstrip("/")
                full_path = Path(config.OUTPUT_DIR) / rel_path
                print(f"[Upscaler] Auto-picking image from disk: {full_path}")
                if not full_path.exists():
                    raise FileNotFoundError(f"Source image not found: {full_path}")
                input_image = Image.open(full_path).convert("RGB")
            else:
                # Standard base64
                header, encoded = image_base64.split(",", 1) if "," in image_base64 else (None, image_base64)
                image_data = base64.b64decode(encoded)
                input_image = Image.open(BytesIO(image_data)).convert("RGB")
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
        model_type = cls._current_model_type or 'sdxl'
        m_name = model_name or os.path.basename(cls._current_model_path) if cls._current_model_path else config.DEFAULT_MODEL
        
        if model_type == 'flux':
            model_path = os.path.join(config.get('FLUX_MODELS_DIR', 'models/flux'), m_name)
        elif model_type in ['sd3', 'sd3_5_turbo']:
            model_path = os.path.join(config.get('SD3_MODELS_DIR', 'models/sd3'), m_name)
        else:
            model_path = os.path.join(config.MODELS_DIR, m_name)

        pipe, compel = cls.get_pipeline(
            model_type=model_type,
            model_path=model_path,
            is_img2img=True
        )
        cls.reset_interrupt()
        
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
        day_folder = now.strftime("%Y-%m-%d")
        
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
            "seed": seed
        }
        
        print(f"[Upscaler] Success: {target_w}x{target_h} in {exec_time:.1f}s")
        return f"{day_folder}/{file_name}", exec_time, used_params


def list_models(model_type: str) -> list:
    """Lists available model files for a given type."""
    if model_type == 'flux':
        directory = Path(config.get('FLUX_MODELS_DIR', ''))
    elif model_type in ['sd3', 'sd3_5_turbo']:
        directory = Path(config.get('SD3_MODELS_DIR', ''))
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
    directory = Path(config.get('VAE_DIR', ''))
    if not directory or not directory.exists():
        return []
    
    vaes = []
    for ext in ['*.safetensors', '*.pt', '*.ckpt']:
        vaes.extend([f.name for f in directory.glob(ext)])
    return sorted(vaes)

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

