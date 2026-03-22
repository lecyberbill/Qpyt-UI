import os
import torch
import gc
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    UNet2DConditionModel,
)
from transformers import AutoTokenizer, PretrainedConfig, CLIPTextModel, CLIPTextModelWithProjection
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from core.config import config
from core.analyzer import analyze_image
import logging

logger = logging.getLogger("qpyt-ui")

class LoRADataset(Dataset):
    def __init__(self, dataset_path, tokenizer, size=1024, caption_ext=".txt"):
        self.dataset_path = Path(dataset_path)
        self.tokenizer = tokenizer
        self.size = size
        self.image_files = []
        for ext in ["*.png", "*.jpg", "*.jpeg", "*.webp"]:
            self.image_files.extend(list(self.dataset_path.glob(ext)))
        
        self.transform = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, i):
        img_path = self.image_files[i]
        image = Image.open(img_path).convert("RGB")
        pixel_values = self.transform(image)
        
        # Load caption
        caption_path = img_path.with_suffix(".txt")
        caption = ""
        if caption_path.exists():
            with open(caption_path, "r", encoding="utf-8") as f:
                caption = f.read().strip()
        
        tokens = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt"
        ).input_ids[0]
        
        return {"pixel_values": pixel_values, "input_ids": tokens}

class LoRATrainer:
    @staticmethod
    def prepare_dataset(input_dir, concept_name, progress_callback=None):
        """Prepares images (crop/resize) and captions them in a 'prepared' subfolder."""
        input_path = Path(input_dir)
        if not input_path.exists():
            raise Exception(f"Input directory {input_dir} not found.")

        # 1. Create staging area
        prepared_path = input_path / "prepared"
        prepared_path.mkdir(exist_ok=True)
        
        images = []
        for ext in ["*.png", "*.jpg", "*.jpeg", "*.webp"]:
            images.extend(list(input_path.glob(ext)))
        
        total = len(images)
        if total == 0:
            raise Exception(f"No images found in {input_dir}")
            
        logger.info(f"[Trainer] Preparing {total} images into {prepared_path}...")

        # Setup transforms for physical preparation
        prep_transform = transforms.Compose([
            transforms.Resize(1024, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(1024),
        ])

        for i, img_path in enumerate(images):
            if progress_callback:
                progress_callback(i / total * 30.0, message=f"Preparing image {i+1}/{total}: {img_path.name}") # 0-30% for prep

            # Target paths in prepared folder
            target_img_path = prepared_path / f"{img_path.stem}.png"
            target_cap_path = prepared_path / f"{img_path.stem}.txt"

            # 1. Process Image (if not already in prepared or to ensure 1024x1024)
            try:
                with Image.open(img_path) as img:
                    img = img.convert("RGB")
                    prepared_img = prep_transform(img)
                    prepared_img.save(target_img_path, "PNG")
            except Exception as e:
                logger.error(f"[Trainer] Failed to process image {img_path}: {e}")
                continue

            # 2. Handle Caption
            caption = ""
            # Check source dir for existing caption
            source_cap = img_path.with_suffix(".txt")
            if source_cap.exists():
                with open(source_cap, "r", encoding="utf-8") as f:
                    caption = f.read().strip()
            
            # Auto-caption if missing
            if not caption:
                logger.info(f"[Trainer] Captioning {img_path.name}...")
                with Image.open(target_img_path) as img:
                    caption = analyze_image(img, task="<DETAILED_CAPTION>")
                
                # Add concept word if not present
                if concept_name.lower() not in caption.lower():
                    caption = f"{concept_name}, {caption}"
                
                # Save to source too for future use? Let's keep source clean and just save to prepared.
                # Actually, user might want to edit it later. Let's save to BOTH if caption was missing.
                with open(source_cap, "w", encoding="utf-8") as f:
                    f.write(caption)

            # Always save to prepared folder
            with open(target_cap_path, "w", encoding="utf-8") as f:
                f.write(caption)

        logger.info(f"[Trainer] Dataset preparation complete in {prepared_path}")
        return str(prepared_path)

    @staticmethod
    def train_sdxl_lora(
        input_dir, 
        output_name, 
        concept_name, 
        steps=1000, 
        lr=4e-5, 
        rank=16, 
        batch_size=1,
        progress_callback=None
    ):
        """Train a LoRA on SDXL UNet."""
        try:
            # 1. Prepare Dataset (returns actual training path)
            train_path = LoRATrainer.prepare_dataset(input_dir, concept_name, progress_callback)
            input_dir = train_path # Use the prepared subfolder for training

            # 2. Setup Device & Dtypes
            device = "cuda" if torch.cuda.is_available() else "cpu"
            weight_dtype = torch.float16 if device == "cuda" else torch.float32

            # 3. Load Models
            model_id = config.get("SDXL_BASE_MODEL", "stabilityai/stable-diffusion-xl-base-1.0")
            logger.info(f"[Trainer] Loading SDXL components from {model_id}...")
            
            tokenizer_one = AutoTokenizer.from_pretrained(model_id, subfolder="tokenizer", use_fast=False)
            tokenizer_two = AutoTokenizer.from_pretrained(model_id, subfolder="tokenizer_2", use_fast=False)
            
            text_encoder_one = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder", torch_dtype=weight_dtype).to(device)
            text_encoder_two = CLIPTextModelWithProjection.from_pretrained(model_id, subfolder="text_encoder_2", torch_dtype=weight_dtype).to(device)
            
            # VAE is very unstable in fp16 on SDXL, forcing fp32
            vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32).to(device)
            unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet", torch_dtype=weight_dtype).to(device)
            noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")

            # Freeze non-trainable
            vae.requires_grad_(False)
            text_encoder_one.requires_grad_(False)
            text_encoder_two.requires_grad_(False)
            unet.requires_grad_(False)

            # Optimizations
            if device == "cuda":
                unet.enable_gradient_checkpointing()

            # 4. Setup LoRA on UNet
            lora_config = LoraConfig(
                r=rank,
                lora_alpha=rank,
                target_modules=["to_q", "to_k", "to_v", "to_out.0"],
                init_lora_weights="gaussian",
            )
            unet = get_peft_model(unet, lora_config)
            unet.train()

            # 5. Optimizer (Use 8-bit Adam if possible)
            try:
                import bitsandbytes as bnb
                optimizer = bnb.optim.AdamW8bit(unet.parameters(), lr=lr)
                logger.info("[Trainer] Using 8-bit Adam optimizer.")
            except ImportError:
                optimizer = torch.optim.AdamW(unet.parameters(), lr=lr)
                logger.info("[Trainer] bitsandbytes not found, using standard AdamW.")

            # 6. Data Loader
            # Custom loader for dual tokenizers
            dataset = LoRADataset(input_dir, tokenizer_one, size=1024)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            # 7. Training Loop
            logger.info(f"[Trainer] Starting training for {steps} steps...")
            global_step = 0
            
            while global_step < steps:
                for batch in dataloader:
                    if global_step >= steps: break
                    
                    optimizer.zero_grad()
                    
                    # Keep pixels in fp32 for VAE, UNet will use weight_dtype
                    pixel_values = batch["pixel_values"].to(device, dtype=torch.float32)
                    input_ids_one = batch["input_ids"].to(device)
                    # For simplicity, we assume same text for both encoders in this V1
                    input_ids_two = tokenizer_two(
                        dataset.tokenizer.batch_decode(batch["input_ids"], skip_special_tokens=True),
                        padding="max_length", truncation=True, max_length=tokenizer_two.model_max_length, return_tensors="pt"
                    ).input_ids.to(device)

                    # Encode latents
                    with torch.no_grad():
                        latents = vae.encode(pixel_values).latent_dist.sample()
                        latents = latents * vae.config.scaling_factor
                        latents = latents.to(weight_dtype)

                        # Encode text
                        prompt_embeds_one = text_encoder_one(input_ids_one, output_hidden_states=True)
                        pooled_prompt_embeds_one = prompt_embeds_one[0]
                        prompt_embeds_one = prompt_embeds_one.hidden_states[-2]

                        prompt_embeds_two = text_encoder_two(input_ids_two, output_hidden_states=True)
                        pooled_prompt_embeds_two = prompt_embeds_two[0]
                        prompt_embeds_two = prompt_embeds_two.hidden_states[-2]

                        # Concat embeddings
                        prompt_embeds = torch.cat([prompt_embeds_one, prompt_embeds_two], dim=-1)
                        add_text_embeds = pooled_prompt_embeds_two
                        # SDXL also needs time_ids (crop, original size, etc.)
                        # Standard values for 1024x1024
                        add_time_ids = torch.tensor([[1024, 1024, 0, 0, 1024, 1024]], device=device, dtype=weight_dtype).repeat(batch_size, 1)

                    # Sample noise
                    noise = torch.randn_like(latents)
                    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.size(0),), device=device).long()
                    noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                    # Predict noise
                    added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
                    model_pred = unet(noisy_latents, timesteps, prompt_embeds, added_cond_kwargs=added_cond_kwargs).sample

                    # Loss
                    loss = torch.nn.functional.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                    loss.backward()
                    optimizer.step()

                    global_step += 1
                    if progress_callback and global_step % 10 == 0:
                        progress_callback(30.0 + (global_step / steps * 70.0), message=f"Training: Step {global_step}/{steps} - Loss: {loss.item():.4f}")
                
                logger.info(f"[Trainer] Step {global_step}/{steps} - Loss: {loss.item():.4f}")

            # 8. Save result as Safetensors
            from safetensors.torch import save_file
            from peft import get_peft_model_state_dict
            out_path = Path(config.LORAS_DIR) / f"{output_name}.safetensors"
            
            # Extract LoRA weights from PEFT model properly
            lora_state_dict = get_peft_model_state_dict(unet)
            
            save_file(lora_state_dict, str(out_path))
            logger.info(f"[Trainer] LoRA saved to {out_path}")

            return {"status": "success", "lora_path": str(out_path), "steps": global_step}

        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.error(f"[Trainer] Training failed: {e}")
            raise e
        finally:
            # Cleanup
            try:
                del unet, vae, text_encoder_one, text_encoder_two
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except: pass
