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
        """Prepares images (crop/resize) and captions them."""
        input_path = Path(input_dir)
        if not input_path.exists():
            raise Exception(f"Input directory {input_dir} not found.")

        # 1. Create staging area if needed
        # (For now we'll process in place or in a 'prepared' subfolder)
        
        images = []
        for ext in ["*.png", "*.jpg", "*.jpeg", "*.webp"]:
            images.extend(list(input_path.glob(ext)))
        
        total = len(images)
        logger.info(f"[Trainer] Preparing {total} images...")

        for i, img_path in enumerate(images):
            if progress_callback:
                progress_callback(i / total * 30.0) # 0-30% for prep

            # Auto-caption if missing
            cap_path = img_path.with_suffix(".txt")
            if not cap_path.exists():
                logger.info(f"[Trainer] Captioning {img_path.name}...")
                img = Image.open(img_path)
                caption = analyze_image(img, task="<DETAILED_CAPTION>")
                # Add concept word if not present
                if concept_name.lower() not in caption.lower():
                    caption = f"{concept_name}, {caption}"
                
                with open(cap_path, "w", encoding="utf-8") as f:
                    f.write(caption)

        logger.info("[Trainer] Dataset preparation complete.")
        return True

    @staticmethod
    def train_sdxl_lora(
        input_dir, 
        output_name, 
        concept_name, 
        steps=1000, 
        lr=1e-4, 
        rank=16, 
        batch_size=1,
        progress_callback=None
    ):
        """Train a LoRA on SDXL UNet."""
        try:
            # 1. Prepare Dataset
            LoRATrainer.prepare_dataset(input_dir, concept_name, progress_callback)

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
            
            vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae", torch_dtype=weight_dtype).to(device)
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
            dataset = LoRATataset(input_dir, tokenizer_one, size=1024)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            # 7. Training Loop
            logger.info(f"[Trainer] Starting training for {steps} steps...")
            global_step = 0
            
            while global_step < steps:
                for batch in dataloader:
                    if global_step >= steps: break
                    
                    optimizer.zero_grad()
                    
                    pixel_values = batch["pixel_values"].to(device, dtype=weight_dtype)
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
                        progress_callback(30.0 + (global_step / steps * 70.0))
                
                logger.info(f"[Trainer] Step {global_step}/{steps} - Loss: {loss.item():.4f}")

            # 8. Save result as Safetensors
            from safetensors.torch import save_file
            out_path = Path(config.LORAS_DIR) / f"{output_name}.safetensors"
            
            # Extract LoRA weights from PEFT model
            state_dict = unet.state_dict()
            lora_dict = {k: v for k, v in state_dict.items() if "lora_" in k}
            
            save_file(lora_dict, str(out_path))
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
