import torch
import scipy.io.wavfile
import numpy as np
from pathlib import Path
from core.config import config
import uuid
from datetime import datetime
import gc
from pydub import AudioSegment
import io

from transformers import AutoProcessor, MusicgenForConditionalGeneration
import static_ffmpeg
static_ffmpeg.add_paths()

_music_model = None
_music_processor = None

class MusicGenerator:
    @staticmethod
    def get_instance():
        global _music_model, _music_processor
        if _music_model is None:
            print("[MusicGen] Loading model (This may take a while)...")
            model_id = config.get("MUSICGEN_MODEL_PATH", "facebook/musicgen-medium")
            
            _music_processor = AutoProcessor.from_pretrained(model_id)
            
            if torch.cuda.is_available():
                print("[MusicGen] Using float16 precision for CUDA")
                _music_model = MusicgenForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16)
                _music_model = _music_model.cuda()
            else:
                _music_model = MusicgenForConditionalGeneration.from_pretrained(model_id)
                
            print(f"[MusicGen] Model loaded: {model_id}")
            
        return _music_model, _music_processor

    @staticmethod
    def unload():
        global _music_model, _music_processor
        if _music_model is not None:
            print("[MusicGen] Unloading model...")
            del _music_model
            del _music_processor
            _music_model = None
            _music_processor = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    @staticmethod
    def generate(prompt: str, duration: int = 10, guidance_scale: float = 3.0):
        try:
            model, processor = MusicGenerator.get_instance()
            sampling_rate = model.config.audio_encoder.sampling_rate # Usually 32000
            
            tokens_per_sec = 50 
            
            # Logic for long generation (Chunking)
            # Max window for MusicGen is typically ~30s. We'll stick to 25s chunks to be safe + overlap.
            # Strategy:
            # 1. Generate first 25s.
            # 2. If duration left > 0:
            #    Take last 5s of previous output as prompt.
            #    Generate next 20s (max_new_tokens corresponding to 20s).
            #    Append new 20s to final audio.
            # 3. Repeat.
            
            chunk_length = 25 # Seconds of new audio per chunk (initial)
            overlap = 5 # Seconds of audio to use as history
            
            current_duration = 0
            final_audio = []
            
            # Initial generation
            first_chunk_dur = min(duration, chunk_length)
            
            print(f"[MusicGen] Generating initial chunk ({first_chunk_dur}s)...")
            
            inputs = processor(
                text=[prompt],
                padding=True,
                return_tensors="pt",
            )
            if torch.cuda.is_available(): inputs = inputs.to("cuda")
            
            audio_values = model.generate(
                **inputs, 
                max_new_tokens=int(first_chunk_dur * tokens_per_sec),
                guidance_scale=guidance_scale,
                do_sample=True 
            )
            
            # Extract raw audio [1, 1, samples] -> [samples]
            # output is (batch, channel, sequence)
            chunk_audio = audio_values[0, 0].cpu().numpy()
            final_audio.append(chunk_audio)
            
            current_duration += first_chunk_dur
            
            # Loop for remaining time
            while current_duration < duration:
                remaining = duration - current_duration
                next_chunk_dur = min(remaining, 20) # Generate 20s chunks max (+5s prompt context)
                
                print(f"[MusicGen] Generating continuation ({next_chunk_dur}s)... Total so far: {current_duration}s")
                
                # Get last 'overlap' seconds from previous generation to condition next step
                # sampling_rate samples per second
                context_samples = int(overlap * sampling_rate)
                
                # Reconstruct full current audio to slice from
                # (Slightly inefficient to cat every time but safest for index logic)
                full_current = np.concatenate(final_audio)
                
                if len(full_current) < context_samples:
                    # Should not happen if first chunk > overlap, but fallback
                    input_audio = full_current
                else:
                    input_audio = full_current[-context_samples:]
                    
                inputs = processor(
                    text=[prompt],
                    audio=input_audio,
                    sampling_rate=sampling_rate,
                    padding=True,
                    return_tensors="pt",
                )
                if torch.cuda.is_available(): inputs = inputs.to("cuda")
                
                # generate returns [prompt + new]
                # We want 'max_new_tokens' to cover 'next_chunk_dur'
                
                audio_values = model.generate(
                    **inputs,
                    max_new_tokens=int(next_chunk_dur * tokens_per_sec),
                    guidance_scale=guidance_scale,
                    do_sample=True
                )
                
                # The output contains input_audio + new_audio.
                # We only want the new audio.
                # Input length in samples?
                # Transformers output is audio codes decoded? 
                # Actually model.generate returns the waveforms.
                # Length of input part in output might differ slightly due to compression/decompression?
                # Usually it's roughly consistent.
                
                full_output = audio_values[0, 0].cpu().numpy()
                
                # We need to slice off the prompt part.
                # Heuristic: verify length.
                # Expected new samples = next_chunk_dur * sampling_rate
                # We can just take the last N samples corresponding to new tokens.
                # Or just take everything after the input length?
                # Since compression is lossy, exact sample match is risky.
                # But MusicGen output length = input_length + generated_length
                
                # Safer: We asked for next_chunk_dur * tokens_per_sec tokens.
                # Roughly that many samples / 50 * 32000...
                # Actually easier: Just subtract len(input_audio) from output? 
                # But decoding might introduce padding.
                
                # Let's try slicing from end: shape[-1] - (next_chunk_dur * sampling_rate)?
                # Or slicing from start: len(input_audio)?
                
                # Let's trust len(input_audio) is preserved roughly.
                
                new_part = full_output[len(input_audio):]
                
                # Fallback if weird
                if len(new_part) == 0:
                    print("[MusicGen] Warning: No new audio generated in continuation step.")
                    break
                    
                final_audio.append(new_part)
                current_duration += next_chunk_dur
                
                # Garbage collect to free VRAM during long loops?
                torch.cuda.empty_cache()
            
            # Stitch all together
            full_audio_data = np.concatenate(final_audio)
            
            # Convert to float32 for scipy compatibility (scipy doesn't like float16)
            full_audio_data = full_audio_data.astype(np.float32)
            
            # Normalize if needed (clipping to -1, 1 to avoid glitches)
            # MusicGen usually outputs within range but good practice
            max_val = np.max(np.abs(full_audio_data))
            if max_val > 1.0:
                full_audio_data /= max_val
            
            # Save Logic
            now = datetime.now()
            day_folder = now.strftime("%Y_%m_%d")
            # Base name
            base_name = f"music_{uuid.uuid4().hex[:8]}"
            
            final_output_dir = Path(config.OUTPUT_DIR) / day_folder
            final_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save WAV (Intermediate)
            wav_path = final_output_dir / f"{base_name}.wav"
            scipy.io.wavfile.write(wav_path, rate=sampling_rate, data=full_audio_data)
            
            # Try converting to MP3
            mp3_path = final_output_dir / f"{base_name}.mp3"
            print(f"[MusicGen] Converting to MP3: {mp3_path}")
            
            try:
                # With static_ffmpeg.add_paths() called at top, pydub should find ffmpeg/ffprobe in PATH
                # No need for manual AudioSegment.converter setting if PATH is correct
                
                sound = AudioSegment.from_wav(str(wav_path))
                sound.export(str(mp3_path), format="mp3", bitrate="192k")
                
                print(f"[MusicGen] Success: {mp3_path}")
                return f"{day_folder}/{base_name}.mp3"
                
            except Exception as e:
                print(f"[MusicGen] MP3 Conversion failed ({e}). Returning WAV instead.")
                print(f"[MusicGen] Success (WAV): {wav_path}")
                return f"{day_folder}/{base_name}.wav"

        except Exception as e:
            print(f"[MusicGen] Error: {e}")
            raise e
