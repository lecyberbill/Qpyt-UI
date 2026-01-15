import torch
import gc
import traceback
from transformers import AutoModelForCausalLM, AutoTokenizer
from core.config import config

class LlmPrompterManager:
    _model = None
    _tokenizer = None
    _model_id = "Qwen/Qwen2.5-1.5B-Instruct"

    @classmethod
    def load_model(cls):
        if cls._model is None:
            print(f"[LLM Prompter] Loading model {cls._model_id} on CPU...")
            try:
                cls._tokenizer = AutoTokenizer.from_pretrained(cls._model_id)
                cls._model = AutoModelForCausalLM.from_pretrained(
                    cls._model_id,
                    torch_dtype="auto",
                    device_map="cpu"
                )
                
                if cls._tokenizer.pad_token is None:
                    cls._tokenizer.pad_token = cls._tokenizer.eos_token
                
                print("[LLM Prompter] Model loaded successfully.")
            except Exception as e:
                print(f"[LLM Prompter] Error loading model: {e}")
                traceback.print_exc()
                cls._model = None
                cls._tokenizer = None
                raise e

    @classmethod
    def enhance_prompt(cls, base_prompt: str, max_new_tokens: int = 512) -> str:
        if not base_prompt:
            return ""

        cls.load_model()
        
        try:
            # Use System prompt for instructions and User for the data
            system_prompt = "You are a professional stable diffusion prompt engineer. Your task is to transform simple ideas into highly detailed, creative, and artistic image generation prompts in English. Reply ONLY with the enhanced prompt. No introduction, no quotes, no explanations."
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Enhance this idea: {base_prompt}"}
            ]

            text_input = cls._tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            model_inputs = cls._tokenizer([text_input], return_tensors="pt").to(cls._model.device)

            generated_ids = cls._model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=cls._tokenizer.pad_token_id,
                eos_token_id=cls._tokenizer.eos_token_id
            )

            # Extract generated text only (remove input prompt)
            new_tokens = generated_ids[0][len(model_inputs.input_ids[0]):]
            raw_output = cls._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

            # Clean output (handle thinking tags or conversational preambles)
            cleaned_output = raw_output
            
            # Remove <think> and </think> if present (for deepseek/qwen thinking models)
            if "</think>" in cleaned_output:
                cleaned_output = cleaned_output.split("</think>")[-1].strip()
            
            # Common preambles removal
            preambles = [
                "Voici le prompt amélioré :", "Here is the enhanced prompt:",
                "Enhanced prompt:", "Prompt amélioré :", "Okay, here's the prompt:",
                "Sure, here is the detailed prompt:", "Detailed prompt:"
            ]
            for preamble in preambles:
                if cleaned_output.lower().startswith(preamble.lower()):
                    cleaned_output = cleaned_output[len(preamble):].strip()
                    break
            
            # Remove quotes
            cleaned_output = cleaned_output.strip('"\'')
            
            print(f"[LLM Prompter] Enhanced: {base_prompt} -> {cleaned_output[:50]}...")
            return cleaned_output if cleaned_output else base_prompt

        except Exception as e:
            print(f"[LLM Prompter] Generation error: {e}")
            traceback.print_exc()
            return base_prompt

    @classmethod
    def unload_model(cls):
        if cls._model is not None:
            print("[LLM Prompter] Unloading model...")
            del cls._model
            del cls._tokenizer
            cls._model = None
            cls._tokenizer = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("[LLM Prompter] Model unloaded.")
