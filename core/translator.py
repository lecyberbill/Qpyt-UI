import torch
import gc
from transformers import pipeline
from core.config import config

class TranslationManager:
    _translator = None
    _model_name = "Helsinki-NLP/opus-mt-fr-en"

    @classmethod
    def get_translator(cls):
        if cls._translator is None:
            print(f"[Translator] Loading model {cls._model_name}...")
            # Use CPU for translation to keep VRAM free for generation
            # These models are small and fast enough on CPU
            try:
                cls._translator = pipeline("translation", model=cls._model_name, device="cpu")
                print(f"[Translator] Model loaded successfully.")
            except Exception as e:
                print(f"[Translator] Failed to load model: {e}")
                raise e
        return cls._translator

    @classmethod
    def translate(cls, text: str) -> str:
        if not text:
            return ""
        
        translator = cls.get_translator()
        try:
            result = translator(text)
            translated_text = result[0]['translation_text']
            print(f"[Translator] {text} -> {translated_text}")
            return translated_text
        except Exception as e:
            print(f"[Translator] Translation error: {e}")
            raise e

    @classmethod
    def unload_model(cls):
        if cls._translator is not None:
            print("[Translator] Unloading model...")
            del cls._translator
            cls._translator = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("[Translator] Model unloaded.")
