import json
import os
from pathlib import Path

class Config:
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent
        self.config_path = self.base_dir / "config.json"
        self.local_config_path = self.base_dir / "config.local.json"
        self.settings = {}
        self.load()

    def load(self):
        # 1. Charger les valeurs par défaut
        if self.config_path.exists():
            with open(self.config_path, "r", encoding="utf-8") as f:
                self.settings.update(json.load(f))
        
        # 2. Surcharger avec les valeurs locales (si présentes)
        if self.local_config_path.exists():
            with open(self.local_config_path, "r", encoding="utf-8") as f:
                self.settings.update(json.load(f))

    def get(self, key, default=None):
        return self.settings.get(key, default)

    def __getattr__(self, name):
        if name in self.settings:
            return self.settings[name]
        raise AttributeError(f"'Config' object has no attribute '{name}'")

# Singleton instance
config = Config()
