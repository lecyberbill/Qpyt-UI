import sys
import os
import torch
import unittest
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.generator import ModelManager
from core.config import config

class TestModelLoader(unittest.TestCase):
    def setUp(self):
        # Ensure we are in a safe state
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def test_manager_class_state(self):
        """Verify ModelManager class attributes and basic state."""
        # Since ModelManager uses class methods as a singleton, we check the class itself
        self.assertIsNone(ModelManager._pipeline)
        self.assertIsNone(ModelManager._compel)
        self.assertEqual(ModelManager._interrupt_flag, False)

    def test_lora_architecture_detection_mock(self):
        """Test LoRA architecture detection logic with mocked safetensors."""
        with patch('core.generator.safe_open') as mock_safe:
            mock_file = MagicMock()
            # Mock the metadata find for architecture
            mock_file.keys.return_value = ["ss_base_model_version"]
            mock_file.get_slice.return_value.get_shape.return_value = [1, 2048] # Mock an SDXL shape
            
            # Mocking the context manager
            mock_safe.return_value.__enter__.return_value = mock_file
            
            from pathlib import Path
            # We need to mock the actual metadata check or the fallback
            # Let's mock a case where it finds 'sdxl' in a key dimension
            
            # Simple test to see if it even calls the right things
            arch = ModelManager._get_lora_architecture(Path("fake.safetensors"))
            # In our implementation, a shape of 2048 in a key like 'attn2.to_k' returns sdxl
            # But the mock needs to return that key
            mock_file.keys.return_value = ["attn2.to_k"]
            arch = ModelManager._get_lora_architecture(Path("fake.safetensors"))
            self.assertEqual(arch, "sdxl")

    def test_apply_loras_empty(self):
        """Verify apply_loras handles empty lists gracefully."""
        # Mock pipeline
        mock_pipe = MagicMock()
        warnings = ModelManager.apply_loras(mock_pipe, [])
        self.assertEqual(warnings, [])

    def test_sampler_list(self):
        """Verify samplers list is populated."""
        from core.generator import list_samplers
        samplers = list_samplers()
        self.assertIn("Euler a", samplers)
        self.assertIn("DPM++ 2M Karras", samplers)

if __name__ == '__main__':
    unittest.main()
