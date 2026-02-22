import diffusers
print(f"Diffusers version: {diffusers.__version__}")
try:
    from diffusers import Flux2Pipeline, Flux2Transformer2DModel
    print("Flux2 classes ARE available")
except ImportError:
    print("Flux2 classes NOT available")
