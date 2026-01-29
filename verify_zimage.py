try:
    from diffusers import ZImagePipeline
    print("ZImagePipeline found!")
except ImportError:
    print("ZImagePipeline NOT found in diffusers.")
    import diffusers
    print(f"Diffusers version: {diffusers.__version__}")
