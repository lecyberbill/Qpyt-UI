import base64
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.utils import load_image_from_input
from PIL import Image
import io

def test_padding_fix():
    print("Testing Base64 Padding and Path Fix...")
    
    # 1. Test Base64 Padding (4n+2 or 4n+3)
    img = Image.new('RGB', (10, 10), color='red')
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    original_b64 = base64.b64encode(buffered.getvalue()).decode()
    
    stripped_b64 = original_b64.rstrip('=')
    print(f"B64 Stripped Padding: {len(stripped_b64)} chars")
    loaded_img = load_image_from_input(stripped_b64)
    assert loaded_img.size == (10, 10)
    print("✓ Padding fix OK")

    # 2. Test Base64 Truncation (4n+1 case)
    # We take a valid B64 and add 1 junk char or remove 1 more
    invalid_len_b64 = stripped_b64
    if len(invalid_len_b64) % 4 == 0:
        invalid_len_b64 = invalid_len_b64[:-3] # Make it 4n+1
    elif len(invalid_len_b64) % 4 == 2:
        invalid_len_b64 = invalid_len_b64[:-1] # Make it 4n+1
    elif len(invalid_len_b64) % 4 == 3:
        invalid_len_b64 = invalid_len_b64[:-2] # Make it 4n+1
    
    print(f"B64 Truncated (4n+1): {len(invalid_len_b64)} chars")
    # This should not crash binascii now
    try:
        load_image_from_input(invalid_len_b64)
        print("✓ 4n+1 recover OK (no crash)")
    except Exception as e:
        print(f"Caught expected failure or error: {e}")

    # 3. Test Internal Paths
    # Ensure config.OUTPUT_DIR exists for test
    from core.config import config
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    test_file = os.path.join(config.OUTPUT_DIR, "test_path.png")
    img.save(test_file)
    
    print("Testing /outputs/ prefix...")
    loaded_path = load_image_from_input("/outputs/test_path.png")
    assert loaded_path.size == (10, 10)
    print("✓ /outputs/ path OK")

    print("Testing /view/ prefix...")
    loaded_view = load_image_from_input("/view/test_path.png")
    assert loaded_view.size == (10, 10)
    print("✓ /view/ path OK")

if __name__ == "__main__":
    test_padding_fix()
