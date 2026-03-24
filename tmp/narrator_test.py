import requests
import json
import time
import sys

# Configuration
BASE_URL = "http://127.0.0.1:8000"
STORY_IDEA = "A small robot finds a glowing flower in a dark forest."

def test_narrator_flow():
    print(f"--- Starting CLI Test for Story Narrator ---")
    
    # 1. IMAGINE THE STORY
    print(f"\n1. Imagining story: '{STORY_IDEA}'...")
    system_prompt = """You are an AI screenwriter. For the provided idea, you MUST create a storyboard of EXACTLY 4 panels. 
    CHARACTER CONSISTENCY IS CRITICAL: Describe the main character's physical traits (color, clothes, features) identically in EVERY prompt.
    Strictly NO other text, NO preamble, NO conversational noise.
    JSON response format only: 
    [
      {"caption": "short description", "prompt": "precise SD prompt with same character details"},
      {"caption": "short description", "prompt": "precise SD prompt with same character details"},
      {"caption": "short description", "prompt": "precise SD prompt with same character details"},
      {"caption": "short description", "prompt": "precise SD prompt with same character details"}
    ]"""

    assistant_payload = {
        "provider": "gemini",
        "messages": [{"role": "user", "content": STORY_IDEA}],
        "system_prompt": system_prompt,
        "temperature": 0.7
    }

    try:
        resp = requests.post(f"{BASE_URL}/prompt/assistant", json=assistant_payload)
        resp.raise_for_status()
        data = resp.json()
        
        content = data['content'].strip()
        # Clean potential markdown
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()
            
        panels = json.loads(content)
        print(f"Success! Imagined 4 panels:")
        for idx, p in enumerate(panels):
            print(f"  [{idx+1}] {p['caption']}")
            
    except Exception as e:
        print(f"ERROR during imagination: {e}")
        if 'resp' in locals(): print(resp.text)
        return

    # 2. PRODUCE THE IMAGES (Testing with SDXL for speed/default)
    print(f"\n2. Producing images (sequential)...")
    
    # Get default SDXL model from server if possible, or use a known one
    # We'll just use 'sdxl' as model_type
    
    for idx, panel in enumerate(panels):
        print(f"\n--- Panel {idx+1}/4: {panel['caption']} ---")
        gen_payload = {
            "prompt": panel['prompt'],
            "model_type": "sdxl",
            "width": 1024,
            "height": 1024,
            "num_inference_steps": 20, # Reduced for faster test
            "guidance_scale": 7.5,
            "output_format": "webp"
        }
        
        try:
            submit_resp = requests.post(f"{BASE_URL}/generate", json=gen_payload)
            submit_resp.raise_for_status()
            task_id = submit_resp.json()['task_id']
            print(f"Task queued: {task_id}")
            
            # Polling
            completed = False
            while not completed:
                status_resp = requests.get(f"{BASE_URL}/queue/status/{task_id}")
                task = status_resp.json()
                status = task.get('status')
                
                if status == 'COMPLETED':
                    print(f"SUCCESS: Panel {idx+1} ready at {task['result']['image_url']}")
                    completed = True
                elif status == 'FAILED':
                    print(f"FAILED: {task.get('error')}")
                    completed = True
                else:
                    print(f"Status: {status}...", end="\r")
                    time.sleep(2)
                    
        except Exception as e:
            print(f"ERROR during generation: {e}")
            break

    print("\n--- CLI Test Completed ---")

if __name__ == "__main__":
    test_narrator_flow()
