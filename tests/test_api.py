import httpx
import pytest
from pathlib import Path

BASE_URL = "http://127.0.0.1:8000"

def test_health_check():
    """Vérifie que le serveur est en ligne."""
    with httpx.Client(base_url=BASE_URL) as client:
        response = client.get("/docs")
        assert response.status_code == 200

def test_image_generation_valid():
    """Teste une génération d'image avec des paramètres valides."""
    payload = {
        "prompt": "Un chat astronaute sur la lune, style cyberpunk",
        "width": 512,
        "height": 512,
        "guidance_scale": 7.5
    }
    with httpx.Client(base_url=BASE_URL) as client:
        response = client.post("/generate", json=payload)
        
        assert response.status_code == 200
        json_resp = response.json()
        assert json_resp["status"] == "queued"
        assert "task_id" in json_resp
        
        # We don't verify the file existence here as it's async

def test_validation_error():
    """Vérifie que le contrat Pydantic bloque les mauvaises entrées."""
    payload = {
        "prompt": "", # Trop court, devrait échouer
        "width": 10000, # Trop large, devrait échouer
    }
    with httpx.Client(base_url=BASE_URL) as client:
        response = client.post("/generate", json=payload)
        assert response.status_code == 422  # Unprocessable Entity