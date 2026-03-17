@echo off
SETLOCAL
echo [Micro-Gradio] Activation de l'environnement virtuel...
IF NOT EXIST ".venv\Scripts\activate.bat" (
    echo [ERREUR] Environnement virtuel .venv non trouve. 
    echo Veuillez lancer l'installation au prealable.
    pause
    exit /b
)

call .venv\Scripts\activate.bat

echo [Micro-Gradio] Lancement du serveur FastAPI...
echo Accedez a l'app sur : http://127.0.0.1:8000/

:: Activation des logs détaillés pour Triton / PyTorch Inductor
set TORCH_LOGS="+dynamo,inductor"
set TORCH_COMPILE_DEBUG=1

python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

pause
ENDLOCAL
