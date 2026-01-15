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
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000 --reload

pause
ENDLOCAL
