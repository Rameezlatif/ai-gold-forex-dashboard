@echo off
setlocal

cd /d "%~dp0"

where python >nul 2>nul
if errorlevel 1 (
    echo Python is not installed or not available in PATH.
    pause
    exit /b 1
)

python -m pip install -r requirements.txt
python -m streamlit run app.py --server.port 8501

pause
