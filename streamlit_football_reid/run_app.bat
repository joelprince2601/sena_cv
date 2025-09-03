@echo off
echo ================================================
echo Football Player Re-ID Streamlit Application
echo ================================================
echo.

cd /d "%~dp0"

echo Installing/updating dependencies...
pip install -r requirements.txt

echo.
echo Starting Streamlit application...
echo The application will open in your default web browser.
echo Press Ctrl+C to stop the application.
echo.

python run_app.py

pause
