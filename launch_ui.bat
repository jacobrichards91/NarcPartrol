@echo off
REM NarcPartrol UI launcher (Windows)
REM Double-click this file or run from cmd:  launch_ui.bat

cd /d "%~dp0"

echo Starting NarcPartrol UI...
echo Your browser will open to http://localhost:8501
echo Press Ctrl+C in this window to stop the server.
echo.

python -m streamlit run app.py --server.headless=false --browser.gatherUsageStats=false

pause
