@echo off
REM NarcPartrol UI launcher (Windows)
REM Finds whichever Python has streamlit installed.

cd /d "%~dp0"
set "PY="

REM Try py launcher with preferred versions in order
py -3.12 -c "import streamlit" >nul 2>&1
if not errorlevel 1 set "PY=py -3.12"

if "%PY%"=="" (
    py -3.11 -c "import streamlit" >nul 2>&1
    if not errorlevel 1 set "PY=py -3.11"
)

if "%PY%"=="" (
    py -3.13 -c "import streamlit" >nul 2>&1
    if not errorlevel 1 set "PY=py -3.13"
)

if "%PY%"=="" (
    py -3.10 -c "import streamlit" >nul 2>&1
    if not errorlevel 1 set "PY=py -3.10"
)

REM Fall back to plain "python" on PATH
if "%PY%"=="" (
    python -c "import streamlit" >nul 2>&1
    if not errorlevel 1 set "PY=python"
)

if "%PY%"=="" (
    echo.
    echo ERROR: Could not find a Python with streamlit installed.
    echo.
    echo Run the installer first using Python 3.12:
    echo     py -3.12 install.py
    echo.
    echo If you don't have Python 3.12, download it from:
    echo     https://www.python.org/downloads/release/python-31210/
    echo.
    pause
    exit /b 1
)

echo Starting NarcPartrol UI with %PY% ...
echo Browser will open to http://localhost:8501
echo Press Ctrl+C to stop the server.
echo.

%PY% -m streamlit run app.py --server.headless=false --browser.gatherUsageStats=false

pause
