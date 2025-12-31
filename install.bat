@echo off
REM FSAE Data Extractor - Windows Installation Script
REM This script sets up the Python environment and installs all dependencies

echo ================================================
echo   FSAE Data Extractor - Installation
echo ================================================
echo.

REM Check for Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.9+ from https://www.python.org/downloads/
    pause
    exit /b 1
)

echo [1/4] Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)

echo [2/4] Activating virtual environment...
call venv\Scripts\activate.bat

echo [3/4] Upgrading pip...
python -m pip install --upgrade pip

echo [4/4] Installing dependencies...
echo       This may take several minutes for OCR models...
echo.

REM Install PaddlePaddle first (often has issues on Windows)
pip install paddlepaddle -i https://pypi.tuna.tsinghua.edu.cn/simple

REM Install remaining requirements
pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo WARNING: Some packages may have failed to install.
    echo The app will fall back to EasyOCR if PaddleOCR is unavailable.
    echo.
)

echo.
echo ================================================
echo   Installation Complete!
echo ================================================
echo.
echo To run the application:
echo   1. Activate the virtual environment: venv\Scripts\activate
echo   2. Run: python run.py
echo.
echo To build a standalone executable:
echo   python build.py
echo.
pause
