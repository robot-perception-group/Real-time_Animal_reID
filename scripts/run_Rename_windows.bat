@echo off
setlocal

REM -------------------------------------------------
REM Move to repository root (location of this file)
REM -------------------------------------------------
cd /d "%~dp0\.."

REM -------------------------------------------------
REM Define paths
REM -------------------------------------------------
set MAMBA_EXE=%cd%\tools\micromamba\micromamba.exe
set ENV_PATH=%cd%\.env

REM -------------------------------------------------
REM Check micromamba exists
REM -------------------------------------------------
if not exist "%MAMBA_EXE%" (
    echo micromamba not found at:
    echo %MAMBA_EXE%
    pause
    exit /b
)

REM -------------------------------------------------
REM Check environment exists
REM -------------------------------------------------
if not exist "%ENV_PATH%" (
    echo Environment not found at:
    echo %ENV_PATH%
    pause
    exit /b
)

REM -------------------------------------------------
REM Run RAPID
REM -------------------------------------------------
echo Starting Rename...
"%MAMBA_EXE%" run -p "%ENV_PATH%" python src\rapid\tools\rename_imgs.py

echo.
echo Rename finished.
echo.

pause
