@echo off
setlocal enabledelayedexpansion


echo.
echo ===============================================
echo        RAPID Windows Installer
echo ===============================================
echo.


REM -------------------------------------------------
REM Step 0 - Move to repository root
REM -------------------------------------------------
cd /d "%~dp0\.."
echo Working directory (repo root): %cd%
echo.


REM -------------------------------------------------
REM Step 1 - Download micromamba (Windows 64-bit)
REM -------------------------------------------------
set MAMBA_DIR=%cd%\tools\micromamba
set MAMBA_EXE=%MAMBA_DIR%\micromamba.exe

if not exist "%MAMBA_EXE%" (
    echo [Step 1] Downloading micromamba...
    mkdir "%MAMBA_DIR%" 2>nul

    powershell -Command "Invoke-WebRequest -Uri 'https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-win-64' -OutFile '%MAMBA_EXE%'"

    if %errorlevel% neq 0 (
        echo Failed to download micromamba.
        pause
        exit /b
    )

    echo Micromamba ready.
) else (
    echo [Step 1] Micromamba already present.
)

echo.


REM -------------------------------------------------
REM Step 2 - Create environment inside project
REM -------------------------------------------------
echo [Step 2] Creating environment in .env folder...

if not exist "windows_requirements.yaml" (
    echo windows_requirements.yaml not found in repository root.
    pause
    exit /b
)

"%MAMBA_EXE%" create -y -f windows_requirements.yaml -p "%cd%\.env"

if %errorlevel% neq 0 (
    echo Failed to create environment.
    pause
    exit /b
)

echo Environment created successfully.
echo.


REM -------------------------------------------------
REM Step 3 - Install pip-only packages
REM -------------------------------------------------
echo [Step 3] Installing pip-only packages...

"%MAMBA_EXE%" run -p "%cd%\.env" python -m pip install --upgrade pip
"%MAMBA_EXE%" run -p "%cd%\.env" python -m pip install pynvml==12.0.0
"%MAMBA_EXE%" run -p "%cd%\.env" python -m pip install uv

if %errorlevel% neq 0 (
    echo Failed to install pip packages.
    pause
    exit /b
)

echo.


REM -------------------------------------------------
REM Step 4 - Install RAPID package
REM -------------------------------------------------
echo [Step 4] Installing RAPID...

"%MAMBA_EXE%" run -p "%cd%\.env" uv pip install --editable .

if %errorlevel% neq 0 (
    echo Failed to install RAPID.
    pause
    exit /b
)

echo.


echo ===============================================
echo      Installation completed successfully!
echo ===============================================
echo.

echo To run RAPID (or FalseTagFinder):
echo 1. update database and query paths in ..\config\config_RAPID (or ..\config\config_FalseTagFinder)
echo 2. double click run_RAPID_windows (or run_FalseTagFinder_windows) in the main folder
echo 3. Enjoy results :)
echo.

pause
