@echo off
setlocal

REM -----------------------------------------------------
REM RAPID Windows Setup Script - Safe for General PCs
REM -----------------------------------------------------

REM Step 0: Go to script directory
echo [Step 0] Entering script folder...
cd /d "%~dp0" || (
    echo Failed to enter script directory.
    pause
    exit /b
)

echo Current directory: %cd%

REM Step 1: Check Python
echo [Step 1] Checking Python version...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python 3.10+ not found. Please install Python first.
    pause
    exit /b
)

echo Python found.

REM Step 2: Check Conda
echo [Step 2] Checking Conda installation...
where conda >nul 2>&1
if %errorlevel% neq 0 (
    echo Conda not found. Please install Miniconda:
    echo https://docs.conda.io/en/latest/miniconda.html
    pause
    exit /b
)

echo Conda found.

REM Step 3: Create conda environment
echo [Step 3] Creating Conda environment 'rapid_env'...
if not exist "%~dp0..\windows_requirements.yaml" (
    echo windows_requirements.yaml not found next to installer.
    pause
    exit /b
)

echo Using YAML file: %~dp0windows_requirements.yaml
call conda env create -f "%~dp0..\windows_requirements.yaml"
if %errorlevel% neq 0 (
    echo Failed to create conda environment.
    pause
    exit /b
)

echo Conda environment created successfully.

REM Step 4: Activate environment
echo [Step 4] Activating environment 'rapid_env'...
call conda activate rapid_env
if %errorlevel% neq 0 (
    echo Failed to activate conda environment.
    pause
    exit /b
)

echo Environment activated.

REM Step 5: Install RAPID
echo [Step 5] Installing RAPID in editable mode...
cd /d "%~dp0.." || (
    echo Failed to enter project root directory.
    pause
    exit /b
)
uv pip install --editable .
if %errorlevel% neq 0 (
    echo Failed to install RAPID.
    pause
    exit /b
)

echo.
echo ===============================================
echo  Installation completed successfully!
echo ===============================================
pause
