@echo off
setlocal

echo ========================================================
echo  Building Audio Filter Design Studio Standalone Executable
echo ========================================================

python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not found in PATH. Please install Python 3.9+.
    exit /b 1
)

python -c "import PyInstaller" >nul 2>&1
if errorlevel 1 (
    echo [INFO] PyInstaller not detected. Installing PyInstaller...
    python -m pip install --upgrade pyinstaller
    if errorlevel 1 (
        echo [ERROR] Failed to install PyInstaller.
        exit /b 1
    )
)

echo [INFO] Compiling single-file windowed executable...
python -m PyInstaller --clean --onefile --noconsole --name FilterDesignStudio app.py

if errorlevel 1 (
    echo [ERROR] Build failed!
    exit /b 1
)

echo.
echo ========================================================
echo  Build Succeeded!
echo  Executable location: %~dp0dist\FilterDesignStudio.exe
echo ========================================================
endlocal
