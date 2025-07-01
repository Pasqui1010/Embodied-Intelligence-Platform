@echo off
echo ================================================
echo   Embodied Intelligence Platform Setup (Windows)
echo ================================================
echo.

:: Check if we're in WSL or need to redirect to WSL
where wsl >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: WSL is required for this project.
    echo Please install WSL2 with Ubuntu and try again.
    echo Visit: https://docs.microsoft.com/en-us/windows/wsl/install
    pause
    exit /b 1
)

echo Checking WSL status...
wsl -- echo "WSL is available"
if %errorlevel% neq 0 (
    echo ERROR: WSL is not properly configured.
    echo Please ensure WSL2 is installed and running.
    pause
    exit /b 1
)

echo.
echo Running setup in WSL environment...
echo This may take several minutes...
echo.

:: Change to WSL path and run the setup script
wsl -- cd /mnt/c/Users/%USERNAME%/OneDrive/PROJECTS/Robot\ Project/Service && chmod +x scripts/setup_dev_env.sh && ./scripts/setup_dev_env.sh

if %errorlevel% neq 0 (
    echo.
    echo ERROR: Setup failed. Please check the error messages above.
    echo.
    echo Common solutions:
    echo 1. Ensure Docker Desktop is running
    echo 2. Ensure WSL2 Ubuntu distribution is installed
    echo 3. Check that virtualization is enabled in BIOS
    echo.
    pause
    exit /b 1
)

echo.
echo ================================================
echo   Setup completed successfully!
echo ================================================
echo.
echo Next steps:
echo 1. Open WSL terminal: wsl
echo 2. Navigate to project: cd /mnt/c/Users/%USERNAME%/OneDrive/PROJECTS/Robot\ Project/Service
echo 3. Start demo: docker-compose up demo-slam
echo.
echo Or use the Quick Start guide: QUICK_START.md
echo.
pause 