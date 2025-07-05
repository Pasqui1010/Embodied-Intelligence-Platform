@echo off
REM Comprehensive Build Script for Embodied Intelligence Platform (Windows)
REM Builds all packages in the correct dependency order

echo 🚀 Building Embodied Intelligence Platform...

REM Function to build packages in a directory
:build_packages
set dir=%1
set packages=%2

if exist "%dir%" (
    echo 📦 Building packages in %dir%...
    cd /d "%dir%"
    
    if not "%packages%"=="" (
        colcon build --packages-select %packages% --event-handlers console_direct+
    ) else (
        colcon build --event-handlers console_direct+
    )
    
    REM Source the workspace (Windows equivalent)
    call install\setup.bat
    cd /d ".."
) else (
    echo ⚠️  Directory %dir% not found, skipping...
)
goto :eof

REM Build order: interfaces first, then core, then intelligence, then integration

REM 1. Build interfaces (dependencies for other packages)
call :build_packages "intelligence" "eip_interfaces"

REM 2. Build core packages
call :build_packages "core" "eip_slam"

REM 3. Build intelligence packages
call :build_packages "intelligence" "eip_safety_arbiter"

REM 4. Build integration packages
call :build_packages "integration" "eip_orchestrator"

echo ✅ All packages built successfully!

REM Test the system
echo 🧪 Running safety benchmarks...
cd benchmarks
python -m pytest safety_benchmarks/ -v --tb=short

echo.
echo 🎯 Build completed successfully!
echo.
echo 🚀 To run the demo:
echo    docker-compose up demo-slam
echo.
echo 🔧 To test individual components:
echo    call core\install\setup.bat
echo    ros2 run eip_slam basic_slam_node.py
echo    ros2 run eip_safety_arbiter safety_monitor_node

pause 