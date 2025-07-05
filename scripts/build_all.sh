#!/bin/bash

# Comprehensive Build Script for Embodied Intelligence Platform
# Builds all packages in the correct dependency order

set -e  # Exit on any error

echo "🚀 Building Embodied Intelligence Platform..."

# Function to build packages in a directory
build_packages() {
    local dir=$1
    local packages=$2
    
    if [ -d "$dir" ]; then
        echo "📦 Building packages in $dir..."
        cd "$dir"
        
        if [ -n "$packages" ]; then
            colcon build --packages-select $packages --event-handlers console_direct+
        else
            colcon build --event-handlers console_direct+
        fi
        
        # Source the workspace
        source install/setup.bash
        cd ..
    else
        echo "⚠️  Directory $dir not found, skipping..."
    fi
}

# Build order: interfaces first, then core, then intelligence, then integration

# 1. Build interfaces (dependencies for other packages)
build_packages "intelligence" "eip_interfaces"

# 2. Build core packages
build_packages "core" "eip_slam"

# 3. Build intelligence packages
build_packages "intelligence" "eip_safety_arbiter"

# 4. Build integration packages
build_packages "integration" "eip_orchestrator"

echo "✅ All packages built successfully!"

# Test the system
echo "🧪 Running safety benchmarks..."
cd benchmarks
python3 -m pytest safety_benchmarks/ -v --tb=short

echo ""
echo "🎯 Build completed successfully!"
echo ""
echo "🚀 To run the demo:"
echo "   docker-compose up demo-slam"
echo ""
echo "🔧 To test individual components:"
echo "   source core/install/setup.bash"
echo "   ros2 run eip_slam basic_slam_node.py"
echo "   ros2 run eip_safety_arbiter safety_monitor_node" 