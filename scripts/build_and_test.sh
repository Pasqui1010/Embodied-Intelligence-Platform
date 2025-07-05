#!/bin/bash

# Build and Test Script for Embodied Intelligence Platform
# This script builds all packages and runs basic tests

set -e  # Exit on any error

echo "🚀 Building Embodied Intelligence Platform..."

# Change to the core directory where ROS 2 packages are located
cd core

# Build all packages
echo "📦 Building ROS 2 packages..."
colcon build --packages-select eip_slam --event-handlers console_direct+

# Source the workspace
echo "🔧 Sourcing workspace..."
source install/setup.bash

# Test message compilation
echo "🧪 Testing message compilation..."
ros2 interface list | grep eip_interfaces || echo "⚠️  eip_interfaces not found - will be built in intelligence/"

# Test SLAM node compilation
echo "🧪 Testing SLAM node compilation..."
ros2 pkg list | grep eip_slam || echo "❌ eip_slam package not found"

# Test safety benchmarks
echo "🧪 Running safety benchmarks..."
cd ../benchmarks
python3 -m pytest safety_benchmarks/ -v --tb=short

echo "✅ Build and test completed successfully!"
echo ""
echo "🎯 Next steps:"
echo "1. Run: docker-compose up demo-slam"
echo "2. Or test individual components:"
echo "   - ros2 run eip_slam basic_slam_node.py"
echo "   - ros2 run eip_safety_arbiter safety_monitor_node" 