#!/bin/bash

# Embodied Intelligence Platform - Development Environment Setup
# This script sets up the development environment for the EIP project

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running on supported OS
check_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        log_info "Detected Linux system"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        log_warning "macOS detected - some features may require Docker"
    else
        log_error "Unsupported OS: $OSTYPE"
        exit 1
    fi
}

# Check if Docker is installed and running
check_docker() {
    log_info "Checking Docker installation..."
    
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed. Please install Docker first."
        echo "Visit: https://docs.docker.com/get-docker/"
        exit 1
    fi
    
    if ! docker info &> /dev/null; then
        log_error "Docker daemon is not running. Please start Docker."
        exit 1
    fi
    
    log_success "Docker is installed and running"
}

# Check if Docker Compose is available
check_docker_compose() {
    log_info "Checking Docker Compose..."
    
    if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
        log_error "Docker Compose is not available. Please install it."
        echo "Visit: https://docs.docker.com/compose/install/"
        exit 1
    fi
    
    log_success "Docker Compose is available"
}

# Setup environment variables
setup_env_vars() {
    log_info "Setting up environment variables..."
    
    # Create .env file if it doesn't exist
    if [ ! -f .env ]; then
        log_info "Creating .env file..."
        cat > .env << EOF
# Display for GUI applications
DISPLAY=${DISPLAY:-:0}

# ROS 2 Domain ID (change if you have multiple robots)
ROS_DOMAIN_ID=42

# API Keys (set these for LLM functionality)
# OPENAI_API_KEY=your_openai_api_key_here
# HF_TOKEN=your_huggingface_token_here

# Development settings
PYTHONPATH=/workspace
NVIDIA_VISIBLE_DEVICES=all
NVIDIA_DRIVER_CAPABILITIES=all
EOF
        log_warning "Created .env file. Please set your API keys for full functionality."
    else
        log_success ".env file already exists"
    fi
}

# Setup pre-commit hooks
setup_pre_commit() {
    log_info "Setting up pre-commit hooks..."
    
    if [ -f .pre-commit-config.yaml ]; then
        if command -v pre-commit &> /dev/null; then
            pre-commit install
            log_success "Pre-commit hooks installed"
        else
            log_warning "pre-commit not found. Install with: pip install pre-commit"
        fi
    else
        log_warning "No pre-commit configuration found"
    fi
}

# Create initial directory structure
create_directories() {
    log_info "Creating directory structure..."
    
    directories=(
        "core/eip_slam"
        "core/eip_perception"
        "core/eip_navigation"
        "core/eip_manipulation"
        "intelligence/eip_llm_interface"
        "intelligence/eip_vlm_grounding"
        "intelligence/eip_task_planning"
        "intelligence/eip_safety_arbiter"
        "social/eip_hri_core"
        "social/eip_social_perception"
        "social/eip_proactive_assistance"
        "social/eip_social_norms"
        "learning/eip_shadow_learning"
        "learning/eip_experience_buffer"
        "learning/eip_model_validation"
        "simulation/environments"
        "simulation/scenarios"
        "simulation/synthetic_data"
        "integration/eip_orchestrator"
        "integration/eip_config_manager"
        "integration/eip_monitoring"
        "hardware/robot_configs"
        "hardware/drivers"
        "hardware/deployment"
        "examples/01_basic_slam"
        "examples/02_simple_commands"
        "examples/03_proactive_assistance"
        "examples/04_continuous_learning"
        "benchmarks/slam_benchmarks"
        "benchmarks/safety_benchmarks"
        "benchmarks/hri_benchmarks"
        "benchmarks/integration_tests"
        "docs/architecture"
        "docs/tutorials"
        "docs/api_reference"
        "docs/research_papers"
        "tools/data_collection"
        "tools/visualization"
        "tools/deployment"
        "docker/development"
        "docker/simulation"
        "docker/deployment"
    )
    
    for dir in "${directories[@]}"; do
        if [ ! -d "$dir" ]; then
            mkdir -p "$dir"
            # Create placeholder files for git
            touch "$dir/.gitkeep"
        fi
    done
    
    log_success "Directory structure created"
}

# Build Docker images
build_docker_images() {
    log_info "Building Docker development image..."
    
    # Create basic Dockerfile for development if it doesn't exist
    if [ ! -f docker/development/Dockerfile ]; then
        log_info "Creating development Dockerfile..."
        cat > docker/development/Dockerfile << 'EOF'
FROM ubuntu:22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    vim \
    build-essential \
    cmake \
    python3 \
    python3-pip \
    python3-dev \
    software-properties-common \
    lsb-release \
    gnupg2 \
    && rm -rf /var/lib/apt/lists/*

# Install ROS 2 Humble
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null \
    && apt-get update \
    && apt-get install -y \
        ros-humble-desktop \
        ros-humble-rmw-fastrtps-cpp \
        ros-humble-rmw-cyclonedx-cpp \
        ros-dev-tools \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip3 install --no-cache-dir \
    torch \
    torchvision \
    transformers \
    opencv-python \
    numpy \
    scipy \
    matplotlib \
    jupyter \
    pytest \
    pytest-benchmark \
    pre-commit \
    sphinx \
    sphinx-autobuild

# Create workspace
WORKDIR /workspace

# Source ROS 2 in bashrc
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc

CMD ["/bin/bash"]
EOF
    fi
    
    # Build the development image
    if ! docker build -t eip-dev:latest docker/development/; then
        log_error "Failed to build development Docker image"
        exit 1
    fi
    
    log_success "Development Docker image built successfully"
}

# Initialize workspace
init_workspace() {
    log_info "Initializing ROS 2 workspace..."
    
    # Create colcon workspace structure if not exists
    if [ ! -f colcon.meta ]; then
        cat > colcon.meta << 'EOF'
{
    "names": {
        "eip_slam": {
            "cmake-args": [
                "-DCMAKE_BUILD_TYPE=RelWithDebInfo"
            ]
        },
        "eip_safety_arbiter": {
            "cmake-args": [
                "-DCMAKE_BUILD_TYPE=Debug",
                "-DENABLE_SAFETY_CHECKS=ON"
            ]
        }
    }
}
EOF
    fi
    
    log_success "Workspace initialized"
}

# Main setup function
main() {
    echo "=================================================="
    echo "  Embodied Intelligence Platform Setup"
    echo "=================================================="
    echo
    
    check_os
    check_docker
    check_docker_compose
    setup_env_vars
    create_directories
    setup_pre_commit
    build_docker_images
    init_workspace
    
    echo
    log_success "Development environment setup complete!"
    echo
    echo "Next steps:"
    echo "1. Set your API keys in .env file for LLM functionality"
    echo "2. Start development environment: docker-compose up dev-env"
    echo "3. Run basic SLAM demo: docker-compose up demo-slam"
    echo "4. Check documentation at: http://localhost:8080 (after running: docker-compose up docs)"
    echo
    echo "For more information, see docs/tutorials/quickstart.md"
}

# Run main function
main "$@" 