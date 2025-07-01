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
    x11-apps \
    mesa-utils \
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
        ros-humble-gazebo-ros-pkgs \
        ros-humble-navigation2 \
        ros-humble-nav2-bringup \
        ros-humble-turtlebot3-gazebo \
        python3-colcon-common-extensions \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages for robotics and AI
RUN pip3 install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    transformers \
    opencv-python \
    numpy \
    scipy \
    matplotlib \
    ultralytics \
    open3d \
    pytest \
    pytest-benchmark

# Install PCL and OpenCV development libraries
RUN apt-get update && apt-get install -y \
    libpcl-dev \
    pcl-tools \
    libopencv-dev \
    python3-opencv \
    && rm -rf /var/lib/apt/lists/*

# Create workspace
WORKDIR /workspace

# Source ROS 2 in bashrc
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
RUN echo "export GAZEBO_MODEL_PATH=/opt/ros/humble/share/turtlebot3_gazebo/models" >> ~/.bashrc

# Environment variables for simulation
ENV TURTLEBOT3_MODEL=waffle_pi
ENV GAZEBO_MODEL_PATH=/opt/ros/humble/share/turtlebot3_gazebo/models

CMD ["/bin/bash"] 