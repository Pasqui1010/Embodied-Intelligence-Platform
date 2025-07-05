cmake_minimum_required(VERSION 3.8)
project(eip_safety_arbiter)

# Find required packages
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)
find_package(action_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(nav2_msgs REQUIRED)
find_package(moveit_msgs REQUIRED)
find_package(eip_interfaces REQUIRED)

# Install Python modules
install(
  DIRECTORY eip_safety_arbiter/
  DESTINATION lib/${PROJECT_NAME}
  USE_SOURCE_PERMISSIONS
)

# Install Python package
install(
  DIRECTORY eip_safety_arbiter/
  DESTINATION lib/python3.10/site-packages/${PROJECT_NAME}
  USE_SOURCE_PERMISSIONS
)

# Install Python package
ament_python_install_package(${PROJECT_NAME})

# Install Python executables
install(
  PROGRAMS
    eip_safety_arbiter/safety_monitor.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package() 