cmake_minimum_required(VERSION 3.8)
project(eip_orchestrator)

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
find_package(eip_interfaces REQUIRED)

# Install Python modules
install(
  DIRECTORY eip_orchestrator/
  DESTINATION lib/${PROJECT_NAME}
  USE_SOURCE_PERMISSIONS
)

# Install Python package
install(
  DIRECTORY eip_orchestrator/
  DESTINATION lib/python3.10/site-packages/${PROJECT_NAME}
  USE_SOURCE_PERMISSIONS
)

# Install launch files
install(
  DIRECTORY launch/
  DESTINATION share/${PROJECT_NAME}/launch
)

# Install resource files
install(
  DIRECTORY resource/
  DESTINATION share/${PROJECT_NAME}/resource
)

# Install Python package
ament_python_install_package(${PROJECT_NAME})

ament_package() 