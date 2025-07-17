cmake_minimum_required(VERSION 3.8)
project(eip_cognitive_architecture)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(rclpy REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(eip_interfaces REQUIRED)
find_package(eip_llm_interface REQUIRED)
find_package(eip_reasoning_engine REQUIRED)
find_package(eip_vlm_grounding REQUIRED)
find_package(eip_adaptive_safety REQUIRED)
find_package(eip_multimodal_safety REQUIRED)

# Install Python modules
ament_python_install_package(${PROJECT_NAME})

# Install Python executables
install(PROGRAMS
  scripts/cognitive_architecture_node
  DESTINATION lib/${PROJECT_NAME}
)

# Install launch files
install(DIRECTORY
  launch
  config
  DESTINATION share/${PROJECT_NAME}
)

# Install Python tests
install(PROGRAMS
  tests/test_attention_mechanism.py
  tests/test_memory_systems.py
  tests/test_executive_control.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package() 