cmake_minimum_required(VERSION 3.8)
project(eip_interfaces)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Define messages and services
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/SafetyVerificationRequest.msg"
  "msg/SafetyVerificationResponse.msg"
  "msg/TaskPlan.msg"
  "msg/TaskStep.msg"
  "msg/SafetyViolation.msg"
  "msg/EmergencyStop.msg"
  "srv/ValidateTaskPlan.srv"
  DEPENDENCIES builtin_interfaces geometry_msgs std_msgs
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package() 