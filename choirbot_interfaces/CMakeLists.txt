cmake_minimum_required(VERSION 3.5)
project(choirbot_interfaces)

# ✅ CORREZIONE: Find packages corretti per interfacce
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(action_msgs REQUIRED)

# ✅ CORREZIONE: Genera interfacce con DIPENDENZE specificate
rosidl_generate_interfaces(${PROJECT_NAME}
  "action/PositionAction.action"
  "msg/PositionTask.msg"
  "msg/PositionTaskArray.msg"
  "srv/PositionTaskService.srv"
  "srv/TaskCompletionService.srv"
  "msg/Keypoint.msg"
  "msg/KeypointArray.msg"
  "msg/KeypointMatch.msg"
  "msg/KeypointPair.msg"
  DEPENDENCIES std_msgs geometry_msgs action_msgs
)

# ✅ CORREZIONE: Export delle dipendenze runtime
ament_export_dependencies(rosidl_default_runtime)

# ✅ CORREZIONE: Testing
if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()