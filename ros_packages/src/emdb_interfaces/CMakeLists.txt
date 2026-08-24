cmake_minimum_required(VERSION 3.10)
project(emdb_interfaces)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/SetAction.srv"
  "srv/SetPosition.srv"
  "srv/SetDeltaAction.srv"
  "srv/StepAction.srv"
  "srv/StepActionRaw.srv"
  "srv/ResetEpisode.srv"
  "srv/SaveDemos.srv"
  "srv/MarkObjectRotten.srv"
  "msg/ObjectState.msg"
  "msg/ObjectStateArray.msg"
  "msg/ObservationEntry.msg"
  "msg/Observation.msg"
  "msg/StepInfo.msg"
  DEPENDENCIES std_msgs geometry_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()