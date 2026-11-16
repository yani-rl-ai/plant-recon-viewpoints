cmake_minimum_required(VERSION 3.8)
project(plant_sim)

find_package(ament_cmake REQUIRED)

install(DIRECTORY launch urdf config models worlds gt scripts maps
  DESTINATION share/${PROJECT_NAME}
)

install(PROGRAMS scripts/teleop_keyboard.py scripts/check_sim.py scripts/gen_world.py scripts/depth_to_cloud.py
  scripts/auto_map.py scripts/check_map.py scripts/map_publisher.py scripts/check_nav.py scripts/cmd_vel_watchdog.py scripts/nav2_bringup.py scripts/scan_manager.py scripts/collect_dataset.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
