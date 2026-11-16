#!/bin/bash
# Stop every plant_sim process (Gazebo, bridge, SLAM, map publisher, RViz of this sim, ...).
#
# Two safety rules:
#  * every pattern is anchored at the START of the command line ("^..."), so it never matches the shell that
#    calls this script (whose command line may contain the text "ros2 launch plant_sim");
#  * only processes that carry IGN_PARTITION=plant_* in their environment are killed, i.e. processes started by
#    the plant_sim launch files / env.sh.  Your other ROS 2 / Gazebo / SLAM sessions are left alone.
PATTERNS=(
  "^/usr/bin/python3 /opt/ros/humble/bin/ros2 launch plant_sim"
  "^/bin/sh -c ruby /usr/bin/ign gazebo"
  "^ign gazebo"
  "^/opt/ros/humble/lib/ros_gz_bridge/parameter_bridge"
  "^/opt/ros/humble/lib/robot_state_publisher/robot_state_publisher"
  "^/opt/ros/humble/lib/slam_toolbox/async_slam_toolbox_node"
  "^/opt/ros/humble/lib/tf2_ros/static_transform_publisher 0 0 0 0 0 0 map odom"
  "^/opt/ros/humble/lib/nav2_"
  "^/opt/ros/humble/lib/twist_mux/"
  "^(/usr/bin/)?python3 .*plant_sim/lib/plant_sim/(depth_to_cloud|map_publisher|cmd_vel_watchdog|nav2_bringup|scan_manager|collect_dataset).py"
  "^/opt/ros/humble/lib/rviz2/rviz2 -d .*/plant_(view|nav|scan).rviz"
)

mine() {   # pids matching the patterns that belong to this simulation
  for p in "${PATTERNS[@]}"; do
    for pid in $(pgrep -f "$p"); do
      [ "$pid" = "$$" ] && continue
      tr '\0' '\n' < "/proc/$pid/environ" 2>/dev/null | grep -q '^IGN_PARTITION=plant_' && echo "$pid"
    done
  done | sort -u
}

pids=$(mine)
[ -n "$pids" ] && kill $pids 2>/dev/null
for i in 1 2 3 4 5 6; do sleep 0.5; [ -z "$(mine)" ] && break; done
pids=$(mine)
[ -n "$pids" ] && kill -9 $pids 2>/dev/null && sleep 1          # stubborn ones (Gazebo ignores SIGTERM at times)
echo "plant_sim processes left: $(mine | wc -l)"
