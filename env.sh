# source ~/plant_ws/env.sh   -- same isolation as plant_sim's launch files, for EVERY terminal you open
# (ROS 2 domain 42, loopback only, own Gazebo transport partition; keeps the simulation away from other ROS/Gazebo
# sessions on the same network)
_PLANT_WS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ROS_DOMAIN_ID=42
export ROS_LOCALHOST_ONLY=1
export IGN_PARTITION=plant_sim
export IGN_IP=127.0.0.1
source /opt/ros/humble/setup.bash
[ -f "$_PLANT_WS/install/setup.bash" ] && source "$_PLANT_WS/install/setup.bash"
unset _PLANT_WS
