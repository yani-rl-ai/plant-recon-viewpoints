"""Simulation + slam_toolbox (online async, mapping mode) for the 2D occupancy map.

    ros2 launch plant_sim mapping.launch.py [headless:=true rviz:=false domain:=43 partition:=plant_test]
    ros2 run plant_sim auto_map.py          # drives the route and saves maps/plant_arena.{pgm,yaml}

slam_toolbox publishes /map and the map -> odom transform; the saved map is NOT loaded here (map:='').
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('plant_sim')
    args = {'headless': 'false', 'rviz': 'true', 'domain': '42', 'partition': 'plant_sim'}
    sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg, 'launch', 'sim.launch.py')),
        launch_arguments={**{k: LaunchConfiguration(k) for k in args}, 'map': ''}.items())
    slam = Node(package='slam_toolbox', executable='async_slam_toolbox_node', name='slam_toolbox',
                output='screen', parameters=[os.path.join(pkg, 'config', 'slam_toolbox_mapping.yaml'),
                                             {'use_sim_time': True}])
    return LaunchDescription([DeclareLaunchArgument(k, default_value=v) for k, v in args.items()] + [sim, slam])
