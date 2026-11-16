"""Live plant scanning: Gazebo GUI + Nav2 + RViz (2D map, reconstruction, DBL-GNG graph, apples, segmentation overlay)
+ the scan manager that drives the stations and reconstructs the plants.

    ros2 launch plant_sim scan.launch.py                          # plants 1,2,5,4,3,0 (a snake through the arena)
    ros2 launch plant_sim scan.launch.py plants:=1 seg:=oracle    # one plant, ground-truth segmentation (upper bound)
    ros2 launch plant_sim scan.launch.py seg:=color               # colour rule instead of MobileSAM (baseline)

Everything is visible (no headless): the Gazebo window shows the robot driving around the plants, RViz the live result.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('plant_sim')
    args = {'headless': 'false', 'domain': '42', 'partition': 'plant_sim'}
    nav = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg, 'launch', 'nav.launch.py')),
        launch_arguments={**{k: LaunchConfiguration(k) for k in args}, 'rviz': 'false'}.items())
    rviz = Node(package='rviz2', executable='rviz2', arguments=['-d', os.path.join(pkg, 'config', 'plant_scan.rviz')],
                output='log', parameters=[{'use_sim_time': True}], condition=IfCondition(LaunchConfiguration('scan_rviz')))
    manager = Node(package='plant_sim', executable='scan_manager.py', name='scan_manager', output='screen',
                   arguments=['--plants', LaunchConfiguration('plants'), '--radius', LaunchConfiguration('radius'), '--azimuths', LaunchConfiguration('azimuths'),
                              '--tilts', LaunchConfiguration('tilts'), '--seg', LaunchConfiguration('seg'),
                              '--pose-source', LaunchConfiguration('pose_source'), '--use-yolo', LaunchConfiguration('use_yolo')])
    return LaunchDescription(
        [DeclareLaunchArgument(k, default_value=v) for k, v in args.items()] +
        [DeclareLaunchArgument('plants', default_value='1,2,5,4,3,0'),
         DeclareLaunchArgument('radius', default_value='1.4'),
         DeclareLaunchArgument('azimuths', default_value='0,90,180,270', description='station azimuths around each plant [deg]: 0,90,180,270 | 0,120,240 | 0,180'),
         DeclareLaunchArgument('tilts', default_value='0.35,0.0,-0.35', description='rad, positive = looking down'),
         DeclareLaunchArgument('seg', default_value='sam', description='sam (MobileSAM) | color | oracle'),
         DeclareLaunchArgument('pose_source', default_value='odom', description='odom (camera + odometry only) | map (AMCL pose on the saved 2-D map)'),
         DeclareLaunchArgument('use_yolo', default_value='false', description='true: YOLO26n-seg boxes as extra MobileSAM prompts; false: MobileSAM only'),
         DeclareLaunchArgument('scan_rviz', default_value='true'),
         nav, rviz, manager])
