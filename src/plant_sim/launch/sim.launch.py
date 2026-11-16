"""Gazebo arena + TB3/pole/tilt-camera + ROS bridge + robot_state_publisher (+ RViz).

    ros2 launch plant_sim sim.launch.py                # GUI + RViz
    ros2 launch plant_sim sim.launch.py headless:=true rviz:=false

The simulation is isolated from other machines on the LAN (a foreign Gazebo/TurtleBot3 sim was seen
there): ROS_DOMAIN_ID=42 + ROS_LOCALHOST_ONLY=1 and a private ign-transport partition on loopback.
Other terminals must use the same:  source ~/plant_ws/env.sh
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (AppendEnvironmentVariable, DeclareLaunchArgument,
                            IncludeLaunchDescription, SetEnvironmentVariable, TimerAction)
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg = get_package_share_directory('plant_sim')
    ros_gz_sim = get_package_share_directory('ros_gz_sim')
    gz_sim_launch = os.path.join(ros_gz_sim, 'launch', 'gz_sim.launch.py')

    world = os.path.join(pkg, 'worlds', 'plant_arena.sdf')
    urdf = os.path.join(pkg, 'urdf', 'tb3_plant.urdf.xacro')
    sdf = os.path.join(pkg, 'urdf', 'tb3_plant.sdf.xacro')
    bridge_yaml = os.path.join(pkg, 'config', 'bridge.yaml')
    rviz_cfg = os.path.join(pkg, 'config', 'plant_view.rviz')
    saved_map = os.path.join(pkg, 'maps', 'plant_arena.yaml')      # written by auto_map.py

    headless = LaunchConfiguration('headless')
    use_map = IfCondition(PythonExpression(["'", LaunchConfiguration('map'), "' not in ('', 'none')"]))
    return LaunchDescription([
        DeclareLaunchArgument('map', default_value=saved_map if os.path.exists(saved_map) else '',
                              description="saved 2D occupancy map (.yaml) to publish on /map; 'none' = no map "
                                          "(mapping.launch.py passes '' because slam_toolbox publishes /map)"),
        DeclareLaunchArgument('headless', default_value='false',
                              description='true: Gazebo server only (no Gazebo GUI)'),
        DeclareLaunchArgument('rviz', default_value='true'),
        DeclareLaunchArgument('x', default_value='0.0'),
        DeclareLaunchArgument('y', default_value='0.0'),
        DeclareLaunchArgument('yaw', default_value='0.0'),
        DeclareLaunchArgument('domain', default_value='42', description='ROS_DOMAIN_ID of the simulation'),
        DeclareLaunchArgument('partition', default_value='plant_sim', description='ign-transport partition'),

        # --- isolation from the LAN (change domain/partition only to run a second sim, e.g. a test)
        SetEnvironmentVariable('ROS_DOMAIN_ID', LaunchConfiguration('domain')),
        SetEnvironmentVariable('ROS_LOCALHOST_ONLY', '1'),
        SetEnvironmentVariable('IGN_PARTITION', LaunchConfiguration('partition')),
        SetEnvironmentVariable('IGN_IP', '127.0.0.1'),
        AppendEnvironmentVariable('IGN_GAZEBO_RESOURCE_PATH', os.path.join(pkg, 'models')),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(gz_sim_launch),
            launch_arguments={'gz_args': ['-r -s -v2 ', world], 'on_exit_shutdown': 'true'}.items()),
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(gz_sim_launch),
            launch_arguments={'gz_args': '-g -v2 ', 'on_exit_shutdown': 'true'}.items(),
            condition=UnlessCondition(headless)),

        Node(package='robot_state_publisher', executable='robot_state_publisher', output='screen',
             parameters=[{'use_sim_time': True,
                          'robot_description': ParameterValue(Command(['xacro ', urdf]), value_type=str)}]),

        TimerAction(period=4.0, actions=[Node(
            package='ros_gz_sim', executable='create', output='screen',
            arguments=['-world', 'plant_arena', '-name', 'tb3_plant',
                       '-string', Command(['xacro ', sdf]),
                       '-x', LaunchConfiguration('x'), '-y', LaunchConfiguration('y'), '-z', '0.01',
                       '-Y', LaunchConfiguration('yaw')])]),

        Node(package='ros_gz_bridge', executable='parameter_bridge', output='screen',
             parameters=[{'config_file': bridge_yaml}]),

        # velocity arbitration: /cmd_vel_teleop (100) > /cmd_vel_nav_out (Nav2, 50) > /cmd_vel (10) -> /cmd_vel_mux
        Node(package='twist_mux', executable='twist_mux', name='twist_mux', output='screen',
             parameters=[os.path.join(pkg, 'config', 'twist_mux.yaml'), {'use_sim_time': True}],
             remappings=[('cmd_vel_out', 'cmd_vel_mux')]),

        Node(package='plant_sim', executable='cmd_vel_watchdog.py', name='cmd_vel_watchdog', output='screen',
             parameters=[{'use_sim_time': True}]),

        # saved occupancy map on /map.  slam_toolbox started at the world origin, so map == odom == world:
        # an identity map -> odom transform places the saved map correctly.
        Node(package='plant_sim', executable='map_publisher.py', name='map_publisher', output='screen',
             parameters=[{'yaml_filename': LaunchConfiguration('map'), 'use_sim_time': True,
                          'keepout_plants': os.path.join(pkg, 'gt', 'plants.json'), 'keepout_radius': 0.95}],
             condition=use_map),
        Node(package='tf2_ros', executable='static_transform_publisher', name='map_to_odom',
             arguments=['0', '0', '0', '0', '0', '0', 'map', 'odom'],
             parameters=[{'use_sim_time': True}], condition=use_map),

        # colour + depth -> PointCloud2 (coloured and black & white), like the ZED2i cloud_registered
        Node(package='plant_sim', executable='depth_to_cloud.py', output='screen',
             parameters=[{'use_sim_time': True}]),

        Node(package='rviz2', executable='rviz2', arguments=['-d', rviz_cfg], output='log',
             parameters=[{'use_sim_time': True}], condition=IfCondition(LaunchConfiguration('rviz'))),
    ])
