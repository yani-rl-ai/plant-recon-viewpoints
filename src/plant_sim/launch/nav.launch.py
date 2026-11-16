"""Gazebo + RViz + saved 2D map + Nav2 (AMCL localisation on the saved map, DWB controller).

    ros2 launch plant_sim nav.launch.py                        # GUI + RViz (top-down, costmaps, goal tool)
    ros2 launch plant_sim nav.launch.py headless:=true rviz:=false domain:=43 partition:=plant_test
    ros2 run plant_sim teleop_keyboard.py                      # keyboard overrides Nav2 while you hold a speed

Everything of sim.launch.py is included (Gazebo, robot, bridge, point clouds, twist_mux, /map from the saved
map via map_publisher.py, map -> odom identity).  Added here: AMCL and the Nav2 servers.  The robot spawns at
the map origin (0, 0, 0), where the map was made, so AMCL starts with that pose: no "2D Pose Estimate" needed.

The Nav2 nodes are activated by scripts/nav2_bringup.py instead of nav2_lifecycle_manager: in this setup the
response to a change_state call is sometimes lost (the node changed state, the manager never hears about it
and waits forever; seen with amcl and with map_server).  nav2_bringup.py re-checks with get_state and retries.
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, SetLaunchConfiguration
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg = get_package_share_directory('plant_sim')
    params = os.path.join(pkg, 'config', 'nav2_params.yaml')
    rviz_cfg = os.path.join(pkg, 'config', 'plant_nav.rviz')
    args = {'headless': 'false', 'domain': '42', 'partition': 'plant_sim', 'x': '0.0', 'y': '0.0', 'yaw': '0.0'}

    # Arguments given to an included launch file leak into this one (launch configurations are shared), and so do the
    # environment variables it sets (domain, partition).  Hence: no scoping, but the user's `rviz` value is saved in
    # `nav_rviz` BEFORE sim.launch.py is told rviz:='false' (RViz is started below with the Nav2 config).
    save_rviz = SetLaunchConfiguration('nav_rviz', LaunchConfiguration('rviz'))
    sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(pkg, 'launch', 'sim.launch.py')),
        launch_arguments={**{k: LaunchConfiguration(k) for k in args}, 'rviz': 'false'}.items())

    sim_time = {'use_sim_time': True}
    log = ['--ros-args', '--log-level', 'info']

    def server(package, executable, name=None, remap=()):
        return Node(package=package, executable=executable, name=name or executable, output='screen',
                    parameters=[params], arguments=log, remappings=list(remap))

    initial_pose = {'initial_pose.x': ParameterValue(LaunchConfiguration('x'), value_type=float),
                    'initial_pose.y': ParameterValue(LaunchConfiguration('y'), value_type=float),
                    'initial_pose.yaw': ParameterValue(LaunchConfiguration('yaw'), value_type=float)}
    amcl = Node(package='nav2_amcl', executable='amcl', name='amcl', output='screen',
                parameters=[params, initial_pose], arguments=log)
    nav_nodes = [
        server('nav2_controller', 'controller_server', remap=[('cmd_vel', 'cmd_vel_nav')]),
        server('nav2_smoother', 'smoother_server'),
        server('nav2_planner', 'planner_server'),
        server('nav2_behaviors', 'behavior_server'),
        server('nav2_bt_navigator', 'bt_navigator'),
        server('nav2_waypoint_follower', 'waypoint_follower'),
        server('nav2_velocity_smoother', 'velocity_smoother',
               remap=[('cmd_vel', 'cmd_vel_nav'), ('cmd_vel_smoothed', 'cmd_vel_nav_out')]),
    ]
    # NOT nav2_lifecycle_manager: it hangs for good when the response to one change_state call is lost (seen
    # with amcl).  nav2_bringup.py verifies each node with get_state and repeats until it is ACTIVE.
    bringup = Node(package='plant_sim', executable='nav2_bringup.py', name='nav2_bringup', output='screen',
                   parameters=[sim_time, {'with_amcl': True}])
    rviz = Node(package='rviz2', executable='rviz2', arguments=['-d', rviz_cfg], output='log', parameters=[sim_time],
                condition=IfCondition(LaunchConfiguration('nav_rviz')))

    return LaunchDescription(
        [DeclareLaunchArgument(k, default_value=v) for k, v in args.items()] +
        [DeclareLaunchArgument('rviz', default_value='true'),
         DeclareLaunchArgument('localizer', default_value='amcl', description='AMCL is the only localiser in this repository; the argument is accepted so that localizer:=amcl can be given'),
         save_rviz, sim, amcl] + nav_nodes + [bringup, rviz])
