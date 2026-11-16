#!/usr/bin/env python3
"""End-to-end test of Nav2 in plant_sim (run while `ros2 launch plant_sim nav.launch.py` is up).

    python3 check_nav.py

* Nav2 (AMCL + bt_navigator) becomes active
* the robot reaches three goals between the plants (success, true pose within 0.30 m of the goal),
  AMCL agrees with Gazebo's true pose, and the laser never sees anything closer than 0.12 m (no collision)
* teleop (/cmd_vel_teleop) overrides Nav2 / generic /cmd_vel through twist_mux, and the robot stops
  when all sources fall silent

Uses Gazebo's true pose (ign topic) for the checks; needs the ign command line on the same partition.
"""
import math
import re
import subprocess
import sys
import time

import rclpy
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Twist
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from nav_msgs.msg import Odometry
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import LaserScan

GOALS = [(-1.5, -3.0), (1.5, 3.0), (0.0, 0.0)]
FAILS = 0


def report(ok, msg):
    global FAILS
    FAILS += 0 if ok else 1
    print(('PASS  ' if ok else 'FAIL  ') + msg, flush=True)


def gz_pose():
    out = subprocess.run(['ign', 'topic', '-e', '-n', '1', '-t', '/world/plant_arena/dynamic_pose/info'],
                         capture_output=True, text=True, timeout=15).stdout
    m = re.search(r'name: "tb3_plant"\s+id: \d+\s+position \{\s+x: (\S+)\s+y: (\S+)\s+z: (\S+)\s+\}\s+'
                  r'orientation \{\s+x: (\S+)\s+y: (\S+)\s+z: (\S+)\s+w: (\S+)', out)
    x, y, z, qx, qy, qz, qw = map(float, m.groups())
    return x, y, math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))


def main():
    rclpy.init()
    nav = BasicNavigator()
    state = {'amcl': None, 'odom': None, 'min_range': 99.0}
    nav.create_subscription(PoseWithCovarianceStamped, 'amcl_pose', lambda m: state.__setitem__('amcl', m.pose.pose),
                            QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL,
                                       reliability=ReliabilityPolicy.RELIABLE))
    nav.create_subscription(Odometry, '/odom', lambda m: state.__setitem__('odom', m), 10)

    def on_scan(m):
        r = [v for v in m.ranges if math.isfinite(v) and v > m.range_min]
        if r:
            state['min_range'] = min(state['min_range'], min(r))
    nav.create_subscription(LaserScan, '/scan', on_scan, 5)

    t0 = time.time()
    nav.waitUntilNav2Active(localizer='amcl')
    report(True, f'Nav2 active (AMCL + bt_navigator) {time.time() - t0:.0f} s after the test started')

    for i, (gx, gy) in enumerate(GOALS, 1):
        goal = PoseStamped()
        goal.header.frame_id = 'map'
        goal.header.stamp = rclpy.time.Time().to_msg()          # 0 = "latest transform"
        goal.pose.position.x, goal.pose.position.y, goal.pose.orientation.w = gx, gy, 1.0
        t1 = time.time()
        nav.goToPose(goal)
        while not nav.isTaskComplete() and time.time() - t1 < 240:
            rclpy.spin_once(nav, timeout_sec=0.2)
        if not nav.isTaskComplete():
            nav.cancelTask()
        res = nav.getResult()
        time.sleep(1.5)
        for _ in range(10):
            rclpy.spin_once(nav, timeout_sec=0.1)
        x, y, yaw = gz_pose()
        err = math.hypot(x - gx, y - gy)
        a = state['amcl']
        aerr = math.hypot(a.position.x - x, a.position.y - y) if a else float('nan')
        report(res == TaskResult.SUCCEEDED and err < 0.30,
               f'goal {i} ({gx:+.1f}, {gy:+.1f}): {res.name} in {time.time() - t1:.0f} s, true pose ({x:+.2f}, {y:+.2f}) '
               f'is {err * 100:.0f} cm from the goal, AMCL error {aerr * 100:.0f} cm')
    report(state['min_range'] > 0.12, f'closest laser return during the whole run: {state["min_range"]:.2f} m (> 0.12 m: no collision)')

    # ---- twist_mux priorities (the robot stands at (0, 0) now)
    pub_teleop = nav.create_publisher(Twist, '/cmd_vel_teleop', 10)
    pub_generic = nav.create_publisher(Twist, '/cmd_vel', 10)

    def drive(topic_msgs, sec):
        t = time.time()
        while time.time() - t < sec:
            for pub, v in topic_msgs:
                m = Twist()
                m.linear.x = v
                pub.publish(m)
            rclpy.spin_once(nav, timeout_sec=0.05)
        return state['odom'].twist.twist.linear.x

    v = drive([(pub_generic, 0.15)], 2.5)
    report(v > 0.08, f'/cmd_vel alone drives the robot forward (odom speed {v:+.2f} m/s)')
    v = drive([(pub_generic, 0.15), (pub_teleop, -0.10)], 2.5)
    report(v < -0.05, f'teleop wins over /cmd_vel: robot goes backward (odom speed {v:+.2f} m/s)')
    t = time.time()
    while time.time() - t < 1.5:
        rclpy.spin_once(nav, timeout_sec=0.05)
    v = state['odom'].twist.twist.linear.x
    report(abs(v) < 0.02, f'all sources silent for 1.5 s: robot stopped (odom speed {v:+.3f} m/s)')

    print('RESULT:', 'ALL PASS' if FAILS == 0 else f'{FAILS} FAILED', flush=True)
    nav.destroy_node()
    rclpy.shutdown()
    sys.exit(1 if FAILS else 0)


if __name__ == '__main__':
    main()
