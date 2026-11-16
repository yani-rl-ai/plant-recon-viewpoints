#!/usr/bin/env python3
"""Where does odometry diverge from Gazebo's true pose?  Spin 360 deg like auto_map, then drive straight."""
import math
import re
import subprocess
import time

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node


def gz_pose():
    out = subprocess.run(['ign', 'topic', '-e', '-n', '1', '-t', '/world/plant_arena/dynamic_pose/info'],
                         capture_output=True, text=True, timeout=15).stdout
    m = re.search(r'name: "tb3_plant"\s+id: \d+\s+position \{\s+x: (\S+)\s+y: (\S+)\s+z: (\S+)\s+\}\s+'
                  r'orientation \{\s+x: (\S+)\s+y: (\S+)\s+z: (\S+)\s+w: (\S+)', out)
    x, y, z, qx, qy, qz, qw = map(float, m.groups())
    return x, y, math.degrees(math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))), qx, qy


class D(Node):
    def __init__(self):
        super().__init__('debug_spin', parameter_overrides=[rclpy.parameter.Parameter(
            'use_sim_time', rclpy.Parameter.Type.BOOL, True)])
        self.odom = None
        self.create_subscription(Odometry, '/odom', lambda m: setattr(self, 'odom', m), 10)
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)

    def sp(self, s=0.02):
        t = time.time()
        while time.time() - t < s:
            rclpy.spin_once(self, timeout_sec=0.01)

    def pose(self):
        o = self.odom.pose.pose
        q = o.orientation
        return o.position.x, o.position.y, math.degrees(math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)))


def show(tag, n):
    ox, oy, oyaw = n.pose()
    gx, gy, gyaw, qx, qy = gz_pose()
    d = (gyaw - oyaw + 180) % 360 - 180
    print(f'{tag:28s} odom ({ox:+.2f},{oy:+.2f}, {oyaw:+7.1f}) | true ({gx:+.2f},{gy:+.2f}, {gyaw:+7.1f}) | '
          f'yaw diff {d:+6.1f} deg, pos diff {math.hypot(gx - ox, gy - oy):.2f} m | roll/pitch q=({qx:+.3f},{qy:+.3f})', flush=True)


rclpy.init()
n = D()
while n.odom is None:
    n.sp(0.2)
n.sp(1.0)
show('start', n)
t = Twist()
w = float(__import__('sys').argv[1]) if len(__import__('sys').argv) > 1 else 0.5
done, last, k = 0.0, n.pose()[2], 0
t.angular.z = w
t0 = time.time()
print(f'--- spinning at {w} rad/s')
while done < 360 - 3:
    n.pub.publish(t)
    n.sp(0.05)
    y = n.pose()[2]
    done += abs((y - last + 180) % 360 - 180)
    last = y
    if time.time() - t0 > 4 * (k + 1):
        k += 1
        show(f'  spin, odom turned {done:5.0f} deg', n)
n.pub.publish(Twist())
n.sp(1.5)
show('after full turn', n)
print('--- straight 1.5 m')
t = Twist()
t.linear.x = 0.22
x0 = n.pose()[0]
while n.pose()[0] - x0 < 1.5:
    n.pub.publish(t)
    n.sp(0.05)
n.pub.publish(Twist())
n.sp(1.5)
show('after 1.5 m straight', n)
rclpy.shutdown()
