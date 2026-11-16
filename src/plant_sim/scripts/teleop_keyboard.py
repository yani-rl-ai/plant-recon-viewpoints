#!/usr/bin/env python3
"""Keyboard teleop for the plant robot: driving + camera tilt.

    source ~/plant_ws/env.sh && ros2 run plant_sim teleop_keyboard.py
    # for MAPPING (slow turns keep the 2D map straight, see docs/MAPPING.md):
    ros2 run plant_sim teleop_keyboard.py --max-lin 0.2 --max-ang 0.25

Keys (the terminal must have focus):
    w / x   forward / backward  (+0.02 / -0.02 m/s per press, max 0.22 m/s like the TB3 Burger)
    a / d   turn left / right   (+0.10 / -0.10 rad/s per press, max 1.0 rad/s)
    (the maxima can be lowered with --max-lin / --max-ang)
    s       stop driving        (space too)
    r / f   camera tilt DOWN / UP by 5 deg      0  camera level
    q       quit (stops the robot)

The command is re-published at 10 Hz while a speed is set (published on /cmd_vel_teleop, twist_mux input
priority 100: it overrides Nav2 while you drive and hands control back ~0.6 s after you stop).
"""
import argparse
import math
import select
import sys
import termios
import tty

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64

MAX_LIN, MAX_ANG = 0.22, 1.0
D_LIN, D_ANG = 0.02, 0.10
TILT_STEP, TILT_LIMIT = math.radians(5.0), math.radians(45.0)

HELP = __doc__.split('Keys')[1].split('The command')[0]


class Teleop(Node):
    def __init__(self, max_lin=MAX_LIN, max_ang=MAX_ANG):
        super().__init__('plant_teleop')
        self.max_lin, self.max_ang = max_lin, max_ang
        self.vel = self.create_publisher(Twist, '/cmd_vel_teleop', 10)   # twist_mux input, priority 100
        self.tilt = self.create_publisher(Float64, '/camera_tilt/cmd', 10)
        self.create_subscription(JointState, '/joint_states', self._js, 10)
        self.lin = self.ang = 0.0
        self.idle = 0
        self.tilt_cmd = 0.0
        self.tilt_now = None

    def _js(self, m):
        if 'camera_tilt_joint' in m.name:
            self.tilt_now = m.position[m.name.index('camera_tilt_joint')]

    def publish(self):
        # While a speed is set, publish it (10 Hz).  After `s` / space send zeros for 0.6 s and then fall
        # silent: twist_mux then hands the robot back to Nav2 instead of being held at standstill by us.
        moving = self.lin != 0.0 or self.ang != 0.0
        self.idle = 0 if moving else self.idle + 1
        if moving or self.idle <= 6:
            t = Twist()
            t.linear.x, t.angular.z = self.lin, self.ang
            self.vel.publish(t)
        self.tilt.publish(Float64(data=self.tilt_cmd))

    def key(self, k):
        if k == 'w':
            self.lin = min(self.lin + D_LIN, self.max_lin)
        elif k == 'x':
            self.lin = max(self.lin - D_LIN, -self.max_lin)
        elif k == 'a':
            self.ang = min(self.ang + D_ANG, self.max_ang)
        elif k == 'd':
            self.ang = max(self.ang - D_ANG, -self.max_ang)
        elif k in ('s', ' '):
            self.lin = self.ang = 0.0
        elif k == 'r':
            self.tilt_cmd = min(self.tilt_cmd + TILT_STEP, TILT_LIMIT)
        elif k == 'f':
            self.tilt_cmd = max(self.tilt_cmd - TILT_STEP, -TILT_LIMIT)
        elif k == '0':
            self.tilt_cmd = 0.0

    def status(self):
        now = 'n/a' if self.tilt_now is None else f'{math.degrees(self.tilt_now):+5.1f}'
        return (f'\rlin {self.lin:+.2f} m/s  ang {self.ang:+.2f} rad/s  |  '
                f'tilt cmd {math.degrees(self.tilt_cmd):+5.1f} deg, actual {now} deg (+ = down)   ')


def main():
    ap = argparse.ArgumentParser(description='keyboard teleop: driving + camera tilt')
    ap.add_argument('--max-lin', type=float, default=MAX_LIN, help=f'max forward speed [m/s] (default {MAX_LIN})')
    ap.add_argument('--max-ang', type=float, default=MAX_ANG, help=f'max turn rate [rad/s] (default {MAX_ANG})')
    args, ros_args = ap.parse_known_args()
    if not sys.stdin.isatty():
        print('teleop_keyboard needs an interactive terminal (stdin is not a tty).')
        sys.exit(1)
    rclpy.init(args=[sys.argv[0]] + ros_args)
    node = Teleop(args.max_lin, args.max_ang)
    old = termios.tcgetattr(sys.stdin)
    print(HELP)
    try:
        tty.setcbreak(sys.stdin.fileno())
        while rclpy.ok():
            if select.select([sys.stdin], [], [], 0.1)[0]:
                k = sys.stdin.read(1)
                if k == 'q':
                    break
                node.key(k)
            node.publish()
            rclpy.spin_once(node, timeout_sec=0.0)
            sys.stdout.write(node.status())
            sys.stdout.flush()
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)
        node.lin = node.ang = 0.0
        for _ in range(5):
            node.vel.publish(Twist())
        print('\nstopped.')
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
