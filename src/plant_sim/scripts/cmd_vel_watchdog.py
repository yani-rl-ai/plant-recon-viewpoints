#!/usr/bin/env python3
"""Dead-man watchdog between twist_mux and Gazebo:  /cmd_vel_mux -> /cmd_vel_gz.

twist_mux only STOPS PUBLISHING when every source has timed out, and Gazebo's DiffDrive then keeps driving at
the last command (a closed teleop terminal or a crashed script would leave the robot running).  This node
forwards fresh commands and publishes zero velocity whenever nothing new arrived for `timeout` seconds.
"""
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node


class Watchdog(Node):
    def __init__(self):
        super().__init__('cmd_vel_watchdog')
        self.timeout = self.declare_parameter('timeout', 0.5).value
        self.last = None
        self.last_time = None
        self.pub = self.create_publisher(Twist, '/cmd_vel_gz', 10)
        self.create_subscription(Twist, '/cmd_vel_mux', self._cb, 10)
        self.create_timer(0.05, self._tick)

    def _cb(self, msg):
        self.last, self.last_time = msg, self.get_clock().now()

    def _tick(self):
        fresh = self.last_time is not None and (self.get_clock().now() - self.last_time).nanoseconds < self.timeout * 1e9
        self.pub.publish(self.last if fresh else Twist())


def main():
    rclpy.init()
    node = Watchdog()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
