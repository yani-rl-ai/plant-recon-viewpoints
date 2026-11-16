#!/usr/bin/env python3
"""Drive the robot along a fixed collision-free route through the arena while slam_toolbox maps it,
then save the 2D occupancy map (+ the slam_toolbox pose graph).

    source ~/plant_ws/env.sh
    ros2 launch plant_sim mapping.launch.py            # sim + slam_toolbox (+ RViz)
    ros2 run plant_sim auto_map.py [--save PREFIX]     # in a second terminal

The route (world frame == odom frame, the robot spawns at the origin) keeps >= 1.0 m from the walls and
the plants and passes every pot on all sides: a full turn on the spot, the outer ring, then two passes
between the plant columns.  An emergency stop on the laser aborts the run if anything is closer than
0.35 m ahead.  Wheel odometry is exact in the simulator, so the route is independent of SLAM.
"""
import argparse
import math
import os
import subprocess
import sys

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformListener

ROUTE = [(-1.5, 0), (-4, 0), (-4, -4), (4, -4), (4, 4), (-4, 4), (-4, 0), (-1.5, 0),
         (-1.5, -4), (-1.5, 4), (1.5, 4), (1.5, -4), (1.5, 0), (0, 0)]
# Turning is deliberately slow: scans taken while the robot rotates fast are inserted with a slightly wrong
# pose (a 0.1 s timing slip at 0.6 rad/s is 3 deg) and smear the walls into the thick, skewed map seen at first.
V_MAX, W_MAX = 0.22, 0.25
STOP_DIST = 0.35


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


class AutoMap(Node):
    def __init__(self, pose_source='slam'):
        super().__init__('auto_map', parameter_overrides=[rclpy.parameter.Parameter(
            'use_sim_time', rclpy.Parameter.Type.BOOL, True)])
        self.pose_source = pose_source
        self.tf_buf = Buffer()
        self.tf_listener = TransformListener(self.tf_buf, self)
        self.warned = False
        self.pose = None
        self.have_scan = False
        self.front_min = float('inf')
        self.create_subscription(Odometry, '/odom', self._odom, 20)
        self.create_subscription(LaserScan, '/scan', self._scan, 5)
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)

    def _odom(self, m):
        q = m.pose.pose.orientation
        yaw = math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
        self.pose = (m.pose.pose.position.x, m.pose.pose.position.y, yaw)

    def _scan(self, m):
        lim = math.radians(35)
        v = [(r, math.degrees(m.angle_min + i * m.angle_increment)) for i, r in enumerate(m.ranges)
             if abs(m.angle_min + i * m.angle_increment) < lim and math.isfinite(r) and r > m.range_min]
        self.front_min = min(v)[0] if v else float('inf')
        self.front_close = sorted(v)[:5]                 # (range, angle deg) of the 5 closest returns, for diagnostics
        self.have_scan = True

    def current_pose(self):
        """Pose used to steer.  Wheel odometry drifts (a 2 % yaw error piles up over the many turns of the
        route), so the default is slam_toolbox's map -> base_footprint, which the laser keeps corrected;
        map == world because SLAM starts at the spawn pose.  Falls back to odometry if SLAM has no TF yet."""
        if self.pose_source == 'slam':
            try:
                t = self.tf_buf.lookup_transform('map', 'base_footprint', rclpy.time.Time())
                q = t.transform.rotation
                return (t.transform.translation.x, t.transform.translation.y,
                        math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z)))
            except Exception:                                   # noqa: BLE001
                if not self.warned:
                    self.get_logger().warn('no map -> base_footprint yet, using odometry')
                    self.warned = True
        return self.pose

    def send(self, v, w):
        t = Twist()
        t.linear.x, t.angular.z = float(v), float(w)
        self.pub.publish(t)

    def spin(self, sec=0.05):
        rclpy.spin_once(self, timeout_sec=sec)

    def stop(self):
        for _ in range(10):
            self.send(0, 0)
            self.spin(0.02)

    def turn_full(self):
        """One full turn on the spot so the first scans already cover every direction."""
        self.get_logger().info('full turn on the spot')
        yaw0 = self.pose[2]
        done, last = 0.0, yaw0
        while done < 2 * math.pi - 0.05 and rclpy.ok():
            self.send(0, W_MAX)
            self.spin()
            done += abs(wrap(self.pose[2] - last))
            last = self.pose[2]
        self.stop()

    def goto(self, gx, gy):
        while rclpy.ok():
            self.spin()
            x, y, yaw = self.current_pose()
            dist = math.hypot(gx - x, gy - y)
            if dist < 0.12:
                return True
            err = wrap(math.atan2(gy - y, gx - x) - yaw)
            if abs(err) > 0.5:                      # turn on the spot first
                self.send(0, math.copysign(W_MAX, err))
                continue
            if self.front_min < STOP_DIST:
                self.stop()
                dump = os.environ.get('AUTO_MAP_DUMP')               # optional: Gazebo's own poses, for diagnosis
                if dump:
                    r = subprocess.run(['ign', 'topic', '-e', '-n', '1', '-t', '/world/plant_arena/dynamic_pose/info'],
                                       capture_output=True, text=True, timeout=15)
                    open(dump, 'w').write(r.stdout)
                self.get_logger().error(f'EMERGENCY STOP: obstacle {self.front_min:.2f} m ahead at '
                                        f'({x:.2f}, {y:.2f}, yaw {math.degrees(yaw):.0f} deg) on the way to ({gx}, {gy}); '
                                        f'closest returns (m, deg): '
                                        f'{[(round(r, 2), round(a)) for r, a in self.front_close]}')
                return False
            self.send(min(V_MAX, 0.6 * dist + 0.05) * max(0.3, math.cos(err)),
                      max(-W_MAX, min(W_MAX, 1.5 * err)))
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--save', default=os.path.expanduser('~/plant_ws/src/plant_sim/maps/plant_arena'),
                    help='map file prefix (.pgm/.yaml/_posegraph are written)')
    ap.add_argument('--no-save', action='store_true')
    ap.add_argument('--pose', choices=['slam', 'odom'], default='slam', help='pose used to steer (default: slam)')
    args = ap.parse_args()
    rclpy.init()
    n = AutoMap(args.pose)
    n.get_logger().info('waiting for /odom and /scan ...')
    while rclpy.ok() and (n.pose is None or not n.have_scan):
        n.spin(0.2)
    n.spin(1.0)
    ok = True
    n.turn_full()
    for i, (gx, gy) in enumerate(ROUTE, 1):
        n.get_logger().info(f'waypoint {i}/{len(ROUTE)} -> ({gx:+.1f}, {gy:+.1f})')
        if not n.goto(gx, gy):
            ok = False
            break
    n.stop()
    if not ok:
        n.get_logger().error('route aborted, map NOT saved')
        sys.exit(2)
    n.get_logger().info('route finished, letting slam_toolbox integrate the last scans ...')
    for _ in range(100):
        n.spin(0.05)
    if not args.no_save:
        os.makedirs(os.path.dirname(args.save), exist_ok=True)
        r1 = subprocess.run(['ros2', 'run', 'nav2_map_server', 'map_saver_cli', '-f', args.save,
                             '--ros-args', '-p', 'save_map_timeout:=20.0', '-p', 'use_sim_time:=true'])
        r2 = subprocess.run(['ros2', 'service', 'call', '/slam_toolbox/serialize_map',
                             'slam_toolbox/srv/SerializePoseGraph', f"{{filename: '{args.save}_posegraph'}}"])
        n.get_logger().info(f'map saved: {args.save}.pgm/.yaml (rc {r1.returncode}), posegraph (rc {r2.returncode})')
    n.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
