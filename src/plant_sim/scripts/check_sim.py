#!/usr/bin/env python3
"""Smoke test of the running plant_sim: topics, camera height in TF, tilt motor, driving stability.

    source ~/plant_ws/env.sh && ros2 launch plant_sim sim.launch.py headless:=true rviz:=false
    python3 check_sim.py --out /tmp/plant_check [--drive]

Prints PASS/FAIL lines and exits non-zero on any failure.  Images are decoded by hand (no cv_bridge).
"""
import argparse
import math
import os
import sys
import time

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, JointState, LaserScan, PointCloud2
from std_msgs.msg import Float64
from tf2_ros import Buffer, TransformListener

COLOR = '/zed/zed_node/rgb/image_rect_color'
DEPTH = '/zed/zed_node/depth/depth_registered'
INFO = '/zed/zed_node/rgb/camera_info'
CLOUD = '/zed/zed_node/point_cloud/cloud_registered'
CLOUD_DTYPE = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('rgb', '<f4')])


def quat_to_R(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class Check(Node):
    def __init__(self):
        super().__init__('check_sim', parameter_overrides=[rclpy.parameter.Parameter(
            'use_sim_time', rclpy.Parameter.Type.BOOL, True)])
        self.color = self.depth = self.info = self.joint = self.odom = None
        self.scan = self.cloud = self.gray = None
        self.cache = {}                                   # (kind, stamp) -> msg, to pair cloud with its images
        self.create_subscription(Image, COLOR, lambda m: self._keep('color', m), 5)
        self.create_subscription(Image, DEPTH, lambda m: self._keep('depth', m), 5)
        self.create_subscription(LaserScan, '/scan', lambda m: setattr(self, 'scan', m), 5)
        self.create_subscription(PointCloud2, CLOUD, lambda m: setattr(self, 'cloud', m), 2)
        self.create_subscription(PointCloud2, CLOUD + '_gray', lambda m: setattr(self, 'gray', m), 2)
        self.create_subscription(CameraInfo, INFO, lambda m: setattr(self, 'info', m), 5)
        self.create_subscription(JointState, '/joint_states', self._js, 20)
        self.create_subscription(Odometry, '/odom', lambda m: setattr(self, 'odom', m), 20)
        self.tilt_pub = self.create_publisher(Float64, '/camera_tilt/cmd', 10)
        self.vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.tf_buf = Buffer()
        self.tf_listener = TransformListener(self.tf_buf, self)
        self.fails = 0

    def _keep(self, kind, m):
        setattr(self, kind, m)
        self.cache[(kind, m.header.stamp.sec, m.header.stamp.nanosec)] = m
        while len(self.cache) > 40:
            self.cache.pop(next(iter(self.cache)))

    def paired(self, cloud=None):
        """(cloud, colour, depth) with identical stamps, or None."""
        cloud = self.cloud if cloud is None else cloud
        if cloud is None:
            return None
        s = cloud.header.stamp
        c, d = self.cache.get(('color', s.sec, s.nanosec)), self.cache.get(('depth', s.sec, s.nanosec))
        return (cloud, c, d) if c is not None and d is not None else None

    def _js(self, m):
        if 'camera_tilt_joint' in m.name:
            self.joint = m.position[m.name.index('camera_tilt_joint')]

    def spin_for(self, sec):
        t0 = time.time()
        while time.time() - t0 < sec:
            rclpy.spin_once(self, timeout_sec=0.05)

    def wait_for(self, cond, timeout, what):
        t0 = time.time()
        while not cond() and time.time() - t0 < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
        ok = cond()
        self.report(ok, f'{what} within {timeout:.0f} s')
        return ok

    def report(self, ok, msg):
        print(('PASS  ' if ok else 'FAIL  ') + msg, flush=True)
        self.fails += 0 if ok else 1

    def cam_pose(self):
        try:
            t = self.tf_buf.lookup_transform('odom', 'camera_optical_frame', rclpy.time.Time())
        except Exception:
            return None
        R = quat_to_R(t.transform.rotation)
        fwd = R[:, 2]                                  # optical z = viewing direction
        return np.array([t.transform.translation.x, t.transform.translation.y, t.transform.translation.z]), \
            math.degrees(math.asin(-fwd[2])), R


def decode(msg):
    if msg.encoding == 'rgb8':
        return np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3)
    if msg.encoding == '32FC1':
        return np.frombuffer(msg.data, np.float32).reshape(msg.height, msg.width)
    raise ValueError(msg.encoding)


def check_scan(n):
    """Laser: 270 deg, 10 m, mounted on the base; world geometry gives known ranges at the start pose
    (robot at the origin facing +x: wall at x = 5 m ahead, pots of the plants at (0, +-2) abeam)."""
    if not n.wait_for(lambda: n.scan is not None, 20, '/scan received'):
        return
    s = n.scan
    n.report(s.header.frame_id == 'base_scan', f'scan frame_id = {s.header.frame_id}')
    n.report(abs(math.degrees(s.angle_max - s.angle_min) - 270) < 0.5 and abs(s.angle_min + s.angle_max) < 1e-3,
             f'scan field of view = {math.degrees(s.angle_max - s.angle_min):.1f} deg, symmetric about the front')
    n.report(abs(s.range_max - 10.0) < 1e-3 and s.range_min < 0.1,
             f'scan range {s.range_min:.2f} .. {s.range_max:.1f} m, {len(s.ranges)} beams, '
             f'{math.degrees(s.angle_increment):.2f} deg step')
    r = np.array(s.ranges)
    fin = np.isfinite(r)

    def at(deg, half=1.0):
        a = s.angle_min + np.arange(len(r)) * s.angle_increment
        m = np.abs(np.degrees(a) - deg) <= half
        return float(np.median(r[m & fin])) if (m & fin).any() else float('nan')

    ahead, left, right = at(0), at(90), at(-90)
    n.report(abs(ahead - 4.97) < 0.10, f'range straight ahead = {ahead:.2f} m (wall at 5 m, sensor 3 cm ahead of the origin)')
    n.report(abs(left - 1.89) < 0.06 and abs(right - 1.89) < 0.06,
             f'range abeam = {left:.2f} m left / {right:.2f} m right (pots at 2 m, radius 0.11 m -> 1.89)')
    n.report(np.nanmin(r[fin]) > 1.0, f'closest return = {np.nanmin(r[fin]):.2f} m (no hits on the robot itself or the pole)')
    n.report(fin.mean() > 0.6, f'{fin.mean() * 100:.0f} % of the beams return a range')
    try:
        t = n.tf_buf.lookup_transform('odom', 'base_scan', rclpy.time.Time())
        z = t.transform.translation.z
        n.report(abs(z - 0.185) < 0.005, f'scan plane {z:.3f} m above the floor (pots are 0.20 m high)')
    except Exception as e:                                  # noqa: BLE001
        n.report(False, f'TF odom -> base_scan: {e}')


def check_clouds(n):
    if not n.wait_for(lambda: n.paired() is not None and n.paired(n.gray) is not None, 20,
                      'cloud_registered (+ _gray) received, each matched to its colour/depth frame'):
        return
    cloud, color, depth = n.paired()
    h, w = depth.height, depth.width
    n.report((cloud.height, cloud.width) == (h, w) and cloud.header.frame_id == 'camera_optical_frame'
             and [f.name for f in cloud.fields] == ['x', 'y', 'z', 'rgb'] and cloud.point_step == 16,
             f'cloud organised {cloud.width}x{cloud.height}, fields x y z rgb, frame {cloud.header.frame_id}')
    P = np.frombuffer(cloud.data, CLOUD_DTYPE).reshape(h, w)
    z = decode(depth)
    ok = np.isfinite(P['z'])
    n.report(np.allclose(P['z'][ok], z[ok]) and (ok == (np.isfinite(z) & (z > 0.3) & (z < 8.0))).all(),
             f'cloud z == depth image on all {ok.sum()} valid pixels ({ok.mean() * 100:.0f} %), invalid pixels are NaN')
    K = np.array(n.info.k).reshape(3, 3)
    vv, uu = np.nonzero(ok)
    ex = np.abs(P['x'][vv, uu] - (uu - K[0, 2]) * P['z'][vv, uu] / K[0, 0]).max()
    ey = np.abs(P['y'][vv, uu] - (vv - K[1, 2]) * P['z'][vv, uu] / K[1, 1]).max()
    n.report(ex < 1e-3 and ey < 1e-3, f'x, y follow the pinhole model of camera_info on all valid pixels '
                                      f'(max error {max(ex, ey) * 1000:.3f} mm)')
    rgb = P['rgb'].copy().view(np.uint32)
    img = decode(color).astype(np.uint32)
    n.report(((rgb[ok] >> 16) & 255 == img[..., 0][ok]).all() and ((rgb[ok] & 255) == img[..., 2][ok]).all(),
             'colour of every point equals the colour image pixel (true colour cloud)')
    g, gcol, gdep = n.paired(n.gray)
    Gc = np.frombuffer(g.data, CLOUD_DTYPE).reshape(h, w)
    gok = np.isfinite(Gc['z'])
    G = Gc['rgb'].copy().view(np.uint32)
    r_, g_, b_ = (G >> 16) & 255, (G >> 8) & 255, G & 255
    ci = decode(gcol).astype(np.float64)
    lum = (0.299 * ci[..., 0] + 0.587 * ci[..., 1] + 0.114 * ci[..., 2])
    same = (r_[gok] == g_[gok]).all() and (g_[gok] == b_[gok]).all()
    close = np.abs(r_[gok].astype(float) - lum[gok]).max() <= 1.0
    n.report(same and close and len(np.unique(r_[gok])) > 20 and np.allclose(Gc['z'][gok], decode(gdep)[gok]),
             f'black & white cloud: r == g == b = luminance of the colour pixel ({len(np.unique(r_[gok]))} grey levels), '
             f'same points as the depth image')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='/tmp/plant_check')
    ap.add_argument('--drive', action='store_true')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rclpy.init()
    n = Check()

    n.wait_for(lambda: n.color is not None and n.depth is not None and n.info is not None, 40,
               'colour, depth and camera_info received')
    if n.color is not None:
        c, d = n.color, n.depth
        n.report(c.encoding == 'rgb8' and (c.width, c.height) == (640, 360), f'colour {c.encoding} {c.width}x{c.height}')
        n.report(d.encoding == '32FC1' and (d.width, d.height) == (640, 360), f'depth {d.encoding} {d.width}x{d.height}')
        n.report(c.header.frame_id == 'camera_optical_frame' and d.header.frame_id == 'camera_optical_frame',
                 f'frame_id colour/depth = {c.header.frame_id} / {d.header.frame_id}')
        K = np.array(n.info.k).reshape(3, 3)
        hfov = 2 * math.degrees(math.atan(320 / K[0, 0]))
        n.report(abs(hfov - 110) < 0.5 and abs(K[0, 2] - 320) < 1 and abs(K[1, 2] - 180) < 1,
                 f'camera_info fx={K[0, 0]:.1f} cx={K[0, 2]:.1f} cy={K[1, 2]:.1f} -> hfov {hfov:.1f} deg')

    n.wait_for(lambda: n.cam_pose() is not None, 20, 'TF odom -> camera_optical_frame available')
    n.spin_for(1.0)
    pose = n.cam_pose()
    if pose:
        p, pitch, _ = pose
        n.report(abs(p[2] - 1.0) < 0.01, f'camera height in TF = {p[2]:.3f} m (expect 1.000)')
        n.report(abs(pitch) < 1.0, f'camera pitch at 0 command = {pitch:+.2f} deg (expect 0)')

    check_scan(n)
    check_clouds(n)

    for target in (0.5, -0.5, 0.0):                    # + = look down
        for _ in range(60):
            n.tilt_pub.publish(Float64(data=target))
            n.spin_for(0.05)
        n.spin_for(0.5)
        pose = n.cam_pose()
        if pose and n.joint is not None:
            err = abs(n.joint - target)
            ok_tf = abs(pose[1] - math.degrees(target)) < 2.0
            n.report(err < 0.02 and ok_tf, f'tilt command {math.degrees(target):+5.1f} deg -> joint {math.degrees(n.joint):+6.2f} deg, '
                                           f'TF pitch {pose[1]:+6.2f} deg')
        else:
            n.report(False, 'tilt: no joint state / TF')
        if target == 0.5 and n.color is not None:
            n.spin_for(0.6)
            cv2.imwrite(os.path.join(args.out, 'color_tilt_down.png'), cv2.cvtColor(decode(n.color), cv2.COLOR_RGB2BGR))
            dep = decode(n.depth)
            fin = np.isfinite(dep)
            dv = np.clip((np.where(fin, dep, 8.0) - 0.3) / 5.7 * 255, 0, 255).astype(np.uint8)
            cv2.imwrite(os.path.join(args.out, 'depth_tilt_down.png'), cv2.applyColorMap(dv, cv2.COLORMAP_TURBO))
            print(f'      depth: finite {fin.mean() * 100:.0f} %, range {np.nanmin(dep[fin]):.2f} .. {np.nanmax(dep[fin]):.2f} m')

    if n.color is not None:
        cv2.imwrite(os.path.join(args.out, 'color_level.png'), cv2.cvtColor(decode(n.color), cv2.COLOR_RGB2BGR))

    if args.drive:
        n.spin_for(0.5)
        x0 = n.odom.pose.pose.position.x
        for _ in range(100):                           # 0.12 m/s for 5 s
            n.vel_pub.publish(Twist(linear=__import__('geometry_msgs.msg', fromlist=['Vector3']).Vector3(x=0.12)))
            n.spin_for(0.05)
        n.vel_pub.publish(Twist())
        n.spin_for(1.0)
        dx = n.odom.pose.pose.position.x - x0
        n.report(0.4 < dx < 0.7, f'drove {dx:.2f} m in 5 s at 0.12 m/s (odom)')
        pose = n.cam_pose()
        n.report(pose is not None and abs(pose[0][2] - 1.0) < 0.02, 'camera height unchanged after driving (no tipping)')
        for _ in range(60):                            # spin in place 0.4 rad/s for 3 s
            n.vel_pub.publish(Twist(angular=__import__('geometry_msgs.msg', fromlist=['Vector3']).Vector3(z=0.4)))
            n.spin_for(0.05)
        n.vel_pub.publish(Twist())
        n.spin_for(1.0)
        pose = n.cam_pose()
        n.report(pose is not None and abs(pose[0][2] - 1.0) < 0.02 and abs(pose[1]) < 3.0,
                 'camera height and pitch unchanged after spinning (no tipping)')

    print('RESULT:', 'ALL PASS' if n.fails == 0 else f'{n.fails} FAILED')
    n.destroy_node()
    rclpy.shutdown()
    sys.exit(1 if n.fails else 0)


if __name__ == '__main__':
    main()
