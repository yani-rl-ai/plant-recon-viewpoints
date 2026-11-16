#!/usr/bin/env python3
"""Localisation accuracy in plant_sim (run while `ros2 launch plant_sim nav.launch.py` is up).

    python3 check_localization.py [--plants 1,2] [--out results/localization_check.json]

Drives the robot with Nav2 to the scan stations of the given plants (4 per plant, radius 1.4 m, as in scan_manager) and, all the
time, compares Gazebo's TRUE pose (/gt/dynamic_pose) with
  * map -> base_footprint from tf   (what the scanner uses as the camera pose in `--pose-source map`)
Prints position / heading errors over the whole drive and of the samples taken when the robot stood still at a station.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from nav2_simple_commander.robot_navigator import BasicNavigator
from tf2_msgs.msg import TFMessage
from tf2_ros import Buffer, TransformListener

from nav_helpers import arc_waypoints, navigate_station

CENTRES = {0: (-3.0, -2.0), 1: (0.0, -2.0), 2: (3.0, -2.0), 3: (-3.0, 2.0), 4: (0.0, 2.0), 5: (3.0, 2.0)}


def yaw_of(q):
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plants', default='1,2')
    ap.add_argument('--azimuths', default='0,90,180,270', help='station azimuths around each plant [deg]')
    ap.add_argument('--out', default=os.path.expanduser('~/plant_ws/results/localization_check.json'))
    ap.add_argument('--timeout', type=float, default=150.0)
    args = ap.parse_args()

    rclpy.init()
    nav = BasicNavigator()
    buf = Buffer()
    TransformListener(buf, nav)
    st = {'gt': None, 'v': 0.0, 'w': 0.0}

    def on_gt(m):
        for t in m.transforms:
            if t.child_frame_id == 'tb3_plant':
                st['gt'] = (t.transform.translation.x, t.transform.translation.y, yaw_of(t.transform.rotation))
    nav.create_subscription(TFMessage, '/gt/dynamic_pose', on_gt, 50)
    nav.create_subscription(Odometry, '/odom', lambda m: st.update(v=abs(m.twist.twist.linear.x), w=abs(m.twist.twist.angular.z)), 20)

    nav._waitForNodeToActivate('bt_navigator')
    t0 = time.time()
    while (st['gt'] is None or not buf.can_transform('map', 'base_footprint', rclpy.time.Time())) and time.time() - t0 < 60:
        rclpy.spin_once(nav, timeout_sec=0.1)
    if st['gt'] is None:
        raise SystemExit('no ground-truth pose on /gt/dynamic_pose')

    samples, still = [], []

    def sample(tag):
        try:
            T = buf.lookup_transform('map', 'base_footprint', rclpy.time.Time())
        except Exception:                                      # noqa: BLE001
            return None
        gt = st['gt']
        tf = (T.transform.translation.x, T.transform.translation.y, yaw_of(T.transform.rotation))
        row = {'t': time.time() - t0, 'tag': tag, 'tf_pos_err': math.hypot(tf[0] - gt[0], tf[1] - gt[1]), 'tf_yaw_err': abs(wrap(tf[2] - gt[2]))}
        return row

    # stations: counter-clockwise from the one nearest to the robot (as scan_manager does)
    here = (0.0, 0.0)
    goals = []
    for k in [int(p) for p in args.plants.split(',')]:
        c = CENTRES[k]
        th0 = math.atan2(here[1] - c[1], here[0] - c[0])
        azim = sorted(math.radians(float(a)) % (2 * math.pi) for a in args.azimuths.split(','))
        i0 = min(range(len(azim)), key=lambda i: abs(wrap(azim[i] - th0)))
        for j in range(len(azim)):
            th = azim[(i0 + j) % len(azim)]
            x, y = c[0] + 1.4 * math.cos(th), c[1] + 1.4 * math.sin(th)
            goals.append((k, j + 1, x, y, th + math.pi, c, azim[(i0 + j - 1) % len(azim)] if j else None, th))
            here = (x, y)

    reached = 0
    for k, s, x, y, yaw, c, th_prev, th in goals:
        def map_pose():
            T_ = buf.lookup_transform('map', 'base_footprint', rclpy.time.Time())
            return T_.transform.translation.x, T_.transform.translation.y, yaw_of(T_.transform.rotation)
        wps = arc_waypoints(c, 1.4, th_prev, th) if th_prev is not None else []
        navigate_station(nav, map_pose, (x, y, yaw), wps, log=print, timeout=args.timeout)
        # wait until the robot really stands still (speed < 0.01 m/s and 0.02 rad/s for 1 s, as scan_manager does), then 2 s of samples
        since, tw = None, time.time()
        while time.time() - tw < 20.0:
            rclpy.spin_once(nav, timeout_sec=0.1)
            if st['v'] < 0.01 and st['w'] < 0.02:
                since = since or time.time()
                if time.time() - since > 1.0:
                    break
            else:
                since = None
        te = time.time()
        while time.time() - te < 2.0:
            rclpy.spin_once(nav, timeout_sec=0.1)
            r = sample('still')
            if r:
                still.append(r)
        gt = st['gt']
        d = math.hypot(gt[0] - x, gt[1] - y)
        reached += d < 0.6
        print(f'plant {k} station {s}: true pose {d:.2f} m from the station | tf error {still[-1]["tf_pos_err"] * 100:.1f} cm, '
              f'{math.degrees(still[-1]["tf_yaw_err"]):.1f} deg' if still else f'plant {k} station {s}', flush=True)

    def stats(rows, key, scale=1.0):
        v = np.array([r[key] for r in rows if key in r]) * scale
        return None if not len(v) else {'mean': float(v.mean()), 'median': float(np.median(v)), 'p95': float(np.percentile(v, 95)), 'max': float(v.max()), 'n': int(len(v))}

    out = {'stations': len(goals), 'reached_within_0.6m': int(reached), 'duration_s': time.time() - t0,
           'tf_pos_err_m_all': stats(samples + still, 'tf_pos_err'), 'tf_pos_err_m_still': stats(still, 'tf_pos_err'),
           'tf_yaw_err_deg_all': stats(samples + still, 'tf_yaw_err', 180 / math.pi), 'tf_yaw_err_deg_still': stats(still, 'tf_yaw_err', 180 / math.pi)}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, 'w'), indent=1)
    print(json.dumps(out, indent=1))
    rclpy.shutdown()


if __name__ == '__main__':
    main()
