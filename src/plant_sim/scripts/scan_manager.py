#!/usr/bin/env python3
"""Scan manager: Nav2 stations around every plant, tilt sweep, YOLO + colour segmentation, accumulation (odom + ICP),
DBL-GNG, live visualisation in RViz on the 2D occupancy map, and metrics against the Gazebo ground truth.

    source ~/plant_ws/env.sh
    ros2 launch plant_sim scan.launch.py plants:=1,2,5,4,3,0      # Gazebo GUI + RViz + Nav2 + this node

Per plant: 4 stations on a circle (default 1.4 m) at 0/90/180/270 deg around the plant, the robot facing the plant
(Nav2 goal, keep-out disk 0.95 m so the camera never enters the canopy); at every station 3 tilt angles
(+20 deg down, level, -20 deg up).  The reconstruction uses ONLY camera + odometry; the 270 deg laser is used
by Nav2 to drive.  The plant positions (field layout) are a prior used to pick the YOLO instance and to crop;
the Gazebo ground truth is used for the metrics, the labelling of the segmentation score and the 'oracle' upper bound.
"""
import argparse
import json
import math
import os
import sys
import threading
import time

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Point, PoseStamped, TransformStamped
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from nav_msgs.msg import Odometry
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, JointState, PointCloud2, PointField
from std_msgs.msg import ColorRGBA, Float64
from tf2_msgs.msg import TFMessage
from tf2_ros import Buffer, TransformBroadcaster, TransformListener
from visualization_msgs.msg import Marker, MarkerArray



def _workspace_root():
    """Folder that holds src/plant_recon and env.sh.  `colcon build --symlink-install` makes the installed script a symlink
    into src/, so the real path of this file leads back to the workspace; falls back to ~/plant_ws."""
    d = os.path.dirname(os.path.realpath(__file__))
    for _ in range(6):
        if os.path.isdir(os.path.join(d, 'src', 'plant_recon')):
            return d
        d = os.path.dirname(d)
    return os.path.expanduser('~/plant_ws')


WS = _workspace_root()
sys.path.insert(0, os.path.join(WS, 'src', 'plant_recon'))
from nav_helpers import arc_waypoints, navigate_station  # noqa: E402
from plant_recon.geom import GroundTruth, T_from_msg, optical_pose_from_gt  # noqa: E402
from plant_recon.pipeline import Config, PlantReconstructor  # noqa: E402
from plant_recon.segment import SamSegmenter  # noqa: E402

COLOR, DEPTH, INFO = '/zed/zed_node/rgb/image_rect_color', '/zed/zed_node/depth/depth_registered', '/zed/zed_node/rgb/camera_info'
CLASS_RGB = {1: (1.0, 0.5, 0.0), 2: (0.1, 0.8, 0.1), 3: (1.0, 0.1, 0.1), 4: (0.45, 0.25, 0.08)}   # pot, plant, apple, soil
CLASS_BGR = {1: (0, 128, 255), 2: (0, 200, 0), 3: (255, 0, 255), 4: (20, 70, 130)}
CLASS_LABEL = {1: 'pot', 2: 'plant', 3: 'apple', 4: 'soil'}
FIELDS = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1) for i, n in enumerate(('x', 'y', 'z', 'rgb'))]
DT = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('rgb', '<f4')])


def cloud_msg(P, C, frame, stamp):
    a = np.empty(len(P), DT)
    a['x'], a['y'], a['z'] = P[:, 0], P[:, 1], P[:, 2]
    c = (np.clip(C, 0, 1) * 255).astype(np.uint32)
    a['rgb'] = ((c[:, 0] << 16) | (c[:, 1] << 8) | c[:, 2]).view(np.float32)
    m = PointCloud2(height=1, width=len(P), fields=FIELDS, is_bigendian=False, point_step=16, row_step=16 * len(P), is_dense=True)
    m.header.frame_id, m.header.stamp = frame, stamp
    m.data = a.tobytes()
    return m


def class_colors(L):
    tab = np.array([(0, 0, 0)] + [CLASS_RGB[i] for i in (1, 2, 3, 4)])
    return tab[np.asarray(L, int)]


def yaw_quat(yaw):
    return 0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)


class ScanManager(Node):
    def __init__(self, args):
        super().__init__('scan_manager', parameter_overrides=[rclpy.parameter.Parameter(
            'use_sim_time', rclpy.Parameter.Type.BOOL, True)])
        self.args = args
        self.frame = args.pose_source                         # frame the reconstruction lives in (odom / map)
        self.lock = threading.Lock()
        self.color = self.depth = self.info = self.odom = None
        self.tilt_now = None
        self.gt_pose = {}
        self.tf_buf = Buffer()
        self.tf_listener = TransformListener(self.tf_buf, self)
        self.tf_br = TransformBroadcaster(self)
        self.create_subscription(Image, COLOR, lambda m: self._set('color', m), 2)
        self.create_subscription(Image, DEPTH, lambda m: self._set('depth', m), 2)
        self.create_subscription(CameraInfo, INFO, lambda m: setattr(self, 'info', m), 2)
        self.create_subscription(Odometry, '/odom', lambda m: setattr(self, 'odom', m), 10)
        self.create_subscription(JointState, '/joint_states', self._js, 10)
        self.create_subscription(TFMessage, '/gt/dynamic_pose', self._gt, 10)
        self.tilt_pub = self.create_publisher(Float64, '/camera_tilt/cmd', 10)
        self.pub_cloud = self.create_publisher(PointCloud2, '/scan/recon_cloud', 1)
        self.pub_cloud_class = self.create_publisher(PointCloud2, '/scan/recon_cloud_class', 1)
        self.pub_done = self.create_publisher(PointCloud2, '/scan/done_cloud', 1)
        self.pub_graph = self.create_publisher(MarkerArray, '/scan/graph', 1)
        self.pub_fruit = self.create_publisher(MarkerArray, '/scan/fruit', 1)
        self.pub_stations = self.create_publisher(MarkerArray, '/scan/stations', 1)
        self.pub_status = self.create_publisher(Marker, '/scan/status', 1)
        self.pub_overlay = self.create_publisher(Image, '/scan/overlay', 1)
        # state shown in RViz (republished at 1 Hz so RViz never sees stale stamps)
        self.cur = None
        self.focus = np.array([0.0, 0.0])
        self.done = []                                        # (P, C) of finished plants
        self.graph_markers = []
        self.fruit_markers = []
        self.station_markers = []
        self.status_text = 'starting'
        self.status_pos = np.array([0.0, 0.0, 2.0])
        self.create_timer(1.0, self._republish)
        self.create_timer(0.1, self._focus_tf)

    # ----------------------------------------------------------------- inputs
    def _set(self, name, msg):
        with self.lock:
            setattr(self, name, (time.time(), msg))

    def _js(self, m):
        if 'camera_tilt_joint' in m.name:
            self.tilt_now = m.position[m.name.index('camera_tilt_joint')]

    def _gt(self, msg):
        for t in msg.transforms:
            self.gt_pose[t.child_frame_id] = T_from_msg(t)

    def stamp_of(self, m):
        return Time(seconds=m.header.stamp.sec, nanoseconds=m.header.stamp.nanosec)

    def lookup(self, target, source, stamp=None, timeout=0.3):
        # The chain map -> odom -> base_footprint can be inconsistent for a moment right after the localiser's first
        # map -> odom message (ExtrapolationException at start-up): try again for up to 5 s before giving up.
        t_end = time.time() + 5.0
        while True:
            try:
                return T_from_msg(self.tf_buf.lookup_transform(target, source, stamp or Time(), Duration(seconds=timeout)))
            except Exception:                                  # noqa: BLE001
                try:
                    return T_from_msg(self.tf_buf.lookup_transform(target, source, Time(), Duration(seconds=1.0)))
                except Exception:                              # noqa: BLE001
                    if time.time() > t_end:
                        raise
                    time.sleep(0.2)

    # ----------------------------------------------------------------- RViz
    def _focus_tf(self):
        t = TransformStamped()
        t.header.stamp, t.header.frame_id, t.child_frame_id = self.get_clock().now().to_msg(), 'map', 'scan_focus'
        t.transform.translation.x, t.transform.translation.y = float(self.focus[0]), float(self.focus[1])
        t.transform.rotation.w = 1.0
        self.tf_br.sendTransform(t)

    def _republish(self):
        try:
            self._republish_impl()
        except Exception as e:                                  # noqa: BLE001  a visualisation glitch must never stop the node
            self.get_logger().warn(f'republish skipped: {e!r}', throttle_duration_sec=5.0)

    def _republish_impl(self):
        now = self.get_clock().now().to_msg()
        cur = self.cur
        if cur is not None:
            P, C, L = cur.P, cur.C, cur.L                       # add_view replaces these one after the other: publish only a consistent set
            if len(P) and len(P) == len(C) == len(L):
                self.pub_cloud.publish(cloud_msg(P, C, self.frame, now))
                self.pub_cloud_class.publish(cloud_msg(P, class_colors(L), self.frame, now))
        if self.done:
            P = np.vstack([d[0] for d in self.done])
            C = np.vstack([d[1] for d in self.done])
            self.pub_done.publish(cloud_msg(P, C, self.frame, now))
        for pub, markers in ((self.pub_graph, self.graph_markers), (self.pub_fruit, self.fruit_markers),
                             (self.pub_stations, self.station_markers)):
            for m in markers:
                m.header.stamp = now
            pub.publish(MarkerArray(markers=markers))
        m = Marker()
        m.header.frame_id, m.header.stamp, m.ns, m.id = 'map', now, 'status', 0
        m.type, m.action = Marker.TEXT_VIEW_FACING, Marker.ADD
        m.pose.position.x, m.pose.position.y, m.pose.position.z = map(float, self.status_pos)
        m.pose.orientation.w = 1.0
        m.scale.z = 0.11
        m.color = ColorRGBA(r=1.0, g=1.0, b=0.2, a=1.0)
        m.text = self.status_text
        self.pub_status.publish(m)

    def set_status(self, text, plant=None):
        self.status_text = text
        if plant is not None:
            self.status_pos = np.array([plant[0], plant[1], 1.95])
        self.get_logger().info(text.replace('\n', ' | '))

    def overlay(self, bgr, cls, region, lines):
        out = bgr.copy()
        for c, v in CLASS_BGR.items():
            m = cls == c
            out[m] = (0.35 * out[m] + 0.65 * np.array(v)).astype(np.uint8)
        cs, _ = cv2.findContours(region.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.polylines(out, cs, True, (0, 255, 255), 2)
        for j, (c, bgr_c) in enumerate(CLASS_BGR.items()):
            cv2.rectangle(out, (6 + 78 * j, out.shape[0] - 22), (14 + 78 * j, out.shape[0] - 12), bgr_c, -1)
            cv2.putText(out, CLASS_LABEL[c], (18 + 78 * j, out.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        for i, t in enumerate(lines):
            cv2.putText(out, t, (6, 16 + 16 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 3)
            cv2.putText(out, t, (6, 16 + 16 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)
        msg = Image(height=out.shape[0], width=out.shape[1], encoding='bgr8', step=3 * out.shape[1])
        msg.header.stamp, msg.header.frame_id = self.get_clock().now().to_msg(), 'camera_optical_frame'
        msg.data = out.tobytes()
        self.pub_overlay.publish(msg)

    def build_graph_markers(self, rec):
        nodes, edges, _ = rec.graph()
        ncls, _ = rec.node_classes(nodes)
        pts = Marker()
        pts.header.frame_id, pts.ns, pts.id, pts.type, pts.action = self.frame, 'gng_nodes', 0, Marker.POINTS, Marker.ADD
        pts.scale.x = pts.scale.y = 0.016
        pts.pose.orientation.w = 1.0
        for p, c in zip(nodes, ncls):
            pts.points.append(Point(x=float(p[0]), y=float(p[1]), z=float(p[2])))
            r, g, b = CLASS_RGB[int(c)]
            pts.colors.append(ColorRGBA(r=r, g=g, b=b, a=1.0))
        ln = Marker()
        ln.header.frame_id, ln.ns, ln.id, ln.type, ln.action = self.frame, 'gng_edges', 1, Marker.LINE_LIST, Marker.ADD
        ln.scale.x = 0.0025
        ln.pose.orientation.w = 1.0
        ln.color = ColorRGBA(r=0.9, g=0.9, b=0.9, a=0.85)
        for a, b in edges:
            for i in (a, b):
                ln.points.append(Point(x=float(nodes[i][0]), y=float(nodes[i][1]), z=float(nodes[i][2])))
        return [pts, ln]

    def build_fruit_markers(self, rec, est):
        ms, i = [], 0
        for k, c, r in ((rec.k, f[1], f[2]) for f in rec.gt.fruits if f[0] == rec.k):
            m = Marker()
            m.header.frame_id, m.ns, m.id, m.type, m.action = 'map', 'gt_apple', i, Marker.SPHERE, Marker.ADD
            m.pose.position.x, m.pose.position.y, m.pose.position.z = map(float, c)
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 2 * r * 1.08
            m.color = ColorRGBA(r=0.1, g=1.0, b=0.1, a=0.35)
            ms.append(m)
            i += 1
        for j, (c, r, n, rms) in enumerate(est):
            m = Marker()
            m.header.frame_id, m.ns, m.id, m.type, m.action = self.frame, 'est_apple', j, Marker.SPHERE, Marker.ADD
            m.pose.position.x, m.pose.position.y, m.pose.position.z = map(float, c)
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 2 * r
            m.color = ColorRGBA(r=1.0, g=0.05, b=0.05, a=0.9)
            ms.append(m)
        return ms

    def build_station_markers(self, c, stations, current):
        ms = []
        for i, (x, y, yaw, th) in enumerate(stations):
            m = Marker()
            m.header.frame_id, m.ns, m.id, m.type, m.action = 'map', 'stations', i, Marker.ARROW, Marker.ADD
            m.pose.position.x, m.pose.position.y, m.pose.position.z = float(x), float(y), 0.05
            m.pose.orientation.z, m.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
            m.scale.x, m.scale.y, m.scale.z = 0.5, 0.07, 0.07
            m.color = ColorRGBA(r=1.0, g=0.9, b=0.0, a=1.0) if i == current else ColorRGBA(r=0.3, g=0.6, b=1.0, a=0.8)
            ms.append(m)
        return ms

    # ----------------------------------------------------------------- robot actions
    def set_tilt(self, angle, timeout=6.0):
        t0, stable = time.time(), None
        while time.time() - t0 < timeout:
            self.tilt_pub.publish(Float64(data=float(angle)))
            time.sleep(0.1)
            if self.tilt_now is not None and abs(self.tilt_now - angle) < 0.03:
                stable = stable or time.time()
                if time.time() - stable > 0.5:
                    return True
            else:
                stable = None
        return False

    def wait_still(self, sec=1.0, timeout=15.0):
        t0, since = time.time(), None
        while time.time() - t0 < timeout:
            o = self.odom
            if o is not None and abs(o.twist.twist.linear.x) < 0.01 and abs(o.twist.twist.angular.z) < 0.02:
                since = since or time.time()
                if time.time() - since > sec:
                    return True
            else:
                since = None
            time.sleep(0.1)
        return False

    def grab(self, after, timeout=8.0):
        t0 = time.time()
        while time.time() - t0 < timeout:
            with self.lock:
                c, d = self.color, self.depth
            if c and d and c[0] > after and d[0] > after and abs(
                    (c[1].header.stamp.sec - d[1].header.stamp.sec) + 1e-9 * (c[1].header.stamp.nanosec - d[1].header.stamp.nanosec)) < 2e-3:
                cm, dm = c[1], d[1]
                st = self.stamp_of(cm)
                bgr = np.frombuffer(cm.data, np.uint8).reshape(cm.height, cm.width, 3)[:, :, ::-1].copy()
                depth = np.frombuffer(dm.data, np.float32).reshape(dm.height, dm.width).copy()
                K = np.array(self.info.k).reshape(3, 3)
                return {'color': bgr, 'depth': depth, 'K': K,
                        'T_odom_opt': self.lookup('odom', 'camera_optical_frame', st),
                        'T_map_opt': self.lookup('map', 'camera_optical_frame', st),
                        'T_odom_map': self.lookup('odom', 'map'),
                        'T_gt_opt': optical_pose_from_gt(self.gt_pose['tb3_plant'], self.gt_pose['camera_tilt_link'])}
            time.sleep(0.05)
        return None


def run(node, nav, args, gt, yolo):
    out = args.out
    os.makedirs(out, exist_ok=True)
    json.dump(vars(args), open(f'{out}/args.json', 'w'), indent=1)
    tilts = [float(t) for t in args.tilts.split(',')]
    plants = [int(p) for p in args.plants.split(',')]
    results = []
    node.set_status('waiting for Nav2 (AMCL + bt_navigator) ...')
    nav._waitForNodeToActivate('bt_navigator')
    t_mission = time.time()

    for k in plants:
        c = gt.centres[k]
        node.focus = c
        T_om = node.lookup('odom', 'map')
        prior = (T_om @ np.array([c[0], c[1], 0.75, 1.0]))[:2] if args.pose_source == 'odom' else c
        rec = PlantReconstructor(k, prior, gt, yolo, Config(seg=args.seg, use_icp=not args.no_icp))
        node.cur, node.graph_markers, node.fruit_markers = rec, [], []
        pdir = f'{out}/plant{k}'
        os.makedirs(pdir, exist_ok=True)
        # stations: 4 sides, starting with the one closest to the robot, counter-clockwise
        rob = node.lookup('map', 'base_footprint')[:2, 3]
        th0 = math.atan2(rob[1] - c[1], rob[0] - c[0])
        # azimuths of the stations around the plant (absolute, from +x, counter-clockwise): the first one visited is the
        # one nearest to the robot, the others follow counter-clockwise.  The default 0/90/180/270 is the former 4-side orbit.
        azim = sorted(math.radians(float(a)) % (2 * math.pi) for a in args.azimuths.split(','))
        i0 = min(range(len(azim)), key=lambda i: abs(math.atan2(math.sin(azim[i] - th0), math.cos(azim[i] - th0))))
        stations = []
        for j in range(len(azim)):
            th = azim[(i0 + j) % len(azim)]
            x, y = c[0] + args.radius * math.cos(th), c[1] + args.radius * math.sin(th)
            stations.append((x, y, th + math.pi, th))
        n_st, n_views = len(stations), len(stations) * len(tilts)
        t_plant = time.time()
        view_logs = []
        for si, (x, y, yaw, th) in enumerate(stations):
            node.station_markers = node.build_station_markers(c, stations, si)
            node.set_status(f'plant {k}   station {si + 1}/{n_st} ({math.degrees(th) % 360:.0f} deg)\ndriving with Nav2 ...', c)
            # Nav2 drives to the station (robot map pose within 0.6 m / 0.6 rad decides, see nav_helpers.navigate_station); between two
            # stations the robot follows the circle round the plant in hops of <= 90 deg, so that a station on the opposite side has
            # an unambiguous route (nav_helpers explains the two DWB failure modes this avoids).
            def map_pose():
                T_ = node.lookup('map', 'base_footprint')
                return T_[0, 3], T_[1, 3], math.atan2(T_[1, 0], T_[0, 0])
            wps = arc_waypoints(c, args.radius, stations[si - 1][3], th) if si > 0 else []
            ok, dist, dyaw, n_try = navigate_station(nav, map_pose, (x, y, yaw), wps, log=node.get_logger().info)
            if ok and n_try > 1:
                node.get_logger().warn(f'station {si + 1}: reached at attempt {n_try} ({dist:.2f} m / {math.degrees(dyaw):.0f} deg from it)')
            if not ok:
                node.get_logger().error(f'plant {k} station {si + 1}: not reachable, skipped')
                continue
            node.wait_still()
            for ti, tilt in enumerate(tilts):
                node.set_status(f'plant {k}   station {si + 1}/{n_st}   tilt {math.degrees(tilt):+.0f} deg\nsegmenting + reconstructing ...', c)
                node.set_tilt(tilt)
                time.sleep(0.4)
                fr = node.grab(time.time())
                if fr is None:
                    node.get_logger().error('no frame')
                    continue
                T = fr['T_odom_opt'] if args.pose_source == 'odom' else fr['T_map_opt']
                if args.pose_source == 'odom':          # the plant position in the odom frame moves as odometry drifts: refresh it
                    rec.prior_xy = (fr['T_odom_map'] @ np.array([c[0], c[1], 0.75, 1.0]))[:2]
                t0 = time.time()
                info = rec.add_view(fr['color'], fr['depth'], fr['K'], T, fr['T_gt_opt'], station=si + 1)
                info['t_total'] = time.time() - t0
                info['station'], info['tilt'] = si + 1, tilt
                Tg = fr['T_gt_opt']
                info['pose_err_mm'] = float(np.linalg.norm(T[:3, 3] - Tg[:3, 3]) * 1000)
                info['pose_err_deg'] = float(np.degrees(np.arccos(np.clip((np.trace(T[:3, :3].T @ Tg[:3, :3]) - 1) / 2, -1, 1))))
                np.savez_compressed(f'{pdir}/view_{info["view"]:02d}.npz', color=fr['color'], depth=fr['depth'].astype(np.float16),
                                    K=fr['K'], T_odom_opt=fr['T_odom_opt'], T_map_opt=fr['T_map_opt'], T_odom_map=fr['T_odom_map'],
                                    T_gt_opt=fr['T_gt_opt'], tilt=tilt, station=si + 1, prior=prior, plant=k)
                est = rec.fruit_estimates()
                node.graph_markers = node.build_graph_markers(rec) if rec.gng is not None else []
                node.fruit_markers = node.build_fruit_markers(rec, est['gng'])
                cl = info.get('cloud', {})
                sv = info.get('seg_vs_gt', {})
                node.overlay(fr['color'], rec.last_cls, rec.last_region,
                             [f'plant {k}  station {si + 1}/{n_st}  tilt {math.degrees(tilt):+.0f} deg',
                              f'plant pixels {info["plant_px"]}   IoU vs ground truth {sv.get("iou", float("nan")):.2f}   '
                              f'precision {sv.get("precision", float("nan")):.2f}  recall {sv.get("recall", float("nan")):.2f}'])
                node.set_status(
                    f'plant {k}   view {info["view"]}/{n_views}   station {si + 1}/{n_st}   tilt {math.degrees(tilt):+.0f} deg   [camera pose: {args.pose_source}]\n'
                    f'model {info["model_points"]} pts (' + ' '.join(f'{CLASS_LABEL[c]} {int((rec.L == c).sum())}' for c in (2, 1, 4, 3)) + f')   DBL-GNG {info["gng_nodes"]} nodes\n'
                    f'F1 vs ground truth, odom frame (with drift): 1 cm {cl.get("f1@10mm", 0):.2f} | 2 cm {cl.get("f1@20mm", 0):.2f} | 5 cm {cl.get("f1@50mm", 0):.2f}\n'
                    f'shape after rigid alignment: 1 cm {info.get("cloud_aligned", {}).get("f1@10mm", 0):.2f} | 2 cm {info.get("cloud_aligned", {}).get("f1@20mm", 0):.2f}'
                    f'   (cam pose err {info["pose_err_mm"]:.0f} mm, {info["pose_err_deg"]:.1f} deg)\n'
                    f'apples localised {len(est["gng"])} / {len(rec.gt_fruit_centres)}', c)
                node.get_logger().info(
                    f'  view {info["view"]:2d} st{si + 1} tilt {math.degrees(tilt):+4.0f}: seg IoU {sv.get("iou", float("nan")):.2f} | '
                    f'pts {info["points_view"]:5d} -> model {info["model_points"]:6d} | icp {"-" if "icp" not in info else ("ON " if info["icp"]["applied"] else "off")} | '
                    f'GNG {info["gng_nodes"]:4d} nodes | F1@1cm {cl.get("f1@10mm", 0):.3f} (obs {cl.get("f1_observed@10mm", 0):.3f}) '
                    f'F1@2cm {cl.get("f1@20mm", 0):.3f} | shape-aligned F1@1cm {info.get("cloud_aligned", {}).get("f1@10mm", 0):.3f} F1@2cm {info.get("cloud_aligned", {}).get("f1@20mm", 0):.3f} | cam-pose err {info["pose_err_mm"]:.0f} mm {info["pose_err_deg"]:.2f} deg | '
                    f'{info["t_total"]:.1f} s')
                view_logs.append(info)
        node.set_status(f'plant {k}: final DBL-GNG fit + metrics ...', c)
        res = rec.finalize()
        res['views_log'] = [{kk: v for kk, v in i.items()} for i in view_logs]
        res['seconds'] = time.time() - t_plant
        json.dump(res, open(f'{pdir}/metrics.json', 'w'), indent=1, default=float)
        np.savez_compressed(f'{pdir}/model.npz', P=rec.P, C=rec.C, L=rec.L)
        if rec.gng is not None:
            nodes, edges, comp = rec.graph()
            ncls, _ = rec.node_classes(nodes)
            np.savez_compressed(f'{pdir}/graph.npz', nodes=nodes, edges=edges, components=comp, node_class=ncls)
        node.done.append((rec.P.copy(), class_colors(rec.L)))
        results.append(res)
        cl, fr_, si_ = res['cloud'], res['fruit'], res['shape_iou']
        pcs = ' '.join(f'{n} {v["f1@20mm"]:.2f}' for n, v in res['per_class'].items())
        ious = '  '.join(f'{m} {si_[m]["iou@20mm"]:.2f}' for m in ('cloud', 'dbl_gng') if m in si_)
        node.set_status(f'plant {k} DONE ({res["seconds"]:.0f} s)\nF1@1cm {cl["f1@10mm"]:.2f}  F1@2cm {cl["f1@20mm"]:.2f}   per class @2cm: {pcs}\n'
                        f'voxel IoU@2cm (aligned): {ious}\n'
                        f'apples {fr_["gng"]["matched"]}/{fr_["n_true"]} localised (DBL-GNG), mean err {fr_["gng"]["mean_err_mm"] or float("nan"):.1f} mm', c)
    write_summary(out, results, time.time() - t_mission)
    node.set_status(f'ALL DONE ({len(results)} plants, {(time.time() - t_mission) / 60:.1f} min)\nresults: {out}', np.array([0.0, 0.0]))
    return results


def write_summary(out, results, seconds):
    rows = []
    for r in results:
        c, a = r['cloud'], r.get('cloud_aligned', {})
        gi, go, fr = r.get('gng_incremental', {}), r.get('gng_offline', {}), r['fruit']
        rows.append((r['plant'], r['model_points'], c['f1@10mm'], c['f1@20mm'], c['f1@50mm'], c.get('f1_observed@10mm', float('nan')),
                     a.get('f1@10mm', float('nan')), gi.get('node_to_surface_mean_mm', float('nan')), gi.get('coverage@20mm', float('nan')),
                     gi.get('nodes', 0), fr['gng']['matched'], fr['n_true'], fr['gng']['false_positive'], fr['gng']['mean_err_mm'], r['seconds']))
    hdr = ('plant | points | F1@1cm | F1@2cm | F1@5cm | F1@1cm observed-part | F1@1cm after rigid align | GNG node-to-surface mm | '
           'GNG coverage@2cm | GNG nodes | apples matched | apples true | FP | apple err mm | seconds')
    lines = [hdr, '|'.join(['---'] * 15)]
    for r in rows:
        lines.append(' | '.join('-' if (isinstance(v, float) and math.isnan(v)) or v is None else (f'{v:.3f}' if isinstance(v, float) else str(v)) for v in r))
    # voxel IoU against the true surface, aligned frame: merged cloud vs DBL-GNG nodes
    names = (('cloud', 'merged cloud'), ('dbl_gng', 'DBL-GNG nodes'))
    lines += ['', '## Voxel IoU vs the true surface (aligned frame; observed-part in brackets)', '',
              'plant | method | points | IoU@1cm | IoU@2cm | IoU@5cm | IoU@2cm pot | plant | apple | soil', '|'.join(['---'] * 10)]
    for r in results:
        si = r.get('shape_iou', {})
        for key, label in names:
            s = si.get(key)
            if not s:
                continue
            pc = s['per_class']
            cls = ' | '.join(f'{pc[n]["iou@20mm"]:.3f}' if n in pc else '-' for n in ('pot', 'plant', 'apple', 'soil'))
            lines.append(f'{r["plant"]} | {label} | {s["points"]} | ' + ' | '.join(
                f'{s[f"iou@{v}mm"]:.3f} ({s.get(f"iou_observed@{v}mm", float("nan")):.3f})' for v in (10, 20, 50)) + f' | {cls}')
    open(f'{out}/summary.md', 'w').write('\n'.join(lines) + f'\n\nmission time {seconds / 60:.1f} min\n')
    json.dump(results, open(f'{out}/summary.json', 'w'), indent=1, default=float)
    print('\n'.join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plants', default='1,2,5,4,3,0')
    ap.add_argument('--radius', type=float, default=1.4)
    ap.add_argument('--azimuths', default='0,90,180,270', help='station azimuths around each plant [deg], e.g. 0,120,240 or 0,180')
    ap.add_argument('--tilts', default='0.35,0.0,-0.35', help='rad, positive = looking down')
    ap.add_argument('--sam', default=os.path.join(WS, 'models', 'mobile_sam.pt'))
    ap.add_argument('--yolo', default=os.path.join(WS, 'models', 'yolo26n-seg.pt'))
    ap.add_argument('--use-yolo', default='false', help='true: YOLO26n-seg detections are extra MobileSAM prompts; false: MobileSAM only (geometry + red blobs)')
    ap.add_argument('--seg', default='sam', choices=['sam', 'color', 'oracle'])
    ap.add_argument('--no-icp', action='store_true')
    ap.add_argument('--pose-source', default='odom', choices=['odom', 'map'])
    ap.add_argument('--out', default=os.path.join(WS, 'results', f'scan_{time.strftime("%Y%m%d_%H%M%S")}'))
    args, ros_args = ap.parse_known_args()
    rclpy.init(args=[sys.argv[0]] + ros_args)
    gt = GroundTruth(os.path.join(WS, 'src', 'plant_sim', 'gt'))
    use_yolo = str(args.use_yolo).lower() in ('1', 'true', 'yes')
    yolo = SamSegmenter(args.sam, args.yolo if use_yolo else None, use_yolo_box=use_yolo, gate_colour=True) if args.seg == 'sam' else None
    node = ScanManager(args)
    ex = MultiThreadedExecutor(num_threads=3)
    ex.add_node(node)

    def spin():
        """An exception in any callback ends ex.spin(); without it no image, TF or pose message is processed any more
        (a stale TF made the robot look 2 m away from a station it had reached), so log it and spin on."""
        while rclpy.ok():
            try:
                ex.spin()
            except Exception as e:                              # noqa: BLE001
                node.get_logger().error(f'executor callback raised {e!r}; spinning on')
                time.sleep(0.05)
    threading.Thread(target=spin, daemon=True).start()
    nav = BasicNavigator()
    node.get_logger().info('waiting for camera, ground-truth poses and TF ...')
    while not (node.color and node.depth and node.info and 'camera_tilt_link' in node.gt_pose and node.odom):
        time.sleep(0.3)
    try:
        run(node, nav, args, gt, yolo)
        node.get_logger().info('finished; keeping the visualisation alive (Ctrl-C to quit)')
        while rclpy.ok():
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
