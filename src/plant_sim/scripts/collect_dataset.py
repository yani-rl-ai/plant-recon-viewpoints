#!/usr/bin/env python3
"""Auto-labelled data for YOLO: the robot is teleported to random viewpoints around the plants (Gazebo set_pose),
a colour+depth frame is captured, and every pixel is labelled by comparing its measured 3D point with the exact
plant surface of the world (plant_sim/gt).  No manual labelling.

    source ~/plant_ws/env.sh
    ros2 launch plant_sim sim.launch.py rviz:=false          # Gazebo GUI stays visible while this runs
    python3 collect_dataset.py --out ~/plant_ws/datasets/plant_seg --n 340

Classes of the YOLO dataset:  0 = plant (the whole plant: pot + stems + leaves + fruit, one instance per plant),
                              1 = apple (one instance per visible apple).
The plant polygon is the plant's pixels closed and dilated into ONE region (leaves are thin and fragmented; the
pipeline later refines the pixels with colour inside this region).  For the test split the per-pixel ground truth
(class / plant id / fruit id images) is stored too, to score every segmentation method pixel by pixel.
"""
import argparse
import collections
import json
import math
import os
import subprocess
import sys
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float64
from tf2_msgs.msg import TFMessage

sys.path.insert(0, os.path.expanduser('~/plant_ws/src/plant_recon'))
from plant_recon.geom import GroundTruth, T_from_msg, optical_pose_from_gt  # noqa: E402

COLOR, DEPTH, INFO = '/zed/zed_node/rgb/image_rect_color', '/zed/zed_node/depth/depth_registered', '/zed/zed_node/rgb/camera_info'


def yaw_of(T):
    return math.atan2(T[1, 0], T[0, 0])


class Collector(Node):
    def __init__(self):
        super().__init__('collect_dataset')
        self.color = self.depth = self.info = None
        self.gt = {}
        self.hist = collections.deque(maxlen=400)           # (wall time, x, y, z, yaw, tilt)
        self.create_subscription(Image, COLOR, lambda m: setattr(self, 'color', (time.time(), m)), 2)
        self.create_subscription(Image, DEPTH, lambda m: setattr(self, 'depth', (time.time(), m)), 2)
        self.create_subscription(CameraInfo, INFO, lambda m: setattr(self, 'info', m), 2)
        self.create_subscription(TFMessage, '/gt/dynamic_pose', self._gt, 20)
        self.tilt_pub = self.create_publisher(Float64, '/camera_tilt/cmd', 10)

    def _gt(self, msg):
        for t in msg.transforms:
            self.gt[t.child_frame_id] = T_from_msg(t)
        if 'tb3_plant' in self.gt and 'camera_tilt_link' in self.gt:
            M, L = self.gt['tb3_plant'], self.gt['camera_tilt_link']
            tilt = math.atan2(-L[2, 0], L[2, 2])           # pitch of the tilt link (+ = looking down)
            self.hist.append((time.time(), M[0, 3], M[1, 3], M[2, 3], yaw_of(M), tilt))

    def spin(self, sec):
        t = time.time()
        while time.time() - t < sec:
            rclpy.spin_once(self, timeout_sec=0.02)

    def set_pose(self, x, y, yaw):
        req = f'name: "tb3_plant", position: {{x: {x:.4f}, y: {y:.4f}, z: 0.06}}, orientation: {{x: 0, y: 0, z: {math.sin(yaw / 2):.6f}, w: {math.cos(yaw / 2):.6f}}}'
        r = subprocess.run(['ign', 'service', '-s', '/world/plant_arena/set_pose', '--reqtype', 'ignition.msgs.Pose',
                            '--reptype', 'ignition.msgs.Boolean', '--timeout', '3000', '--req', req],
                           capture_output=True, text=True, timeout=10)
        return 'true' in r.stdout

    def settled(self, x, y, yaw, tilt):
        now = time.time()
        recent = [h for h in self.hist if now - h[0] < 0.4]
        if not self.hist:
            return False
        last = self.hist[-1]
        near = abs(last[1] - x) < 0.03 and abs(last[2] - y) < 0.03 and abs(math.remainder(last[4] - yaw, 2 * math.pi)) < 0.05
        tilt_ok = abs(last[5] - tilt) < 0.03
        still = all(abs(h[1] - last[1]) < 0.002 and abs(h[2] - last[2]) < 0.002 and abs(h[3] - last[3]) < 0.002
                    and abs(h[5] - last[5]) < 0.002 for h in recent)
        return near and tilt_ok and still and last[3] < 0.02 and now - self.hist[-1][0] >= 0.0

    def capture(self, x, y, yaw, tilt):
        """Teleport, settle, return (color BGR, depth, K, T_world_optical) or None."""
        self.tilt_pub.publish(Float64(data=float(tilt)))
        if not self.set_pose(x, y, yaw):
            return None
        t0 = time.time()
        stable_since = None
        while time.time() - t0 < 6.0:
            self.tilt_pub.publish(Float64(data=float(tilt)))
            self.spin(0.05)
            if self.settled(x, y, yaw, tilt):
                stable_since = stable_since or time.time()
                if time.time() - stable_since > 0.5:
                    break
            else:
                stable_since = None
        else:
            return None
        t_ok = time.time()
        while time.time() - t_ok < 3.0:
            self.spin(0.05)
            if self.color and self.depth and self.color[0] > t_ok and self.depth[0] > t_ok and \
                    abs((self.color[1].header.stamp.sec + 1e-9 * self.color[1].header.stamp.nanosec) -
                        (self.depth[1].header.stamp.sec + 1e-9 * self.depth[1].header.stamp.nanosec)) < 2e-3:
                break
        else:
            return None
        c, d = self.color[1], self.depth[1]
        color = np.frombuffer(c.data, np.uint8).reshape(c.height, c.width, 3)[:, :, ::-1].copy()
        depth = np.frombuffer(d.data, np.float32).reshape(d.height, d.width).copy()
        K = np.array(self.info.k).reshape(3, 3)
        T = optical_pose_from_gt(self.gt['tb3_plant'], self.gt['camera_tilt_link'])
        return color, depth, K, T


def contour_polygon(mask, min_area):
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cs = [c for c in cs if cv2.contourArea(c) >= min_area]
    if not cs:
        return None
    c = cv2.approxPolyDP(max(cs, key=cv2.contourArea), 0.8, True).reshape(-1, 2)
    return c if len(c) >= 3 else None


def yolo_labels(pid, fid, h, w):
    lines = []
    for k in sorted(set(pid[pid > 0].tolist())):
        m = (pid == k).astype(np.uint8)
        m = cv2.dilate(cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2), np.ones((9, 9), np.uint8))
        poly = contour_polygon(m, 150)
        if poly is not None:
            lines.append('0 ' + ' '.join(f'{px / w:.5f} {py / h:.5f}' for px, py in poly))
    for j in sorted(set(fid[fid > 0].tolist())):
        poly = contour_polygon(fid == j, 25)
        if poly is not None:
            lines.append('1 ' + ' '.join(f'{px / w:.5f} {py / h:.5f}' for px, py in poly))
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.expanduser('~/plant_ws/datasets/plant_seg'))
    ap.add_argument('--n', type=int, default=340)
    ap.add_argument('--n-test', type=int, default=40)
    ap.add_argument('--seed', type=int, default=11)
    args = ap.parse_args()
    gt = GroundTruth(os.path.expanduser('~/plant_ws/src/plant_sim/gt'))
    rng = np.random.default_rng(args.seed)
    for s in ('train', 'val', 'test'):
        os.makedirs(f'{args.out}/images/{s}', exist_ok=True)
        os.makedirs(f'{args.out}/labels/{s}', exist_ok=True)
    os.makedirs(f'{args.out}/test_gt', exist_ok=True)
    rclpy.init()
    n = Collector()
    n.get_logger().info('waiting for camera and ground-truth poses ...')
    while not (n.color and n.depth and n.info and 'camera_tilt_link' in n.gt):
        n.spin(0.2)
        n.tilt_pub.publish(Float64(data=0.0))
    meta = open(f'{args.out}/meta.jsonl', 'a')
    made, tries, t_start = 0, 0, time.time()
    while made < args.n and tries < 3 * args.n:
        tries += 1
        k = made % 6
        c = gt.centres[k]
        r, th = rng.uniform(1.0, 2.8), rng.uniform(0, 2 * math.pi)
        x, y = c[0] + r * math.cos(th), c[1] + r * math.sin(th)
        yaw = math.atan2(c[1] - y, c[0] - x) + rng.uniform(-0.3, 0.3)
        tilt = float(rng.choice([-0.3, 0.0, 0.25, 0.5]))
        others = np.delete(gt.centres, k, axis=0)
        if abs(x) > 4.3 or abs(y) > 4.3 or np.min(np.linalg.norm(others - [x, y], axis=1)) < 1.0:
            continue
        got = n.capture(x, y, yaw, tilt)
        if got is None:
            n.get_logger().warn('capture failed, skipping')
            continue
        color, depth, K, T = got
        cls, pid, fid = gt.label_pixels(depth, K, T)
        if (pid > 0).sum() < 150:
            continue                                         # plant (almost) not in view
        split = 'test' if made >= args.n - args.n_test else ('val' if made % 7 == 3 else 'train')
        name = f'{made:04d}'
        cv2.imwrite(f'{args.out}/images/{split}/{name}.png', color)
        lines = yolo_labels(pid, fid, *depth.shape)
        open(f'{args.out}/labels/{split}/{name}.txt', 'w').write('\n'.join(lines) + ('\n' if lines else ''))
        if split == 'test':
            cv2.imwrite(f'{args.out}/test_gt/{name}_cls.png', cls)
            cv2.imwrite(f'{args.out}/test_gt/{name}_pid.png', pid)
            cv2.imwrite(f'{args.out}/test_gt/{name}_fid.png', fid)
            np.save(f'{args.out}/test_gt/{name}_depth.npy', depth.astype(np.float16))
        meta.write(json.dumps({'name': name, 'split': split, 'plant': k, 'x': x, 'y': y, 'yaw': yaw, 'tilt': tilt, 'dist': r,
                               'K': K.tolist(), 'T_world_optical': T.tolist(), 'plant_px': int((pid > 0).sum()),
                               'apples': len(set(fid[fid > 0].tolist()))}) + '\n')
        meta.flush()
        made += 1
        if made % 10 == 0 or made == 1:
            n.get_logger().info(f'{made}/{args.n}  ({(time.time() - t_start) / made:.1f} s/sample)  last: plant {k} at {r:.1f} m, '
                                f'tilt {tilt:+.2f}, {int((pid > 0).sum())} plant px, {len(lines)} labels')
    ys = [''] * 0
    open(f'{args.out}/data.yaml', 'w').write(
        f'path: {args.out}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: plant\n  1: apple\n')
    n.get_logger().info(f'done: {made} samples in {(time.time() - t_start) / 60:.1f} min -> {args.out}')
    n.set_pose(0.0, 0.0, 0.0)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
