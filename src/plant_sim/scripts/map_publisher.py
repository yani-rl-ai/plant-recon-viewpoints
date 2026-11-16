#!/usr/bin/env python3
"""Publish a saved occupancy map (.yaml + .pgm from nav2 map_saver_cli) on /map, latched.

    ros2 run plant_sim map_publisher.py --ros-args -p yaml_filename:=/path/to/map.yaml

Used instead of nav2 map_server + lifecycle_manager because
  * the lifecycle manager asks map_server to configure before the node is ready; the request can get lost and
    then the map is never loaded (seen in the plant_sim launch, intermittently);
  * with the saver's default free_thresh 0.25 the nav2 loader reads the grey "unknown" pixel (205 -> 0.196)
    as FREE.  Here:  pixel 0 / occupied_thresh -> 100 (occupied), pixel >= 250 -> 0 (free), everything else
    (205) -> -1 (unknown).
Frame: map.  The robot starts at the map origin of the SLAM run, hence map == odom == world (identity TF).
"""
import json
import os

import cv2
import numpy as np
import rclpy
import yaml
from nav_msgs.msg import MapMetaData, OccupancyGrid
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy


def load_grid(yaml_path):
    meta = yaml.safe_load(open(yaml_path))
    if meta.get('mode', 'trinary') != 'trinary':
        raise ValueError("only mode 'trinary' maps (the map_saver_cli default) are supported")
    img = cv2.imread(os.path.join(os.path.dirname(os.path.abspath(yaml_path)), meta['image']), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(meta['image'])
    occ_prob = img / 255.0 if meta.get('negate', 0) else (255 - img) / 255.0
    grid = np.full(img.shape, -1, np.int8)
    free = img >= 250 if not meta.get('negate', 0) else img <= 5
    grid[free] = 0
    grid[occ_prob > float(meta['occupied_thresh'])] = 100
    return grid[::-1], float(meta['resolution']), meta['origin']       # image row 0 = top, grid row 0 = bottom


class MapPublisher(Node):
    def __init__(self):
        super().__init__('map_publisher')
        path = self.declare_parameter('yaml_filename', '').value
        self.pub = self.create_publisher(OccupancyGrid, '/map', QoSProfile(
            depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE))
        grid, res, origin = load_grid(path)
        self.msg = OccupancyGrid()
        self.msg.header.frame_id = 'map'
        self.msg.info = MapMetaData(resolution=res, width=grid.shape[1], height=grid.shape[0])
        self.msg.info.origin.position.x, self.msg.info.origin.position.y = float(origin[0]), float(origin[1])
        self.msg.info.origin.orientation.w = 1.0
        self.msg.data = grid.ravel().tolist()
        self.get_logger().info(f'{path}: {grid.shape[1]} x {grid.shape[0]} cells @ {res} m, origin {origin[:2]}, '
                               f'{int((grid == 100).sum())} occupied / {int((grid == 0).sum())} free / '
                               f'{int((grid == -1).sum())} unknown')
        plants = self.declare_parameter('keepout_plants', '').value          # path of gt/plants.json (positions only)
        radius = self.declare_parameter('keepout_radius', 0.95).value
        self.nav_msg = None
        if plants:
            # The laser plane (0.185 m) only meets the pots, but the canopy is up to 0.62 m wide at camera height and
            # the camera needs 0.3 m of free depth: planners must keep out of a disk around each plant.
            nav = grid.copy()
            res_m = res
            yy, xx = np.mgrid[0:grid.shape[0], 0:grid.shape[1]]
            wx, wy = origin[0] + (xx + 0.5) * res_m, origin[1] + (yy + 0.5) * res_m          # grid row 0 = bottom
            for pl in json.load(open(plants))['plants']:
                nav[np.hypot(wx - pl['xy'][0], wy - pl['xy'][1]) <= radius] = 100
            self.nav_msg = OccupancyGrid()
            self.nav_msg.header.frame_id = 'map'
            self.nav_msg.info = self.msg.info
            self.nav_msg.data = nav.ravel().tolist()
            self.nav_pub = self.create_publisher(OccupancyGrid, '/nav_map', QoSProfile(
                depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE))
            self.get_logger().info(f'/nav_map: keep-out radius {radius} m around {len(json.load(open(plants))["plants"])} plants')
        self.publish()      # once: the transient-local QoS hands it to every later subscriber.  Re-sending it would
                            # make AMCL (and the costmaps) re-initialise on each copy.

    def publish(self):
        self.msg.header.stamp = self.get_clock().now().to_msg()
        self.msg.info.map_load_time = self.msg.header.stamp
        self.pub.publish(self.msg)
        if self.nav_msg is not None:
            self.nav_msg.header.stamp = self.msg.header.stamp
            self.nav_pub.publish(self.nav_msg)


def main():
    rclpy.init()
    node = MapPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
