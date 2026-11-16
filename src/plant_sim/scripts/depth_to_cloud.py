#!/usr/bin/env python3
"""Colour + depth -> organised PointCloud2 (the equivalent of the ZED2i `cloud_registered`).

The ZED wrapper publishes three different things that are easy to confuse:
  depth/depth_registered         sensor_msgs/Image 32FC1  one distance [m] per pixel -> can only be
                                 shown as grey / a colour map, it carries no 3D points and no colour
  rgb/image_rect_color           sensor_msgs/Image        the colour picture
  point_cloud/cloud_registered   sensor_msgs/PointCloud2  x y z + packed rgb per pixel, computed
                                 from the two above (registered = depth already aligned to colour)
This node produces the last one from the first two, so colour and depth stay the only sensor topics:

  /zed/zed_node/point_cloud/cloud_registered        x y z rgb  (true colour)
  /zed/zed_node/point_cloud/cloud_registered_gray   x y z rgb  (same points, grey = black & white)

Frame = frame of the images (camera_optical_frame: x right, y down, z forward).  Like the ZED topic
the cloud is organised (height x width = image size) and invalid pixels are NaN.
"""
import message_filters
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image, PointCloud2, PointField

DTYPE = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('rgb', '<f4')])
FIELDS = [PointField(name=n, offset=4 * i, datatype=PointField.FLOAT32, count=1)
          for i, n in enumerate(('x', 'y', 'z', 'rgb'))]


class DepthToCloud(Node):
    def __init__(self):
        super().__init__('depth_to_cloud')
        p = self.declare_parameter
        color = p('color_topic', '/zed/zed_node/rgb/image_rect_color').value
        depth = p('depth_topic', '/zed/zed_node/depth/depth_registered').value
        info = p('info_topic', '/zed/zed_node/rgb/camera_info').value
        self.near = p('min_depth', 0.3).value
        self.far = p('max_depth', 8.0).value
        self.pub_rgb = self.create_publisher(
            PointCloud2, p('cloud_topic', '/zed/zed_node/point_cloud/cloud_registered').value, 2)
        self.pub_gray = self.create_publisher(
            PointCloud2, p('gray_topic', '/zed/zed_node/point_cloud/cloud_registered_gray').value, 2)
        self.K = None
        self.grid = None
        self.create_subscription(CameraInfo, info, self._info, 5)
        sync = message_filters.ApproximateTimeSynchronizer(
            [message_filters.Subscriber(self, Image, color), message_filters.Subscriber(self, Image, depth)],
            queue_size=5, slop=0.05)
        sync.registerCallback(self._frames)
        self.get_logger().info(f'{color} + {depth} -> cloud_registered (+ _gray)')

    def _info(self, m):
        K = np.array(m.k).reshape(3, 3)
        if self.K is None or not np.allclose(K, self.K):
            self.K = K
            self.grid = None

    def _make_grid(self, h, w):
        u, v = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
        self.grid = ((u - self.K[0, 2]) / self.K[0, 0], (v - self.K[1, 2]) / self.K[1, 1], (h, w))

    @staticmethod
    def _pack(r, g, b):
        return ((r.astype(np.uint32) << 16) | (g.astype(np.uint32) << 8) | b.astype(np.uint32)).view(np.float32)

    def _frames(self, cmsg, dmsg):
        if self.K is None:
            return
        h, w = dmsg.height, dmsg.width
        if cmsg.encoding != 'rgb8' or dmsg.encoding != '32FC1' or (cmsg.height, cmsg.width) != (h, w):
            self.get_logger().warn('expected rgb8 + 32FC1 of identical size', throttle_duration_sec=5.0)
            return
        if self.grid is None or self.grid[2] != (h, w):
            self._make_grid(h, w)
        z = np.frombuffer(dmsg.data, np.float32).reshape(h, w)
        z = np.where(np.isfinite(z) & (z > self.near) & (z < self.far), z, np.nan).astype(np.float32)
        pts = np.empty(h * w, DTYPE)
        pts['x'] = (self.grid[0] * z).ravel()
        pts['y'] = (self.grid[1] * z).ravel()
        pts['z'] = z.ravel()
        rgb = np.frombuffer(cmsg.data, np.uint8).reshape(h * w, 3)
        for pub, packed in ((self.pub_rgb, lambda: self._pack(rgb[:, 0], rgb[:, 1], rgb[:, 2])),
                            (self.pub_gray, lambda: self._pack(*(3 * [(0.299 * rgb[:, 0] + 0.587 * rgb[:, 1]
                                                                       + 0.114 * rgb[:, 2]).astype(np.uint8)])))):
            if pub.get_subscription_count() == 0:
                continue
            pts['rgb'] = packed()
            msg = PointCloud2(header=dmsg.header, height=h, width=w, fields=FIELDS, is_bigendian=False,
                              point_step=DTYPE.itemsize, row_step=DTYPE.itemsize * w, is_dense=False)
            msg.data = pts.tobytes()
            pub.publish(msg)


def main():
    rclpy.init()
    node = DepthToCloud()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.try_shutdown()


if __name__ == '__main__':
    main()
