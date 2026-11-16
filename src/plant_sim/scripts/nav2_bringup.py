#!/usr/bin/env python3
"""Bring the Nav2 lifecycle nodes to ACTIVE, robust against lost service responses.

Why not nav2_lifecycle_manager: in this setup (FastDDS, ROS_LOCALHOST_ONLY) the response to a `change_state`
call is sometimes dropped (`failed to send response to /amcl/change_state (timeout)`: the client's response
reader is not matched yet) although the node DID change state; the manager then waits forever and the node
is never activated.  Here every step is verified with `get_state` and repeated until the node reports ACTIVE,
so a lost response costs one retry instead of a dead navigation stack.

    ros2 run plant_sim nav2_bringup.py            # waits for the nodes, activates them in order, then exits

(No bond to a manager is created; nothing in this stack depends on it.)
"""
import sys
import time

import rclpy
from lifecycle_msgs.msg import State, Transition
from lifecycle_msgs.srv import ChangeState, GetState
from rclpy.node import Node

NODES = ['controller_server', 'smoother_server', 'planner_server', 'behavior_server',
         'bt_navigator', 'waypoint_follower', 'velocity_smoother']


class Bringup(Node):
    def __init__(self):
        super().__init__('nav2_bringup')
        self.declare_parameter('with_amcl', True)        # AMCL is a lifecycle node and is activated first
        self.names = (['amcl'] if self.get_parameter('with_amcl').value else []) + NODES
        self.lc = {n: (self.create_client(GetState, f'/{n}/get_state'),
                            self.create_client(ChangeState, f'/{n}/change_state')) for n in self.names}

    def call(self, client, request, timeout):
        fut = client.call_async(request)
        t0 = time.time()
        while not fut.done() and time.time() - t0 < timeout:
            rclpy.spin_once(self, timeout_sec=0.05)
        return fut.result() if fut.done() else None

    def state(self, name):
        r = self.call(self.lc[name][0], GetState.Request(), 2.0)
        return None if r is None else r.current_state.id

    def activate(self, name, deadline):
        while time.time() < deadline:
            s = self.state(name)
            if s == State.PRIMARY_STATE_ACTIVE:
                return True
            step = {State.PRIMARY_STATE_UNCONFIGURED: Transition.TRANSITION_CONFIGURE,
                    State.PRIMARY_STATE_INACTIVE: Transition.TRANSITION_ACTIVATE}.get(s)
            if step is not None:
                req = ChangeState.Request()
                req.transition.id = step
                self.call(self.lc[name][1], req, 3.0)        # the response may be lost: re-checked next loop
            time.sleep(0.3)
        return False


def main():
    rclpy.init()
    n = Bringup()
    t0 = time.time()
    ok = True
    for name in n.names:
        while not n.lc[name][0].wait_for_service(timeout_sec=1.0) and time.time() - t0 < 90:
            pass
        if n.activate(name, t0 + 90):
            n.get_logger().info(f'{name}: ACTIVE')
        else:
            n.get_logger().error(f'{name}: could not be activated within 90 s')
            ok = False
    n.get_logger().info(f'NAV2 READY: all {len(n.names)} nodes active after {time.time() - t0:.1f} s' if ok
                        else 'NAV2 NOT READY: see errors above')
    n.destroy_node()
    rclpy.try_shutdown()
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
