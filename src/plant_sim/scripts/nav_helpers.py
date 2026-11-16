"""Navigation helpers shared by scan_manager.py and check_localization.py (Nav2 driving to the scan stations).

Two failure modes of Nav2/DWB were found with station layouts of 2 and 3 stations and are handled here:

1. DWB stands still when the robot starts more than ~90 deg off the first segment of the path: turning in place and turning
   slowly get the same score on the costmap grid (forward_point_distance 0.1), so it picks the slow one (measured: 0.03 m in 80 s;
   a farther forward point ends inside the inflated keep-out disc of a station -> 'No valid trajectories').
   -> pre_rotate(): turn in place towards the path (Nav2 spin) before the goal is sent.
2. When the next station is the opposite point of the circle, the way round the plant is a tie between the two sides.  Two plans
   made one after the other (the one that sets the pre-rotation and the one of the goal) can choose different sides, and the
   robot starts ~180 deg off its path again.
   -> arc_waypoints(): the robot follows the circle round the plant (counter-clockwise, like the orbit itself) in hops of at most
   90 deg, so every plan is short and unambiguous; a watchdog (stall_s) cancels a goal that does not move the robot and retries.
"""
import math
import time

from geometry_msgs.msg import PoseStamped
from rclpy.time import Time


def wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def make_pose(x, y, yaw):
    p = PoseStamped()
    p.header.frame_id, p.header.stamp = 'map', Time().to_msg()
    p.pose.position.x, p.pose.position.y = float(x), float(y)
    p.pose.orientation.z, p.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
    return p


def arc_waypoints(c, r, th_a, th_b, max_step=math.radians(90.0)):
    """Points on the circle (centre c, radius r) between the azimuths th_a and th_b, counter-clockwise, spaced <= max_step.
    Returns the intermediate points only (none if the sweep is <= max_step)."""
    d = (th_b - th_a) % (2 * math.pi)
    n = max(1, math.ceil(d / max_step - 1e-6))
    return [(c[0] + r * math.cos(th_a + d * i / n), c[1] + r * math.sin(th_a + d * i / n)) for i in range(1, n)]


def path_heading_error(nav, goal, pose_fn, look=0.6):
    """Angle [rad] between the heading of the robot and the first ~look m of the path Nav2 plans to goal (None: no path)."""
    x, y, yaw = pose_fn()
    try:
        path = nav.getPath(make_pose(x, y, yaw), goal, use_start=True)
    except Exception:                                                # noqa: BLE001
        return None
    if path is None or not path.poses:
        return None
    tgt = path.poses[-1].pose.position
    for p in path.poses:
        if math.hypot(p.pose.position.x - x, p.pose.position.y - y) >= look:
            tgt = p.pose.position
            break
    if math.hypot(tgt.x - x, tgt.y - y) < 0.2:                       # already at the goal: the final heading is DWB's job
        return None
    return wrap(math.atan2(tgt.y - y, tgt.x - x) - yaw)


def pre_rotate(nav, goal, pose_fn, min_err_deg=45.0, log=None):
    """Turn in place towards the path to `goal` if the heading error exceeds min_err_deg.  Returns the angle removed [deg]."""
    err = path_heading_error(nav, goal, pose_fn)
    if err is None or abs(err) < math.radians(min_err_deg):
        return 0.0
    nav.spin(spin_dist=float(err), time_allowance=20)
    t0 = time.time()
    while not nav.isTaskComplete() and time.time() - t0 < 25:
        time.sleep(0.1)
    if not nav.isTaskComplete():
        nav.cancelTask()
        time.sleep(1.0)
    if log:
        after = path_heading_error(nav, goal, pose_fn)
        log(f'pre-rotation: path heading was {math.degrees(err):+.0f} deg off, now {"?" if after is None else f"{math.degrees(after):+.0f}"} deg')
    return math.degrees(abs(err))


def drive(nav, pose_fn, x, y, yaw, timeout, log=None, stall_s=25.0, stall_m=0.12):
    """Pre-rotate, send the goal and wait.  Returns 'done', 'timeout' or 'stalled' (moved < stall_m in stall_s while farther
    than 0.35 m from the goal: the goal is cancelled so that the caller can retry with a fresh plan)."""
    goal = make_pose(x, y, yaw)
    pre_rotate(nav, goal, pose_fn, log=log)
    nav.goToPose(goal)
    t0 = time.time()
    ref_t, ref = t0, pose_fn()[:2]
    while not nav.isTaskComplete():
        time.sleep(0.2)
        now = time.time()
        status = None
        if now - t0 > timeout:
            status = 'timeout'
        else:
            try:
                px, py, _ = pose_fn()
            except Exception:                                        # noqa: BLE001
                continue
            if math.hypot(px - ref[0], py - ref[1]) > stall_m or math.hypot(px - x, py - y) < 0.35:
                ref_t, ref = now, (px, py)                          # moving, or turning in place at the goal
            elif now - ref_t > stall_s:
                status = 'stalled'
        if status:
            nav.cancelTask()
            time.sleep(1.0)                                          # let the cancel finish before another goal is sent
            if log and status == 'stalled':
                e = path_heading_error(nav, goal, pose_fn)
                log(f'stalled: moved < {stall_m} m in {stall_s:.0f} s, path heading now {"?" if e is None else f"{math.degrees(e):+.0f}"} deg off')
            return status
    return 'done'


def navigate_station(nav, pose_fn, station, waypoints=(), log=None, attempts=3, timeout=180.0, tol_m=0.6, tol_rad=0.6):
    """Drive to the station (x, y, yaw) through the waypoints (x, y), each hop oriented along the direction of travel.
    The robot's map pose decides whether it is there (Nav2's result code is stale after a cancel).
    Returns (reached, distance [m], heading error [rad], number of attempts used)."""
    sx, sy, syaw = station
    dist = dyaw = float('nan')
    for attempt in range(attempts):
        if attempt == 0:
            pts = list(waypoints) + [(sx, sy)]
            for i, (wx, wy) in enumerate(pts):
                last = i == len(pts) - 1
                if last:
                    status = drive(nav, pose_fn, sx, sy, syaw, timeout, log)
                else:
                    nx, ny = pts[i + 1]
                    drive(nav, pose_fn, wx, wy, math.atan2(ny - wy, nx - wx), 90.0, log)   # best effort: only a routing aid
        else:
            status = drive(nav, pose_fn, sx, sy, syaw, timeout, log)
        x, y, yaw = pose_fn()
        dist, dyaw = math.hypot(x - sx, y - sy), abs(wrap(yaw - syaw))
        if dist <= tol_m and dyaw <= tol_rad:
            return True, dist, dyaw, attempt + 1
        if log:
            log(f'station attempt {attempt + 1}: robot is {dist:.2f} m / {math.degrees(dyaw):.0f} deg from the station ({status})')
    return False, dist, dyaw, attempts
