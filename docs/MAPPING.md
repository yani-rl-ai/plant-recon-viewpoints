# Making a new 2-D map of the arena (optional)

The repository already contains a map, `src/plant_sim/maps/plant_arena.pgm/.yaml` (5 cm per cell), made with SLAM Toolbox
from the 270° laser scan. **You do not need this document to run the missions.** Use it if you change the world
(for example with `gen_world.py`) or want to make your own map.

The laser's scan plane is 0.185 m above the floor, so the map contains the **walls and the six pots** (0.22 m wide
circles), but not the foliage.

> The map is created with the robot starting at the origin (0, 0) heading 0°, and everything else assumes that the
> map frame equals the Gazebo world frame. Do not change the spawn pose (`x:=`, `y:=`, `yaw:=`) while mapping.

## 1. Prepare

```bash
cd ~/plant_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select plant_sim
~/plant_ws/stop_sim.sh          # no old simulation may be running
```

Run `source ~/plant_ws/env.sh` in every terminal below.

## 2. Start the mapping (3 terminals)

**Terminal 1: simulation, SLAM and RViz**

```bash
source ~/plant_ws/env.sh
ros2 launch plant_sim mapping.launch.py
```

Wait about 15 s. Gazebo and RViz open, the robot stands at the centre of the arena, red dots are the laser scan, and the
map starts to grow.

**Terminal 2: keyboard teleop with limited speeds**

```bash
source ~/plant_ws/env.sh
ros2 run plant_sim teleop_keyboard.py --max-lin 0.2 --max-ang 0.25
```

Click this terminal so that it has focus.

| Key | Action |
|---|---|
| `w` / `x` | forward / backward (±0.02 m/s per press, max 0.2) |
| `a` / `d` | turn left / right (±0.1 rad/s per press, max 0.25) |
| `s` or space | **stop** |
| `q` | quit |

Terminal 3 is used later to save the map.

## 3. How to drive (this decides the map quality)

In the author's experiments, fast in-place turns (0.6 rad/s) gave a map that was skewed by 3–5° with doubled walls and
shifted pots (9 of 10 checks failed); with all turns slow (0.25 rad/s) the map matched the true world (median error
2.5 cm, six pots found with a centre error of about 3 cm). The exact cause was not established; slowing the turns fixed it.

1. **Stop (`s`) before turning.** Turn on the spot slowly, then drive on. Do not turn while driving fast.
2. **Slow turns:** with `--max-ang 0.25`, press `a` or `d` only 2–3 times (0.1 / 0.2 / 0.25 rad/s).
3. **Drive at most 0.2 m/s.**
4. **Keep at least 1 m from walls and pots.** The pots have physical collision and the teleop does not protect you.
   The pots are at (−3, ±2), (0, ±2), (3, ±2).
5. Do not close the Gazebo window: that ends the SLAM node and the map is lost.

Suggested route (about 10–15 minutes; coordinates are approximately the RViz grid, (0, 0) = start):

| # | Go to | Note |
|---|---|---|
| 0 | (0, 0) | **turn 360° slowly** (about 25 s), then `s` |
| 1 | (−1.5, 0) | |
| 2 | (−4, 0) | |
| 3 | (−4, −4) | along the west wall to the south |
| 4 | (4, −4) | south wall |
| 5 | (4, 4) | east wall |
| 6 | (−4, 4) | north wall |
| 7 | (−4, 0) | |
| 8 | (−1.5, 0) | |
| 9 | (−1.5, −4) | corridor between the plants (x = −1.5) |
| 10 | (−1.5, 4) | |
| 11 | (1.5, 4) | |
| 12 | (1.5, −4) | corridor between the plants (x = 1.5) |
| 13 | (1.5, 0) | |
| 14 | (0, 0) | done |

The corridors at x = ±1.5 m pass the inner side of the pots, the outer ring sees their outer side, so that every pot is
seen from all sides and becomes a complete circle on the map.

| Good | Bad: stop and start again |
|---|---|
| walls are straight, **one** thin line | walls doubled, thick or layered |
| the arena is a square aligned with the grid | the arena looks tilted or like a parallelogram |
| every pot is one small circle | pots doubled or shifted |

The cause is almost always turning too fast or turning while driving.

## 4. (Optional) Automatic mapping

The robot can drive the route itself, following the SLAM pose and stopping if something is closer than 0.35 m ahead. The
reference map `plant_arena.*` was made this way. In terminal 2 instead of the teleop:

```bash
source ~/plant_ws/env.sh
ros2 run plant_sim auto_map.py --save ~/plant_ws/src/plant_sim/maps/auto_map
```

## 5. Save the map

> **Save before you close Terminal 1, Gazebo or RViz.** The map lives inside the running `slam_toolbox` node.

Terminal 3:

```bash
source ~/plant_ws/env.sh
ros2 run nav2_map_server map_saver_cli -f ~/plant_ws/src/plant_sim/maps/my_map \
  --ros-args -p save_map_timeout:=20.0
```

It prints `Map saved successfully` and creates `my_map.pgm` (white = free, black = occupied, grey = unknown) and
`my_map.yaml`:

```yaml
image: my_map.pgm
mode: trinary
resolution: 0.05            # 5 cm per cell
origin: [-10.4, -10.3, 0]   # lower-left corner of the image in the map frame [m]
negate: 0
occupied_thresh: 0.65
free_thresh: 0.25
```

Optionally save the SLAM pose graph (about 30 MB, only useful to continue mapping later):

```bash
ros2 service call /slam_toolbox/serialize_map slam_toolbox/srv/SerializePoseGraph \
  "{filename: '$HOME/plant_ws/src/plant_sim/maps/my_map_posegraph'}"
```

Use a name other than `plant_arena` so that the reference map is not overwritten. Pose graphs are ignored by git.

## 6. Check the map

The script compares the map with the exact geometry of this world (walls at ±5 m, six pots of radius 0.11 m):

```bash
python3 ~/plant_ws/src/plant_sim/scripts/check_map.py \
  --map ~/plant_ws/src/plant_sim/maps/my_map.yaml --png /tmp/my_map_check.png
```

A good map prints `RESULT: ALL PASS`. For the reference map:

```
PASS  occupied cells vs real surfaces: median 2.5 cm, 95 % 7.5 cm ...
PASS  walls: 100.0 % of the 40 m of wall perimeter has an occupied cell within 8 cm
PASS  pot 0 at (-3.0, -2.0): 18 cells, fitted centre error 3.0 cm, radius 10.7 cm ...
...
RESULT: ALL PASS
```

`/tmp/my_map_check.png` overlays the map (grey/black) with the true walls (cyan line) and pots (green circles): the walls
must lie on the cyan line and each pot inside its green circle. If a check fails, look at the PNG and map again with
slower turns.

## 7. Use the map

View it in RViz with the simulation (the file is read directly, no rebuild):

```bash
source ~/plant_ws/env.sh
ros2 launch plant_sim sim.launch.py map:=$HOME/plant_ws/src/plant_sim/maps/my_map.yaml
```

Without `map:=` the simulation loads `maps/plant_arena.yaml`. To make your map the default (back up the old one first):

```bash
cd ~/plant_ws/src/plant_sim/maps
mkdir -p ~/plant_ws/maps_backup && cp plant_arena.* ~/plant_ws/maps_backup/
cp my_map.pgm plant_arena.pgm
sed 's/my_map.pgm/plant_arena.pgm/' my_map.yaml > plant_arena.yaml
cd ~/plant_ws && source /opt/ros/humble/setup.bash && colcon build --symlink-install --packages-select plant_sim
```

## 8. Common problems

| Symptom | Cause and fix |
|---|---|
| `ros2 topic list` shows no `/scan` or `/odom` | `env.sh` not sourced in that terminal |
| RViz shows no map | wait a few seconds or move the robot a little; Terminal 1 must run `mapping.launch.py`, not `sim.launch.py` |
| `map_saver_cli` fails or times out | SLAM is not running or the domain differs: check `env.sh` and that Terminal 1 is alive |
| Teleop does not move the robot | the teleop window is not focused, or `env.sh` is missing |
| Robot hit a pot and is stuck | back up with `x`; if the map is damaged, start again |
| Tilted map or doubled walls | turns were too fast, see section 3 |
| Everything died at once | the Gazebo window was closed, which ends the whole launch |
| Stop everything | `~/plant_ws/stop_sim.sh` |
