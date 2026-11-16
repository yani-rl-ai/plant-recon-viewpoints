#!/usr/bin/env python3
"""Generate the Gazebo world with 6 potted fruit plants and their exact ground truth.

    python3 gen_world.py [--seed 7]        # run from anywhere, writes into the package

Writes (relative to the package root):
    worlds/plant_arena.sdf                      10 x 10 m walled arena, 6 plants
    models/plant_assets/meshes/leaf.obj         shared (double-sided) leaf mesh
    gt/plants_gt.npz                            ground-truth surface points of every plant
    gt/plants.json                              plant poses, fruit centres / radii, orbit centres

Every plant is built from primitives only (cylinders, spheres, one leaf mesh), so its
surface can be sampled analytically: the ground truth used later for F1 / fruit
localisation error is the real rendered geometry, not an approximation.

Labels (also given to Gazebo's Label system for segmentation cameras):
    0 pot + soil    (label 3 in Gazebo)
    1 foliage       (stem, branches, leaves, fruit stalks; label 1 in Gazebo)
    2 fruit         (apples; label 2 in Gazebo)
"""
import argparse
import copy
import json
import os

import numpy as np

PKG = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# plant positions (x, y): two rows of three, 3 m apart -> orbit circles of r = 1.2 m never touch
PLANT_XY = [(-3.0, -2.0), (0.0, -2.0), (3.0, -2.0), (-3.0, 2.0), (0.0, 2.0), (3.0, 2.0)]
ARENA = 10.0                 # inner side length [m]
WALL_T, WALL_H = 0.2, 1.5
POT_R, POT_H = 0.11, 0.20
GT_DENSITY = 1.5e5           # surface samples per m^2 (about 2.6 mm spacing)
GZ_LABEL = {"pot": 3, "foliage": 1, "fruit": 2}


# ------------------------------------------------------------------ geometry helpers
def unit(v):
    return v / np.linalg.norm(v)


def rot_z_to(d):
    """Rotation matrix whose z axis is d."""
    z = unit(np.asarray(d, float))
    a = np.array([1.0, 0, 0]) if abs(z[0]) < 0.9 else np.array([0, 1.0, 0])
    x = unit(np.cross(a, z))
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)


def rpy_from_R(R):
    """SDF rpy for R = Rz(yaw) Ry(pitch) Rx(roll)."""
    pitch = -np.arcsin(np.clip(R[2, 0], -1, 1))
    if abs(R[2, 0]) < 0.999999:
        return np.arctan2(R[2, 1], R[2, 2]), pitch, np.arctan2(R[1, 0], R[0, 0])
    return 0.0, pitch, np.arctan2(-R[0, 1], R[1, 1])


def Rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def axis_angle(axis, a):
    k = unit(axis)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(a) * K + (1 - np.cos(a)) * K @ K


# ------------------------------------------------------------------ leaf mesh
def leaf_mesh(nx=12):
    """Unit leaf: x = 0..1 (base -> tip), y in [-0.5, 0.5] (width), z up; V-fold + droop."""
    t = np.linspace(0, 1, nx + 1)
    half = 0.5 * np.sin(np.pi * t ** 0.8) ** 0.9
    V = []
    for ti, hw in zip(t, half):
        droop = -0.12 * ti ** 2
        for s in (-1, 0, 1):
            V.append([ti, s * hw, droop - 0.35 * abs(s) * hw * 0.5])
    V = np.array(V)
    F = []
    for i in range(nx):
        a, b = 3 * i, 3 * (i + 1)
        for j in (0, 1):                      # left half, right half
            p0, p1, p2, p3 = a + j, a + j + 1, b + j + 1, b + j
            F += [[p0, p1, p2], [p0, p2, p3]]
    F = np.array(F)
    # make sure all faces look up (+z) for the front side
    n = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    flip = n[:, 2] < 0
    F[flip] = F[flip][:, ::-1]
    return V, F


def write_leaf_obj(path):
    V, F = leaf_mesh()
    fn = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
    F, fn = F[np.linalg.norm(fn, axis=1) > 1e-9], fn[np.linalg.norm(fn, axis=1) > 1e-9]   # drop degenerate faces
    vn = np.zeros_like(V)
    for k in range(3):
        np.add.at(vn, F[:, k], fn)
    norm = np.linalg.norm(vn, axis=1, keepdims=True)
    vn = np.where(norm > 1e-12, vn / np.maximum(norm, 1e-12), np.array([0.0, 0.0, 1.0]))
    n = len(V)
    lines = ["# procedural leaf, double sided (second copy = flipped normals/winding)"]
    lines += [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in V]
    lines += [f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in V]
    lines += [f"vn {x:.6f} {y:.6f} {z:.6f}" for x, y, z in vn]
    lines += [f"vn {-x:.6f} {-y:.6f} {-z:.6f}" for x, y, z in vn]
    for a, b, c in F:
        lines.append(f"f {a + 1}//{a + 1} {b + 1}//{b + 1} {c + 1}//{c + 1}")
    for a, b, c in F:
        lines.append(f"f {c + 1 + n}//{c + 1 + n} {b + 1 + n}//{b + 1 + n} {a + 1 + n}//{a + 1 + n}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ------------------------------------------------------------------ surface sampling
def sample_triangles(P, F, rng, density):
    tri = P[F]
    area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
    n = int(round(area.sum() * density))
    idx = rng.choice(len(F), size=max(n, 1), p=area / area.sum())
    u, v = rng.random((len(idx), 1)), rng.random((len(idx), 1))
    flip = (u + v) > 1
    u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
    t = tri[idx]
    return t[:, 0] + u * (t[:, 1] - t[:, 0]) + v * (t[:, 2] - t[:, 0])


def sample_sphere(c, r, rng, density):
    n = max(int(4 * np.pi * r * r * density), 1)
    d = rng.standard_normal((n, 3))
    return c + r * d / np.linalg.norm(d, axis=1, keepdims=True)


def sample_disk(r, z, rng, density):
    n = max(int(np.pi * r * r * density), 1)
    rr, tt = r * np.sqrt(rng.random(n)), rng.random(n) * 2 * np.pi
    return np.stack([rr * np.cos(tt), rr * np.sin(tt), np.full(n, z)], axis=1)


def sample_cylinder(p0, p1, r, rng, density, caps=False):
    axis = np.asarray(p1, float) - np.asarray(p0, float)
    L = np.linalg.norm(axis)
    R = rot_z_to(axis)
    n = max(int(2 * np.pi * r * L * density), 1)
    th, h = rng.random(n) * 2 * np.pi, rng.random(n) * L
    pts = np.stack([r * np.cos(th), r * np.sin(th), h], axis=1)
    if caps:
        m = max(int(np.pi * r * r * density), 1)
        rr, tt = r * np.sqrt(rng.random(m)), rng.random(m) * 2 * np.pi
        pts = np.vstack([pts, np.stack([rr * np.cos(tt), rr * np.sin(tt), np.full(m, L)], axis=1)])
    return np.asarray(p0, float) + pts @ R.T


# ------------------------------------------------------------------ plant construction
TARGET_TOP = 1.50          # total height of EVERY plant, pot included (top of the highest leaf) [m]


def build_plant(rng, H):
    """Primitives of one plant in the plant frame (origin = pot centre on the floor).
    H = height of the main stem above the pot; the random draws do not depend on H, so build_tall_plant can
    tune H until the whole plant is TARGET_TOP high."""
    z0 = POT_H
    n_nodes = 10
    pts = [np.array([0.0, 0.0, z0])]
    for i in range(1, n_nodes + 1):
        pts.append(pts[-1] + np.array([rng.normal(0, 0.010), rng.normal(0, 0.010), H / n_nodes]))
    stem = np.array(pts)
    cyl = [(stem[i], stem[i + 1], 0.015 - 0.008 * i / n_nodes) for i in range(n_nodes)]   # (p0, p1, radius)
    leaves = []                                          # (pos, R, L, W)
    fruit_anchors = []                                   # (attach point, outward dir)

    def add_leaf(pos, az, elev, roll, L):
        d = np.array([np.cos(elev) * np.cos(az), np.cos(elev) * np.sin(az), np.sin(elev)])
        n0 = np.array([0, 0, 1.0]) - d[2] * d
        n0 = unit(n0) if np.linalg.norm(n0) > 1e-6 else np.array([1.0, 0, 0])
        n0 = axis_angle(d, roll) @ n0
        R = np.stack([d, np.cross(n0, d), n0], axis=1)
        leaves.append((pos, R, L, L * rng.uniform(0.38, 0.52)))

    n_br = int(rng.integers(7, 10))
    az0 = rng.uniform(0, 2 * np.pi)
    for b in range(n_br):
        u = rng.uniform(0.22, 0.90)
        i = min(int(u * n_nodes), n_nodes - 1)
        start = stem[i] + (u * n_nodes - i) * (stem[i + 1] - stem[i])
        az = az0 + b * 2 * np.pi / n_br + rng.normal(0, 0.25)
        el = np.deg2rad(rng.uniform(20, 50))
        L = rng.uniform(0.25, 0.45)
        d1 = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
        el2 = el - np.deg2rad(rng.uniform(15, 30))
        d2 = np.array([np.cos(el2) * np.cos(az), np.cos(el2) * np.sin(az), np.sin(el2)])
        p1, p2 = start + 0.5 * L * d1, start + 0.5 * L * d1 + 0.5 * L * d2
        cyl += [(start, p1, 0.0075), (p1, p2, 0.006)]
        for p, dd in ((p1, d1), (p2, d2)):
            for _ in range(int(rng.integers(2, 4))):
                add_leaf(p, az + rng.normal(0, 0.9), np.deg2rad(rng.uniform(-15, 50)),
                         rng.normal(0, 0.45), rng.uniform(0.12, 0.22))
        fruit_anchors.append((p2 + 0.02 * d2, np.array([np.cos(az), np.sin(az), 0.0])))
    for i in range(3, n_nodes):                           # leaves along the main stem
        for _ in range(2):
            add_leaf(stem[i], rng.uniform(0, 2 * np.pi), np.deg2rad(rng.uniform(-10, 40)),
                     rng.normal(0, 0.4), rng.uniform(0.12, 0.20))
    for _ in range(int(rng.integers(5, 9))):              # crown
        add_leaf(stem[-1], rng.uniform(0, 2 * np.pi), np.deg2rad(rng.uniform(30, 70)),
                 rng.normal(0, 0.4), rng.uniform(0.10, 0.18))

    n_fruit = int(rng.integers(4, 8))
    order = rng.permutation(len(fruit_anchors))[:n_fruit]
    fruits, stalks = [], []
    for k in order:
        a, out = fruit_anchors[k]
        r = rng.uniform(0.034, 0.044)
        stalk_len = 0.03
        c = a + np.array([0, 0, -stalk_len - 0.9 * r])
        if all(np.linalg.norm(c - f[0]) > 2 * (r + f[1]) + 0.01 for f in fruits):
            fruits.append((c, r))
            stalks.append((a, a + np.array([0, 0, -stalk_len]), 0.002))
    return {"cyl": cyl + stalks, "leaves": leaves, "fruits": fruits}


def plant_top(plant):
    """Highest point of the plant geometry."""
    V, _ = leaf_mesh()
    z = [max(p0[2], p1[2]) + r for p0, p1, r in plant["cyl"]] + [c[2] + r for c, r in plant["fruits"]]
    for pos, R, L, W in plant["leaves"]:
        z.append(float((pos + (V * np.array([L, W, L])) @ R.T)[:, 2].max()))
    return max(z)


def build_tall_plant(rng, target=TARGET_TOP):
    """A random plant whose top is exactly `target` above the floor (H is tuned, the shape stays random)."""
    state = copy.deepcopy(rng.bit_generator.state)
    H = target - POT_H - 0.18
    for _ in range(8):
        r2 = np.random.default_rng()
        r2.bit_generator.state = copy.deepcopy(state)
        plant = build_plant(r2, H)
        top = plant_top(plant)
        if abs(top - target) < 0.003:
            break
        H += target - top
    rng.bit_generator.state = r2.bit_generator.state
    return plant, top


# ------------------------------------------------------------------ SDF writers
def f6(v):
    return " ".join(f"{x:.5f}" for x in v)


def cyl_visual(name, p0, p1, r, rgb):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    c, L = 0.5 * (p0 + p1), np.linalg.norm(p1 - p0)
    roll, pitch, yaw = rpy_from_R(rot_z_to(p1 - p0))
    return (f'<visual name="{name}"><pose>{f6(c)} {roll:.5f} {pitch:.5f} {yaw:.5f}</pose><geometry><cylinder>'
            f'<radius>{r:.5f}</radius><length>{L:.5f}</length></cylinder></geometry>'
            f'<material><ambient>{f6(rgb)} 1</ambient><diffuse>{f6(rgb)} 1</diffuse></material></visual>')


def sphere_visual(name, c, r, rgb):
    return (f'<visual name="{name}"><pose>{f6(c)} 0 0 0</pose><geometry><sphere><radius>{r:.5f}</radius></sphere>'
            f'</geometry><material><ambient>{f6(rgb)} 1</ambient><diffuse>{f6(rgb)} 1</diffuse>'
            f'<specular>0.3 0.3 0.3 1</specular></material></visual>')


def leaf_visual(name, pos, R, L, W, rgb):
    roll, pitch, yaw = rpy_from_R(R)
    return (f'<visual name="{name}"><pose>{f6(pos)} {roll:.5f} {pitch:.5f} {yaw:.5f}</pose><geometry><mesh>'
            f'<uri>model://plant_assets/meshes/leaf.obj</uri><scale>{L:.5f} {W:.5f} {L:.5f}</scale></mesh>'
            f'</geometry><material><ambient>{f6(rgb)} 1</ambient><diffuse>{f6(rgb)} 1</diffuse></material></visual>')


def model_xml(name, pose, label, body, collision=""):
    return (f'<model name="{name}"><static>true</static><pose>{f6(pose)}</pose>'
            f'<plugin filename="ignition-gazebo-label-system" name="ignition::gazebo::systems::Label">'
            f'<label>{label}</label></plugin><link name="link">{collision}{body}</link></model>')


def box_model(name, pose, size, rgb):
    return (f'<model name="{name}"><static>true</static><pose>{f6(pose)} 0 0 0</pose><link name="link">'
            f'<collision name="c"><geometry><box><size>{f6(size)}</size></box></geometry></collision>'
            f'<visual name="v"><geometry><box><size>{f6(size)}</size></box></geometry>'
            f'<material><ambient>{f6(rgb)} 1</ambient><diffuse>{f6(rgb)} 1</diffuse></material></visual>'
            f'</link></model>')


WORLD_HEAD = """<?xml version="1.0"?>
<!-- GENERATED by scripts/gen_world.py - do not edit by hand. Gazebo Fortress (ignition-gazebo 6). -->
<sdf version="1.8">
  <world name="plant_arena">
    <physics type="ode">
      <max_step_size>0.002</max_step_size>
      <real_time_factor>1.0</real_time_factor>
    </physics>
    <plugin filename="ignition-gazebo-physics-system" name="ignition::gazebo::systems::Physics"/>
    <plugin filename="ignition-gazebo-user-commands-system" name="ignition::gazebo::systems::UserCommands"/>
    <plugin filename="ignition-gazebo-scene-broadcaster-system" name="ignition::gazebo::systems::SceneBroadcaster"/>
    <plugin filename="ignition-gazebo-sensors-system" name="ignition::gazebo::systems::Sensors">
      <render_engine>ogre2</render_engine>
    </plugin>
    <plugin filename="ignition-gazebo-imu-system" name="ignition::gazebo::systems::Imu"/>
    <scene>
      <ambient>0.55 0.55 0.55 1</ambient>
      <background>0.75 0.82 0.9 1</background>
      <shadows>true</shadows>
    </scene>
    <light type="directional" name="sun">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.8 0.8 0.8 1</diffuse>
      <specular>0.2 0.2 0.2 1</specular>
      <direction>-0.4 0.2 -0.9</direction>
    </light>
    <model name="ground_plane"><static>true</static><link name="link">
      <collision name="c"><geometry><plane><normal>0 0 1</normal><size>40 40</size></plane></geometry></collision>
      <visual name="v"><geometry><plane><normal>0 0 1</normal><size>40 40</size></plane></geometry>
        <material><ambient>0.62 0.60 0.55 1</ambient><diffuse>0.62 0.60 0.55 1</diffuse></material></visual>
    </link></model>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    gt_rng = np.random.default_rng(args.seed + 1000)

    write_leaf_obj(os.path.join(PKG, "models/plant_assets/meshes/leaf.obj"))
    Vleaf, Fleaf = leaf_mesh()

    xml = [WORLD_HEAD]
    half = ARENA / 2 + WALL_T / 2
    for i, (p, s) in enumerate([((0, half, WALL_H / 2), (ARENA + 2 * WALL_T, WALL_T, WALL_H)),
                                ((0, -half, WALL_H / 2), (ARENA + 2 * WALL_T, WALL_T, WALL_H)),
                                ((half, 0, WALL_H / 2), (WALL_T, ARENA, WALL_H)),
                                ((-half, 0, WALL_H / 2), (WALL_T, ARENA, WALL_H))]):
        xml.append(box_model(f"wall{i}", p, s, (0.86, 0.84, 0.78)))

    meta = {"seed": args.seed, "arena": ARENA, "plants": []}
    gt_pts, gt_lab, gt_pid = [], [], []
    for k, (px, py) in enumerate(PLANT_XY):
        yaw = float(rng.uniform(0, 2 * np.pi))
        plant, top = build_tall_plant(rng)
        pot_body = (cyl_visual("pot", (0, 0, 0), (0, 0, POT_H), POT_R, (0.72, 0.36, 0.20)) +
                    cyl_visual("soil", (0, 0, POT_H - 0.002), (0, 0, POT_H + 0.003), POT_R - 0.006, (0.25, 0.17, 0.10)))
        pot_col = (f'<collision name="c"><pose>0 0 {POT_H / 2:.4f} 0 0 0</pose><geometry><cylinder>'
                   f'<radius>{POT_R}</radius><length>{POT_H}</length></cylinder></geometry></collision>')
        fol = []
        for i, (p0, p1, r) in enumerate(plant["cyl"]):
            fol.append(cyl_visual(f"stem{i}", p0, p1, r, (0.33, 0.45, 0.18)))
        for i, (pos, R, L, W) in enumerate(plant["leaves"]):
            g = rng.uniform(-0.05, 0.05)
            fol.append(leaf_visual(f"leaf{i}", pos, R, L, W, (0.10 + g, 0.48 + g, 0.12 + g)))
        fr = []
        for i, (c, r) in enumerate(plant["fruits"]):
            fr.append(sphere_visual(f"apple{i}", c, r, (0.78 + rng.uniform(-0.08, 0.08), 0.06, 0.05)))
        pose = (px, py, 0.0, 0.0, 0.0, yaw)
        xml.append(model_xml(f"plant{k}_pot", pose, GZ_LABEL["pot"], "".join(pot_body), pot_col))
        xml.append(model_xml(f"plant{k}_foliage", pose, GZ_LABEL["foliage"], "".join(fol)))
        xml.append(model_xml(f"plant{k}_fruit", pose, GZ_LABEL["fruit"], "".join(fr)))

        # ---- ground truth in WORLD frame
        Rw, tw = Rz(yaw), np.array([px, py, 0.0])
        parts = {0: [sample_cylinder((0, 0, 0), (0, 0, POT_H), POT_R, gt_rng, GT_DENSITY),
                     sample_disk(POT_R - 0.006, POT_H + 0.003, gt_rng, GT_DENSITY)],   # soil top
                 1: [], 2: []}
        for p0, p1, r in plant["cyl"]:
            parts[1].append(sample_cylinder(p0, p1, r, gt_rng, GT_DENSITY))
        for pos, R, L, W in plant["leaves"]:
            P = pos + (Vleaf * np.array([L, W, L])) @ R.T
            parts[1].append(sample_triangles(P, Fleaf, gt_rng, GT_DENSITY))
        for c, r in plant["fruits"]:
            parts[2].append(sample_sphere(np.asarray(c), r, gt_rng, GT_DENSITY))
        for lab, lst in parts.items():
            if lst:
                P = np.vstack(lst) @ Rw.T + tw
                gt_pts.append(P.astype(np.float32))
                gt_lab.append(np.full(len(P), lab, np.uint8))
                gt_pid.append(np.full(len(P), k, np.uint8))
        meta["plants"].append({
            "id": k, "xy": [px, py], "yaw": yaw,
            "stem_top_z": float(max(p1[2] for _, p1, _ in plant["cyl"])), "top_z": float(top),
            "fruits": [{"center": (Rw @ np.asarray(c) + tw).tolist(), "radius": float(r)} for c, r in plant["fruits"]],
            "n_leaves": len(plant["leaves"])})
    xml.append("  </world>\n</sdf>\n")

    with open(os.path.join(PKG, "worlds/plant_arena.sdf"), "w") as f:
        f.write("\n".join(xml))
    np.savez_compressed(os.path.join(PKG, "gt/plants_gt.npz"), points=np.vstack(gt_pts),
                        label=np.concatenate(gt_lab), plant=np.concatenate(gt_pid))
    with open(os.path.join(PKG, "gt/plants.json"), "w") as f:
        json.dump(meta, f, indent=1)
    n = sum(len(p) for p in gt_pts)
    print(f"world written: {len(PLANT_XY)} plants, {sum(len(p['fruits']) for p in meta['plants'])} apples, "
          f"{n} ground-truth points")


if __name__ == "__main__":
    main()
