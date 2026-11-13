import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

os.environ.setdefault(
    "OPENAI_API_KEY", "cached-query-placeholder"
)

from main import Main

ROOT = Path.cwd()
OUT = ROOT / "project2_evidence/12_dual_station/route_preflight"
OUT.mkdir(parents=True, exist_ok=True)

scene_file = ROOT / "configs/og_scene_file_dual_station.json"
assert scene_file.is_file(), "Dual station scene file missing"

print("[1/4] Reload saved dual-station scene", flush=True)

app = Main(str(scene_file), visualize=False)
env = app.env
env.reset()
scene = env.og_env.scene


def to_np(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def get_object(name):
    obj = scene.object_registry("name", name)
    assert obj is not None, f"Object missing after reload: {name}"
    return obj


names = [
    "logistics_station_A",
    "logistics_station_B",
    "logistics_box_1",
    "station_b_cyan_marker"
]

objects = {
    name: get_object(name)
    for name in names
}

print("OBJECTS_RELOADED =", names, flush=True)

print("[2/4] Check station geometry", flush=True)

def aabb(obj):
    low, high = obj.aabb
    return (
        to_np(low).astype(float),
        to_np(high).astype(float)
    )


a_low, a_high = aabb(objects["logistics_station_A"])
b_low, b_high = aabb(objects["logistics_station_B"])

a_center = (a_low + a_high) / 2
b_center = (b_low + b_high) / 2

box_position, _ = objects[
    "logistics_box_1"
].get_position_orientation()

box_position = to_np(box_position).astype(float)

box_xy_error = float(
    np.linalg.norm(
        box_position[:2] - a_center[:2]
    )
)

print("STATION_A_CENTER =", a_center.round(4), flush=True)
print("STATION_B_CENTER =", b_center.round(4), flush=True)
print("BOX_POSITION =", box_position.round(4), flush=True)
print("BOX_TO_A_XY_M =", box_xy_error, flush=True)

assert box_xy_error < 0.04, (
    "Box is not positioned on station A"
)

print("[3/4] Inspect robot-base route clearance", flush=True)

base_position, _ = env.robot.get_position_orientation()
base_position = to_np(base_position).astype(float)

def distance_to_rectangle(point_xy, low, high):
    delta = np.maximum(
        np.maximum(low[:2] - point_xy, point_xy - high[:2]),
        0.0
    )
    return float(np.linalg.norm(delta))


# 仅检查旧方案的世界 X 正方向直线轨迹。
# 这是几何间隙估计，不等于真实碰撞检测。
route = [
    base_position[:2] + np.array([dx, 0.0])
    for dx in np.linspace(0, 0.22, 23)
]

min_a = min(
    distance_to_rectangle(p, a_low, a_high)
    for p in route
)

min_b = min(
    distance_to_rectangle(p, b_low, b_high)
    for p in route
)

print("BASE_INITIAL_XYZ =", base_position.round(4), flush=True)
print("ROUTE_TO_A_EDGE_MIN_M =", round(min_a, 4), flush=True)
print("ROUTE_TO_B_EDGE_MIN_M =", round(min_b, 4), flush=True)

print("[4/4] Save reloaded scene image", flush=True)

obs = env.get_cam_obs()
cam = obs[0]

rgb = to_np(cam["rgb"])[..., :3]

if rgb.dtype.kind == "f" and np.nanmax(rgb) <= 1.0:
    rgb = rgb * 255

rgb = np.nan_to_num(rgb).clip(0, 255).astype(np.uint8)

Image.fromarray(rgb).save(
    OUT / "reloaded_scene.png"
)

report = {
    "reload_verified": True,
    "objects_found": names,
    "station_A_center": a_center.tolist(),
    "station_B_center": b_center.tolist(),
    "box_position": box_position.tolist(),
    "box_to_A_xy_error_m": box_xy_error,
    "robot_base_position": base_position.tolist(),
    "old_route_min_distance_to_A_edge_m": min_a,
    "old_route_min_distance_to_B_edge_m": min_b,
    "real_collision_check_completed": False,
    "transport_executed": False
}

(OUT / "report.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2),
    encoding="utf-8"
)


# ====================================================
# Measure actual Fetch ground-level body geometry
# ====================================================

footprint_links = []

for name, link in env.robot.links.items():

    lname = name.lower()

    if not any(word in lname for word in (
        "base_link", "wheel", "caster"
    )):
        continue

    try:
        low, high = link.aabb
        low = to_np(low).astype(float)
        high = to_np(high).astype(float)

        footprint_links.append({
            "name": name,
            "low": low.tolist(),
            "high": high.tolist()
        })

    except Exception as exc:
        print("LINK_AABB_WARNING:", name, repr(exc))

if not footprint_links:
    raise RuntimeError(
        "Cannot determine actual robot base geometry"
    )

foot_low = np.min(
    [row["low"] for row in footprint_links],
    axis=0
)

foot_high = np.max(
    [row["high"] for row in footprint_links],
    axis=0
)

# Conservative circular envelope around base body.
offset = np.maximum(
    np.abs(foot_low[:2] - base_position[:2]),
    np.abs(foot_high[:2] - base_position[:2])
)

radius = float(np.linalg.norm(offset))

# Include a 3 cm safety margin.
required_clearance = radius + 0.03

# Reconstruct the old forward-motion assumption:
# approximately 5.1 cm before grasp + 22 cm carrying.
candidate_route = [
    base_position[:2] + np.array([delta, 0.0])
    for delta in np.linspace(0.0, 0.271, 55)
]

dist_a = min(
    distance_to_rectangle(p, a_low, a_high)
    for p in candidate_route
)

dist_b = min(
    distance_to_rectangle(p, b_low, b_high)
    for p in candidate_route
)

min_clearance = min(dist_a, dist_b)

safe = bool(
    min_clearance >= required_clearance
)

print("BASE_FOOTPRINT_LINKS =", [
    row["name"] for row in footprint_links
], flush=True)

print(
    "BASE_ENVELOPE_RADIUS_M =",
    round(radius, 4),
    flush=True
)

print(
    "REQUIRED_CLEARANCE_M =",
    round(required_clearance, 4),
    flush=True
)

print(
    "OLD_ROUTE_MIN_CENTER_DISTANCE_M =",
    round(min_clearance, 4),
    flush=True
)

print(
    "OLD_ROUTE_GEOMETRICALLY_SAFE =",
    safe,
    flush=True
)

# Save RGB-D from the actual dual-station scene,
# for later target segmentation and B reobservation.
np.savez_compressed(
    OUT / "dual_station_rgbd.npz",
    rgb=to_np(cam["rgb"]),
    points=to_np(cam["points"]),
    seg=to_np(cam["seg"])
)

report.update({
    "footprint_links": footprint_links,
    "base_envelope_radius_m": radius,
    "required_clearance_m": required_clearance,
    "old_route_min_center_distance_m": min_clearance,
    "old_route_geometrically_safe": safe,
    "route_assumption": "world_positive_x",
    "route_real_collision_tested": False,
    "dual_station_rgbd_saved": True
})

(OUT / "report.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

print("DUAL_STATION_ROUTE_PREFLIGHT_DONE", flush=True)

