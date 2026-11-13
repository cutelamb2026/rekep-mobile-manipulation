import json
from pathlib import Path

import numpy as np
from utils import get_config
from environment import ReKepOGEnv

root = Path.cwd()

source = root / (
    "project2_evidence/02_mobile/rgbd_run01/"
    "forward_keypoints.npy"
)

summary_file = root / (
    "project2_evidence/02_mobile/rgbd_run01/"
    "summary.json"
)

out = root / "project2_evidence/03_task_modeling"
out.mkdir(parents=True, exist_ok=True)

keypoints = np.load(source, allow_pickle=False)
summary = json.loads(summary_file.read_text(encoding="utf-8"))

assert keypoints.shape == (9, 3)
assert np.isfinite(keypoints).all()

def to_numpy(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x, dtype=float)

config = get_config(
    config_path=str(root / "configs/config.yaml")
)

scene = root / "configs/og_scene_file_red_pen.json"

print("[1/4] Loading simulation", flush=True)

env = ReKepOGEnv(
    config["env"],
    str(scene),
    verbose=False
)
env.reset()

start, _ = env.robot.get_position_orientation()
start = to_numpy(start)

print("[2/4] Moving base", flush=True)

action = np.zeros(env.robot.action_dim)
action[0] = 0.10

for _ in range(10):
    env.og_env.step(action)

moved, _ = env.robot.get_position_orientation()
moved = to_numpy(moved)

expected = np.asarray(
    summary["snapshots"]["forward"]["base_position"],
    dtype=float
)

pose_error = float(np.linalg.norm(moved - expected))

print("BASE_POSITION:", np.round(moved, 5), flush=True)
print("BASE_REPLAY_ERROR_M:", pose_error, flush=True)

report = {
    "scene": str(scene),
    "base_position": moved.tolist(),
    "base_replay_error_m": pose_error,
    "keypoints": keypoints.tolist(),
    "owners": []
}

if pose_error > 0.03:
    report["status"] = "base_pose_mismatch"
    print(
        "STOP: replay pose differs by over 3 cm",
        flush=True
    )
else:
    print("[3/4] Registering keypoints", flush=True)

    env.register_keypoints(keypoints)

    print("===== KEYPOINT OWNERS =====", flush=True)

    for i in range(len(keypoints)):
        obj = env.get_object_by_keypoint(i)

        if obj is None:
            name = "NONE"
        elif isinstance(obj, str):
            name = obj
        else:
            name = str(getattr(obj, "name", obj))

        report["owners"].append({
            "index": i,
            "owner": name,
            "position": keypoints[i].tolist()
        })

        print(
            f"KP{i}: owner={name}, "
            f"position={np.round(keypoints[i], 4)}",
            flush=True
        )

    report["status"] = "registered"

report_path = out / "red_pen_keypoint_owners.json"

report_path.write_text(
    json.dumps(report, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

print("[4/4] REPORT_SAVED:", report_path, flush=True)
