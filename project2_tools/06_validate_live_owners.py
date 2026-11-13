import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.optimize import linear_sum_assignment

os.environ.setdefault(
    "OPENAI_API_KEY", "cached-query-placeholder"
)

from main import Main

root = Path.cwd()
out = root / "project2_evidence/03_task_modeling/live_owner_run01"
out.mkdir(parents=True, exist_ok=True)

# 初始化现有仿真，不修改原 main.py
main = Main(
    "./configs/og_scene_file_red_pen.json",
    visualize=False
)
env = main.env
env.reset()

# 底盘移动 10 步
start_pos, _ = env.robot.get_position_orientation()

action = np.zeros(env.robot.action_dim, dtype=float)
action[0] = 0.10

for _ in range(10):
    env.og_env.step(action)

end_pos, _ = env.robot.get_position_orientation()

# 移动后重新感知并生成关键点
camera = env.get_cam_obs()[main.config["vlm_camera"]]

keypoints, overlay = main.keypoint_proposer.get_keypoints(
    camera["rgb"],
    camera["points"],
    camera["seg"]
)

keypoints = np.asarray(keypoints, dtype=float)
assert keypoints.ndim == 2 and keypoints.shape[1] == 3
assert np.isfinite(keypoints).all()

image = (
    overlay.detach().cpu().numpy()
    if hasattr(overlay, "detach")
    else np.asarray(overlay)
)

Image.fromarray(
    np.clip(image[:, :, :3], 0, 255).astype(np.uint8)
).save(out / "live_keypoints.png")

# 注册关键点，检查实际关联的物体
env.register_keypoints(keypoints)
registered = np.asarray(
    env.get_keypoint_positions(), dtype=float
)

owners = []

print("===== KEYPOINT OWNERS =====", flush=True)

for i, (p, q) in enumerate(zip(keypoints, registered)):
    owner = env.get_object_by_keypoint(i)
    owners.append(str(owner.name))

    print(
        f"KP{i}: {np.round(p, 4)} "
        f"owner={owner.name} "
        f"snap_m={np.linalg.norm(q-p):.4f}",
        flush=True
    )

report = {
    "scene": "./configs/og_scene_file_red_pen.json",
    "base_motion_m": float(
        np.linalg.norm(
            np.asarray(end_pos) - np.asarray(start_pos)
        )
    ),
    "live_keypoints": keypoints.tolist(),
    "registered_keypoints": registered.tolist(),
    "object_owners": owners,
    "snap_distance_m": np.linalg.norm(
        registered - keypoints, axis=1
    ).tolist(),
    "offline_mapping": {}
}

# 与上一轮保存的 9 个关键点建立候选对应关系
ref_file = (
    root
    / "project2_evidence/02_mobile/rgbd_run01/"
      "forward_keypoints.npy"
)

if ref_file.exists():
    ref = np.load(ref_file, allow_pickle=False)

    distances = np.linalg.norm(
        ref[:, None, :] - keypoints[None, :, :],
        axis=2
    )

    rows, cols = linear_sum_assignment(distances)
    mapping = {
        int(i): int(j)
        for i, j in zip(rows, cols)
    }

    report["offline_mapping"] = {
        str(i): {
            "live_index": mapping[i],
            "distance_m": float(distances[i, mapping[i]])
        }
        for i in mapping
    }

    print("===== CANDIDATE MAPPING =====", flush=True)

    for i in [1, 3, 5, 6, 7, 8]:
        if i in mapping:
            j = mapping[i]
            print(
                f"OLD_KP{i} -> LIVE_KP{j}, "
                f"d={distances[i,j]:.4f}m, "
                f"owner={owners[j]}",
                flush=True
            )

path = out / "live_owner_report.json"

path.write_text(
    json.dumps(report, indent=2, ensure_ascii=False),
    encoding="utf-8"
)

print("REPORT_SAVED:", path, flush=True)
