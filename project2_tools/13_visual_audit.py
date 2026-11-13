import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
EVAL = ROOT / "project2_evidence/07_evaluation"
RUNS = EVAL / "runs"

# 1. 分析已经执行的全部 Episode
rows = []

for p in sorted(RUNS.glob("*/episode_summary.json")):
    rows.append(json.loads(p.read_text(encoding="utf-8")))

print("===== EVALUATION AUDIT =====")
print("TOTAL:", len(rows))

success = sum(r.get("program_success") is True for r in rows)
strict = sum(
    r.get("program_success") is True
    and r.get("goal_xy_error_m") is not None
    and r.get("goal_z_error_m") is not None
    and r["goal_xy_error_m"] <= 0.03
    and r["goal_z_error_m"] <= 0.02
    for r in rows
)

print("PROGRAM_SUCCESS:", success)
print("STRICT_3CM_SUCCESS:", strict)
print("EXIT_CODES:", dict(Counter(
    str(r.get("process_exit_code")) for r in rows
)))

for r in rows:
    if r.get("program_success") is True:
        continue

    print("\nFAILED_EPISODE:", r["episode_id"])
    print("STATUS:", r.get("program_status"))

    result_path = ROOT / r.get("result_path", "")
    if result_path.is_file():
        detail = json.loads(
            result_path.read_text(encoding="utf-8")
        )
        error = detail.get("error")
        if error:
            print("ERROR_TAIL:")
            print("\n".join(error.splitlines()[-10:]))

# 2. 审计两种物体的 RGB-D 感知数据
cases = {
    "logistics": (
        ROOT / "project2_evidence/05_logistics/transfer_run03"
    ),
    "retail": (
        ROOT / "project2_evidence/06_retail/pick_run01"
    ),
}

audit = {}

for name, folder in cases.items():
    print("\n===== VISION:", name.upper(), "=====")

    camera_path = folder / "camera.npz"
    report_path = folder / "result.json"
    kp_path = folder / "proposed_keypoints.npy"

    if not camera_path.is_file():
        print("CAMERA_NPZ_MISSING:", camera_path)
        continue

    with np.load(camera_path, allow_pickle=False) as data:
        rgb = np.asarray(data["rgb"])
        points = np.asarray(data["points"])
        seg = np.asarray(data["seg"]).squeeze()

    print("RGB_SHAPE:", rgb.shape)
    print("POINTS_SHAPE:", points.shape)
    print("SEG_SHAPE:", seg.shape)

    if seg.ndim == 3:
        seg = seg[..., 0]

    labels, counts = np.unique(seg, return_counts=True)
    order = np.argsort(counts)[::-1][:12]

    label_counts = [
        (int(labels[i]), int(counts[i]))
        for i in order
    ]

    print("SEG_LABELS_TOP12:", label_counts)

    valid = (
        np.isfinite(points).all(axis=-1)
        & (np.linalg.norm(points, axis=-1) > 0.001)
    )
    print("VALID_3D_PIXELS:", int(valid.sum()))

    result = json.loads(
        report_path.read_text(encoding="utf-8")
    )

    target_center = np.asarray(
        result["box_start"], dtype=float
    )

    summary = {
        "rgb_shape": list(rgb.shape),
        "points_shape": list(points.shape),
        "top_segmentation_labels": label_counts,
        "valid_3d_pixels": int(valid.sum()),
    }

    if kp_path.is_file():
        kp = np.asarray(np.load(kp_path))
        print("DINO_KEYPOINT_COUNT:", len(kp))

        if kp.ndim == 2 and kp.shape[1] == 3:
            distance = np.linalg.norm(
                kp - target_center[None, :],
                axis=1
            )

            print(
                "DINO_TO_TARGET_CENTER_M:",
                np.round(distance, 4).tolist()
            )

            near_count = int((distance <= 0.08).sum())

            print("DINO_WITHIN_8CM:", near_count)

            summary["dino_keypoint_count"] = len(kp)
            summary["dino_target_distances_m"] = (
                distance.tolist()
            )
            summary["dino_near_target_count"] = near_count
    else:
        print("DINO_KEYPOINT_FILE_MISSING")

    audit[name] = summary

output = EVAL / "visual_v2_input_audit.json"
output.write_text(
    json.dumps(audit, ensure_ascii=False, indent=2),
    encoding="utf-8"
)

print("\nVISUAL_AUDIT_SAVED:", output)
print("NO_SIMULATION_RESTARTED")
