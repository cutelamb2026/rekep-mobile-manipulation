import json
from pathlib import Path

import numpy as np
from scipy import ndimage
from PIL import Image, ImageDraw

ROOT = Path.cwd()
OUT = ROOT / "project2_evidence/08_visual_v2"
OUT.mkdir(parents=True, exist_ok=True)

CASES = {
    "logistics": ROOT / "project2_evidence/05_logistics/transfer_run03",
    "retail": ROOT / "project2_evidence/06_retail/pick_run01",
}


def rgb_uint8(arr):
    arr = np.asarray(arr)[..., :3]
    if arr.dtype.kind == "f" and np.nanmax(arr) <= 1.0:
        arr = arr * 255.0
    return np.nan_to_num(arr).clip(0, 255).astype(np.uint8)


for task, folder in CASES.items():

    print("\n==========", task.upper(), "==========")

    dest = OUT / task
    dest.mkdir(parents=True, exist_ok=True)

    with np.load(folder / "camera.npz") as data:
        rgb = rgb_uint8(data["rgb"])
        points = np.asarray(data["points"], dtype=float)
        seg = np.asarray(data["seg"]).squeeze()

    assert seg.shape == rgb.shape[:2]
    assert points.shape == (*seg.shape, 3)

    labels, counts = np.unique(seg, return_counts=True)

    print("ALL_LABELS:")
    for lab, count in sorted(
        zip(labels, counts),
        key=lambda x: x[1]
    ):
        print(" ", int(lab), "pixels =", int(count))

    # 原型启发式：选最少见的分割类别作为物体候选
    object_label = int(labels[np.argmin(counts)])

    binary = seg == object_label

    components, n = ndimage.label(
        binary,
        structure=np.ones((3, 3), dtype=np.uint8)
    )

    candidates = []

    for component_id in range(1, n + 1):

        ys, xs = np.where(components == component_id)

        if len(xs) < 30:
            continue

        xyz = points[ys, xs]

        valid = (
            np.isfinite(xyz).all(axis=1)
            & (np.linalg.norm(xyz, axis=1) > 0.001)
        )

        xyz = xyz[valid]

        if len(xyz) < 30:
            continue

        colors = rgb[ys, xs][valid].astype(float) / 255.0

        # 从可见表面点云估计位置
        estimated_center = np.median(xyz, axis=0)

        # 顶部点云估计抓取候选点
        top_level = np.quantile(xyz[:, 2], 0.85)
        top_points = xyz[xyz[:, 2] >= top_level]
        top_point = np.median(top_points, axis=0)

        # 零售蓝色 SKU 的颜色区分指标
        # 蓝色越明显，该分数通常越大
        blue_score = float(
            np.median(colors[:, 2] - colors[:, 1])
        )

        candidates.append({
            "component_id": component_id,
            "pixels": int(len(xyz)),
            "bbox_xyxy": [
                int(xs.min()),
                int(ys.min()),
                int(xs.max()),
                int(ys.max())
            ],
            "estimated_center": estimated_center.tolist(),
            "top_grasp_candidate": top_point.tolist(),
            "blue_score": blue_score,
        })

    if not candidates:
        print("NO_VALID_OBJECT_COMPONENT")
        continue

    if task == "logistics":
        # 目前只有一个物流目标物体
        selected = max(
            candidates,
            key=lambda c: c["pixels"]
        )
    else:
        # 从蓝色目标商品、红色干扰商品和蓝绿色
        # 暂存区中选取最符合蓝色商品外观的实例
        selected = max(
            candidates,
            key=lambda c: c["blue_score"]
        )

    selected_mask = (
        components == selected["component_id"]
    )

    mask_image = np.zeros_like(rgb)
    mask_image[selected_mask] = (255, 255, 255)

    Image.fromarray(mask_image).save(
        dest / "target_mask.png"
    )

    overlay = Image.fromarray(rgb).convert("RGB")

    tint = rgb.copy()
    tint[selected_mask] = (
        0.4 * tint[selected_mask]
        + 0.6 * np.array([0, 255, 60])
    ).astype(np.uint8)

    overlay = Image.fromarray(tint)

    draw = ImageDraw.Draw(overlay)
    draw.rectangle(
        selected["bbox_xyxy"],
        outline=(255, 255, 0),
        width=3
    )

    overlay.save(dest / "target_overlay.png")

    # 以下真实中心只用于事后误差评估
    result = json.loads(
        (folder / "result.json").read_text(encoding="utf-8")
    )

    ground_truth = np.asarray(
        result["box_start"],
        dtype=float
    )

    estimated = np.asarray(
        selected["estimated_center"],
        dtype=float
    )

    localization_error = float(
        np.linalg.norm(estimated - ground_truth)
    )

    dino_path = folder / "proposed_keypoints.npy"

    if dino_path.exists():
        dino = np.asarray(np.load(dino_path))

        dino_distances = (
            np.linalg.norm(
                dino - estimated[None, :],
                axis=1
            ).tolist()
        )
    else:
        dino_distances = []

    report = {
        "task": task,
        "method": "rgbd_semantic_component_prototype",
        "uses_ground_truth_for_prediction": False,
        "segmentation_label_heuristic": "rarest_class",
        "inferred_object_label": object_label,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "selected": selected,
        "ground_truth_for_evaluation_only":
            ground_truth.tolist(),
        "localization_error_m": localization_error,
        "within_6cm": localization_error <= 0.06,
        "dino_distance_to_visual_estimate_m":
            dino_distances,
        "ready_for_live_execution": False
    }

    (dest / "localization_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )

    print("CANDIDATE_COUNT =", len(candidates))
    print("SELECTED_LABEL =", object_label)
    print(
        "ESTIMATED_CENTER =",
        np.round(estimated, 4)
    )
    print(
        "TOP_GRASP_POINT =",
        np.round(
            selected["top_grasp_candidate"], 4
        )
    )
    print(
        "LOCALIZATION_ERROR_M =",
        round(localization_error, 5)
    )
    print(
        "WITHIN_6CM =",
        localization_error <= 0.06
    )
    print(
        "DINO_DISTANCES_M =",
        np.round(dino_distances, 4)
    )
    print("OVERLAY_SAVED =", dest / "target_overlay.png")

print("\nOFFLINE_VISUAL_LOCALIZATION_DONE")
