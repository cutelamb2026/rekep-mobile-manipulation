import json
from pathlib import Path

import numpy as np
from scipy import ndimage
from PIL import Image, ImageDraw


def to_numpy(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def localize_from_camera(
    rgb, points, seg, task, half_height, output_dir
):
    """Estimate world-frame target position from RGB-D data."""

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rgb = to_numpy(rgb)[..., :3]
    if rgb.dtype.kind == "f" and np.nanmax(rgb) <= 1.0:
        rgb = rgb * 255
    rgb = np.nan_to_num(rgb).clip(0, 255).astype(np.uint8)

    points = to_numpy(points).astype(np.float64)
    seg = np.squeeze(to_numpy(seg))

    if seg.ndim == 3:
        seg = seg[..., 0]

    if (
        rgb.shape[:2] != seg.shape
        or points.shape != (*seg.shape, 3)
    ):
        raise ValueError("RGB-D / segmentation shape mismatch")

    # 沿用已经通过离线验证的稀有分割类别策略
    labels, counts = np.unique(seg, return_counts=True)

    valid_labels = [
        (int(label), int(count))
        for label, count in zip(labels, counts)
        if 100 <= count <= seg.size * 0.15
    ]

    if not valid_labels:
        raise RuntimeError("No plausible target label")

    target_label, _ = min(
        valid_labels,
        key=lambda x: x[1]
    )

    components, n = ndimage.label(
        seg == target_label,
        structure=np.ones((3, 3), dtype=np.uint8)
    )

    candidates = []

    for component_id in range(1, n + 1):
        ys, xs = np.where(components == component_id)

        if len(xs) < 60:
            continue

        xyz = points[ys, xs]

        valid = (
            np.isfinite(xyz).all(axis=1)
            & (np.linalg.norm(xyz, axis=1) > 0.001)
        )

        xyz = xyz[valid]
        xs = xs[valid]
        ys = ys[valid]

        if len(xyz) < 60:
            continue

        if np.any(np.ptp(xyz[:, :2], axis=0) > 0.18):
            continue

        colors = rgb[ys, xs].astype(float) / 255.0

        # 相机点云估计物体顶面中心
        top = [
            float(np.median(xyz[:, 0])),
            float(np.median(xyz[:, 1])),
            float(np.quantile(xyz[:, 2], 0.90)),
        ]

        candidates.append({
            "component_id": int(component_id),
            "pixels": int(len(xyz)),
            "top_xyz": top,
            "blue_score": float(
                np.median(colors[:, 2] - colors[:, 1])
            ),
            "bbox": [
                int(xs.min()),
                int(ys.min()),
                int(xs.max()),
                int(ys.max()),
            ]
        })

    if not candidates:
        raise RuntimeError("No valid object component")

    if task == "retail":
        selected = max(
            candidates,
            key=lambda c: c["blue_score"]
        )

        if selected["blue_score"] < 0.08:
            raise RuntimeError(
                "Blue target not reliably identified"
            )

    elif task == "logistics":
        selected = max(
            candidates,
            key=lambda c: c["pixels"]
        )

    else:
        raise ValueError("Unknown task: " + task)

    top = np.array(selected["top_xyz"], dtype=float)

    # 利用任务中已知的物体高度恢复中心
    # 不读取仿真真实物体位姿
    center = top.copy()
    center[2] -= float(half_height)

    selected_mask = (
        components == selected["component_id"]
    )

    mask_img = (
        selected_mask.astype(np.uint8) * 255
    )

    Image.fromarray(mask_img).save(
        out / "vision_mask.png"
    )

    overlay = rgb.copy()

    overlay[selected_mask] = (
        0.45 * overlay[selected_mask]
        + 0.55 * np.array([0, 255, 70])
    ).astype(np.uint8)

    img = Image.fromarray(overlay)

    ImageDraw.Draw(img).rectangle(
        selected["bbox"],
        outline=(255, 255, 0),
        width=3
    )

    img.save(out / "vision_overlay.png")

    report = {
        "task": task,
        "method": "rgbd_segmented_surface",
        "uses_ground_truth_pose_for_prediction": False,
        "selected_label": target_label,
        "candidate_count": len(candidates),
        "selected": selected,
        "visual_top_xyz": top.tolist(),
        "visual_center_xyz": center.tolist(),
        "known_half_height_m": float(half_height),
        "simulator_mesh_registration_still_required": True
    }

    (out / "vision_detection.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    return top, center, report
