import json
from pathlib import Path

import numpy as np
from scipy import ndimage
from PIL import Image


def as_np(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def localize_from_camera(
    rgb, points, seg, task, half_height, output_dir
):
    if task != "logistics":
        raise ValueError("This module is for logistics only")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rgb = as_np(rgb)[..., :3]
    if rgb.dtype.kind == "f" and np.nanmax(rgb) <= 1:
        rgb = rgb * 255
    rgb = np.nan_to_num(rgb).clip(0, 255).astype(np.uint8)

    points = as_np(points).astype(float)

    if points.shape != (*rgb.shape[:2], 3):
        raise ValueError("RGB-D dimensions mismatch")

    r = rgb[..., 0].astype(float)
    g = rgb[..., 1].astype(float)
    b = rgb[..., 2].astype(float)

    # Brown logistics material against gray/blue stations
    mask = (
        (r > 60)
        & (g > 35)
        & (r > 1.12 * g)
        & (g > 1.12 * b)
        & (r > 1.45 * b)
    )

    mask = ndimage.binary_opening(mask)

    labels, count = ndimage.label(mask)
    candidates = []

    for idx in range(1, count + 1):
        ys, xs = np.where(labels == idx)

        if len(xs) < 40:
            continue

        xyz = points[ys, xs]

        valid = (
            np.isfinite(xyz).all(axis=1)
            & (xyz[:, 2] > 0.68)
            & (xyz[:, 2] < 0.90)
        )

        xyz = xyz[valid]

        if len(xyz) < 40:
            continue

        if np.any(np.ptp(xyz[:, :2], axis=0) > 0.15):
            continue

        top = np.array([
            np.median(xyz[:, 0]),
            np.median(xyz[:, 1]),
            np.quantile(xyz[:, 2], 0.90)
        ])

        candidates.append((len(xyz), idx, top))

    if not candidates:
        raise RuntimeError(
            "Brown logistics box not found in RGB-D"
        )

    pixels, idx, top = max(candidates)

    center = top.copy()
    center[2] -= half_height

    selected = labels == idx

    Image.fromarray(
        (selected * 255).astype(np.uint8)
    ).save(out / "vision_box_mask.png")

    overlay = rgb.copy()
    overlay[selected] = (
        overlay[selected].astype(float) * 0.4
        + np.array([0, 255, 0]) * 0.6
    ).astype(np.uint8)

    Image.fromarray(overlay).save(
        out / "vision_box_overlay.png"
    )

    report = {
        "method": "brown_rgbd_component",
        "uses_sim_true_pose": False,
        "pixels": pixels,
        "top": top.tolist(),
        "center": center.tolist()
    }

    (out / "vision_box_detection.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )

    return top, center, report
