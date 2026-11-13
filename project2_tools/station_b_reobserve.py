import json
from pathlib import Path

import numpy as np
from scipy import ndimage
from PIL import Image


def to_np(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def locate_station_b(rgb, points, output_dir):
    """
    Find cyan station marker in a fresh RGB-D frame.
    Returns marker center in world coordinates.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rgb = to_np(rgb)[..., :3]

    if rgb.dtype.kind == "f" and np.nanmax(rgb) <= 1.0:
        rgb = rgb * 255.0

    rgb = np.nan_to_num(rgb).clip(
        0, 255
    ).astype(np.uint8)

    points = to_np(points).astype(float)

    if points.shape != (*rgb.shape[:2], 3):
        raise ValueError("RGB-D shape mismatch")

    r, g, b = [
        rgb[..., i].astype(np.int16)
        for i in range(3)
    ]

    # Cyan marker, clearly separated from brown box
    mask = (
        (g > 110)
        & (b > 110)
        & (g > r + 55)
        & (b > r + 55)
    )

    mask = ndimage.binary_opening(
        mask, iterations=1
    )

    labels, n = ndimage.label(mask)

    regions = []

    for idx in range(1, n + 1):
        ys, xs = np.where(labels == idx)

        if len(xs) < 60:
            continue

        xyz = points[ys, xs]

        valid = (
            np.isfinite(xyz).all(axis=1)
            & (np.linalg.norm(xyz, axis=1) > 0.001)
        )

        xyz = xyz[valid]

        if len(xyz) < 60:
            continue

        # Reject implausibly large regions
        if np.any(
            np.ptp(xyz[:, :2], axis=0) > 0.22
        ):
            continue

        regions.append((
            len(xyz),
            idx,
            np.median(xyz, axis=0)
        ))

    Image.fromarray(rgb).save(
        out / "post_transport_rgb.png"
    )

    if not regions:
        Image.fromarray(
            (mask * 255).astype(np.uint8)
        ).save(out / "station_b_mask.png")

        raise RuntimeError(
            "Station B marker not detected"
        )

    count, idx, estimated = max(
        regions,
        key=lambda row: row[0]
    )

    selected = labels == idx

    Image.fromarray(
        (selected * 255).astype(np.uint8)
    ).save(out / "station_b_mask.png")

    overlay = rgb.copy()

    overlay[selected] = (
        0.35 * overlay[selected]
        + 0.65 * np.array([255, 70, 255])
    ).astype(np.uint8)

    Image.fromarray(overlay).save(
        out / "station_b_overlay.png"
    )

    report = {
        "method": "cyan_marker_rgbd_world_points",
        "estimated_world_xyz": estimated.tolist(),
        "pixels": int(count),
        "uses_marker_true_pose_for_estimation": False
    }

    (out / "station_b_detection.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8"
    )

    return estimated, report
