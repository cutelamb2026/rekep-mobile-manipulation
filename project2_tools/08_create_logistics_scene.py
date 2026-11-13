import json
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from utils import get_config
from environment import ReKepOGEnv
from omnigibson.objects import PrimitiveObject

ROOT = Path.home() / "Rekep"
OUT = ROOT / "project2_evidence/05_logistics/scene_run01"
OUT.mkdir(parents=True, exist_ok=True)


def to_np(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def rgb8(x):
    x = to_np(x)[..., :3]
    if np.issubdtype(x.dtype, np.floating):
        finite = x[np.isfinite(x)]
        if finite.size and finite.max() <= 1.0:
            x = x * 255.0
    return np.nan_to_num(x).clip(0, 255).astype(np.uint8)


def save_report(report):
    (OUT / "scene_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )


report = {
    "task": "logistics_small_box_transfer",
    "status": "started",
    "scene_type": "primitive_box_prototype",
    "grasp_verified": False,
    "placement_verified": False
}

try:
    config = get_config(
        config_path=str(ROOT / "configs/config.yaml")
    )

    print("[1/5] 加载原有仿真场景", flush=True)

    env = ReKepOGEnv(
        config["env"],
        str(ROOT / "configs/og_scene_file_red_pen.json"),
        verbose=False
    )
    env.reset()

    scene = env.og_env.scene

    def find(name):
        obj = scene.object_registry("name", name)
        if obj is None:
            raise RuntimeError(f"Object not found: {name}")
        return obj

    # 保留原场景桌面和 Fetch
    pen = find("pen_2")
    holder = find("pencil_holder_1")
    find("table_1")

    pen_pos = to_np(
        pen.get_position_orientation()[0]
    ).astype(float)

    holder_pos = to_np(
        holder.get_position_orientation()[0]
    ).astype(float)

    print("PEN_SOURCE_POS =", pen_pos, flush=True)
    print("HOLDER_GOAL_POS =", holder_pos, flush=True)

    print("[2/5] 移除笔和笔筒", flush=True)

    for name in ("pen_1", "pen_2", "pencil_holder_1"):
        obj = find(name)
        scene.remove_object(obj)
        print("REMOVED =", name, flush=True)

    print("[3/5] 创建小型物料箱", flush=True)

    # 6.5 cm 立方体，供 Fetch 抓取实验使用
    # 这是工业物料箱的缩比仿真原型
    box = PrimitiveObject(
        name="logistics_box_1",
        primitive_type="Cube",
        category="object",
        size=0.08,
        rgba=(0.72, 0.46, 0.22, 1.0),
        fixed_base=False,
        visual_only=False
    )

    scene.add_object(box)

    # 在原红笔区域附近放置箱体
    start_xyz = pen_pos.copy()
    start_xyz[2] += 0.075

    box.set_position_orientation(
        position=torch.as_tensor(
            start_xyz, dtype=torch.float32
        ),
        orientation=torch.tensor(
            [0.0, 0.0, 0.0, 1.0],
            dtype=torch.float32
        )
    )

    action = np.zeros(
        env.robot.action_dim, dtype=float
    )

    for _ in range(15):
        env.og_env.step(action)

    box_pos = to_np(
        box.get_position_orientation()[0]
    ).astype(float)

    print("BOX_POSITION =", box_pos, flush=True)

    # B 区域先使用原笔筒附近的桌面位置
    goal_xyz = [
        float(holder_pos[0]),
        float(holder_pos[1]),
        float(box_pos[2])
    ]

    print("GOAL_POSITION =", goal_xyz, flush=True)

    print("[4/5] 获取场景图像", flush=True)

    obs = env.get_cam_obs()
    cam_id = config["main"]["vlm_camera"]

    cam = (
        obs[cam_id]
        if cam_id in obs
        else obs[str(cam_id)]
    )

    rgb = rgb8(cam["rgb"])

    Image.fromarray(rgb).save(
        OUT / "logistics_scene_rgb.png"
    )

    np.savez_compressed(
        OUT / "logistics_camera.npz",
        rgb=rgb,
        points=to_np(cam["points"]),
        seg=to_np(cam["seg"])
    )

    registered = find("logistics_box_1")

    report.update({
        "status": "scene_created",
        "box_name": registered.name,
        "box_category": "object",
        "box_size_m": 0.08,
        "box_position": box_pos.tolist(),
        "goal_position": goal_xyz,
        "camera_rgb_shape": list(rgb.shape),
        "scene_verified": True
    })

    save_report(report)

    print("[5/5] 场景创建完成", flush=True)
    print("SCENE_CREATED", flush=True)
    print("REPORT_SAVED =", OUT / "scene_report.json", flush=True)

except Exception:
    report["status"] = "failed"
    report["error"] = traceback.format_exc()
    save_report(report)
    traceback.print_exc()
    raise
