import json
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from main import Main
from omnigibson.objects import PrimitiveObject
from project2_tools.station_b_reobserve import locate_station_b


ROOT = Path.cwd()
OUT = ROOT / "project2_evidence/12_dual_station"
SCENE_FILE = ROOT / "configs/og_scene_file_dual_station.json"
REPORT = OUT / "scene_report.json"

OUT.mkdir(parents=True, exist_ok=True)

# 沿用已验证的物料起点与目标区域
A = np.array([-0.139824, -0.150199])
B = np.array([-0.299981,  0.150022])

# 两座独立实体平台
SIDE = 0.22
TOP_Z = 0.686
HEIGHT = 0.686

BOX_SIZE = 0.065


def arr(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def save_report(**kwargs):
    if REPORT.exists():
        report = json.loads(
            REPORT.read_text(encoding="utf-8")
        )
    else:
        report = {}

    report.update(kwargs)

    REPORT.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )


def set_xyz(obj, xyz):
    obj.set_position_orientation(
        position=torch.tensor(xyz, dtype=torch.float32),
        orientation=torch.tensor(
            [0.0, 0.0, 0.0, 1.0],
            dtype=torch.float32
        )
    )


def get_aabb(obj):
    low, high = obj.aabb
    return arr(low).astype(float), arr(high).astype(float)


def rgb_image(rgb):
    x = arr(rgb)[..., :3]

    if x.dtype.kind == "f" and np.nanmax(x) <= 1.0:
        x = x * 255.0

    return np.nan_to_num(x).clip(0, 255).astype(np.uint8)


try:
    save_report(
        status="building",
        task="industrial_logistics_dual_station",
        station_type="two_physical_supports",
        task_executed=False
    )

    print("[1/5] 加载原始仿真场景", flush=True)

    app = Main(
        "./configs/og_scene_file_red_pen.json",
        visualize=False
    )

    env = app.env
    env.reset()
    scene = env.og_env.scene

    # 移除原桌面和演示物体
    # 不修改原场景 JSON
    for name in (
        "pen_1",
        "pen_2",
        "pencil_holder_1",
        "table_1"
    ):
        obj = scene.object_registry("name", name)

        if obj is None:
            raise RuntimeError(
                "Missing original object: " + name
            )

        scene.remove_object(obj)

    print("[2/5] 创建两个独立物理工位", flush=True)

    def make_station(name, xy, color):
        station = PrimitiveObject(
            name=name,
            primitive_type="Cube",
            category="object",
            size=1.0,
            scale=[SIDE, SIDE, HEIGHT],
            rgba=color,
            fixed_base=True,
            visual_only=False
        )

        scene.add_object(station)

        set_xyz(
            station,
            [xy[0], xy[1], TOP_Z - HEIGHT / 2]
        )

        return station

    station_a = make_station(
        "logistics_station_A",
        A,
        (0.48, 0.52, 0.57, 1.0)
    )

    station_b = make_station(
        "logistics_station_B",
        B,
        (0.30, 0.49, 0.68, 1.0)
    )

    # 创建具有真实碰撞的物流物料箱
    box = PrimitiveObject(
        name="logistics_box_1",
        primitive_type="Cube",
        category="object",
        size=BOX_SIZE,
        rgba=(0.72, 0.46, 0.22, 1.0),
        fixed_base=False,
        visual_only=False
    )

    scene.add_object(box)

    set_xyz(
        box,
        [
            A[0],
            A[1],
            TOP_Z + BOX_SIZE / 2 + 0.012
        ]
    )

    # B 工位的视觉定位标记
    # 标记无碰撞，实际承重由 station_B 提供
    marker = PrimitiveObject(
        name="station_b_cyan_marker",
        primitive_type="Cylinder",
        category="object",
        radius=0.065,
        height=0.004,
        rgba=(0.02, 0.94, 0.86, 1.0),
        fixed_base=True,
        visual_only=True
    )

    scene.add_object(marker)

    set_xyz(
        marker,
        [B[0], B[1], TOP_Z + 0.002]
    )

    print("[3/5] 验证实体工位几何与物料稳定性",
          flush=True)

    action = np.zeros(env.robot.action_dim)

    for _ in range(40):
        env.og_env.step(action)

    a_low, a_high = get_aabb(station_a)
    b_low, b_high = get_aabb(station_b)

    box_xyz, _ = box.get_position_orientation()
    box_xyz = arr(box_xyz).astype(float)

    # 两平台的 XY 投影不能重叠
    xy_disjoint = bool(
        np.any(
            a_high[:2] < b_low[:2] - 0.005
        )
        or np.any(
            b_high[:2] < a_low[:2] - 0.005
        )
    )

    expected_box_z = TOP_Z + BOX_SIZE / 2

    box_height_error = abs(
        box_xyz[2] - expected_box_z
    )

    box_xy_error = float(
        np.linalg.norm(box_xyz[:2] - A)
    )

    box_stable = bool(
        box_height_error < 0.020
        and box_xy_error < 0.030
    )

    print(
        "STATIONS_DISJOINT =", xy_disjoint,
        flush=True
    )
    print(
        "BOX_STABLE =", box_stable,
        flush=True
    )

    print("[4/5] 检测 B 工位视觉标记", flush=True)

    obs = env.get_cam_obs()

    # 保存两个固定相机的场景图
    for cam_id in (0, 1):
        cam = obs[cam_id]

        Image.fromarray(
            rgb_image(cam["rgb"])
        ).save(OUT / f"dual_station_cam{cam_id}.png")

    b_detected = False
    b_error = None
    b_visual_xyz = None
    b_error_message = None

    try:
        cam = obs[0]

        detected, detection = locate_station_b(
            cam["rgb"],
            cam["points"],
            OUT
        )

        b_visual_xyz = arr(detected).tolist()

        b_error = float(
            np.linalg.norm(detected[:2] - B)
        )

        b_detected = bool(b_error <= 0.03)

    except Exception as e:
        b_error_message = repr(e)

    print(
        "B_VISUAL_DETECTED =", b_detected,
        flush=True
    )
    print(
        "B_VISUAL_XY_ERROR_M =", b_error,
        flush=True
    )

    # 在场景验证通过后，保存可复用场景
    scene_ok = bool(
        xy_disjoint
        and box_stable
        and b_detected
    )

    save_report(
        status=(
            "scene_verified"
            if scene_ok
            else "scene_needs_adjustment"
        ),
        station_A_center=A.tolist(),
        station_B_center=B.tolist(),
        station_A_aabb=[
            a_low.tolist(), a_high.tolist()
        ],
        station_B_aabb=[
            b_low.tolist(), b_high.tolist()
        ],
        station_center_distance_m=float(
            np.linalg.norm(A - B)
        ),
        stations_physically_separate=xy_disjoint,
        station_supports_collidable=True,
        box_xyz=box_xyz.tolist(),
        box_stable=box_stable,
        box_height_error_m=float(box_height_error),
        box_xy_error_m=box_xy_error,
        station_B_visual_detected=b_detected,
        station_B_visual_xyz=b_visual_xyz,
        station_B_visual_error_m=b_error,
        station_B_visual_error_message=b_error_message,
        scene_verified=scene_ok,
        task_executed=False
    )

    if not scene_ok:
        print(
            "SCENE_VALIDATION_FAILED: results preserved",
            flush=True
        )
    else:
        print("[5/5] 保存独立双工位场景", flush=True)

        if SCENE_FILE.exists():
            raise RuntimeError(
                "Refusing to overwrite existing scene: "
                + str(SCENE_FILE)
            )

        import omnigibson as og
        og.sim.save([str(SCENE_FILE)])
        assert SCENE_FILE.is_file(), 'Simulator did not create scene JSON'

        save_report(
            saved_scene_file=str(SCENE_FILE)
        )

        print(
            "DUAL_STATION_SCENE_SAVED =",
            SCENE_FILE,
            flush=True
        )

    print("DUAL_STATION_BUILD_FINISHED", flush=True)

except Exception:
    save_report(
        status="failed",
        error=traceback.format_exc()
    )
    traceback.print_exc()
    raise
