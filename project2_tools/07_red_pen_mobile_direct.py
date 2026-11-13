import json
import os
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

os.environ.setdefault(
    "OPENAI_API_KEY",
    "cached-query-placeholder"
)

from main import Main
from project2_tools.mobile_frame_sync import sync_mobile_frames


ROOT = Path.cwd()

OUT = ROOT / (
    "project2_evidence/04_red_pen_mobile/run01"
)

PROGRAM = OUT / "red_pen_program"
PROGRAM.mkdir(parents=True, exist_ok=True)

STATUS = OUT / "status.json"


def to_np(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def record(**updates):
    if STATUS.exists():
        data = json.loads(STATUS.read_text())
    else:
        data = {}

    data.update(updates)

    STATUS.write_text(
        json.dumps(data, indent=2, ensure_ascii=False)
    )


def write_constraint(filename, content):
    compile(content, filename, "exec")

    (PROGRAM / filename).write_text(
        content,
        encoding="utf-8"
    )


try:
    record(status="started")

    print("[1/6] 初始化仿真", flush=True)

    main = Main(
        "./configs/og_scene_file_red_pen.json",
        visualize=False
    )

    env = main.env
    env.reset()

    print("[2/6] Fetch 底盘移动", flush=True)

    before = to_np(
        env.robot.get_position_orientation()[0]
    ).astype(float)

    action = np.zeros(env.robot.action_dim)
    action[0] = 0.10

    for _ in range(10):
        env.og_env.step(action)

    after = to_np(
        env.robot.get_position_orientation()[0]
    ).astype(float)

    motion = float(
        np.linalg.norm(after - before)
    )

    print(
        "BASE_MOVEMENT_M =",
        motion,
        flush=True
    )

    record(
        status="base_moved",
        base_movement_m=motion
    )

    if motion < 0.005:
        raise RuntimeError(
            "底盘移动不足 5mm，停止后续规划"
        )

    print(
        "[3/6] 同步 IK 坐标变换",
        flush=True
    )

    sync = sync_mobile_frames(main)

    print(
        "FRAME_SYNC =",
        sync["status"],
        flush=True
    )

    record(
        status="frame_synchronized",
        frame_sync=sync["status"]
    )

    print(
        "[4/6] 实时 RGB-D 与关键点",
        flush=True
    )

    obs = env.get_cam_obs()
    cam_id = main.config["vlm_camera"]

    if cam_id in obs:
        cam = obs[cam_id]
    else:
        cam = obs[str(cam_id)]

    # 使用原始 Tensor 输入，适配当前 ReKep
    keypoints, overlay = (
        main.keypoint_proposer.get_keypoints(
            cam["rgb"],
            cam["points"],
            cam["seg"]
        )
    )

    keypoints = to_np(keypoints).astype(float)

    if (
        keypoints.ndim != 2
        or keypoints.shape[1] != 3
        or not np.isfinite(keypoints).all()
    ):
        raise ValueError("3D 关键点无效")

    np.save(
        OUT / "live_keypoints.npy",
        keypoints
    )

    img = to_np(overlay)[..., :3]

    if (
        img.dtype.kind == "f"
        and np.nanmax(img) <= 1.0
    ):
        img = img * 255.0

    Image.fromarray(
        np.nan_to_num(img)
        .clip(0, 255)
        .astype("uint8")
    ).save(OUT / "live_keypoints.png")

    print(
        "KEYPOINT_COUNT =",
        len(keypoints),
        flush=True
    )

    print(
        "[5/6] 验证红笔和笔筒归属",
        flush=True
    )

    env.register_keypoints(keypoints)

    owners = []

    for i in range(len(keypoints)):
        obj = env.get_object_by_keypoint(i)

        if isinstance(obj, str):
            name = obj
        else:
            name = getattr(obj, "name", str(obj))

        owners.append(str(name))

        print(
            f"KP{i} -> {name}",
            flush=True
        )

    # 只操作红笔 pen_2
    red = [
        i for i, name in enumerate(owners)
        if name == "pen_2"
    ]

    # 目标是黑色笔筒
    holder = [
        i for i, name in enumerate(owners)
        if name == "pencil_holder_1"
    ]

    record(
        status="owners_checked",
        owners=owners,
        red_indices=red,
        holder_indices=holder
    )

    if len(red) < 2 or len(holder) < 2:
        raise RuntimeError(
            "红笔或笔筒候选点不足，停止规划"
        )

    # 确定红笔轴线
    red_xyz = keypoints[red]

    pair_distance = np.linalg.norm(
        red_xyz[:, None] - red_xyz[None, :],
        axis=2
    )

    a, b = np.unravel_index(
        np.argmax(pair_distance),
        pair_distance.shape
    )

    axis_a = red[int(a)]
    axis_b = red[int(b)]

    # 选择靠近红笔候选点中心的抓取点
    center = red_xyz.mean(axis=0)

    grasp = red[
        int(
            np.argmin(
                np.linalg.norm(
                    red_xyz - center,
                    axis=1
                )
            )
        )
    ]

    # 计算笔筒上方悬停目标
    # 尚未验证其为精确开口中心
    holder_xyz = keypoints[holder]

    holder_xy = holder_xyz[:, :2].mean(
        axis=0
    )

    hover = [
        float(holder_xy[0]),
        float(holder_xy[1]),
        float(
            holder_xyz[:, 2].max() + 0.15
        )
    ]

    # 生成三阶段任务元数据
    meta = {
        "init_keypoint_positions":
            keypoints.tolist(),
        "num_keypoints":
            len(keypoints),
        "num_stages":
            3,
        "grasp_keypoints":
            [grasp, -1, -1],
        "release_keypoints":
            [-1, -1, grasp]
    }

    (PROGRAM / "metadata.json").write_text(
        json.dumps(meta, indent=2)
    )

    # Stage 1：对齐并抓取红笔
    write_constraint(
        "stage1_subgoal_constraints.txt",
        f'''def stage1_subgoal_constraint1(end_effector, keypoints):
    return np.linalg.norm(end_effector - keypoints[{grasp}])
'''
    )

    # Stage 2：保持抓取
    write_constraint(
        "stage2_path_constraints.txt",
        f'''def stage2_path_constraint1(end_effector, keypoints):
    return get_grasping_cost_by_keypoint_idx({grasp})
'''
    )

    # Stage 2：红笔姿态调整
    write_constraint(
        "stage2_subgoal_constraints.txt",
        f'''def stage2_subgoal_constraint1(end_effector, keypoints):
    axis = keypoints[{axis_b}] - keypoints[{axis_a}]
    return np.linalg.norm(np.cross(axis, np.array([0., 0., 1.])))
'''
    )

    # Stage 3：转运期间保持抓取
    write_constraint(
        "stage3_path_constraints.txt",
        f'''def stage3_path_constraint1(end_effector, keypoints):
    return get_grasping_cost_by_keypoint_idx({grasp})
'''
    )

    # Stage 3：将红笔移到笔筒上方
    write_constraint(
        "stage3_subgoal_constraints.txt",
        f'''def stage3_subgoal_constraint1(end_effector, keypoints):
    hover = np.array({hover!r})
    return np.linalg.norm(keypoints[{grasp}] - hover)

def stage3_subgoal_constraint2(end_effector, keypoints):
    axis = keypoints[{axis_b}] - keypoints[{axis_a}]
    return np.linalg.norm(np.cross(axis, np.array([0., 0., 1.])))
'''
    )

    record(
        status="program_ready",
        grasp_index=grasp,
        axis_indices=[axis_a, axis_b],
        holder_hover=hover,
        opening_verified=False
    )

    print(
        "PROGRAM_READY =",
        PROGRAM,
        flush=True
    )

    print(
        "GRASP_IDX =",
        grasp,
        "HOVER =",
        hover,
        flush=True
    )

    print(
        "[6/6] ReKep 规划和机械臂执行",
        flush=True
    )

    record(status="executing")

    # 不调用 perform_task，避免再次 reset 底盘
    main._execute(
        str(PROGRAM),
        disturbance_seq=None
    )

    record(
        status="execution_returned_success_unverified"
    )

    print(
        "EXECUTION_RETURNED",
        flush=True
    )

except Exception:
    record(
        status="failed",
        error=traceback.format_exc()
    )

    traceback.print_exc()
    raise
