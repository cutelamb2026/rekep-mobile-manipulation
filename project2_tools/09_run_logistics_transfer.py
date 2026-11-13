import json
import os
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image

os.environ.setdefault(
    "OPENAI_API_KEY",
    "cached-query-placeholder"
)

from main import Main
from environment import T
from omnigibson.objects import PrimitiveObject
from project2_tools.mobile_frame_sync import sync_mobile_frames


ROOT = Path.cwd()
OUT = ROOT / "project2_evidence/05_logistics/transfer_run03"
PROGRAM = OUT / "program"

PROGRAM.mkdir(parents=True, exist_ok=True)

REPORT = OUT / "result.json"


def as_np(x):
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def image_u8(x):
    a = as_np(x)[..., :3]

    if (
        a.dtype.kind == "f"
        and np.isfinite(a).any()
        and np.nanmax(a) <= 1
    ):
        a = a * 255

    return np.nan_to_num(a).clip(0, 255).astype(np.uint8)


def pose(obj):
    p, q = obj.get_position_orientation()
    return (
        as_np(p).astype(float),
        as_np(q).astype(float)
    )


def save(**fields):
    if REPORT.exists():
        result = json.loads(
            REPORT.read_text(encoding="utf-8")
        )
    else:
        result = {}

    result.update(fields)

    REPORT.write_text(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )


def write(name, code):
    compile(code, name, "exec")

    (PROGRAM / name).write_text(
        code,
        encoding="utf-8"
    )


try:
    # ==========================================
    # 1. 初始化工业物流物料箱场景
    # ==========================================

    save(
        status="starting",
        task="industrial_logistics_material_transfer",
        perception_used_for_grasp=False,
        success=False
    )

    print("[1/7] 初始化场景与物料箱", flush=True)

    app = Main(
        "./configs/og_scene_file_red_pen.json",
        visualize=False
    )

    env = app.env
    env.reset()

    scene = env.og_env.scene

    def get_obj(name):
        obj = scene.object_registry("name", name)

        if obj is None:
            raise RuntimeError(
                "Missing scene object: " + name
            )

        return obj

    source_xyz, _ = pose(get_obj("pen_2"))
    target_xyz, _ = pose(get_obj("pencil_holder_1"))

    # 删除原始演示物体，保留桌面与机器人
    for name in (
        "pen_1",
        "pen_2",
        "pencil_holder_1"
    ):
        scene.remove_object(get_obj(name))

    # 创建工业物流物料箱仿真原型
    box = PrimitiveObject(
        name="logistics_box_1",
        primitive_type="Cube",
        category="object",
        size=0.065,
        rgba=(0.72, 0.46, 0.22, 1.0),
        fixed_base=False,
        visual_only=False
    )

    scene.add_object(box)

    spawn = source_xyz.copy()
    spawn[2] += 0.055

    box.set_position_orientation(
        position=torch.tensor(
            spawn,
            dtype=torch.float32
        ),
        orientation=torch.tensor(
            [0., 0., 0., 1.],
            dtype=torch.float32
        )
    )

    neutral = np.zeros(
        env.robot.action_dim,
        dtype=float
    )

    # 让箱体稳定落到桌面
    for _ in range(15):
        env.og_env.step(neutral)

    initial, box_quat = pose(box)

    # B 区域使用原笔筒所在的桌面位置
    goal = np.array([
        target_xyz[0],
        target_xyz[1],
        initial[2]
    ])

    print(
        "BOX_START =", initial.round(4),
        flush=True
    )

    print(
        "GOAL_B =", goal.round(4),
        flush=True
    )

    save(
        status="box_created",
        box_start=initial.tolist(),
        goal_B=goal.tolist(),
        size_m=0.065
    )

    # ==========================================
    # 2. 底盘移动与 IK 坐标同步
    # ==========================================

    print("[2/7] 底盘移动与坐标同步", flush=True)

    before, _ = pose(env.robot)

    forward = neutral.copy()
    forward[0] = 0.10

    for _ in range(10):
        env.og_env.step(forward)

    after, _ = pose(env.robot)

    moved = float(
        np.linalg.norm(after - before)
    )

    if moved < 0.005:
        raise RuntimeError(
            "Base displacement below 5 mm"
        )

    sync = sync_mobile_frames(app)

    print(
        "BASE_M =", moved,
        "FRAME_SYNC =", sync["status"],
        flush=True
    )

    save(
        status="mobile_synced",
        base_displacement_m=moved,
        frame_sync=sync["status"]
    )

    # ==========================================
    # 3. 重新感知并保存 DINOv2 候选点
    # ==========================================

    print("[3/7] 重新获取 RGB-D", flush=True)

    obs = env.get_cam_obs()
    cam_id = app.config["vlm_camera"]

    cam = (
        obs[cam_id]
        if cam_id in obs
        else obs[str(cam_id)]
    )

    Image.fromarray(
        image_u8(cam["rgb"])
    ).save(OUT / "scene_rgb.png")

    np.savez_compressed(
        OUT / "camera.npz",
        rgb=as_np(cam["rgb"]),
        points=as_np(cam["points"]),
        seg=as_np(cam["seg"])
    )

    try:
        proposed, overlay = (
            app.keypoint_proposer.get_keypoints(
                cam["rgb"],
                cam["points"],
                cam["seg"]
            )
        )

        np.save(
            OUT / "proposed_keypoints.npy",
            as_np(proposed)
        )

        Image.fromarray(
            image_u8(overlay)
        ).save(OUT / "proposed_keypoints.png")

        print(
            "VISION_KP =", len(proposed),
            flush=True
        )

        save(
            vision_keypoints_count=len(proposed)
        )

    except Exception as exc:
        print(
            "VISION_WARNING =", repr(exc),
            flush=True
        )

        save(
            vision_keypoints_error=repr(exc)
        )

    # ==========================================
    # 4. 根据箱体表面几何构建约束关键点
    # ==========================================

    print("[4/7] 构建箱体抓取关键点", flush=True)

    center, quaternion = pose(box)
    rot = T.quat2mat(quaternion)

    direction = before - center
    direction[2] = 0

    direction = direction / (
        np.linalg.norm(direction) + 1e-12
    )

    # KP0：机器人一侧的箱体表面候选抓取点
    grasp_point = center + rot @ np.array([0.0, 0.0, 0.0325])

    # KP1：箱体顶面中心
    top_point = center + rot @ np.array([
        0., 0., 0.0325
    ])

    # KP2、KP3：顶面辅助几何点
    upper_x = center + rot @ np.array([
        0.02, 0., 0.0325
    ])

    upper_y = center + rot @ np.array([
        0., 0.02, 0.0325
    ])

    anchors = np.stack([
        grasp_point,
        top_point,
        upper_x,
        upper_y
    ])

    # 关键点必须确实绑定到物料箱
    env.register_keypoints(anchors)

    owners = [
        getattr(
            env.get_object_by_keypoint(i),
            "name",
            ""
        )
        for i in range(len(anchors))
    ]

    print(
        "ANCHOR_OWNERS =", owners,
        flush=True
    )

    if any(
        name != "logistics_box_1"
        for name in owners
    ):
        raise RuntimeError(
            "Geometry anchors do not all belong "
            "to logistics_box_1"
        )

    registered = np.asarray(
        env.get_keypoint_positions(),
        dtype=float
    )

    snap_errors = np.linalg.norm(
        registered - anchors,
        axis=1
    )

    print(
        "ANCHOR_SNAP_M =",
        snap_errors.round(4),
        flush=True
    )

    if np.max(snap_errors) > 0.04:
        raise RuntimeError(
            "Keypoint attachment shifted "
            "more than 4 cm"
        )

    save(
        status="anchors_checked",
        anchor_owners=owners,
        anchor_positions=anchors.tolist(),
        anchor_snap_m=snap_errors.tolist()
    )

    # ==========================================
    # 5. 生成 ReKep 三阶段约束程序
    # ==========================================

    print("[5/7] 生成三阶段任务约束", flush=True)

    # Stage 2：转运时的箱体顶面悬停位置
    hover = [
        float(goal[0]),
        float(goal[1]),
        float(top_point[2] + 0.12)
    ]

    # Stage 3：释放前降低箱体高度
    place = [
        float(goal[0]),
        float(goal[1]),
        float(top_point[2] + 0.025)
    ]

    metadata = {
        "init_keypoint_positions": anchors.tolist(),
        "num_keypoints": 4,
        "num_stages": 3,
        "grasp_keypoints": [0, -1, -1],
        "release_keypoints": [-1, -1, 0]
    }

    (PROGRAM / "metadata.json").write_text(
        json.dumps(
            metadata,
            indent=2
        ),
        encoding="utf-8"
    )

    # Stage 1：对准箱体表面并抓取
    write(
        "stage1_subgoal_constraints.txt",
        """def stage1_subgoal_constraint1(end_effector, keypoints):
    return np.linalg.norm(end_effector - keypoints[0])
"""
    )

    # Stage 2、3：转运中必须保持抓取
    for stage in (2, 3):
        write(
            f"stage{stage}_path_constraints.txt",
            """def keep_holding_box(end_effector, keypoints):
    return get_grasping_cost_by_keypoint_idx(0)
"""
        )

    # Stage 2：将物料箱转运到 B 区域上方
    write(
        "stage2_subgoal_constraints.txt",
        f"""def stage2_subgoal_constraint1(end_effector, keypoints):
    target = np.array({hover!r})
    return np.linalg.norm(keypoints[1] - target)
"""
    )

    # Stage 3：移动到释放前的放置位置
    write(
        "stage3_subgoal_constraints.txt",
        f"""def stage3_subgoal_constraint1(end_effector, keypoints):
    target = np.array({place!r})
    return np.linalg.norm(keypoints[1] - target)
"""
    )

    save(
        status="program_ready",
        program_dir=str(PROGRAM),
        hover_goal=hover,
        prerelease_goal=place
    )

    print(
        "PROGRAM_READY =", PROGRAM,
        flush=True
    )

    # ==========================================
    # 6. ReKep 执行抓取、转运、释放
    # ==========================================

    print(
        "[6/7] ReKep 规划与物料搬运",
        flush=True
    )

    original_grasp = app._execute_grasp_action
    original_release = app._execute_release_action

    # 监控真实抓取状态
    def checked_grasp():
        from scipy.spatial.transform import Rotation

        box_center, _ = pose(box)
        ee0 = np.asarray(env.get_ee_pose(), dtype=float)

        rot0 = T.quat2mat(ee0[3:])
        approach = np.asarray(rot0[:, 0], dtype=float)

        # 如果夹爪没有明显朝下，切换为上方接近姿态
        orientation_adjusted = bool(approach[2] > -0.6)

        if orientation_adjusted:
            y = np.asarray(rot0[:, 1], dtype=float).copy()
            y[2] = 0.0

            if np.linalg.norm(y) < 0.2:
                y = np.array([0.0, 1.0, 0.0])

            y /= np.linalg.norm(y)
            approach = np.array([0.0, 0.0, -1.0])

            z = np.cross(approach, y)
            desired_rot = np.column_stack(
                [approach, y, z]
            )
            desired_quat = Rotation.from_matrix(
                desired_rot
            ).as_quat()
        else:
            desired_quat = ee0[3:].copy()

        depth = float(app.config["grasp_depth"])

        # 箱体中心略偏上：首次尝试的接触目标
        contact = box_center + np.array(
            [0.0, 0.0, 0.008]
        )

        # 根据夹爪实际前进轴计算预抓取位置
        pre_target = contact - approach * depth
        pre_pose = np.r_[pre_target, desired_quat]

        planned_adjustment = float(
            np.linalg.norm(pre_target - ee0[:3])
        )

        print(
            "GRASP_PRE_TARGET =",
            np.round(pre_target, 4),
            flush=True
        )
        print(
            "GRASP_BOX_CENTER =",
            np.round(box_center, 4),
            flush=True
        )
        print(
            "GRASP_APPROACH_AXIS =",
            np.round(approach, 4),
            flush=True
        )
        print(
            "GRASP_ORIENTATION_ADJUSTED =",
            orientation_adjusted,
            flush=True
        )
        print(
            "GRASP_PRE_ADJUSTMENT_M =",
            planned_adjustment,
            flush=True
        )

        save(
            grasp_target_center=contact.tolist(),
            grasp_pre_target=pre_target.tolist(),
            grasp_pre_adjustment_m=planned_adjustment,
            grasp_approach_axis=approach.tolist(),
            grasp_orientation_adjusted=orientation_adjusted
        )

        def abort(reason):
            save(
                status="grasp_alignment_failed",
                grasp_failure_reason=reason
            )

            try:
                video = env.save_video()
                save(failure_video_path=str(video))
                print(
                    "FAILURE_VIDEO_SAVED =",
                    video,
                    flush=True
                )
            except Exception as error:
                print(
                    "VIDEO_SAVE_ERROR =",
                    repr(error),
                    flush=True
                )

            raise RuntimeError(reason)

        # 防止目标位置异常时产生过大动作
        if planned_adjustment > 0.20:
            abort(
                "Stage1: correction exceeds 20cm"
            )

        # 先保持夹爪打开，修正预抓取位姿
        pre_error = float("inf")

        for attempt in range(2):
            env.execute_action(
                np.r_[
                    pre_pose,
                    env.get_gripper_null_action()
                ],
                precise=True
            )

            actual_pre = np.asarray(
                env.get_ee_pose(),
                dtype=float
            )

            pre_error = float(
                np.linalg.norm(
                    actual_pre[:3] - pre_target
                )
            )

            print(
                f"PREGRASP_ERROR_M_attempt{attempt + 1}"
                f" = {pre_error:.5f}",
                flush=True
            )

            if pre_error <= 0.035:
                break

        save(pregrasp_error_m=pre_error)

        if pre_error > 0.035:
            abort(
                "Stage1: pregrasp pose not reached"
            )

        # 到位后才进行接近和闭合
        env.execute_action(
            np.r_[
                contact,
                desired_quat,
                env.get_gripper_close_action()
            ],
            precise=True
        )

        final_ee = np.asarray(
            env.get_ee_pose(),
            dtype=float
        )

        contact_error = float(
            np.linalg.norm(
                final_ee[:3] - contact
            )
        )

        box_final, _ = pose(box)
        held = bool(env.is_grasping(box))

        save(
            grasp_verified=held,
            grasp_contact_error_m=contact_error,
            post_grasp_position=box_final.tolist()
        )

        print(
            "GRASP_CONTACT_ERROR_M =",
            round(contact_error, 5),
            flush=True
        )
        print(
            "GRASP_VERIFIED =",
            held,
            flush=True
        )

        if not held:
            abort(
                "Stage1: box not grasped after closing"
            )

    # 记录释放前的真实位姿
    def checked_release():
        before_release, _ = pose(box)

        original_release()

        save(
            release_called=True,
            box_prerelease_position=before_release.tolist()
        )

        print(
            "RELEASE_CALLED",
            flush=True
        )

    app._execute_grasp_action = checked_grasp
    app._execute_release_action = checked_release

    # 记录 ReKep 保存的视频
    original_video = env.save_video

    def logged_video():
        path = original_video()

        save(
            video_path=str(path)
        )

        return path

    env.save_video = logged_video

    original_depth = float(app.config["grasp_depth"])
    app.config["grasp_depth"] = 0.04
    print("ORIGINAL_GRASP_DEPTH_M =", original_depth,
          flush=True)
    print("TEST_GRASP_DEPTH_M = 0.04", flush=True)

    save(
        status="executing",
        original_grasp_depth_m=original_depth,
        test_grasp_depth_m=0.04
    )

    # 不调用 perform_task，防止重置移动后的底盘
    app._execute(
        str(PROGRAM),
        disturbance_seq=None
    )

    # ==========================================
    # 7. 检查箱体是否真的搬到了 B
    # ==========================================

    print(
        "[7/7] 检查最终落位结果",
        flush=True
    )

    env.sleep(1.0)

    final, _ = pose(box)

    xy_error = float(
        np.linalg.norm(final[:2] - goal[:2])
    )

    z_error = float(
        abs(final[2] - initial[2])
    )

    state = json.loads(
        REPORT.read_text(encoding="utf-8")
    )

    # 必须抓取、释放且满足落位条件
    achieved = bool(
        xy_error <= 0.08
        and z_error <= 0.06
        and state.get("grasp_verified")
        and state.get("release_called")
    )

    save(
        status="finished",
        box_final=final.tolist(),
        goal_xy_error_m=xy_error,
        goal_z_error_m=z_error,
        success=achieved,
        success_criterion=(
            "Grasp + release + "
            "final XY<=0.08m and Z<=0.06m"
        )
    )

    print(
        "TRANSFER_SUCCESS =", achieved,
        "XY_ERROR_M =", xy_error,
        "Z_ERROR_M =", z_error,
        flush=True
    )

except Exception:
    save(
        status="failed",
        success=False,
        error=traceback.format_exc()
    )

    traceback.print_exc()
    raise
