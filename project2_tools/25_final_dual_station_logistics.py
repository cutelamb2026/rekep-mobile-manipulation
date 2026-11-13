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
OUT = ROOT / "project2_evidence/13_final_dual_station/logistics_run02"
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
        "./configs/og_scene_file_dual_station_v2.json",
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


    station_a = get_obj("logistics_station_A")
    station_b = get_obj("logistics_station_B")
    marker = get_obj("station_b_cyan_marker")
    box = get_obj("logistics_box_1")

    initial, box_quat = pose(box)

    a_low, a_high = station_a.aabb
    b_low, b_high = station_b.aabb

    a_low = as_np(a_low).astype(float)
    a_high = as_np(a_high).astype(float)
    b_low = as_np(b_low).astype(float)
    b_high = as_np(b_high).astype(float)

    b_center = (b_low + b_high) / 2

    goal = np.array([
        b_center[0],
        b_center[1],
        initial[2]
    ])

    neutral = np.zeros(
        env.robot.action_dim, dtype=float
    )

    save(
        status="dual_station_loaded",
        box_start=initial.tolist(),
        goal_B=goal.tolist(),
        station_A_aabb=[a_low.tolist(), a_high.tolist()],
        station_B_aabb=[b_low.tolist(), b_high.tolist()]
    )

    print("DUAL_STATIONS_LOADED", flush=True)
    print("BOX_START =", initial, flush=True)
    print("GOAL_B =", goal, flush=True)

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

    # Visual target localization in world coordinates
    from project2_tools.dual_station_box_vision import localize_from_camera

    vision_top, vision_center, vision_info = localize_from_camera(
        rgb=cam["rgb"],
        points=cam["points"],
        seg=cam["seg"],
        task='logistics',
        half_height=0.0325,
        output_dir=OUT
    )

    # Ground truth used only for audit metrics, not grasp planning
    vision_xy_audit = float(
        np.linalg.norm(vision_center[:2] - initial[:2])
    )
    vision_3d_audit = float(
        np.linalg.norm(vision_center - initial)
    )

    print("VISION_CENTER =", vision_center.round(5), flush=True)
    print("VISION_XY_AUDIT_M =", vision_xy_audit, flush=True)
    print("VISION_3D_AUDIT_M =", vision_3d_audit, flush=True)

    save(
        vision_driven_grasp=True,
        grasp_target_source="rgbd_segmented_surface",
        dino_used_to_choose_grasp=False,
        vision_center=vision_center.tolist(),
        vision_top=vision_top.tolist(),
        vision_xy_audit_m=vision_xy_audit,
        vision_3d_audit_m=vision_3d_audit,
        simulator_mesh_registration_used=True
    )

    center = vision_center.copy()
    rot = np.eye(3, dtype=float)


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
    anchors_requested = anchors.copy()
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
        registered - anchors_requested,
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

        box_center = vision_center.copy()
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


    def grasp_then_transport():
        # A. Execute the already validated vision-driven grasp.
        checked_grasp()




        if not env.is_grasping(box):
            raise RuntimeError("Object not held after Stage 1")

        def abort_transport(reason):
            save(
                transport_verified=False,
                transport_failure_reason=reason
            )
            try:
                video = env.save_video()
                save(failure_video_path=str(video))
            except Exception as exc:
                print("TRANSPORT_VIDEO_ERROR =", repr(exc))
            raise RuntimeError(reason)

        # B. Lift the held object before moving the base.
        ee_before = np.asarray(env.get_ee_pose(), dtype=float)
        lift_pose = ee_before.copy()
        lift_pose[2] += 0.10

        env.execute_action(
            np.r_[lift_pose, env.get_gripper_null_action()],
            precise=True
        )

        ee_after = np.asarray(env.get_ee_pose(), dtype=float)
        actual_lift = float(ee_after[2] - ee_before[2])

        print("CARRY_LIFT_M =", actual_lift, flush=True)

        if actual_lift < 0.04:
            abort_transport("Insufficient lifting clearance")

        if not env.is_grasping(box):
            abort_transport("Object lost during lifting")

        # C. Validate the actual robot action mapping.
        order = list(env.robot.controller_order)

        if not order or not order[-1].startswith("gripper"):
            abort_transport(
                "Gripper is not the final controller"
            )

        gripper_controller = env.robot.controllers[order[-1]]
        gripper_dim = int(
            getattr(gripper_controller, "command_dim", 0)
        )

        if gripper_dim not in (1, 2):
            abort_transport("Unexpected gripper command dimension")

        closed_command = float(
            env.get_last_og_gripper_action()
        )

        if abs(closed_command) > 1e-6:
            abort_transport("Gripper is not in closed-command state")


        # D. Controlled loaded transport between physical stations.
        base_before, _ = pose(env.robot)
        object_before, _ = pose(box)

        from pathlib import Path
        import json as _json

        preflight = _json.loads(
            Path(
                "project2_evidence/12_dual_station/"
                "route_preflight/report.json"
            ).read_text(encoding="utf-8")
        )

        required = float(preflight["required_clearance_m"])

        def rectangle_distance(xy, low, high):
            delta = np.maximum(
                np.maximum(low[:2] - xy, xy - high[:2]),
                0.0
            )
            return float(np.linalg.norm(delta))

        def clearance(xy):
            return min(
                rectangle_distance(xy, a_low, a_high),
                rectangle_distance(xy, b_low, b_high)
            )

        # First base approach is already executed before grasping.
        # The route is two segments: move in Y, then approach B in X.
        current_xy = base_before[:2].copy()

        waypoints = [
            np.array([current_xy[0], goal[1]]),
            np.array([goal[0] - 0.59, goal[1]])
        ]

        # Check the entire proposed path before any movement.
        route = [current_xy] + waypoints

        for p0, p1 in zip(route[:-1], route[1:]):
            for p in np.linspace(p0, p1, 101):
                if clearance(p) < required:
                    abort_transport(
                        "Route fails conservative clearance"
                    )

        save(
            route_waypoints=[
                p.tolist() for p in waypoints
            ],
            route_clearance_required_m=required,
            route_control="differential_drive_feedback"
        )

        print(
            "DUAL_STATION_ROUTE_PRECHECK_PASS",
            flush=True
        )

        def wrap_angle(a):
            return float(np.arctan2(
                np.sin(a), np.cos(a)
            ))

        def move_to(target, waypoint_index):
            reached = False

            for step in range(360):
                position, quat = pose(env.robot)

                delta = target - position[:2]
                distance = float(np.linalg.norm(delta))

                if distance <= 0.035:
                    reached = True
                    break

                rot = T.quat2mat(quat)
                yaw = float(np.arctan2(
                    rot[1, 0], rot[0, 0]
                ))

                desired = float(np.arctan2(
                    delta[1], delta[0]
                ))

                angle_error = wrap_angle(desired - yaw)

                command = np.zeros(
                    env.robot.action_dim,
                    dtype=float
                )

                if abs(angle_error) > 0.16:
                    # Rotate in place before forward motion.
                    command[0] = 0.0
                    command[1] = float(np.clip(
                        1.0 * angle_error, -0.5, 0.5
                    ))
                else:
                    command[0] = float(np.clip(
                        0.65 * distance, 0.035, 0.12
                    ))
                    command[1] = float(np.clip(
                        1.2 * angle_error, -0.25, 0.25
                    ))

                command[-gripper_dim:] = closed_command

                env._step(command)

                actual, _ = pose(env.robot)

                if clearance(actual[:2]) < required:
                    abort_transport(
                        "Base entered restricted clearance zone"
                    )

                if (step + 1) % 5 == 0:
                    if not env.is_grasping(box):
                        abort_transport(
                            "Material dropped during transport"
                        )

            if not reached:
                abort_transport(
                    f"Waypoint {waypoint_index} not reached"
                )

            print(
                "WAYPOINT_REACHED =",
                waypoint_index,
                "POSITION =",
                position[:2],
                flush=True
            )

            save(
                **{
                    f"waypoint_{waypoint_index}_reached": True,
                    f"waypoint_{waypoint_index}_xy":
                        position[:2].tolist()
                }
            )

        for idx, waypoint in enumerate(waypoints, 1):
            move_to(waypoint, idx)

        base_after, _ = pose(env.robot)
        object_after, _ = pose(box)

        base_move = float(np.linalg.norm(
            base_after[:2] - base_before[:2]
        ))

        object_move = float(np.linalg.norm(
            object_after[:2] - object_before[:2]
        ))

        print(
            "CARRIED_BASE_MOVE_M =", base_move,
            flush=True
        )
        print(
            "CARRIED_OBJECT_MOVE_M =", object_move,
            flush=True
        )

        # E. Refresh world-to-robot and both IK solver frames.
        sync_report = sync_mobile_frames(app)

        save(
            loaded_transport_attempted=True,
            carry_lift_m=actual_lift,
            carried_base_move_m=base_move,
            carried_object_move_m=object_move,
            post_transport_frame_sync=sync_report["status"],
            transport_grasp_retained=bool(env.is_grasping(box))
        )

        if base_move < 0.12:
            abort_transport("Base displacement below 12 cm")

        if object_move < 0.08:
            abort_transport("Carried object did not follow the base")

        if sync_report["status"] != "synchronized":
            abort_transport("Frame synchronization failed")

        if not env.is_grasping(box):
            abort_transport("Object lost after transport")

        save(transport_verified=True)

        print("LOADED_TRANSPORT_VERIFIED", flush=True)

        # Fresh RGB-D observation AFTER base movement
        from project2_tools.station_b_reobserve import (
            locate_station_b
        )


        # Post-transport B detection from BOTH fixed RGB-D cameras.
        # Station coordinates only define a search/safety region.
        # Planning targets come from observed 3D camera points.
        observations = env.get_cam_obs()
        candidates = []
        diagnostics = []

        for camera_id, camera_data in observations.items():
            rgb_b = image_u8(camera_data["rgb"])
            xyz_b = as_np(
                camera_data["points"]
            ).astype(np.float64)

            if xyz_b.shape != (*rgb_b.shape[:2], 3):
                diagnostics.append({
                    "camera": str(camera_id),
                    "error": "RGB-D shape mismatch"
                })
                continue

            r = rgb_b[..., 0].astype(np.int16)
            g = rgb_b[..., 1].astype(np.int16)
            b = rgb_b[..., 2].astype(np.int16)

            cyan = (
                (g > 110)
                & (b > 110)
                & (g > r + 50)
                & (b > r + 50)
            )

            finite = np.isfinite(xyz_b).all(axis=-1)

            # B station is a known task destination.
            # This ROI rejects other cyan surfaces.
            xy_distance = np.linalg.norm(
                xyz_b[..., :2]
                - goal[None, None, :2],
                axis=-1
            )

            near_station = xy_distance < 0.09

            # The marker lies just above B support surface.
            near_surface = (
                (xyz_b[..., 2] >= b_high[2] - 0.02)
                & (xyz_b[..., 2] <= b_high[2] + 0.035)
            )

            mask = (
                cyan & finite
                & near_station & near_surface
            )

            pixels = int(mask.sum())

            Image.fromarray(rgb_b).save(
                OUT / f"station_b_camera_{camera_id}.png"
            )

            Image.fromarray(
                (mask.astype(np.uint8) * 255)
            ).save(
                OUT / f"station_b_mask_camera_{camera_id}.png"
            )

            item = {
                "camera": str(camera_id),
                "valid_cyan_pixels": pixels
            }

            if pixels >= 60:
                observed = np.median(
                    xyz_b[mask], axis=0
                )

                error = float(np.linalg.norm(
                    observed[:2] - goal[:2]
                ))

                item["observed_xyz"] = observed.tolist()
                item["xy_check_error_m"] = error

                if error <= 0.03:
                    candidates.append({
                        **item,
                        "xyz": observed
                    })

            diagnostics.append(item)

            print(
                "B_CAMERA_DIAGNOSTIC =",
                item,
                flush=True
            )

        save(
            station_b_camera_diagnostics=diagnostics,
            station_b_detection_attempted=True
        )

        if not candidates:
            abort_transport(
                "Station B not reliably visible in either camera"
            )

        # Prefer the observation nearest the expected
        # B work area, but DO NOT use goal as the new target.
        selected = min(
            candidates,
            key=lambda c: c["xy_check_error_m"]
        )

        b_xyz = selected["xyz"]
        audit_error = selected["xy_check_error_m"]

        b_detection = {
            "method": "dual_camera_cyan_rgbd_roi",
            "camera": selected["camera"],
            "valid_pixels": selected["valid_cyan_pixels"],
            "estimated_world_xyz": b_xyz.tolist()
        }

        save(
            station_b_reobserved=True,
            station_b_visual_xyz=b_xyz.tolist(),
            station_b_audit_xy_error_m=audit_error,
            station_b_detection=b_detection,
            station_b_selected_camera=selected["camera"]
        )

        print(
            "STATION_B_REOBSERVED =",
            b_xyz.round(5),
            flush=True
        )
        print(
            "STATION_B_XY_ERROR_M =",
            audit_error,
            flush=True
        )

        # New constraint functions use the newly observed B.
        def target_constraint(target):
            target = np.asarray(
                target, dtype=float
            ).copy()

            def constraint(end_effector, keypoints):
                return float(
                    np.linalg.norm(
                        keypoints[1] - target
                    )
                )

            return constraint

        if (
            len(app.constraint_fns[2]["subgoal"]) != 1
            or len(app.constraint_fns[3]["subgoal"]) != 1
        ):
            abort_transport(
                "Unexpected ReKep constraint structure"
            )

        box_height = 0.065

        stage2_target = np.array([
            b_xyz[0],
            b_xyz[1],
            b_xyz[2] + box_height + 0.12
        ])

        stage3_target = np.array([
            b_xyz[0],
            b_xyz[1],
            b_xyz[2] + box_height + 0.025
        ])

        # _execute reads these functions on subsequent stages.
        app.constraint_fns[2]["subgoal"] = [
            target_constraint(stage2_target)
        ]

        app.constraint_fns[3]["subgoal"] = [
            target_constraint(stage3_target)
        ]

        save(
            runtime_constraints_updated=True,
            stage2_visual_target=stage2_target.tolist(),
            stage3_visual_target=stage3_target.tolist(),
            station_b_sensor_source="post_transport_rgbd"
        )

        print(
            "DYNAMIC_REKEP_GOALS_UPDATED",
            flush=True
        )

        print("RESUMING_REKEP_STAGE_2", flush=True)

    app._execute_grasp_action = grasp_then_transport

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
