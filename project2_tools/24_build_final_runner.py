from pathlib import Path

src = Path(
    "project2_tools/19_reobserve_station_b_logistics.py"
)

dst = Path(
    "project2_tools/24_final_dual_station_logistics.py"
)

assert src.is_file(), "Missing validated transport runner"
assert not dst.exists(), "Final runner exists; will not overwrite"

s = src.read_text(encoding="utf-8")


def replace_once(old, new):
    global s
    assert s.count(old) == 1, (
        "Source structure mismatch: " + repr(old)
    )
    s = s.replace(old, new, 1)


replace_once(
    "./configs/og_scene_file_red_pen.json",
    "./configs/og_scene_file_dual_station_v2.json"
)

replace_once(
    "project2_evidence/11_station_b_reobserve/logistics_run01",
    "project2_evidence/13_final_dual_station/logistics_run01"
)

replace_once(
    "from project2_tools.vision_rgbd_localizer import localize_from_camera",
    "from project2_tools.dual_station_box_vision import localize_from_camera"
)

# Existing A/B platforms and object are already saved in the scene.
start = s.index(
    '    source_xyz, _ = pose(get_obj("pen_2"))'
)

end = s.index(
    "    # ==========================================\n"
    "    # 2. 底盘移动与 IK 坐标同步",
    start
)

new_init = '''
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

'''

s = s[:start] + new_init + s[end:]

# The saved scene already contains the cyan B marker.
marker_start = (
    "        # B destination marker on existing workbench"
)

marker_end = '''        save(
            station_b_design_xyz=marker_xyz.tolist()
        )'''

assert s.count(marker_start) == 1
assert s.count(marker_end) == 1

i = s.index(marker_start)
j = s.index(marker_end, i) + len(marker_end)

s = s[:i] + s[j:]


# Replace fixed loaded-base motion with odometry-feedback
# waypoint following and per-step clearance guards.
motion_start = (
    "        # D. Measure actual loaded base motion."
)

motion_end = (
    "        # E. Refresh world-to-robot and both IK solver frames."
)

assert s.count(motion_start) == 1
assert s.count(motion_end) == 1

i = s.index(motion_start)
j = s.index(motion_end, i)

new_motion = '''
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

'''

s = s[:i] + new_motion + s[j:]

# Make sure unsafe original 40-step motion is gone.
assert "for step in range(40):" not in s
assert "waypoint_2_reached" not in s  # generated at runtime

compile(s, str(dst), "exec")
dst.write_text(s, encoding="utf-8")

print("FINAL_DUAL_STATION_RUNNER_READY")
