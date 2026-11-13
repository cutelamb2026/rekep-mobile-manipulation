from pathlib import Path

src = Path(
    "project2_tools/18_loaded_transport_logistics.py"
)

dst = Path(
    "project2_tools/19_reobserve_station_b_logistics.py"
)

assert src.exists(), "Validated runner missing"
assert not dst.exists(), "Refusing to overwrite runner"

s = src.read_text(encoding="utf-8")

old_dir = (
    "project2_evidence/10_mobile_transport/"
    "logistics_run01"
)

new_dir = (
    "project2_evidence/11_station_b_reobserve/"
    "logistics_run01"
)

grasp_marker = "        checked_grasp()\n"

resume_marker = (
    '        print("RESUMING_REKEP_STAGE_2", flush=True)'
)

for needle in (
    old_dir,
    grasp_marker,
    resume_marker
):
    assert s.count(needle) == 1, (
        f"Source structure mismatch: {needle}"
    )

s = s.replace(old_dir, new_dir, 1)

# Add visual destination marker AFTER initial object detection
marker_code = '''
        # B destination marker on existing workbench
        marker = PrimitiveObject(
            name="station_b_cyan_marker",
            primitive_type="Cylinder",
            category="object",
            radius=0.075,
            height=0.004,
            rgba=(0.02, 0.94, 0.86, 1.0),
            fixed_base=True,
            visual_only=True
        )

        scene.add_object(marker)

        marker_xyz = np.array([
            float(goal[0]),
            float(goal[1]),
            float(initial[2] - 0.0325 + 0.004)
        ])

        marker.set_position_orientation(
            position=torch.tensor(
                marker_xyz,
                dtype=torch.float32
            ),
            orientation=torch.tensor(
                [0., 0., 0., 1.],
                dtype=torch.float32
            )
        )

        save(
            station_b_design_xyz=marker_xyz.tolist()
        )

'''

s = s.replace(
    grasp_marker,
    grasp_marker + marker_code,
    1
)

# Insert after loaded motion + frame synchronization,
# just before Stage 2 is resumed
reobserve_code = '''
        # Fresh RGB-D observation AFTER base movement
        from project2_tools.station_b_reobserve import (
            locate_station_b
        )

        camera_after = env.get_cam_obs()[
            app.config["vlm_camera"]
        ]

        b_xyz, b_detection = locate_station_b(
            camera_after["rgb"],
            camera_after["points"],
            OUT
        )

        # Configuration is used only for a safety check,
        # not to generate the new planning target.
        audit_error = float(
            np.linalg.norm(
                b_xyz[:2] - goal[:2]
            )
        )

        print(
            "STATION_B_VISUAL_XYZ =",
            b_xyz.round(5),
            flush=True
        )

        print(
            "STATION_B_AUDIT_XY_M =",
            audit_error,
            flush=True
        )

        save(
            station_b_reobserved=True,
            station_b_visual_xyz=b_xyz.tolist(),
            station_b_detection=b_detection,
            station_b_audit_xy_error_m=audit_error
        )

        if audit_error > 0.045:
            abort_transport(
                "Station B visual detection is inconsistent"
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

'''

s = s.replace(
    resume_marker,
    reobserve_code + resume_marker,
    1
)

compile(s, str(dst), "exec")
dst.write_text(s, encoding="utf-8")

print("STATION_B_RUNNER_READY")
