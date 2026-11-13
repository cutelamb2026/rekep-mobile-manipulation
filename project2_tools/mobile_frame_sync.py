import numpy as np
from environment import T


def sync_mobile_frames(main):
    """
    Refresh world-to-robot transforms after Fetch base motion.

    Must be called AFTER moving the base and BEFORE
    IK planning or arm execution.
    """
    env = main.env

    # Read the robot's current pose in the world frame.
    robot_pose = env.robot.get_position_orientation()

    new_world2robot = np.asarray(
        T.pose_inv(T.pose2mat(robot_pose)),
        dtype=np.float64
    )

    old_world2robot = np.asarray(
        env.world2robot_homo,
        dtype=np.float64
    ).copy()

    assert new_world2robot.shape == (4, 4)
    assert np.isfinite(new_world2robot).all()

    # 1. Update execution controller.
    env.world2robot_homo = new_world2robot.copy()

    # 2. Update IK solvers used by both planners.
    for solver in (main.subgoal_solver, main.path_solver):
        solver.ik_solver.world2robot_homo = (
            new_world2robot.copy()
        )

        # Discard optimization result from previous robot pose.
        if hasattr(solver, "last_opt_result"):
            solver.last_opt_result = None

    # 3. Verify all components use the same transform.
    np.testing.assert_allclose(
        env.world2robot_homo,
        main.subgoal_solver.ik_solver.world2robot_homo
    )
    np.testing.assert_allclose(
        env.world2robot_homo,
        main.path_solver.ik_solver.world2robot_homo
    )

    return {
        "status": "synchronized",
        "transform_change_norm": float(
            np.linalg.norm(
                new_world2robot - old_world2robot
            )
        ),
        "world2robot_homo": new_world2robot.tolist()
    }
