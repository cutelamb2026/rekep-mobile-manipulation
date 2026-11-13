import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np
from PIL import Image


def arr(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def rgb8(x):
    x = arr(x)[..., :3]
    if np.issubdtype(x.dtype, np.floating):
        finite = x[np.isfinite(x)]
        if finite.size and finite.min() >= 0 and finite.max() <= 1:
            x = x * 255
    return np.nan_to_num(x).clip(0, 255).astype(np.uint8)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--repo", required=True)
    p.add_argument("--scene", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--speed", type=float, default=0.10)
    p.add_argument("--keypoints", action="store_true")
    args = p.parse_args()

    if not 1 <= args.steps <= 30 or not 0 < args.speed <= 0.2:
        raise ValueError("Use 1-30 steps and speed between 0 and 0.2")

    repo = Path(args.repo).resolve()
    out = Path(args.outdir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(repo))

    from utils import get_config
    from environment import ReKepOGEnv

    config = get_config(config_path=str(repo / "configs/config.yaml"))
    scene = Path(args.scene)
    if not scene.is_absolute():
        scene = repo / scene

    print("[INIT] Starting simulation", flush=True)
    env = ReKepOGEnv(config["env"], str(scene), verbose=False)
    env.reset()

    robot = env.robot
    camera_id = config["main"]["vlm_camera"]

    result = {
        "scene": str(scene),
        "action_dim": int(robot.action_dim),
        "steps": args.steps,
        "speed": args.speed,
        "snapshots": {},
        "observations": {}
    }

    def capture(label):
        pos, quat = robot.get_position_orientation()
        result["snapshots"][label] = {
            "base_position": arr(pos).tolist(),
            "base_orientation": arr(quat).tolist(),
            "ee_pose": arr(env.get_ee_pose()).tolist()
        }

        obs = env.get_cam_obs()
        if camera_id in obs:
            cam = obs[camera_id]
        elif str(camera_id) in obs:
            cam = obs[str(camera_id)]
        else:
            raise KeyError(f"Camera {camera_id} unavailable: {list(obs)}")

        data = {}
        info = {}

        for key in ("rgb", "depth", "points", "seg",
                    "intrinsic", "extrinsic"):
            if key not in cam:
                info[key] = "missing"
                continue
            value = arr(cam[key])
            info[key] = list(value.shape)
            if value.dtype.kind in "biuf":
                data[key] = value

        np.savez_compressed(out / f"{label}_camera.npz", **data)

        if "rgb" in data:
            Image.fromarray(rgb8(data["rgb"])).save(
                out / f"{label}_rgb.png"
            )

        result["observations"][label] = info
        print(f"[{label}] camera saved", flush=True)
        return data

    pre = capture("pre")

    action = np.zeros(robot.action_dim, dtype=float)
    action[0] = args.speed
    for _ in range(args.steps):
        env.og_env.step(action)

    forward = capture("forward")

    action[0] = -args.speed
    for _ in range(args.steps):
        env.og_env.step(action)

    returned = capture("return")

    start = np.array(result["snapshots"]["pre"]["base_position"])
    moved = np.array(result["snapshots"]["forward"]["base_position"])
    end = np.array(result["snapshots"]["return"]["base_position"])

    result["forward_displacement_m"] = float(
        np.linalg.norm(moved - start)
    )
    result["return_error_m"] = float(
        np.linalg.norm(end - start)
    )

    if "rgb" in pre and "rgb" in forward:
        a = rgb8(pre["rgb"]).astype(float)
        b = rgb8(forward["rgb"]).astype(float)
        if a.shape == b.shape:
            result["rgb_mean_abs_difference"] = float(
                np.abs(a - b).mean()
            )

    def save_summary():
        (out / "summary.json").write_text(
            json.dumps(result, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )

    save_summary()
    print("[SAVED] RGB-D and poses", flush=True)

    if args.keypoints:
        try:
            from keypoint_proposal import KeypointProposer

            proposer = KeypointProposer(config["keypoint_proposer"])
            points, overlay = proposer.get_keypoints(
                forward["rgb"],
                forward["points"],
                forward["seg"]
            )

            points = arr(points)
            np.save(out / "forward_keypoints.npy", points)
            Image.fromarray(rgb8(overlay)).save(
                out / "forward_keypoints.png"
            )

            result["keypoint_count"] = len(points)
            print("[KEYPOINT] count =", len(points), flush=True)

        except Exception:
            result["keypoint_error"] = traceback.format_exc()
            traceback.print_exc()
            save_summary()
            return 2

    save_summary()

    print("DONE", flush=True)
    print("FORWARD_DISPLACEMENT_M =", result["forward_displacement_m"])
    print("RETURN_ERROR_M =", result["return_error_m"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
