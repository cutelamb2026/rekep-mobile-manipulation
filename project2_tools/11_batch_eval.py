import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path.home() / "Rekep"
BASE = ROOT / "project2_evidence/07_evaluation"

SOURCES = {
    "logistics": ROOT / "project2_tools/09_run_logistics_transfer.py",
    "retail": ROOT / "project2_tools/10_run_retail_pick.py",
}

# 12 种 A/B 布局：3 种 A 区位置 × 4 种 B 区位置
LAYOUTS = [
    (x, y)
    for x in (-0.03, 0.0, 0.03)
    for y in (-0.045, -0.015, 0.015, 0.045)
]

# 8 类目标摆放：2 种 X 偏移 × 4 种 Y 偏移
STATES = [
    (x, y)
    for x in (-0.02, 0.02)
    for y in (-0.03, -0.01, 0.01, 0.03)
]

FIELDS = [
    "episode_id", "task", "layout_id", "placement_id",
    "source_offset_x_m", "source_offset_y_m",
    "goal_offset_x_m", "goal_offset_y_m",
    "process_exit_code", "program_status",
    "grasp_verified", "release_called",
    "goal_xy_error_m", "goal_z_error_m",
    "program_success", "log_path", "result_path"
]


def records():
    for task in SOURCES:
        for li, (ax, by) in enumerate(LAYOUTS, 1):
            for si, (sx, sy) in enumerate(STATES, 1):
                yield {
                    "episode_id": f"{task}_L{li:02d}_P{si:02d}",
                    "task": task,
                    "layout_id": f"L{li:02d}",
                    "placement_id": f"P{si:02d}",
                    "source_offset_x_m": round(ax + sx, 4),
                    "source_offset_y_m": round(sy, 4),
                    "goal_offset_x_m": 0.0,
                    "goal_offset_y_m": round(by, 4),
                    "layout_A_dx_m": ax,
                    "placement_dx_m": sx,
                    "placement_dy_m": sy,
                }


def prepare():
    BASE.mkdir(parents=True, exist_ok=True)
    manifest = BASE / "scenario_matrix_proposed.csv"

    with manifest.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(next(records()).keys())
        )
        writer.writeheader()
        writer.writerows(records())

    print("MATRIX_CREATED:", manifest)
    print("PLANNED_EPISODES:", len(list(records())))
    print("NOTE: configurations are not yet validated or executed")


def make_source(case, out):
    original = SOURCES[case["task"]].read_text(
        encoding="utf-8"
    )

    if case["task"] == "logistics":
        old_out = (
            "project2_evidence/05_logistics/transfer_run03"
        )
    else:
        old_out = (
            "project2_evidence/06_retail/pick_run01"
        )

    new_out = (
        "project2_evidence/07_evaluation/runs/"
        + case["episode_id"]
    )

    if original.count(old_out) != 1:
        raise RuntimeError(
            f"Expected one output path {old_out}; "
            f"found {original.count(old_out)}"
        )

    program = original.replace(old_out, new_out, 1)

    marker = (
        '    target_xyz, _ = '
        'pose(get_obj("pencil_holder_1"))'
    )

    if program.count(marker) != 1:
        raise RuntimeError(
            "Cannot find source/goal initialization"
        )

    inject = (
        "\n    # Episode-specific A/B configuration\n"
        f'    source_xyz[:2] += np.array(['
        f'{case["source_offset_x_m"]}, '
        f'{case["source_offset_y_m"]}])\n'
        f'    target_xyz[:2] += np.array(['
        f'{case["goal_offset_x_m"]}, '
        f'{case["goal_offset_y_m"]}])\n'
        f'    print("EVAL_CASE", '
        f'{case["episode_id"]!r}, flush=True)\n'
    )

    program = program.replace(
        marker, marker + inject, 1
    )

    compile(program, str(out), "exec")
    out.write_text(program, encoding="utf-8")


def run_one(case_id):
    all_cases = {
        c["episode_id"]: c for c in records()
    }

    if case_id not in all_cases:
        raise SystemExit(
            "Unknown episode: " + case_id
        )

    case = all_cases[case_id]
    out = BASE / "runs" / case_id

    if (out / "result.json").exists():
        raise SystemExit(
            "Existing result; refusing to overwrite: "
            + str(out)
        )

    out.mkdir(parents=True, exist_ok=True)

    (out / "config.json").write_text(
        json.dumps(case, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    runner = out / "runner.py"
    make_source(case, runner)

    logfile = out / "run.log"

    env = os.environ.copy()
    env["PYTHONPATH"] = (
        str(ROOT) + os.pathsep
        + env.get("PYTHONPATH", "")
    )

    with logfile.open("w", encoding="utf-8") as log:
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-X", "faulthandler",
                    "-u", str(runner)
                ],
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=1800,
                check=False
            )
            exit_code = result.returncode

        except subprocess.TimeoutExpired:
            exit_code = 124

    result_file = out / "result.json"

    raw = (
        json.loads(result_file.read_text(encoding="utf-8"))
        if result_file.exists()
        else {}
    )

    row = {
        **{
            k: case.get(k)
            for k in FIELDS
            if k in case
        },
        "process_exit_code": exit_code,
        "program_status": raw.get("status"),
        "grasp_verified": raw.get("grasp_verified"),
        "release_called": raw.get("release_called"),
        "goal_xy_error_m": raw.get("goal_xy_error_m"),
        "goal_z_error_m": raw.get("goal_z_error_m"),
        "program_success": (
            raw.get("success") is True
            and raw.get("status") == "finished"
        ),
        "log_path": str(logfile.relative_to(ROOT)),
        "result_path": (
            str(result_file.relative_to(ROOT))
            if result_file.exists()
            else ""
        )
    }

    (out / "episode_summary.json").write_text(
        json.dumps(row, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print(json.dumps(row, ensure_ascii=False, indent=2))


def collect():
    rows = []

    for case in records():
        p = (
            BASE / "runs" / case["episode_id"]
            / "episode_summary.json"
        )
        if p.exists():
            rows.append(
                json.loads(p.read_text(encoding="utf-8"))
            )

    summary = BASE / "observed_episodes.csv"

    with summary.open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=FIELDS,
            extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)

    successful = sum(
        1 for r in rows if r["program_success"]
    )

    print("RECORDED_EPISODES:", len(rows))
    print("PROGRAM_SUCCESSES:", successful)

    if rows:
        print(
            "OBSERVED_PROGRAM_SUCCESS_RATE:",
            f"{successful / len(rows):.2%}"
        )

    print("CSV:", summary)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "command",
        choices=["prepare", "run", "collect"]
    )
    parser.add_argument("--episode")

    args = parser.parse_args()

    if args.command == "prepare":
        prepare()

    elif args.command == "run":
        if not args.episode:
            parser.error("--episode required")
        run_one(args.episode)

    else:
        collect()
