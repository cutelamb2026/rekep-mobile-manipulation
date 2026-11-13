#!/usr/bin/env python3
"""Recompute exported evidence without OmniGibson / Isaac Sim.

This deliberately separates the recorded program-level success rule from
strict XY <= 3 cm. It does not infer unobserved episodes or process stability.
"""
import csv
import json
import statistics
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "project2_evidence/07_evaluation/observed_episodes.csv"
DUAL = ROOT / "project2_evidence/13_final_dual_station/logistics_run02/result.json"
RETAIL = ROOT / "project2_evidence/09_vision_execution/retail_v1/result.json"


def is_true(s):
    return s is True or str(s).strip().lower() == "true"


def evaluate(rows):
    program_success = sum(is_true(r["program_success"]) for r in rows)
    strict_success = sum(
        is_true(r["program_success"])
        and r.get("goal_xy_error_m") not in ("", None)
        and float(r["goal_xy_error_m"]) <= 0.03
        for r in rows
    )
    errors = [
        float(r["goal_xy_error_m"])
        for r in rows if r.get("goal_xy_error_m") not in ("", None)
    ]
    return {
        "n": len(rows),
        "program_success": program_success,
        "strict_3cm_success": strict_success,
        "tasks": dict(Counter(r["task"] for r in rows)),
        "layout_ids": sorted({r["layout_id"] for r in rows}),
        "placement_ids": sorted({r["placement_id"] for r in rows}),
        "exit_codes": dict(Counter(r["process_exit_code"] for r in rows)),
        "mean_xy_error_m_among_observed": statistics.mean(errors),
        "median_xy_error_m_among_observed": statistics.median(errors),
        "fail_episodes": [
            r["episode_id"] for r in rows
            if not is_true(r["program_success"])
        ],
        "strict_fail_episodes": [
            r["episode_id"] for r in rows
            if not (
                is_true(r["program_success"])
                and r.get("goal_xy_error_m") not in ("", None)
                and float(r["goal_xy_error_m"]) <= 0.03
            )
        ],
    }


def main():
    with CSV.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    data = evaluate(rows)
    dual = json.loads(DUAL.read_text(encoding="utf-8"))
    retail = json.loads(RETAIL.read_text(encoding="utf-8"))
    assert data["n"] == 26, "Unexpected evidence count; re-audit before publication"
    assert data["program_success"] == 25
    assert data["strict_3cm_success"] == 24
    print("EXPORT_ONLY: does not simulate robot actions")
    print(f"PROGRAM_SUCCESS = {data['program_success']}/{data['n']} = "
          f"{data['program_success']/data['n']:.2%}")
    print(f"STRICT_XY_3CM = {data['strict_3cm_success']}/{data['n']} = "
          f"{data['strict_3cm_success']/data['n']:.2%}")
    print("TASK_COUNTS =", data["tasks"])
    print("DISTINCT_LAYOUT_IDS =", len(data["layout_ids"]))
    print("DISTINCT_PLACEMENT_IDS =", len(data["placement_ids"]), data["placement_ids"])
    print("PROCESS_EXIT_CODES =", data["exit_codes"])
    print("FAIL_EPISODES =", data["fail_episodes"])
    print("STRICT_FAIL_EPISODES =", data["strict_fail_episodes"])
    print("DUAL_BASE_CARRIED_M =", dual.get("carried_base_move_m"))
    print("DUAL_B_CONFIG_CHECK_XY_M =", dual.get("station_b_audit_xy_error_m"))
    print("DUAL_FINAL_XY_M =", dual.get("goal_xy_error_m"))
    print("DUAL_PROGRAM_CRITERION =", dual.get("success_criterion"))
    print("DUAL_STATUS =", dual.get("status"), dual.get("success"))
    print("RETAIL_FINAL_XY_M =", retail.get("goal_xy_error_m"))


if __name__ == "__main__":
    main()
