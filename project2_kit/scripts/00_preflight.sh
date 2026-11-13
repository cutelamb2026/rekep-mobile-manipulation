#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 2 ]; then echo "Usage: bash 00_preflight.sh REKEP_ROOT EVIDENCE_DIR" >&2; exit 2; fi
ROOT=$(realpath "$1")
OUT="$2/00_environment"
mkdir -p "$OUT"
cd "$ROOT"
{
 echo "repo_path=$ROOT"
 echo "date_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
 echo "python=$(command -v python || true)"
 python --version 2>&1 || true
 echo "git_commit=$(git rev-parse HEAD 2>/dev/null || echo NOT_GIT)"
 echo "git_dirty=$(test -n "$(git status --porcelain 2>/dev/null || true)" && echo yes || echo no)"
 echo "display_set=$(test -n "${DISPLAY:-}" && echo yes || echo no)"
 uname -a
} > "$OUT/environment_summary.txt"
{
 echo "=== GPU ==="
 nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>&1 || true
 echo "=== CONDA ==="
 conda env list 2>&1 || true
 echo "=== PACKAGES ==="
 python -m pip show omnigibson isaacsim boto3 botocore s3transfer 2>&1 || true
} > "$OUT/runtime_versions.txt"
for f in main.py environment.py keypoint_proposal.py constraint_generation.py ik_solver.py subgoal_solver.py path_solver.py configs/config.yaml configs/og_scene_file_pen.json; do
 if [ -f "$f" ]; then echo "[FOUND] $f"; else echo "[MISSING] $f"; fi
done > "$OUT/core_files.txt"
{
 echo "=== CONFIG FILE NAMES ==="; find configs -maxdepth 2 -type f -printf '%P\n' 2>/dev/null || true
 echo "=== CACHED PROGRAM FILE NAMES ==="; find vlm_query -maxdepth 3 -type f -printf '%P\n' 2>/dev/null || true
} > "$OUT/file_manifest.txt"
# Do not export user environment variables / credentials.
echo "Preflight saved in: $OUT"
