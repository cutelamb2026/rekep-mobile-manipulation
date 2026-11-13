#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 6 ];then
 echo 'Usage: bash 10_run_custom_task.sh REKEP_ROOT EVIDENCE_DIR SCENARIO SCENE_FILE PROGRAM_DIR INSTRUCTION' >&2
 exit 2
fi
ROOT=$(realpath "$1"); EV="$2";SCEN="$3";SCENE=$(realpath "$4");PROGRAM=$(realpath "$5");INSTR="$6"
[ -f "$SCENE" ] || { echo "Missing scene: $SCENE" >&2;exit 2; }
[ -f "$PROGRAM/metadata.json" ] || { echo "Missing cached/new program metadata in: $PROGRAM" >&2;exit 2; }
case "$SCEN" in logistics|retail) ;; *)echo 'scenario must be logistics|retail' >&2;exit 2;;esac
RUN_ID="${SCEN}_$(date -u +%Y%m%dT%H%M%SZ)_$$"
DIR="$EV/04_scenarios/$SCEN/$RUN_ID";mkdir -p "$DIR"
cd "$ROOT"
printf 'Run custom scene=%s program_dir=%s\n' "$SCENE" "$PROGRAM" > "$DIR/command.txt"
set +e
set -o pipefail
python - "$SCENE" "$PROGRAM" "$INSTR" <<'PY' 2>&1 | tee "$DIR/run.log"
import sys
from main import Main
scene,program,instruction=sys.argv[1:]
main=Main(scene_file=scene,visualize=False)
main.perform_task(instruction,rekep_program_dir=program)
PY
RC=${PIPESTATUS[0]}
set -e
echo "$RC" > "$DIR/exit_code.txt"
echo "run_id=$RUN_ID; exit=$RC; log=$DIR/run.log"
exit "$RC"
