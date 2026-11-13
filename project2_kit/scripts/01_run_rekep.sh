#!/usr/bin/env bash
set -euo pipefail
if [ "$#" -ne 3 ]; then echo "Usage: bash 01_run_rekep.sh REKEP_ROOT EVIDENCE_DIR baseline|disturbance" >&2; exit 2; fi
ROOT=$(realpath "$1")
EVIDENCE="$2"
MODE="$3"
case "$MODE" in
 baseline) LABEL=01_baseline; CMD=(python main.py --use_cached_query);;
 disturbance) LABEL=03_disturbance; CMD=(python main.py --use_cached_query --apply_disturbance);;
 *) echo 'Mode must be baseline or disturbance' >&2; exit 2;;
esac
cd "$ROOT"
[ -f main.py ] || { echo "main.py not found in $ROOT" >&2; exit 2; }
RUN_ID="${MODE}_$(date -u +%Y%m%dT%H%M%SZ)_$$"
DIR="$EVIDENCE/$LABEL/$RUN_ID"
mkdir -p "$DIR"
MARKER="$DIR/before_run.marker"
touch "$MARKER"
printf '%q ' "${CMD[@]}" > "$DIR/command.txt"
printf '\n' >> "$DIR/command.txt"
echo "RUN_ID=$RUN_ID"
echo "Running in $ROOT, output: $DIR"
START=$(date -u +%Y-%m-%dT%H:%M:%SZ)
set +e
set -o pipefail
"${CMD[@]}" 2>&1 | tee "$DIR/run.log"
RC=${PIPESTATUS[0]}
set -e
END=$(date -u +%Y-%m-%dT%H:%M:%SZ)
echo "$RC" > "$DIR/exit_code.txt"
find videos -type f -name '*.mp4' -newer "$MARKER" -print 2>/dev/null > "$DIR/new_videos.txt" || true
export P2_RUN_ID="$RUN_ID" P2_START="$START" P2_END="$END" P2_RC="$RC" P2_ROOT="$ROOT" P2_DIR="$DIR" P2_MODE="$MODE"
python - <<'PY'
import json,os,pathlib
pathlib.Path(os.environ['P2_DIR'],'run_manifest.json').write_text(json.dumps({
 'run_id':os.environ['P2_RUN_ID'], 'start_utc':os.environ['P2_START'],
 'end_utc':os.environ['P2_END'], 'exit_code':int(os.environ['P2_RC']),
 'repo_path':os.environ['P2_ROOT'], 'mode':os.environ['P2_MODE'],
 'success':None, 'note':'Success requires final state review; video saved != success'
},ensure_ascii=False,indent=2),encoding='utf-8')
PY
echo "Finished exit=$RC; inspect $DIR/new_videos.txt and $DIR/run.log"
exit "$RC"
