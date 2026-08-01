#!/usr/bin/env bash

set -euo pipefail

if [[ "$#" -lt 4 ]]; then
  echo "usage: $0 SOURCE_ROOT OUTPUT_ROOT LOG_ROOT CANDIDATE [CANDIDATE ...]" >&2
  exit 2
fi

SOURCE_ROOT="$1"
OUTPUT_ROOT="$2"
LOG_ROOT="$3"
shift 3
CANDIDATES=("$@")
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"

cd "$PROJECT_ROOT"
mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"

for FOLD in 0 1; do
  LOG_FILE="$LOG_ROOT/fold${FOLD}_seed42.log"
  echo "===== START SVG-v2 combinations fold=$FOLD candidates=${CANDIDATES[*]} =====" | tee -a "$LOG_FILE"
  "$PYTHON_BIN" -u scripts/run_svg_v2_five_day_fold.py \
    --source-crossfit-root "$SOURCE_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --fold "$FOLD" \
    --candidates "${CANDIDATES[@]}" \
    --mode screen \
    --device "$DEVICE" \
    --seed 42 \
    --model-epochs 60 \
    --num-workers 0 \
    2>&1 | tee -a "$LOG_FILE"
  echo "===== FINISH SVG-v2 combinations fold=$FOLD =====" | tee -a "$LOG_FILE"
done

printf '%s\n' "${CANDIDATES[@]}" > "$OUTPUT_ROOT/COMBINATION_CANDIDATES"
date -Is > "$OUTPUT_ROOT/COMBINATION_SCREEN_COMPLETE"

