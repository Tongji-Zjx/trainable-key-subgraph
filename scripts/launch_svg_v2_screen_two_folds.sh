#!/usr/bin/env bash

set -euo pipefail

if [[ "$#" -lt 3 || "$#" -gt 4 ]]; then
  echo "usage: $0 SOURCE_CROSSFIT_ROOT OUTPUT_ROOT LOG_ROOT [DEVICE]" >&2
  exit 2
fi

SOURCE_CROSSFIT_ROOT="$1"
OUTPUT_ROOT="$2"
LOG_ROOT="$3"
DEVICE="${4:-cuda}"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

cd "$PROJECT_ROOT"
mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"

for FOLD in 0 1; do
  LOG_FILE="$LOG_ROOT/fold${FOLD}_seed42.log"
  echo "===== START SVG-v2 screen fold=$FOLD =====" | tee -a "$LOG_FILE"
  "$PYTHON_BIN" -u scripts/run_svg_v2_five_day_fold.py \
    --source-crossfit-root "$SOURCE_CROSSFIT_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --fold "$FOLD" \
    --candidates A1 B1 C3 F1 G2 \
    --mode screen \
    --device "$DEVICE" \
    --seed 42 \
    --model-epochs 60 \
    --num-workers 0 \
    2>&1 | tee -a "$LOG_FILE"
  echo "===== FINISH SVG-v2 screen fold=$FOLD =====" | tee -a "$LOG_FILE"
done

date -Is > "$OUTPUT_ROOT/SCREEN_COMPLETE"
echo "SVG-v2 two-fold screen complete: $OUTPUT_ROOT"
