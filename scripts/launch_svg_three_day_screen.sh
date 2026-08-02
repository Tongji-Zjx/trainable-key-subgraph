#!/usr/bin/env bash

set -euo pipefail

if [[ "$#" -ne 3 ]]; then
  echo "usage: $0 SOURCE_ROOT OUTPUT_ROOT LOG_ROOT" >&2
  exit 2
fi

SOURCE_ROOT="$1"
OUTPUT_ROOT="$2"
LOG_ROOT="$3"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
INCLUDE_E1="${INCLUDE_E1:-1}"

CANDIDATES=(D1 H1)
if [[ "$INCLUDE_E1" == "1" ]]; then
  CANDIDATES+=(E1)
fi

cd "$PROJECT_ROOT"
mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"

for FOLD in 0 1; do
  LOG_FILE="$LOG_ROOT/screen_fold${FOLD}_seed42.log"
  "$PYTHON_BIN" -u scripts/run_svg_three_day_fold.py \
    --source-crossfit-root "$SOURCE_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --fold "$FOLD" \
    --candidates "${CANDIDATES[@]}" \
    --mode screen \
    --device "$DEVICE" \
    --seed 42 \
    --selection-seed 42 \
    --model-epochs 60 \
    --num-workers 0 \
    2>&1 | tee -a "$LOG_FILE"
done

date -Is > "$OUTPUT_ROOT/SCREEN_COMPLETE"
