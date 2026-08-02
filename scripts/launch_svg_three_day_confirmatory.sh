#!/usr/bin/env bash

set -euo pipefail

if [[ "$#" -ne 4 ]]; then
  echo "usage: $0 SOURCE_ROOT OUTPUT_ROOT LOG_ROOT FROZEN_CANDIDATE" >&2
  exit 2
fi

SOURCE_ROOT="$1"
OUTPUT_ROOT="$2"
LOG_ROOT="$3"
CANDIDATE="$4"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"

cd "$PROJECT_ROOT"
mkdir -p "$OUTPUT_ROOT" "$LOG_ROOT"

for SEED in 42 43 44; do
  for FOLD in 0 1 2; do
    LOG_FILE="$LOG_ROOT/${CANDIDATE}_fold${FOLD}_seed${SEED}.log"
    "$PYTHON_BIN" -u scripts/run_svg_three_day_fold.py \
      --source-crossfit-root "$SOURCE_ROOT" \
      --output-root "$OUTPUT_ROOT" \
      --fold "$FOLD" \
      --candidates BASELINE "$CANDIDATE" \
      --mode confirmatory \
      --device "$DEVICE" \
      --seed "$SEED" \
      --selection-seed 42 \
      --model-epochs 60 \
      --num-workers 0 \
      2>&1 | tee -a "$LOG_FILE"
  done
done

"$PYTHON_BIN" -u scripts/summarize_svg_three_day_confirmatory.py \
  --source-crossfit-root "$SOURCE_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --candidates BASELINE "$CANDIDATE" \
  --seeds 42 43 44

printf '%s\n' "$CANDIDATE" > "$OUTPUT_ROOT/FROZEN_CANDIDATE"
date -Is > "$OUTPUT_ROOT/CONFIRMATORY_COMPLETE"
