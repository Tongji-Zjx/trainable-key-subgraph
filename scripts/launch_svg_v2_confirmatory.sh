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
    echo "===== START SVG-v2 confirmatory candidate=$CANDIDATE fold=$FOLD seed=$SEED =====" | tee -a "$LOG_FILE"
    "$PYTHON_BIN" -u scripts/run_svg_v2_five_day_fold.py \
      --source-crossfit-root "$SOURCE_ROOT" \
      --output-root "$OUTPUT_ROOT" \
      --fold "$FOLD" \
      --candidates "$CANDIDATE" \
      --mode confirmatory \
      --device "$DEVICE" \
      --seed "$SEED" \
      --model-epochs 60 \
      --num-workers 0 \
      2>&1 | tee -a "$LOG_FILE"
    echo "===== FINISH SVG-v2 confirmatory candidate=$CANDIDATE fold=$FOLD seed=$SEED =====" | tee -a "$LOG_FILE"
  done
done

printf '%s\n' "$CANDIDATE" > "$OUTPUT_ROOT/FROZEN_CANDIDATE"
date -Is > "$OUTPUT_ROOT/CONFIRMATORY_COMPLETE"

