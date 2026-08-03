#!/usr/bin/env bash
set -euo pipefail

# Run the corrected neuralized S/V experiment on an existing 3-fold cross-fit
# root, then fit validation-only F0 fusion weights against a frozen short-term
# branch and summarize paired outer-test predictions.
#
# Required environment variables:
#   SOURCE_CROSSFIT_ROOT, OUTPUT_ROOT, DATASET_NAME, SHORT_TERM_SEED
# Optional:
#   MODEL_SEED=42, DEVICE=cuda, EPOCHS=80, BATCH_SIZE=4,
#   GRADIENT_ACCUMULATION_STEPS=2, GPU_WAIT_MIB=22000,
#   GPU_WAIT_SECONDS=600, PYTHON_BIN=python, TRAIN_MISSING_SELECTOR=1,
#   SELECTOR_EPOCHS=80, SELECTOR_NUM_WORKERS=2

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

: "${SOURCE_CROSSFIT_ROOT:?SOURCE_CROSSFIT_ROOT is required}"
: "${OUTPUT_ROOT:?OUTPUT_ROOT is required}"
: "${DATASET_NAME:?DATASET_NAME is required}"
: "${SHORT_TERM_SEED:?SHORT_TERM_SEED is required}"

MODEL_SEED="${MODEL_SEED:-42}"
DEVICE="${DEVICE:-cuda}"
EPOCHS="${EPOCHS:-80}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
GPU_WAIT_MIB="${GPU_WAIT_MIB:-22000}"
GPU_WAIT_SECONDS="${GPU_WAIT_SECONDS:-600}"
GPU_INDEX="${GPU_INDEX:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TRAIN_MISSING_SELECTOR="${TRAIN_MISSING_SELECTOR:-1}"
SELECTOR_EPOCHS="${SELECTOR_EPOCHS:-80}"
SELECTOR_NUM_WORKERS="${SELECTOR_NUM_WORKERS:-2}"
LOG_ROOT="${LOG_ROOT:-logs/neuralized_sv/${DATASET_NAME}_seed${MODEL_SEED}}"
FUSION_ROOT="${FUSION_ROOT:-${OUTPUT_ROOT}/fusion}"

mkdir -p "$OUTPUT_ROOT" "$FUSION_ROOT" "$LOG_ROOT"

wait_for_gpu() {
  if [[ "$DEVICE" != cuda* ]]; then
    return
  fi
  while true; do
    local free_mib
    free_mib="$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
      -i "$GPU_INDEX" | tr -d ' ')"
    if [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib >= GPU_WAIT_MIB )); then
      echo "GPU ${GPU_INDEX} ready: ${free_mib} MiB free"
      return
    fi
    echo "GPU ${GPU_INDEX} waiting: ${free_mib:-unknown} MiB free; need ${GPU_WAIT_MIB} MiB"
    sleep "$GPU_WAIT_SECONDS"
  done
}

export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

for fold in 0 1 2; do
  selector_checkpoint="$SOURCE_CROSSFIT_ROOT/fold_${fold}/selector/best_checkpoint.pt"
  if [[ ! -f "$selector_checkpoint" ]]; then
    if [[ "$TRAIN_MISSING_SELECTOR" != "1" ]]; then
      echo "Missing selector checkpoint: $selector_checkpoint" >&2
      exit 1
    fi
    wait_for_gpu
    echo "START selector fold ${fold}"
    "$PYTHON_BIN" -u scripts/train_dual_selector.py \
      --protocol "$SOURCE_CROSSFIT_ROOT/fold_${fold}/protocol/data_protocol.json" \
      --output-dir "$SOURCE_CROSSFIT_ROOT/fold_${fold}/selector" \
      --device "$DEVICE" \
      --epochs "$SELECTOR_EPOCHS" \
      --batch-size 1 \
      --num-workers "$SELECTOR_NUM_WORKERS" \
      --seed "$MODEL_SEED" \
      --learning-rate 0.001 \
      --weight-decay 0.0001 \
      --gradient-clip 1.0 \
      --early-stopping-patience 15 \
      --selector-objective current \
      2>&1 | tee "$LOG_ROOT/selector_fold_${fold}.log"
    if [[ ! -f "$selector_checkpoint" ]]; then
      echo "Selector training created no checkpoint: $selector_checkpoint" >&2
      exit 1
    fi
    echo "FINISH selector fold ${fold}"
  fi
  completion="$OUTPUT_ROOT/fold_${fold}/models/NSV_safe_residual_seed${MODEL_SEED}/test_evaluation.json"
  if [[ -f "$completion" ]]; then
    echo "SKIP neuralized S/V fold ${fold}: ${completion} exists"
    continue
  fi
  wait_for_gpu
  "$PYTHON_BIN" -u scripts/run_neuralized_sv_fold.py \
    --crossfit-root "$SOURCE_CROSSFIT_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --fold "$fold" \
    --device "$DEVICE" \
    --seed "$MODEL_SEED" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH_SIZE" \
    --gradient-accumulation-steps "$GRADIENT_ACCUMULATION_STEPS" \
    2>&1 | tee "$LOG_ROOT/fold_${fold}.log"
done

for variant in NS_static_spectral NV_dynamic_evolution NSV_safe_residual; do
  for fold in 0 1 2; do
    "$PYTHON_BIN" -u scripts/run_neuralized_sv_short_term_fusion_fold.py \
      --source-crossfit-root "$SOURCE_CROSSFIT_ROOT" \
      --neuralized-root "$OUTPUT_ROOT" \
      --output-root "$FUSION_ROOT" \
      --fold "$fold" \
      --variant "$variant" \
      --short-term-seed "$SHORT_TERM_SEED" \
      --neural-seed "$MODEL_SEED" \
      2>&1 | tee "$LOG_ROOT/fusion_${variant}_fold_${fold}.log"
  done
done

"$PYTHON_BIN" -u scripts/summarize_neuralized_sv_short_term.py \
  --source-crossfit-root "$SOURCE_CROSSFIT_ROOT" \
  --fusion-root "$FUSION_ROOT" \
  --fold-assignments "$SOURCE_CROSSFIT_ROOT/assignments/fold_assignments.json" \
  --output-dir "$OUTPUT_ROOT/summary" \
  --dataset "$DATASET_NAME" \
  --short-term-seed "$SHORT_TERM_SEED" \
  --neural-seed "$MODEL_SEED" \
  --bootstrap-repeats 10000 \
  --bootstrap-seed 20260803 \
  2>&1 | tee "$LOG_ROOT/summary.log"

echo "COMPLETE: $OUTPUT_ROOT/summary/summary.md"
