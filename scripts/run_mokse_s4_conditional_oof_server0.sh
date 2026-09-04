#!/usr/bin/env bash
set -euo pipefail

# Conditional development OOF for the downstream C4/XGB readout and S4
# background branch. The selector and trajectory caches remain frozen.

PROJECT_ROOT="${PROJECT_ROOT:-/home/jdx/data_adhd/trainable-key-subgraph}"
PYTHON_BIN="${PYTHON_BIN:-/home/jdx/miniconda3/envs/sgh5090/bin/python}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/tmp/jdx_mokse_s4_conditional_oof_v1}"
RESULT_ROOT="${RESULT_ROOT:-$PROJECT_ROOT/outputs/mokse_bg_s4_test_guided_v1/fusion_conditional_oof}"
LOG_ROOT="${LOG_ROOT:-$PROJECT_ROOT/logs/mokse_bg_s4_test_guided_v1/fusion_conditional_oof}"

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$SCRATCH_ROOT" "$RESULT_ROOT" "$LOG_ROOT"

run_dataset() {
  local dataset="$1" gpu="$2" source_name="$3" global_root="$4"
  local tge_config="$5" xgb_config="$6" subgraph_name="$7"
  local fold
  export CUDA_VISIBLE_DEVICES="$gpu"
  for fold in 0 1 2 3; do
    "$PYTHON_BIN" -u scripts/run_mokse_s4_conditional_oof_fold.py \
      --dataset "$dataset" \
      --fold "$fold" \
      --source-fold-dir "outputs/mokse_bg_sources/$source_name/fold_$fold" \
      --tge-config "$tge_config" \
      --xgb-config "$xgb_config" \
      --global-root "$global_root" \
      --static-source-root "outputs/mokse_bg_s4_test_guided_v1/$dataset" \
      --subgraph-test-dir \
        "outputs/mokse_bg_safe_s0s3_v1/$dataset/$subgraph_name/fold_$fold" \
      --feature-cache-dir \
        "outputs/mokse_bg_s4_test_guided_v1/cache/$dataset/lane_0" \
      --output-dir "$SCRATCH_ROOT/$dataset/fold_$fold" \
      --device cuda \
      2>&1 | tee -a "$LOG_ROOT/${dataset}_fold${fold}.log"
  done
  touch "$SCRATCH_ROOT/$dataset/FOLDS_COMPLETE"
}

run_dataset \
  adhd 0 adhd_historical_best /home/jdx/data_adhd/adhd_global \
  configs/tge_gnn_c4_rank_xgb_a_neural_seed43_v1.json \
  configs/adhd_c4_rank_xgb_fixed_conditional_oof_v1.json \
  frozen_subgraph_historical_best_predictions \
  > "$LOG_ROOT/adhd_worker.log" 2>&1 &
adhd_pid=$!

run_dataset \
  wmrc 1 wmrc_latest /home/jdx/data_adhd/WMRC_general/WMRC_general \
  configs/wmrc_c4_rank_neural_seed42_conditional_oof_v1.json \
  configs/wmrc_c4_rank_xgb_fixed_conditional_oof_v1.json \
  frozen_subgraph_final_predictions \
  > "$LOG_ROOT/wmrc_worker.log" 2>&1 &
wmrc_pid=$!

wait "$adhd_pid"
wait "$wmrc_pid"

run_selection() {
  local dataset="$1"
  local seed_args=() subgraph_args=() fold seed
  for fold in 0 1 2 3; do
    for seed in 43 44 45; do
      seed_args+=(
        --seed-fold-dir
        "$fold:$seed:$SCRATCH_ROOT/$dataset/fold_$fold/static/seed_$seed"
      )
    done
    subgraph_args+=(
      --subgraph-prediction-dir
      "$SCRATCH_ROOT/$dataset/fold_$fold/subgraph/predictions"
    )
  done
  "$PYTHON_BIN" -u scripts/select_mokse_s4_anchored_fusion.py \
    --dataset "$dataset" \
    --selection-role development_oof \
    "${seed_args[@]}" \
    "${subgraph_args[@]}" \
    --minimum-mean-auc-gain 0.0 \
    --output-dir "$RESULT_ROOT/$dataset" \
    --evaluate-test \
    2>&1 | tee "$LOG_ROOT/${dataset}_selection.log"
}

run_selection adhd
run_selection wmrc
touch "$RESULT_ROOT/CONDITIONAL_OOF_COMPLETE"
echo "MOKSE_S4_CONDITIONAL_OOF_COMPLETE"
