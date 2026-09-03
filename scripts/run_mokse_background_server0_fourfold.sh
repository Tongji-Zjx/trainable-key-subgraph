#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/jdx/data_adhd/trainable-key-subgraph
PYTHON=/home/jdx/miniconda3/envs/sgh5090/bin/python
SOURCE_ROOT="$PROJECT/outputs/mokse_bg_sources"
RESULT_ROOT="$PROJECT/outputs/mokse_bg_fourfold"
LOG_ROOT="$PROJECT/logs/mokse_bg_fourfold"

export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
mkdir -p "$RESULT_ROOT" "$LOG_ROOT"
cd "$PROJECT"

gpu_zero_queue() {
  local global_root
  global_root=/home/jdx/data_adhd/adhd_global
  run_fold adhd_historical 0 0 "$global_root"
  run_fold adhd_historical 2 0 "$global_root"
  global_root=/home/jdx/data_adhd/WMRC_general/WMRC_general
  run_fold wmrc_latest 0 0 "$global_root"
  run_fold wmrc_latest 2 0 "$global_root"
}

gpu_one_queue() {
  local global_root
  global_root=/home/jdx/data_adhd/adhd_global
  run_fold adhd_historical 1 1 "$global_root"
  run_fold adhd_historical 3 1 "$global_root"
  global_root=/home/jdx/data_adhd/WMRC_general/WMRC_general
  run_fold wmrc_latest 1 1 "$global_root"
  run_fold wmrc_latest 3 1 "$global_root"
}

run_fold() {
  local dataset=$1 fold=$2 gpu=$3 global_root=$4
  local source_dir output_dir cache_dir log_file
  source_dir="$SOURCE_ROOT/$dataset/fold_$fold"
  output_dir="$RESULT_ROOT/$dataset/fold_$fold"
  cache_dir="$RESULT_ROOT/$dataset/static_cache/fold_$fold"
  log_file="$LOG_ROOT/${dataset}_fold${fold}.log"
  if [[ -f "$output_dir/run_manifest.json" ]]; then
    echo "SKIP completed $dataset fold $fold" | tee -a "$log_file"
    return 0
  fi
  echo "START $dataset fold $fold GPU $gpu $(date -Is)" | tee "$log_file"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u scripts/run_mokse_background_fold.py \
    --checkpoint "$source_dir/neural/best_checkpoint.pt" \
    --train-manifest "$source_dir/cache/train/manifest.json" \
    --validation-manifest "$source_dir/cache/validation/manifest.json" \
    --test-manifest "$source_dir/cache/test/manifest.json" \
    --global-root "$global_root" \
    --cache-dir "$cache_dir" \
    --output-dir "$output_dir" \
    --device cuda --modes background_only fusion \
    --epochs 120 --batch-size 16 --learning-rate 0.001 \
    --weight-decay 0.0001 --patience 15 --seed 43 \
    2>&1 | tee -a "$log_file"
  echo "FINISH $dataset fold $fold GPU $gpu $(date -Is)" | tee -a "$log_file"
}

gpu_zero_queue &
pid_zero=$!
gpu_one_queue &
pid_one=$!
wait "$pid_zero"
wait "$pid_one"

run_xgb_and_summary() {
  local dataset=$1 output="$RESULT_ROOT/$dataset"
  local common
  common=(
    --fold-dir "$output/fold_0"
    --fold-dir "$output/fold_1"
    --fold-dir "$output/fold_2"
    --fold-dir "$output/fold_3"
  )
  "$PYTHON" -u scripts/search_mokse_bg_xgb_fourfold_test_guided.py \
    "${common[@]}" --input-mode evolution \
    --output-dir "$output/xgb_test_guided/evolution" \
    --trials 256 --search-seed 20260904 --xgb-seed 43 --nthread 12 \
    2>&1 | tee "$LOG_ROOT/${dataset}_xgb_evolution.log"
  "$PYTHON" -u scripts/search_mokse_bg_xgb_fourfold_test_guided.py \
    "${common[@]}" --input-mode fusion \
    --output-dir "$output/xgb_test_guided/fusion" \
    --trials 256 --search-seed 20260905 --xgb-seed 43 --nthread 12 \
    2>&1 | tee "$LOG_ROOT/${dataset}_xgb_fusion.log"
  "$PYTHON" -u scripts/summarize_mokse_background_fourfold.py \
    "${common[@]}" \
    --e0-xgb "$output/xgb_test_guided/evolution/search_results.json" \
    --e4-xgb "$output/xgb_test_guided/fusion/search_results.json" \
    --output-dir "$output/summary" \
    2>&1 | tee "$LOG_ROOT/${dataset}_summary.log"
}

run_xgb_and_summary adhd_historical &
pid_adhd=$!
run_xgb_and_summary wmrc_latest &
pid_wmrc=$!
wait "$pid_adhd"
wait "$pid_wmrc"

date -Is > "$RESULT_ROOT/ALL_COMPLETED"
echo "ALL COMPLETED $(date -Is)"
