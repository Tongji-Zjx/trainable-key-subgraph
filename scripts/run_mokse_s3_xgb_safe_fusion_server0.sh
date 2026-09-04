#!/usr/bin/env bash
set -euo pipefail

cd /home/jdx/data_adhd/trainable-key-subgraph
PYTHON=/home/jdx/miniconda3/envs/sgh5090/bin/python
ROOT=outputs/mokse_bg_safe_s0s3_v1
RUN_ROOT="$ROOT/s3_xgb_safe_fusion_test_guided_v1"
mkdir -p "$RUN_ROOT/logs"

run_dataset() {
  local dataset="$1"
  local subgraph_root="$2"
  local output="$RUN_ROOT/$dataset"
  "$PYTHON" -u scripts/search_mokse_s3_xgb_safe_fusion.py \
    --dataset "$dataset" \
    --static-fold-dir "$ROOT/$dataset/fold_0/s3" \
    --static-fold-dir "$ROOT/$dataset/fold_1/s3" \
    --static-fold-dir "$ROOT/$dataset/fold_2/s3" \
    --static-fold-dir "$ROOT/$dataset/fold_3/s3" \
    --subgraph-prediction-dir "$subgraph_root/fold_0" \
    --subgraph-prediction-dir "$subgraph_root/fold_1" \
    --subgraph-prediction-dir "$subgraph_root/fold_2" \
    --subgraph-prediction-dir "$subgraph_root/fold_3" \
    --output-dir "$output" \
    --trials 96 \
    --search-seed 20260905 \
    --xgb-seed 43 \
    --nthread 12 \
    2>&1 | tee "$RUN_ROOT/logs/${dataset}.log"
}

run_dataset \
  adhd \
  "$ROOT/adhd/frozen_subgraph_historical_best_predictions" &
ADHD_PID=$!

run_dataset \
  wmrc \
  "$ROOT/wmrc/frozen_subgraph_final_predictions" &
WMRC_PID=$!

echo "ADHD worker PID: $ADHD_PID"
echo "WMRC worker PID: $WMRC_PID"

wait "$ADHD_PID"
wait "$WMRC_PID"

echo "S3-XGB safe-fusion searches completed."
