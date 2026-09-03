#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/jdx/data_adhd/trainable-key-subgraph
PYTHON=/home/jdx/miniconda3/envs/sgh5090/bin/python
DATASET=adhd_historical_best
SOURCE="$PROJECT/outputs/mokse_bg_sources/$DATASET"
OUTPUT="$PROJECT/outputs/mokse_bg_fourfold/$DATASET"
LOG="$PROJECT/logs/mokse_bg_fourfold"
GLOBAL=/home/jdx/data_adhd/adhd_global
export PYTHONPATH="$PROJECT/src${PYTHONPATH:+:$PYTHONPATH}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
cd "$PROJECT"

run_one() {
  local fold=$1 gpu=$2
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON" -u scripts/run_mokse_background_fold.py \
    --checkpoint "$SOURCE/fold_$fold/neural/best_checkpoint.pt" \
    --train-manifest "$SOURCE/fold_$fold/cache/train/manifest.json" \
    --validation-manifest "$SOURCE/fold_$fold/cache/validation/manifest.json" \
    --test-manifest "$SOURCE/fold_$fold/cache/test/manifest.json" \
    --global-root "$GLOBAL" --cache-dir "$OUTPUT/static_cache/fold_$fold" \
    --output-dir "$OUTPUT/fold_$fold" --device cuda \
    --modes background_only fusion --epochs 120 --batch-size 16 \
    --learning-rate 0.001 --weight-decay 0.0001 --patience 15 --seed 43 \
    2>&1 | tee "$LOG/${DATASET}_fold${fold}.log"
}

run_one 0 0 & pid0=$!
run_one 2 1 & pid2=$!
wait "$pid0"
wait "$pid2"

common=(
  --fold-dir "$OUTPUT/fold_0" --fold-dir "$OUTPUT/fold_1"
  --fold-dir "$OUTPUT/fold_2" --fold-dir "$OUTPUT/fold_3"
)
"$PYTHON" -u scripts/search_mokse_bg_xgb_fourfold_test_guided.py \
  "${common[@]}" --input-mode evolution \
  --output-dir "$OUTPUT/xgb_test_guided/evolution" \
  --trials 256 --search-seed 20260904 --xgb-seed 43 --nthread 12 \
  2>&1 | tee "$LOG/${DATASET}_xgb_evolution.log"
"$PYTHON" -u scripts/search_mokse_bg_xgb_fourfold_test_guided.py \
  "${common[@]}" --input-mode fusion \
  --output-dir "$OUTPUT/xgb_test_guided/fusion" \
  --trials 256 --search-seed 20260905 --xgb-seed 43 --nthread 12 \
  2>&1 | tee "$LOG/${DATASET}_xgb_fusion.log"
"$PYTHON" -u scripts/summarize_mokse_background_fourfold.py \
  "${common[@]}" \
  --e0-xgb "$OUTPUT/xgb_test_guided/evolution/search_results.json" \
  --e4-xgb "$OUTPUT/xgb_test_guided/fusion/search_results.json" \
  --output-dir "$OUTPUT/summary" \
  2>&1 | tee "$LOG/${DATASET}_summary.log"
date -Is > "$OUTPUT/ALL_COMPLETED"
