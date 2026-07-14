# Key-subgraph project run guide

## Current workflow: all-sample exploratory analysis

The active workflow uses 938 valid samples (class 0: 582; class 1: 356). All
samples train the intermediate extractor and are then passed through hard
subgraph extraction and structural difference analysis. This is an explicitly
in-sample exploratory design: its classifier metrics are optimization
diagnostics, not estimates of generalization performance.

Prepare and freeze the expanded cohort:

```bash
python scripts/build_sample_index.py \
  --data-root data/adhd_5_0.5 \
  --output-dir outputs/index_no_coords

python scripts/prepare_all_sample_protocol.py
```

Preflight checks:

```bash
python -m unittest discover -s tests -v
python scripts/check_data_loading.py \
  --protocol configs/data_protocol_all_samples.json \
  --batch-size 16 --full-scan
python scripts/check_feature_construction.py \
  --protocol configs/data_protocol_all_samples.json
python scripts/check_model_flow.py \
  --protocol configs/data_protocol_all_samples.json --device cuda
```

Expected results are 31 passing tests, one `all` partition containing 938
samples, node/edge feature dimensions 9/23, and nonzero gradients for both
scorers.

Formal training command:

```bash
mkdir -p logs
set -o pipefail
python -u scripts/train_all_samples.py \
  --protocol configs/data_protocol_all_samples.json \
  --device cuda \
  --epochs 50 \
  --batch-size 1 \
  --num-workers 4 \
  --seed 42 \
  --learning-rate 0.001 \
  --weight-decay 0.0001 \
  --target-node-ratio 0.30 \
  --target-edge-ratio 0.30 \
  --budget-weight 1.0 \
  --gradient-clip 5.0 \
  --output-dir outputs/training/all_samples_seed42 \
  2>&1 | tee logs/all_samples_seed42.log
```

The best checkpoint is selected by the lowest inference-mode loss on the full
cohort. History uses the key `cohort`, not `validation`. Progress is flushed
after every epoch. Resume with the same command plus:

```bash
--resume outputs/training/all_samples_seed42/last_checkpoint.pt
```

Export hard subgraphs for every sample:

```bash
python scripts/export_hard_subgraphs.py \
  --protocol configs/data_protocol_all_samples.json \
  --checkpoint outputs/training/all_samples_seed42/best_checkpoint.pt \
  --split all \
  --device cuda \
  --output-dir outputs/hard_subgraphs/all_samples_seed42
```

Run structural statistics, controls, tables, and figures:

```bash
python scripts/analyze_structural_differences.py \
  --protocol configs/data_protocol_all_samples.json \
  --export-dir outputs/hard_subgraphs/all_samples_seed42 \
  --split all \
  --output-dir outputs/structural_analysis/all_samples_seed42 \
  --random-repeats 100 \
  --seed 42
```

All resulting reports must state that the analysis is exploratory and
in-sample. It may describe structural differences in the analyzed cohort, but
must not claim held-out predictive performance or external generalization.

## 1. Historical frozen scope and migration warning

The earlier experiment used the 307 samples recorded in:

- `outputs/index/sample_index.csv`
- `outputs/splits/splits.csv`
- `configs/data_protocol.json`

This 307-sample protocol describes the earlier coordinate-filtered experiment.
It must not be reused for a new expanded-cohort experiment. Coordinate validity
is no longer an inclusion criterion; a separate rebuilt index is required.
The community-anomalous sample remains excluded.

The frozen split is:

| split | total | class 0 | class 1 |
|---|---:|---:|---:|
| train | 215 | 139 | 76 |
| validation | 46 | 30 | 16 |
| test | 46 | 30 | 16 |

The edge rule is fixed before training:

```text
edge exists iff abs(A_ij) > 0.0 and i != j
```

Positive and negative edges are both valid. The `.pt` `global_threshold` field
is retained as source metadata and is not re-estimated during training or
held-out analysis.

## 2. Server environment

Create the server environment from the committed specification:

```bash
conda env create -f environment_server.yml
conda activate web
python -m pip install -e . --no-deps
```

Required compatibility:

- Python 3.7
- PyTorch 1.13.1
- CUDA 11.7

Do not install PyTorch Geometric for this baseline. It uses list-based batches
and has no PyG dependency.

## 3. Preflight checks

Run these before any formal experiment:

```bash
python scripts/freeze_data_protocol.py
python -m unittest discover -s tests -v
python scripts/check_data_loading.py --batch-size 16 --full-scan
python scripts/check_feature_construction.py --samples-per-split 1
python scripts/check_model_flow.py --device cuda
```

Expected high-level results:

- protocol hashes are valid and reused;
- 31 unit tests pass in the current codebase;
- train/validation/test contain 215/46/46 samples;
- all 307 samples load without truncation;
- node feature dimension is 9 (spatial coordinates are excluded);
- edge feature dimension is 23;
- node and edge scorers both receive nonzero gradients.

If any protocol hash differs, stop. Do not overwrite the protocol or split as
part of a training run.

## 4. Formal training

Example baseline command:

```bash
python scripts/train_soft_extractor.py \
  --device cuda \
  --epochs 50 \
  --batch-size 1 \
  --num-workers 4 \
  --seed 42 \
  --learning-rate 0.001 \
  --weight-decay 0.0001 \
  --target-node-ratio 0.30 \
  --target-edge-ratio 0.30 \
  --budget-weight 1.0 \
  --selection-metric roc_auc \
  --output-dir outputs/training/formal_seed42
```

The default training path is `soft_graph`. Hard Top-q, L-hop, node/edge
compression, and Top-K are not part of the differentiable training path.

Training writes:

```text
outputs/training/formal_seed42/
├── best_checkpoint.pt
├── last_checkpoint.pt
└── history.json
```

The best checkpoint is selected only with validation metrics. The training
loop never creates a new split and never accesses the test DataLoader.

To resume an interrupted run, keep the same total configuration and use:

```bash
python scripts/train_soft_extractor.py \
  --device cuda \
  --epochs 50 \
  --batch-size 1 \
  --num-workers 4 \
  --seed 42 \
  --output-dir outputs/training/formal_seed42 \
  --resume outputs/training/formal_seed42/last_checkpoint.pt
```

`--smoke` is debug-only. Never report metrics from a smoke run.

## 5. Validation and model freezing

Inspect the frozen best checkpoint on validation explicitly:

```bash
python scripts/evaluate_checkpoint.py \
  --checkpoint outputs/training/formal_seed42/best_checkpoint.pt \
  --split validation \
  --device cuda \
  --batch-size 1 \
  --num-workers 4
```

Hyperparameters, extraction thresholds, Top-K configuration, and analysis
rules must be finalized after validation and before test evaluation.

## 6. One-time test evaluation

Only after the model and all rules are frozen:

```bash
python scripts/evaluate_checkpoint.py \
  --checkpoint outputs/training/formal_seed42/best_checkpoint.pt \
  --split test \
  --device cuda \
  --batch-size 1 \
  --num-workers 4
```

Do not use test labels to alter the model, budget ratios, extraction settings,
candidate weights, controls, metrics, or significance threshold.

## 7. Hard key-subgraph export

Export held-out key subgraphs from the frozen checkpoint:

```bash
python scripts/export_hard_subgraphs.py \
  --checkpoint outputs/training/formal_seed42/best_checkpoint.pt \
  --split test \
  --output-dir outputs/hard_subgraphs/formal_seed42 \
  --device cuda \
  --seeds-per-community 1 \
  --neighborhood-hops 1 \
  --max-nodes 20 \
  --max-edges 40 \
  --min-nodes 2 \
  --min-edges 1 \
  --top-k 5 \
  --overlap-threshold 1.0
```

Each sample export contains:

- the complete valid candidate pool;
- selected Top-K subgraphs and `subgraph_mask`;
- original node IDs and names;
- original signed edge weights;
- node/edge scores;
- temporal differences and masks;
- checkpoint and data-protocol hashes.

Candidates are never duplicated to fill Top-K.

## 8. Structural difference analysis

Run the same metrics on key subgraphs and all three matched controls:

```bash
python scripts/analyze_structural_differences.py \
  --export-dir outputs/hard_subgraphs/formal_seed42/test \
  --split test \
  --output-dir outputs/structural_analysis/formal_seed42 \
  --random-repeats 100 \
  --seed 42
```

The analysis produces:

```text
subgraph_level_metrics.csv
sample_level_metrics.csv
univariate_test_results.csv
control_group_comparison.csv
analysis_summary.json
figures/boxplots/*.png
figures/discrepancy_heatmap.png
```

The primary statistical unit is the sample, not the individual subgraph.
Missing sign-specific or first-timepoint dynamic metrics remain missing and are
aggregated with per-metric valid masks. They are never replaced by zero.

## 9. Interpretation boundary

The structural module can show that frozen key subgraphs retain stable,
class-related structural differences. It does not establish causality or prove
that the classifier is globally optimal. The statements below apply only to
the historical 307-sample experiment, not to the expanded-cohort workflow.

All formal reports should mention:

- only 307 coordinate-valid samples were analyzed in the historical run;
- site/class imbalance remains a possible confounder;
- test evaluation was performed only after freezing all choices;
- Random, Top-degree, and Low-score controls used the same held-out samples and
  valid timepoints.
