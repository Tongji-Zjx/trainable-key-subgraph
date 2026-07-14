# Implementation breakpoint

Status: resumed and closed on 2026-07-14.

Final data decision:

- this implementation uses only the 307 coordinate-valid samples;
- the 632 all-zero-coordinate samples remain excluded;
- no coordinate mapping, imputation, or node-index guessing is performed;
- the frozen sample index, splits, and data protocol remain unchanged.

Completed and tested:

- sample inspection, index, exclusions, and reproducible group-aware splits;
- frozen data protocol and variable-length list-based Dataset/DataLoader;
- temporal node alignment and signed node/edge features;
- soft node/edge scoring, signed soft graph encoder, classifier, and budget loss;
- training/validation, checkpoint save/load/resume, and frozen evaluation entry point;
- frozen hard candidate extraction, deduplication, Top-K masks, and JSON export;
- Random, Top-degree, and Low-score controls;
- signed structural metrics, sample aggregation, Mann-Whitney U, BH-FDR,
  discrepancy/effect sizes, and visualization;
- 25 unit tests plus local real-data smoke checks.

Last completed check:

- one-batch validation evaluation of `outputs/smoke_training/best_checkpoint.pt`;
- output: `outputs/smoke_training/validation_evaluation.json`;
- this is debug-only and is not a formal experimental result.

After resuming, all 25 unit tests, the 307-sample loading scan, real feature and
backward checks, and the validation-only integration smoke pipeline were rerun.

Formal GPU training and held-out test analysis are intentionally not performed
locally. Follow `docs/PROJECT_RUN_GUIDE.md` on the server.
