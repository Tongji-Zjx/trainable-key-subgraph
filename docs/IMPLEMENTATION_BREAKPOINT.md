# Implementation breakpoint

Status: item 2 completed on 2026-07-14; paused before item 3.

Final data decision:

- spatial coordinates have been removed from the model and data adapter;
- coordinate validity is no longer an inclusion criterion;
- no coordinate mapping, imputation, or node-index guessing is performed;
- raw and neighbor spatial coordinates are not model features;
- a new sample index must be used for the expanded cohort; the old 307-sample
  index, splits, checkpoints, and protocol remain historical artifacts only.

Completed and tested:

- rebuilt coordinate-independent sample index: 938 of 939 samples included
  (class 0: 582; class 1: 356), with only the known community/empty-graph
  anomaly excluded;
- all 938 included samples pass the Dataset payload adapter, covering 33,562
  timepoints without truncation or padding;
- sample inspection, index, exclusions, and reproducible group-aware splits;
- frozen data protocol and variable-length list-based Dataset/DataLoader;
- temporal node alignment and signed node/edge features;
- soft node/edge scoring, signed soft graph encoder, classifier, and budget loss;
- training/validation, checkpoint save/load/resume, and frozen evaluation entry point;
- frozen hard candidate extraction, deduplication, Top-K masks, and JSON export;
- Random, Top-degree, and Low-score controls;
- signed structural metrics, sample aggregation, Mann-Whitney U, BH-FDR,
  discrepancy/effect sizes, and visualization;
- 28 unit tests plus local real-data smoke checks.

Last completed check:

- one-batch validation evaluation of `outputs/smoke_training/best_checkpoint.pt`;
- output: `outputs/smoke_training/validation_evaluation.json`;
- this is debug-only and is not a formal experimental result.

After removing coordinates, all 28 unit tests pass and the expanded 938-sample
index passes a complete payload-adaptation scan.

Item 3 (the requested all-sample training and extraction workflow) has not yet
been implemented. Do not use the historical 307-sample protocol for a new run.
