# Implementation breakpoint

Status: item 3 completed on 2026-07-14.

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
- immutable `all_samples_exploratory` protocol assigns all 938 samples to the
  explicit `all` partition;
- `train_all_samples.py` trains on all samples and selects the best checkpoint
  by lowest full-cohort inference loss, never labelling it validation loss;
- checkpoint evaluation, hard export, and structural analysis support `all` and
  mark their outputs as exploratory/in-sample;
- sample inspection, index, exclusions, and reproducible group-aware splits;
- frozen data protocol and variable-length list-based Dataset/DataLoader;
- temporal node alignment and signed node/edge features;
- soft node/edge scoring, signed soft graph encoder, classifier, and budget loss;
- training/validation, checkpoint save/load/resume, and frozen evaluation entry point;
- frozen hard candidate extraction, deduplication, Top-K masks, and JSON export;
- Random, Top-degree, and Low-score controls;
- signed structural metrics, sample aggregation, Mann-Whitney U, BH-FDR,
  discrepancy/effect sizes, and visualization;
- 32 unit tests plus local real-data smoke checks.

Last completed checks:

- all 32 unit tests pass (the symlink-only test is skipped on Windows hosts
  without symbolic-link privileges and runs on Linux servers);
- the expanded protocol loads all 938 samples and 33,562 timepoints;
- real-data CPU smoke training writes best/last checkpoints and cohort history;
- hard extraction exported 167 subgraphs from 60 timepoints across two samples;
- the structural module produced CSV tables and figures from those exports.

Smoke outputs are debug-only. The server must run the complete 938-sample
training and export commands in `docs/PROJECT_RUN_GUIDE.md`.
