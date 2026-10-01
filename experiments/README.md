# DTTNet experiment records

This directory contains lightweight, versioned experiment records only:

- `run_history.tsv` records start and completion events emitted by the pipeline.
- `results_summary.csv` records completed test-set cSDR/uSDR evaluations.

The active experiment artifacts remain outside the repository under
`D:\CZJ\experiments\dttnet_drff`. Checkpoints, TensorBoard event files,
per-track audio metrics, fusion-weight dumps, datasets, and virtual environments
are intentionally not committed because they are large or machine-local.

The vocals `dynamic` seed-2022 run was still active when this snapshot was
created; its completion should be appended to `run_history.tsv` and its result
added to `results_summary.csv` after evaluation.
