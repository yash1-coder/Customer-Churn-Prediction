# Architecture

## Design intent

Keep a thin boundary between **immutable raw inputs** (`data/raw`), **reproducible transforms** (`src/`), and **presentation or reporting** (`reports/`, optional `app/`).

## Python packages

Packages live directly under `src/` (`data`, `features`, `models`, `evaluation`, `utils`) so imports match the folder names you see in the tree.

Trade-off: the `data` package name is generic. If this ever collides with tooling or habits (`import data` ambiguity), rename the package to a project-specific namespace (for example `churn_data`) and update imports once.

## Current building blocks

- `utils.paths.project_root` — locate the repository root without hard-coded paths.
- `data.load.load_raw_csv` — read-only CSV ingest for raw files.

## Planned extensions (not implemented yet)

- `features` — training matrix construction and feature metadata.
- `models` — baseline and challenger trainers, plus model registry on disk.
- `evaluation` — thresholding, calibration checks, and error analysis helpers.
