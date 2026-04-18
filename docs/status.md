# Status

## Milestone 0 — Repository scaffold (complete)

- Standard portfolio layout, Python 3.11 tooling, pre-commit, and CI are in place.
- Starter library code: `utils.paths.project_root`, `data.load.load_raw_csv`.
- Smoke tests cover path resolution and CSV loading.

## Milestone 1 — Data contract (next)

- Choose the dataset and document the schema in `docs/architecture.md`.
- Add a small **non-sensitive** sample or synthetic fixture under `tests/fixtures/` if real data cannot be committed.
- Add validation for expected columns and types before any modelling work.
