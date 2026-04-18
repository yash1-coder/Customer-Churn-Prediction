# Customer churn prediction

Portfolio-style Python project for churn modelling: reproducible pipelines in `src/`, exploratory work in `notebooks/`, and documented evaluation in `docs/`.

## Quickstart

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
pytest
```

## Layout

- `src/` — reusable library code (`data`, `features`, `models`, `evaluation`, `utils`)
- `data/raw` — immutable inputs (not committed)
- `data/processed` — derived tables ready for modelling
- `configs/` — parameters and run metadata
- `reports/` — exported figures and tables
- `app/` — optional UI entrypoints (for example Streamlit)

## Tooling

Formatting and linting use **Black** and **Ruff**, static typing uses **mypy**, tests use **pytest**, and **pre-commit** wires the hooks locally. CI mirrors the same commands.

## Status

See `docs/status.md` for the current milestone and `TODO.md` for the next tasks.
