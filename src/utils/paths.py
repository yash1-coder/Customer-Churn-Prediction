from pathlib import Path


def project_root() -> Path:
    """Return repository root (directory containing ``pyproject.toml``)."""
    return Path(__file__).resolve().parents[2]
