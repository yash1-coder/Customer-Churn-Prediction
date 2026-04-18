from utils.paths import project_root


def test_project_root_contains_pyproject() -> None:
    root = project_root()
    assert (root / "pyproject.toml").is_file()
