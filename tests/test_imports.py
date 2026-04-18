"""Smoke tests: every public sub-package and key symbols are importable."""


def test_import_data_load() -> None:
    from data.load import load_raw_csv

    assert callable(load_raw_csv)


def test_import_features_preprocess() -> None:
    from features.preprocess import build_feature_matrix

    assert callable(build_feature_matrix)


def test_import_models_train() -> None:
    from models.train import save_model, train_baseline

    assert callable(train_baseline)
    assert callable(save_model)


def test_import_evaluation_metrics() -> None:
    from evaluation.metrics import classification_report

    assert callable(classification_report)


def test_import_utils_paths() -> None:
    from utils.paths import project_root

    assert callable(project_root)
