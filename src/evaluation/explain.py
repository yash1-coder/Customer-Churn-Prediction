"""SHAP-based model explainability for the churn pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from utils.paths import project_root

matplotlib.use("Agg")

REPORTS_DIR = project_root() / "reports"


def shap_summary(
    model: Any,
    X: pd.DataFrame,
    *,
    output_path: Path | None = None,
    max_display: int = 15,
) -> shap.Explanation:
    """Generate a SHAP summary plot and save it to *output_path*.

    Returns the SHAP Explanation object for downstream use.
    """
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)

    if output_path is None:
        output_path = REPORTS_DIR / "shap_summary.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    shap.summary_plot(shap_values, X, max_display=max_display, show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    return shap_values


def top_feature_importances(
    shap_values: shap.Explanation,
    n: int = 10,
) -> pd.DataFrame:
    """Return the top-*n* features by mean |SHAP value|."""
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    names = shap_values.feature_names
    ranking = pd.DataFrame({"feature": names, "mean_abs_shap": mean_abs}).sort_values(
        "mean_abs_shap", ascending=False
    )
    return ranking.head(n).reset_index(drop=True)
