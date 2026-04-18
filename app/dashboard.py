"""Streamlit dashboard for the customer churn prediction project.

Launch with:
    streamlit run app/dashboard.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Ensure src/ is on the path when running via `streamlit run`
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from models.train import load_model  # noqa: E402
from utils.paths import project_root  # noqa: E402

ROOT = project_root()
REPORTS = ROOT / "reports"
ARTIFACTS = ROOT / "artifacts"


def _load_metrics() -> dict[str, float] | None:
    path = REPORTS / "baseline_metrics.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def main() -> None:
    st.set_page_config(page_title="Churn Prediction", layout="wide")
    st.title("Customer Churn Prediction")
    st.markdown(
        "Interactive dashboard for exploring model performance, "
        "feature importance, and individual predictions."
    )

    # --- Metrics overview ---
    st.header("Baseline Model Metrics")
    metrics = _load_metrics()
    if metrics is None:
        st.warning(
            "No metrics found. Run `python -m run_train` from `src/` first."
        )
    else:
        cols = st.columns(len(metrics))
        for col, (name, value) in zip(cols, metrics.items()):
            col.metric(name.upper(), f"{value:.4f}")

    # --- SHAP summary ---
    st.header("Feature Importance (SHAP)")
    shap_img = REPORTS / "shap_summary.png"
    if shap_img.exists():
        st.image(str(shap_img), use_container_width=True)
    else:
        st.info(
            "SHAP summary not generated yet. Run the training pipeline "
            "with explainability enabled."
        )

    # --- Single-customer prediction ---
    st.header("Predict for a Single Customer")
    model_path = ARTIFACTS / "model.pkl"
    if not model_path.exists():
        st.info("Train a model first to enable individual predictions.")
        return

    model = load_model(model_path)
    feature_names = model.feature_names_in_

    with st.form("predict_form"):
        st.markdown("Enter feature values (numeric, matching training columns):")
        input_data: dict[str, float] = {}
        col_a, col_b = st.columns(2)
        for i, feat in enumerate(feature_names):
            target_col = col_a if i % 2 == 0 else col_b
            input_data[feat] = target_col.number_input(feat, value=0.0)
        submitted = st.form_submit_button("Predict")

    if submitted:
        row = pd.DataFrame([input_data])
        prob = model.predict_proba(row)[0, 1]
        label = "Churn" if prob >= 0.5 else "No Churn"
        st.subheader(f"Prediction: **{label}** (probability {prob:.2%})")


if __name__ == "__main__":
    main()
