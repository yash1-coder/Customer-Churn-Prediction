"""Streamlit dashboard for the customer churn prediction project.

Launch with:
    streamlit run app/dashboard.py
"""

from __future__ import annotations

import json
import sys
from html import escape
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure src/ is on the path when running via `streamlit run`.
_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from data.load import load_telco_churn  # noqa: E402
from models.train import load_model  # noqa: E402
from utils.paths import project_root  # noqa: E402

ROOT = project_root()
REPORTS = ROOT / "reports"
ARTIFACTS = ROOT / "artifacts"
MODEL_PATH = ARTIFACTS / "model.pkl"
CHALLENGER_PATH = ARTIFACTS / "challenger_xgb.pkl"

PALETTE = {
    "ink": "#17202a",
    "muted": "#667085",
    "line": "#d8dee7",
    "surface": "#f7f9fc",
    "cream": "#fff8ed",
    "blue": "#2364aa",
    "teal": "#0f766e",
    "coral": "#e85d4f",
    "gold": "#d99a20",
    "red": "#b42318",
    "amber": "#b7791f",
    "green": "#15803d",
}


def _load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


@st.cache_data(show_spinner=False)
def _load_dataset() -> pd.DataFrame | None:
    try:
        df = load_telco_churn()
    except (FileNotFoundError, ValueError):
        return None
    return df.assign(
        ChurnFlag=lambda data: data["Churn"].map({"Yes": 1, "No": 0}),
        TotalChargesNumeric=lambda data: pd.to_numeric(
            data["TotalCharges"], errors="coerce"
        ).fillna(0.0),
    )


@st.cache_resource(show_spinner=False)
def _load_default_model() -> Any | None:
    if not MODEL_PATH.exists():
        return None
    return load_model(MODEL_PATH)


def _percent(value: float | int | None) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.1%}"


def _metric_value(value: float | int | None, *, percentage: bool = True) -> str:
    if value is None:
        return "N/A"
    if percentage:
        return f"{float(value):.1%}"
    return f"{float(value):,.0f}"


def _metric_card(label: str, value: str, caption: str, accent: str) -> str:
    return f"""
    <div class="metric-card" style="--accent: {accent};">
      <div class="metric-label">{label}</div>
      <div class="metric-value">{value}</div>
      <div class="metric-caption">{caption}</div>
    </div>
    """


def _section_heading(label: str, title: str, caption: str) -> None:
    st.markdown(
        f"""
        <div class="section-heading">
          <div class="section-kicker">{label}</div>
          <h2>{title}</h2>
          <p>{caption}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _risk_band(probability: float) -> tuple[str, str]:
    if probability >= 0.70:
        return "High risk", PALETTE["red"]
    if probability >= 0.45:
        return "Watchlist", PALETTE["amber"]
    return "Low risk", PALETTE["green"]


def _chart_style(theme: str) -> dict[str, str]:
    if theme == "Dark":
        return {
            "font": "#e6edf7",
            "grid": "rgba(148, 163, 184, 0.22)",
            "zero": "rgba(226, 232, 240, 0.38)",
            "surface": "rgba(0,0,0,0)",
        }
    return {
        "font": PALETTE["ink"],
        "grid": "#edf1f7",
        "zero": PALETTE["line"],
        "surface": "rgba(255,255,255,0)",
    }


def _styled_page(theme: str) -> None:
    dark_overrides = ""
    if theme == "Dark":
        dark_overrides = f"""
        .stApp {{
            background:
                linear-gradient(
                    90deg,
                    rgba(96, 165, 250, 0.055) 1px,
                    transparent 1px
                ),
                radial-gradient(
                    circle at top right,
                    rgba(15, 118, 110, 0.18),
                    transparent 34%
                ),
                linear-gradient(180deg, #07111f 0%, #0b1220 48%, #111827 100%);
            color: #e6edf7;
        }}
        section[data-testid="stSidebar"] {{
            background: linear-gradient(180deg, #0b1220 0%, #101827 100%);
            border-right: 1px solid rgba(148, 163, 184, 0.24);
        }}
        section[data-testid="stSidebar"] h1,
        h1, h2, h3,
        .app-title,
        .metric-value,
        .hero-stat strong,
        .insight-title,
        .focus-card strong {{
            color: #f8fafc;
        }}
        .app-subtitle,
        .metric-caption,
        .metric-label,
        .hero-stat span,
        .section-heading p,
        .insight-copy,
        .focus-card p,
        .prediction-label {{
            color: #a8b3c7;
        }}
        .app-header {{
            border-color: rgba(148, 163, 184, 0.26);
            background:
                linear-gradient(135deg, rgba(15, 23, 42, 0.92), rgba(30, 41, 59, 0.78)),
                linear-gradient(90deg, {PALETTE["blue"]}, {PALETTE["teal"]});
            box-shadow: 0 20px 48px rgba(0, 0, 0, 0.32);
        }}
        .hero-panel {{
            border-color: rgba(148, 163, 184, 0.24);
            background: rgba(15, 23, 42, 0.72);
        }}
        .hero-stat,
        .metric-card,
        .insight-box,
        .focus-card,
        .prediction-result,
        div[data-testid="stMetric"] {{
            border-color: rgba(148, 163, 184, 0.24);
            background: linear-gradient(
                180deg,
                rgba(15, 23, 42, 0.94),
                rgba(17, 24, 39, 0.9)
            );
            box-shadow: 0 14px 34px rgba(0, 0, 0, 0.28);
        }}
        .hero-stat.wide,
        .focus-card .tag {{
            background: rgba(217, 154, 32, 0.14);
        }}
        .status-pill {{
            border-color: rgba(45, 212, 191, 0.26);
            background: rgba(15, 23, 42, 0.78);
            color: #e6edf7;
        }}
        .risk-track {{
            background: rgba(148, 163, 184, 0.28);
        }}
        input, textarea, select {{
            color: #e6edf7;
        }}
        """

    st.markdown(
        f"""
        <style>
        .stApp {{
            background:
                linear-gradient(90deg, rgba(35, 100, 170, 0.045) 1px, transparent 1px),
                linear-gradient(180deg, #f7f9fc 0%, #ffffff 34%);
            background-size: 28px 28px, auto;
            color: {PALETTE["ink"]};
        }}
        section[data-testid="stSidebar"] {{
            background:
                linear-gradient(180deg, #ffffff 0%, #f9fbff 100%);
            border-right: 1px solid {PALETTE["line"]};
        }}
        section[data-testid="stSidebar"] h1 {{
            color: {PALETTE["ink"]};
            font-size: 1.45rem;
        }}
        h1, h2, h3 {{
            letter-spacing: 0;
        }}
        .block-container {{
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1240px;
        }}
        .app-header {{
            border: 1px solid {PALETTE["line"]};
            border-radius: 8px;
            padding: 0;
            background:
                linear-gradient(
                    135deg,
                    rgba(255, 248, 237, 0.95),
                    rgba(255, 255, 255, 0.96)
                ),
                linear-gradient(90deg, {PALETTE["blue"]}, {PALETTE["teal"]});
            box-shadow: 0 18px 45px rgba(23, 32, 42, 0.08);
            margin-bottom: 20px;
            overflow: hidden;
            position: relative;
        }}
        .app-header::before {{
            content: "";
            display: block;
            height: 5px;
            background: linear-gradient(
                90deg,
                {PALETTE["blue"]},
                {PALETTE["teal"]},
                {PALETTE["gold"]},
                {PALETTE["coral"]}
            );
        }}
        .hero-grid {{
            display: grid;
            grid-template-columns: minmax(0, 1.45fr) minmax(260px, 0.8fr);
            gap: 18px;
            align-items: stretch;
            padding: 24px;
        }}
        .eyebrow {{
            color: {PALETTE["teal"]};
            font-size: 0.76rem;
            font-weight: 800;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.4rem;
        }}
        .app-title {{
            font-size: clamp(2rem, 4vw, 3.1rem);
            line-height: 1;
            font-weight: 800;
            margin: 0;
            color: {PALETTE["ink"]};
        }}
        .app-subtitle {{
            margin-top: 0.85rem;
            color: {PALETTE["muted"]};
            max-width: 780px;
            font-size: 1rem;
            line-height: 1.55;
        }}
        .hero-panel {{
            border: 1px solid rgba(216, 222, 231, 0.82);
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.72);
            padding: 16px;
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
        }}
        .hero-stat {{
            border-radius: 8px;
            padding: 12px;
            background: #ffffff;
            box-shadow: inset 0 0 0 1px rgba(216, 222, 231, 0.72);
        }}
        .hero-stat span {{
            display: block;
            color: {PALETTE["muted"]};
            font-size: 0.72rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }}
        .hero-stat strong {{
            display: block;
            color: {PALETTE["ink"]};
            font-size: 1.35rem;
            margin-top: 6px;
        }}
        .hero-stat.wide {{
            grid-column: 1 / -1;
            background: linear-gradient(135deg, #ffffff, {PALETTE["cream"]});
        }}
        .status-row {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 18px;
        }}
        .status-pill {{
            border: 1px solid rgba(15, 118, 110, 0.22);
            border-radius: 999px;
            padding: 7px 11px;
            background: rgba(255, 255, 255, 0.74);
            color: {PALETTE["ink"]};
            font-size: 0.82rem;
            font-weight: 650;
        }}
        .metric-card {{
            border: 1px solid {PALETTE["line"]};
            border-radius: 8px;
            border-top: 4px solid var(--accent);
            padding: 16px;
            min-height: 126px;
            background:
                linear-gradient(
                    180deg,
                    rgba(255, 255, 255, 0.98),
                    rgba(247, 249, 252, 0.88)
                );
            box-shadow: 0 12px 30px rgba(23, 32, 42, 0.07);
            transition: transform 160ms ease, box-shadow 160ms ease;
        }}
        .metric-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 16px 36px rgba(23, 32, 42, 0.10);
        }}
        .metric-label {{
            color: {PALETTE["muted"]};
            font-size: 0.78rem;
            font-weight: 750;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }}
        .metric-value {{
            color: {PALETTE["ink"]};
            font-size: 2rem;
            line-height: 1.1;
            margin-top: 8px;
            font-weight: 800;
        }}
        .metric-caption {{
            color: {PALETTE["muted"]};
            font-size: 0.86rem;
            line-height: 1.4;
            margin-top: 8px;
        }}
        .section-heading {{
            margin: 18px 0 12px;
        }}
        .section-heading h2 {{
            color: {PALETTE["ink"]};
            font-size: 1.45rem;
            margin: 0;
        }}
        .section-heading p {{
            color: {PALETTE["muted"]};
            margin: 5px 0 0;
            max-width: 760px;
            line-height: 1.45;
        }}
        .section-kicker {{
            color: {PALETTE["coral"]};
            font-size: 0.72rem;
            font-weight: 850;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 5px;
        }}
        .insight-box {{
            border: 1px solid {PALETTE["line"]};
            border-left: 4px solid {PALETTE["teal"]};
            border-radius: 8px;
            padding: 15px 16px;
            background: linear-gradient(180deg, #ffffff, #fbfcff);
            margin-bottom: 10px;
            box-shadow: 0 10px 24px rgba(23, 32, 42, 0.045);
        }}
        .insight-title {{
            color: {PALETTE["ink"]};
            font-weight: 800;
            margin-bottom: 4px;
        }}
        .insight-copy {{
            color: {PALETTE["muted"]};
            line-height: 1.45;
        }}
        .focus-card {{
            border: 1px solid {PALETTE["line"]};
            border-radius: 8px;
            background: #ffffff;
            padding: 16px;
            min-height: 138px;
            box-shadow: 0 12px 28px rgba(23, 32, 42, 0.06);
        }}
        .focus-card .tag {{
            display: inline-block;
            border-radius: 999px;
            padding: 5px 9px;
            background: {PALETTE["cream"]};
            color: {PALETTE["amber"]};
            font-size: 0.72rem;
            font-weight: 850;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }}
        .focus-card strong {{
            display: block;
            color: {PALETTE["ink"]};
            font-size: 1.15rem;
            margin-top: 12px;
        }}
        .focus-card p {{
            color: {PALETTE["muted"]};
            line-height: 1.4;
            margin: 8px 0 0;
        }}
        .prediction-result {{
            border: 1px solid {PALETTE["line"]};
            border-radius: 8px;
            padding: 20px;
            background:
                linear-gradient(135deg, #ffffff 0%, #fff8ed 100%);
            box-shadow: 0 16px 38px rgba(23, 32, 42, 0.08);
        }}
        .prediction-label {{
            font-size: 0.8rem;
            color: {PALETTE["muted"]};
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }}
        .prediction-score {{
            font-size: 2.8rem;
            line-height: 1;
            font-weight: 850;
            margin-top: 8px;
        }}
        .risk-track {{
            width: 100%;
            height: 12px;
            border-radius: 999px;
            background: #e6ebf2;
            overflow: hidden;
            margin-top: 16px;
        }}
        .risk-fill {{
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(
                90deg,
                {PALETTE["green"]},
                {PALETTE["gold"]},
                {PALETTE["coral"]}
            );
        }}
        div[data-testid="stMetric"] {{
            background: #ffffff;
            border: 1px solid {PALETTE["line"]};
            border-radius: 8px;
            padding: 13px 15px;
            box-shadow: 0 8px 22px rgba(23, 32, 42, 0.04);
        }}
        .stButton > button {{
            border-radius: 8px;
            border: 1px solid {PALETTE["blue"]};
            background: linear-gradient(135deg, {PALETTE["blue"]}, {PALETTE["teal"]});
            color: #ffffff;
            font-weight: 800;
            min-height: 2.8rem;
        }}
        @media (max-width: 760px) {{
            .hero-grid {{
                grid-template-columns: 1fr;
                padding: 18px;
            }}
            .hero-panel {{
                grid-template-columns: 1fr;
            }}
        }}
        {dark_overrides}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _comparison_frame(comparison: dict[str, Any] | None) -> pd.DataFrame:
    if not comparison:
        return pd.DataFrame()
    baseline = comparison.get("baseline", {})
    challenger = comparison.get("challenger_xgb", {})
    rows = []
    for metric in baseline:
        rows.append(
            {
                "Metric": metric.replace("_", " ").title(),
                "Logistic Regression": baseline[metric],
                "XGBoost": challenger.get(metric),
            }
        )
    return pd.DataFrame(rows)


def _coefficient_frame(model: Any | None) -> pd.DataFrame:
    if model is None or not hasattr(model, "named_steps"):
        return pd.DataFrame()
    clf = model.named_steps.get("clf")
    if clf is None or not hasattr(clf, "coef_"):
        return pd.DataFrame()
    return pd.DataFrame(
        {
            "Feature": model.feature_names_in_,
            "Coefficient": clf.coef_[0],
        }
    ).assign(
        Direction=lambda df: df["Coefficient"].apply(
            lambda value: "Raises churn risk" if value > 0 else "Lowers churn risk"
        ),
        Strength=lambda df: df["Coefficient"].abs(),
        Label=lambda df: df["Feature"].str.replace("_", " ", regex=False),
    )


def _segment_frame(df: pd.DataFrame, column: str) -> pd.DataFrame:
    grouped = (
        df.groupby(column, dropna=False)
        .agg(Customers=("customerID", "count"), ChurnRate=("ChurnFlag", "mean"))
        .reset_index()
        .sort_values("ChurnRate", ascending=False)
    )
    grouped["Label"] = grouped[column].astype(str)
    return grouped


def _render_focus_cards(df: pd.DataFrame | None) -> None:
    if df is None:
        return

    focus_columns = [
        ("Contract", "Contract exposure"),
        ("InternetService", "Service exposure"),
        ("PaymentMethod", "Payment exposure"),
    ]
    cols = st.columns(3)
    for col, (field, title) in zip(cols, focus_columns, strict=True):
        row = _segment_frame(df, field).iloc[0]
        label = escape(str(row["Label"]))
        churn_rate = float(row["ChurnRate"])
        customers = int(row["Customers"])
        col.markdown(
            f"""
            <div class="focus-card">
              <span class="tag">{title}</span>
              <strong>{label}</strong>
              <p>{churn_rate:.1%} observed churn across {customers:,} customers.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _profile_to_features(
    profile: dict[str, Any], feature_names: list[str]
) -> pd.DataFrame:
    row: dict[str, float] = {name: 0.0 for name in feature_names}

    direct_fields = {
        "SeniorCitizen": float(profile["SeniorCitizen"]),
        "Partner": 1.0 if profile["Partner"] == "Yes" else 0.0,
        "Dependents": 1.0 if profile["Dependents"] == "Yes" else 0.0,
        "tenure": float(profile["tenure"]),
        "PhoneService": 1.0 if profile["PhoneService"] == "Yes" else 0.0,
        "PaperlessBilling": 1.0 if profile["PaperlessBilling"] == "Yes" else 0.0,
        "MonthlyCharges": float(profile["MonthlyCharges"]),
        "TotalCharges": float(profile["TotalCharges"]),
    }
    for name, value in direct_fields.items():
        if name in row:
            row[name] = value

    categorical_fields = [
        "gender",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaymentMethod",
    ]
    for field in categorical_fields:
        encoded_name = f"{field}_{profile[field]}"
        if encoded_name in row:
            row[encoded_name] = 1.0

    return pd.DataFrame([row], columns=feature_names)


def _prediction_form(model: Any | None) -> None:
    if model is None:
        st.info("Train a model first to enable individual predictions.")
        return

    feature_names = list(model.feature_names_in_)

    with st.form("prediction_form"):
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior = st.selectbox("Senior citizen", ["No", "Yes"])
            partner = st.selectbox("Partner", ["No", "Yes"])
            dependents = st.selectbox("Dependents", ["No", "Yes"])
            tenure = st.slider("Tenure months", 0, 72, 12)
        with col_b:
            phone_service = st.selectbox("Phone service", ["Yes", "No"])
            multiple_lines = st.selectbox(
                "Multiple lines", ["No", "Yes", "No phone service"]
            )
            internet_service = st.selectbox(
                "Internet service", ["DSL", "Fiber optic", "No"]
            )
            online_security = st.selectbox(
                "Online security", ["No", "Yes", "No internet service"]
            )
            tech_support = st.selectbox(
                "Tech support", ["No", "Yes", "No internet service"]
            )
        with col_c:
            contract = st.selectbox(
                "Contract", ["Month-to-month", "One year", "Two year"]
            )
            paperless = st.selectbox("Paperless billing", ["Yes", "No"])
            payment = st.selectbox(
                "Payment method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )
            monthly = st.number_input(
                "Monthly charges", min_value=0.0, max_value=200.0, value=74.0, step=1.0
            )
            total = st.number_input(
                "Total charges",
                min_value=0.0,
                max_value=10000.0,
                value=float(tenure * monthly),
                step=50.0,
            )

        expander = st.expander("Optional service details")
        with expander:
            svc_a, svc_b, svc_c = st.columns(3)
            online_backup = svc_a.selectbox(
                "Online backup", ["No", "Yes", "No internet service"]
            )
            device_protection = svc_b.selectbox(
                "Device protection", ["No", "Yes", "No internet service"]
            )
            streaming_tv = svc_c.selectbox(
                "Streaming TV", ["No", "Yes", "No internet service"]
            )
            streaming_movies = svc_a.selectbox(
                "Streaming movies", ["No", "Yes", "No internet service"]
            )

        submitted = st.form_submit_button("Score customer", type="primary")

    if not submitted:
        return

    profile = {
        "gender": gender,
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": multiple_lines,
        "InternetService": internet_service,
        "OnlineSecurity": online_security,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_protection,
        "TechSupport": tech_support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
    }
    input_row = _profile_to_features(profile, feature_names)
    probability = float(model.predict_proba(input_row)[0, 1])
    label, color = _risk_band(probability)

    st.markdown(
        f"""
        <div class="prediction-result">
          <div class="prediction-label">Prediction</div>
          <div class="prediction-score" style="color: {color};">{label}</div>
          <div class="app-subtitle">
            Estimated churn probability: <b>{probability:.1%}</b>
          </div>
          <div class="risk-track">
            <div class="risk-fill" style="width: {probability:.0%};"></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_header(
    model: Any | None, df: pd.DataFrame | None, metrics: dict[str, Any] | None
) -> None:
    model_status = "Model loaded" if model is not None else "Model missing"
    data_status = f"{len(df):,} customers" if df is not None else "Dataset unavailable"
    challenger_status = (
        "Challenger available" if CHALLENGER_PATH.exists() else "No challenger model"
    )
    customers = f"{len(df):,}" if df is not None else "N/A"
    churn_rate = _percent(float(df["ChurnFlag"].mean())) if df is not None else "N/A"
    roc_auc = _percent(metrics.get("roc_auc")) if metrics else "N/A"
    precision = _percent(metrics.get("precision")) if metrics else "N/A"
    st.markdown(
        f"""
        <div class="app-header">
          <div class="hero-grid">
            <div>
              <div class="eyebrow">Retention analytics</div>
              <h1 class="app-title">Customer Churn Command Center</h1>
              <div class="app-subtitle">
                Model performance, churn drivers, customer segments, and
                individual risk scoring from the trained Telco churn pipeline.
              </div>
              <div class="status-row">
                <span class="status-pill">{model_status}</span>
                <span class="status-pill">{challenger_status}</span>
                <span class="status-pill">{data_status}</span>
              </div>
            </div>
            <div class="hero-panel">
              <div class="hero-stat">
                <span>Customers</span>
                <strong>{customers}</strong>
              </div>
              <div class="hero-stat">
                <span>Churn rate</span>
                <strong>{churn_rate}</strong>
              </div>
              <div class="hero-stat">
                <span>ROC-AUC</span>
                <strong>{roc_auc}</strong>
              </div>
              <div class="hero-stat">
                <span>Precision</span>
                <strong>{precision}</strong>
              </div>
              <div class="hero-stat wide">
                <span>Decision posture</span>
                <strong>Balanced retention targeting</strong>
              </div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_overview(
    metrics: dict[str, Any] | None,
    comparison_df: pd.DataFrame,
    df: pd.DataFrame | None,
    theme: str,
) -> None:
    chart_style = _chart_style(theme)
    _section_heading(
        "Executive overview",
        "Performance Snapshot",
        "A concise read on model quality, churn prevalence, and where retention "
        "teams should look first.",
    )
    row_a, row_b, row_c, row_d = st.columns(4)
    churn_rate = float(df["ChurnFlag"].mean()) if df is not None else None
    customers = len(df) if df is not None else None
    roc_auc = metrics.get("roc_auc") if metrics else None
    recall = metrics.get("recall") if metrics else None

    row_a.markdown(
        _metric_card(
            "Portfolio",
            _metric_value(customers, percentage=False),
            "customers",
            PALETTE["blue"],
        ),
        unsafe_allow_html=True,
    )
    row_b.markdown(
        _metric_card(
            "Observed churn",
            _metric_value(churn_rate),
            "raw dataset rate",
            PALETTE["coral"],
        ),
        unsafe_allow_html=True,
    )
    row_c.markdown(
        _metric_card(
            "ROC-AUC",
            _metric_value(roc_auc),
            "default model rank quality",
            PALETTE["teal"],
        ),
        unsafe_allow_html=True,
    )
    row_d.markdown(
        _metric_card(
            "Recall",
            _metric_value(recall),
            "churners captured at 0.50",
            PALETTE["gold"],
        ),
        unsafe_allow_html=True,
    )

    _render_focus_cards(df)

    left, right = st.columns([1.2, 1])
    with left:
        _section_heading(
            "Model comparison",
            "Default vs Challenger",
            "The deployed Logistic Regression model is compared with the "
            "recall-oriented XGBoost challenger on the same test split.",
        )
        if comparison_df.empty:
            st.info("Run the training pipeline to generate model comparison metrics.")
        else:
            melted = comparison_df.melt(
                id_vars="Metric", var_name="Model", value_name="Score"
            )
            fig = px.bar(
                melted,
                x="Metric",
                y="Score",
                color="Model",
                barmode="group",
                color_discrete_map={
                    "Logistic Regression": PALETTE["blue"],
                    "XGBoost": PALETTE["teal"],
                },
                text=melted["Score"].map(lambda value: f"{value:.2f}"),
            )
            fig.update_layout(
                height=430,
                margin=dict(l=10, r=10, t=30, b=10),
                yaxis_tickformat=".0%",
                legend_title_text="",
                font=dict(color=chart_style["font"]),
                plot_bgcolor=chart_style["surface"],
                paper_bgcolor=chart_style["surface"],
                bargap=0.22,
            )
            fig.update_yaxes(gridcolor=chart_style["grid"])
            fig.update_xaxes(tickangle=-18)
            fig.update_traces(textposition="outside", cliponaxis=False)
            st.plotly_chart(fig, use_container_width=True)

    with right:
        _section_heading(
            "Operating read",
            "What This Means",
            "Plain-language interpretation for recruiter review and interview "
            "discussion.",
        )
        if metrics:
            st.markdown(
                f"""
                <div class="insight-box">
                  <div class="insight-title">Default model</div>
                  <div class="insight-copy">
                    Logistic Regression is the deployed model with ROC-AUC
                    {_percent(metrics.get("roc_auc"))} and precision
                    {_percent(metrics.get("precision"))}.
                  </div>
                </div>
                <div class="insight-box">
                  <div class="insight-title">Recall tradeoff</div>
                  <div class="insight-copy">
                    The XGBoost challenger captures more churners but introduces
                    more false positives, so it is better for recall-led outreach.
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        if df is not None:
            top_contract = _segment_frame(df, "Contract").iloc[0]
            st.markdown(
                f"""
                <div class="insight-box">
                  <div class="insight-title">Highest contract risk</div>
                  <div class="insight-copy">
                    {top_contract["Label"]}: {top_contract["ChurnRate"]:.1%}
                    churn across {int(top_contract["Customers"]):,} customers.
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _render_segments(df: pd.DataFrame | None, theme: str) -> None:
    chart_style = _chart_style(theme)
    _section_heading(
        "Segmentation",
        "Churn Hotspots",
        "Compare observed churn rates across major customer dimensions from "
        "the raw dataset.",
    )
    if df is None:
        st.info("Add the raw Telco CSV to data/raw to enable segment analysis.")
        return

    segment_col = st.selectbox(
        "Segment",
        ["Contract", "InternetService", "PaymentMethod", "PaperlessBilling"],
        index=0,
    )
    segment_df = _segment_frame(df, segment_col)
    chart = px.bar(
        segment_df,
        x="ChurnRate",
        y="Label",
        orientation="h",
        color="ChurnRate",
        color_continuous_scale=["#dbeafe", PALETTE["red"]],
        text=segment_df["ChurnRate"].map(lambda value: f"{value:.1%}"),
        hover_data={"Customers": ":,", "ChurnRate": ":.1%", "Label": False},
    )
    chart.update_layout(
        height=390,
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis_tickformat=".0%",
        yaxis_title="",
        xaxis_title="Churn rate",
        coloraxis_showscale=False,
        font=dict(color=chart_style["font"]),
        plot_bgcolor=chart_style["surface"],
        paper_bgcolor=chart_style["surface"],
    )
    chart.update_xaxes(gridcolor=chart_style["grid"])
    chart.update_yaxes(categoryorder="total ascending")
    chart.update_traces(textposition="outside", cliponaxis=False)
    st.plotly_chart(chart, use_container_width=True)

    table = segment_df[["Label", "Customers", "ChurnRate"]].rename(
        columns={"Label": segment_col, "ChurnRate": "Churn rate"}
    )
    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Customers": st.column_config.NumberColumn(format="%d"),
            "Churn rate": st.column_config.ProgressColumn(
                format="%.1f%%", min_value=0, max_value=1
            ),
        },
    )


def _render_drivers(model: Any | None, theme: str) -> None:
    chart_style = _chart_style(theme)
    _section_heading(
        "Explainability",
        "Strongest Churn Drivers",
        "Logistic regression coefficients show which encoded features push "
        "risk higher or lower after preprocessing.",
    )
    coef_df = _coefficient_frame(model)
    if coef_df.empty:
        st.info("Model coefficients are unavailable for the current artifact.")
        return

    top = (
        coef_df.sort_values("Strength", ascending=False)
        .head(14)
        .sort_values("Coefficient")
    )
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=top["Coefficient"],
            y=top["Label"],
            orientation="h",
            marker_color=[
                PALETTE["red"] if value > 0 else PALETTE["teal"]
                for value in top["Coefficient"]
            ],
            hovertemplate="%{y}<br>Coefficient: %{x:.3f}<extra></extra>",
        )
    )
    fig.add_vline(x=0, line_width=1, line_color=chart_style["zero"])
    fig.update_layout(
        height=520,
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis_title="Standardized coefficient",
        yaxis_title="",
        font=dict(color=chart_style["font"]),
        plot_bgcolor=chart_style["surface"],
        paper_bgcolor=chart_style["surface"],
    )
    fig.update_xaxes(gridcolor=chart_style["grid"])
    st.plotly_chart(fig, use_container_width=True)

    pos = coef_df.sort_values("Coefficient", ascending=False).head(3)
    neg = coef_df.sort_values("Coefficient").head(3)
    left, right = st.columns(2)
    left.caption("Largest risk lifts")
    left.dataframe(
        pos[["Label", "Coefficient"]],
        use_container_width=True,
        hide_index=True,
    )
    right.caption("Largest protective signals")
    right.dataframe(
        neg[["Label", "Coefficient"]],
        use_container_width=True,
        hide_index=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="Churn Command Center",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    metrics = _load_json(REPORTS / "baseline_metrics.json")
    comparison = _load_json(REPORTS / "model_comparison.json")
    comparison_df = _comparison_frame(comparison)
    raw_df = _load_dataset()
    model = _load_default_model()

    with st.sidebar:
        st.title("Churn Ops")
        st.caption("Portfolio-grade ML dashboard")
        dark_mode = st.toggle("Dark theme", value=False)
        theme = "Dark" if dark_mode else "Light"
        view = st.radio(
            "View",
            ["Overview", "Segments", "Drivers", "Score Customer"],
            label_visibility="collapsed",
        )
        st.divider()
        st.metric("Default model", "Logistic Regression")
        if metrics:
            st.metric("F1 score", f"{metrics['f1']:.1%}")
        if raw_df is not None:
            st.metric("Dataset churn", f"{raw_df['ChurnFlag'].mean():.1%}")

    _styled_page(theme)
    _render_header(model, raw_df, metrics)

    if view == "Overview":
        _render_overview(metrics, comparison_df, raw_df, theme)
    elif view == "Segments":
        _render_segments(raw_df, theme)
    elif view == "Drivers":
        _render_drivers(model, theme)
    else:
        _section_heading(
            "Risk scoring",
            "Score a Customer",
            "Enter raw customer attributes and the app will transform them into "
            "the exact model feature schema used by the trained artifact.",
        )
        _prediction_form(model)


if __name__ == "__main__":
    main()
