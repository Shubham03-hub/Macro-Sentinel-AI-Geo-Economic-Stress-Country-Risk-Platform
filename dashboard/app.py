"""
Streamlit dashboard for Macro Sentinel — the geo-economic stress prediction platform.

Run:
    streamlit run dashboard/app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.utils.helper import load_config, resolve_path, categorize_risk, risk_category_color
from src.utils.model_loader import load_champion_bundle

st.set_page_config(
    page_title="Macro Sentinel",
    page_icon="🌍",
    layout="wide",
)


def apply_custom_theme():
    """
    Netflix-inspired dark cinematic theme: deep crimson -> near-black
    gradient background, Georgia serif display headings with a subtle
    glow, and glass cards for metrics/plots.
    """
    st.markdown(
        """
        <style>
        /* ---- page background: crimson -> near-black gradient ---- */
        .stApp {
            background: linear-gradient(160deg, #8e0e00 0%, #1f1c18 65%, #140f0c 100%);
            background-attachment: fixed;
        }

        /* ---- base typography ---- */
        html, body, [class*="css"] {
            font-family: "Inter", "Segoe UI", sans-serif;
        }
        p, label, span, .stMarkdown, .stCaption, small {
            color: #f2ece6 !important;
        }
        .stCaption, small {
            color: rgba(242,236,230,0.75) !important;
        }

        /* ---- Netflix-style display headings: Georgia + glow + tracking ---- */
        h1, h2, h3, h4 {
            font-family: "Georgia", "Times New Roman", serif !important;
            color: #ffffff !important;
            font-weight: 700 !important;
            letter-spacing: 0.5px;
            text-shadow: 0 2px 18px rgba(229, 9, 20, 0.55), 0 1px 0 rgba(0,0,0,0.4);
        }
        h1 {
            text-transform: uppercase;
            letter-spacing: 1.5px;
            border-bottom: 3px solid #e50914;
            padding-bottom: 0.4rem;
            display: inline-block;
        }
        h2, h3 {
            border-left: 4px solid #e50914;
            padding-left: 0.6rem;
        }

        /* ---- anchor targets shouldn't hide behind Streamlit's sticky header ---- */
        .anchor-target {
            position: relative;
            top: -70px;
            visibility: hidden;
        }

        /* ---- glass cards for metrics ---- */
        div[data-testid="stMetric"] {
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(229, 9, 20, 0.35);
            border-radius: 16px;
            padding: 1rem 1.2rem;
            backdrop-filter: blur(10px);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.45);
        }
        div[data-testid="stMetric"] label,
        div[data-testid="stMetric"] div {
            color: #ffffff !important;
            font-family: "Georgia", serif !important;
        }

        /* ---- glass containers: selectboxes, expanders, dataframes ---- */
        div[data-baseweb="select"] > div,
        .stTextInput > div > div,
        .stMultiSelect > div > div {
            background: rgba(255, 255, 255, 0.08) !important;
            border-radius: 12px !important;
            border: 1px solid rgba(229, 9, 20, 0.35) !important;
            color: #ffffff !important;
        }

        .streamlit-expanderHeader {
            background: rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            color: #ffffff !important;
            font-family: "Georgia", serif !important;
        }

        /* ---- plot containers get a cinematic dark card wrapper ---- */
        div[data-testid="stPlotlyChart"] {
            background: rgba(0, 0, 0, 0.35);
            border-radius: 18px;
            padding: 0.75rem;
            border: 1px solid rgba(229, 9, 20, 0.3);
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
        }

        /* ---- dividers ---- */
        hr {
            border-color: rgba(229, 9, 20, 0.35) !important;
        }

        /* ---- buttons / download button: Netflix red accent ---- */
        .stDownloadButton button, .stButton button {
            background: #e50914;
            color: #ffffff;
            border-radius: 6px;
            border: none;
            font-family: "Georgia", serif !important;
            font-weight: 700;
            letter-spacing: 0.5px;
            box-shadow: 0 4px 18px rgba(229, 9, 20, 0.5);
        }
        .stDownloadButton button:hover, .stButton button:hover {
            background: #f6121d;
            color: #ffffff;
            box-shadow: 0 4px 24px rgba(229, 9, 20, 0.8);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


PLOTLY_TRANSPARENT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_color="#f2ece6",
    font_family="Georgia, serif",
    legend_font_color="#f2ece6",
)

CFG = load_config()
MODELS_DIR = resolve_path(CFG["paths"]["models_dir"])
PROCESSED_DIR = resolve_path(CFG["paths"]["processed_dir"])
ENTITY_COL = CFG["feature_engineering"]["entity_col"]
TIME_COL = CFG["feature_engineering"]["time_col"]
TARGET_COL = CFG["feature_engineering"]["target_col"]


def anchor(anchor_id: str):
    """Invisible anchor so #anchor_id links (e.g. localhost:8501/#export) scroll to this section."""
    st.markdown(f'<div id="{anchor_id}" class="anchor-target"></div>', unsafe_allow_html=True)


@st.cache_data
def load_feature_dataset() -> pd.DataFrame:
    path = PROCESSED_DIR / CFG["processed_files"]["feature_dataset"]
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data
def load_predictions() -> pd.DataFrame:
    path = PROCESSED_DIR / CFG["processed_files"]["predictions"]
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_resource
def load_bundle():
    try:
        return load_champion_bundle(MODELS_DIR)
    except FileNotFoundError:
        return None


def artifacts_missing_notice():
    st.warning(
        "No trained model or predictions found yet. From the project root, run:\n\n"
        "```\npython main.py --stage all\n```\n\n"
        "then relaunch this dashboard."
    )


def render_header():
    st.title("🌍 Macro Sentinel — AI Geo-Economic Stress & Country Risk Platform")
    st.caption(
        "Country-level economic stress scoring, forecasting, and driver analysis "
        "for investment, policy, and treasury risk teams."
    )


def render_kpi_cards(predictions: pd.DataFrame, bundle: dict):
    col1, col2, col3, col4 = st.columns(4)

    avg_score = predictions["predicted_stress_score"].mean()
    severe_count = (predictions["risk_category"].isin(["High", "Severe"])).sum()
    best_model_name = bundle["metadata"]["champion_name"] if bundle else "n/a"
    model_mae = bundle["metadata"]["metrics"]["mae"] if bundle else float("nan")

    col1.metric("Countries Monitored", f"{len(predictions)}")
    col2.metric("Avg. Global Stress Score", f"{avg_score:.1f} / 100")
    col3.metric("Countries at High/Severe Risk", f"{severe_count}")
    col4.metric(f"Champion Model ({best_model_name})", f"MAE {model_mae:.2f}")


def render_world_overview(predictions: pd.DataFrame):
    anchor("global-risk-overview")
    st.subheader("Global Risk Overview")
    fig = px.choropleth(
        predictions,
        locations="iso3",
        color="predicted_stress_score",
        hover_name="country",
        hover_data={"risk_category": True, "predicted_stress_score": ":.1f", "iso3": False},
        color_continuous_scale=["#1f1c18", "#8e0e00", "#e50914", "#ff5a4e"],
        range_color=(0, 100),
        labels={"predicted_stress_score": "Stress Score"},
    )
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=430, **PLOTLY_TRANSPARENT_LAYOUT)
    fig.update_geos(bgcolor="rgba(0,0,0,0)", lakecolor="#140f0c", landcolor="#2a241f")
    st.plotly_chart(fig, use_container_width=True)


def render_country_explorer(predictions: pd.DataFrame, feature_df: pd.DataFrame):
    st.subheader("Country Explorer")
    left, right = st.columns([1, 2])

    with left:
        countries = sorted(predictions["country"].unique())
        default_idx = countries.index("United States") if "United States" in countries else 0
        selected_country = st.selectbox("Country", countries, index=default_idx)

        row = predictions[predictions["country"] == selected_country].iloc[0]
        color = risk_category_color(row["risk_category"])

        st.markdown(
            f"""
            <div style="padding:1.2rem;border-radius:16px;border:1px solid rgba(229,9,20,0.35);
                        background:rgba(0,0,0,0.35);backdrop-filter:blur(10px);
                        box-shadow:0 8px 32px rgba(0,0,0,0.5);">
                <div style="font-size:0.85rem;color:rgba(242,236,230,0.75);font-family:Georgia,serif;">CURRENT RISK CATEGORY</div>
                <div style="font-size:1.6rem;font-weight:700;color:{color};font-family:Georgia,serif;">{row['risk_category']}</div>
                <div style="font-size:0.85rem;color:rgba(242,236,230,0.75);margin-top:0.5rem;font-family:Georgia,serif;">PREDICTED STRESS SCORE</div>
                <div style="font-size:1.3rem;font-weight:600;color:#ffffff;font-family:Georgia,serif;">{row['predicted_stress_score']:.1f} / 100</div>
                <div style="font-size:0.85rem;color:rgba(242,236,230,0.75);margin-top:0.5rem;font-family:Georgia,serif;">FORECASTED NEXT-YEAR SCORE</div>
                <div style="font-size:1.3rem;font-weight:600;color:#ffffff;font-family:Georgia,serif;">{row['forecasted_stress_next_year']:.1f} / 100
                    <span style="font-size:0.9rem;color:{risk_category_color(row['forecasted_risk_category'])};">
                        ({row['forecasted_risk_category']})
                    </span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        if not feature_df.empty:
            iso3 = row[ENTITY_COL]
            hist = feature_df[feature_df[ENTITY_COL] == iso3].sort_values(TIME_COL)
            if TARGET_COL in hist.columns and hist[TARGET_COL].notna().any():
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=hist[TIME_COL], y=hist[TARGET_COL],
                    mode="lines+markers", name="Historical Stress Score",
                    line=dict(color="#ff5a4e", width=2),
                ))
                fig.add_trace(go.Scatter(
                    x=[row[TIME_COL] + 1], y=[row["forecasted_stress_next_year"]],
                    mode="markers", name="Forecast (next year)",
                    marker=dict(color="#e50914", size=12, symbol="diamond"),
                ))
                fig.update_layout(
                    title=f"{selected_country} — Stress Score Trend & Forecast",
                    xaxis_title="Year", yaxis_title="Stress Score (0-100)",
                    height=380, margin=dict(t=50, b=10),
                    yaxis_range=[0, 100],
                    **PLOTLY_TRANSPARENT_LAYOUT,
                )
                st.plotly_chart(fig, use_container_width=True)


def render_driver_importance():
    st.subheader("Key Macroeconomic Drivers (Global Model)")
    reports_dir = resolve_path(CFG["paths"]["reports_dir"])
    driver_path = reports_dir / "driver_importance_top20.csv"
    if not driver_path.exists():
        st.info("Run evaluation stage to populate driver importance: `python main.py --stage evaluate`")
        return
    drivers = pd.read_csv(driver_path).head(12).iloc[::-1]
    fig = px.bar(
        drivers, x="importance_pct", y="feature", orientation="h",
        labels={"importance_pct": "Importance (%)", "feature": ""},
        color="importance_pct", color_continuous_scale=["#1f1c18", "#8e0e00", "#e50914"],
    )
    fig.update_layout(height=420, margin=dict(t=10, b=10), coloraxis_showscale=False, **PLOTLY_TRANSPARENT_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)


def render_indicator_trends(feature_df: pd.DataFrame, predictions: pd.DataFrame):
    st.subheader("Indicator Trend Analysis")
    if feature_df.empty:
        st.info("No feature dataset found yet.")
        return

    left, right = st.columns([1, 3])
    with left:
        countries = sorted(predictions["country"].unique())
        chosen = st.multiselect("Compare countries", countries, default=countries[:3] if len(countries) >= 3 else countries)
        indicator_options = [
            "gdp_growth_pct", "inflation_pct", "unemployment_pct", "debt_to_gdp_pct",
            "fx_volatility_index", "current_account_pct_gdp", "gdp_per_capita_usd",
        ]
        indicator = st.selectbox("Indicator", indicator_options)

    with right:
        if chosen:
            iso_map = predictions.set_index("country")[ENTITY_COL].to_dict()
            subset = feature_df[feature_df[ENTITY_COL].isin([iso_map[c] for c in chosen if c in iso_map])].copy()
            fig = px.line(
                subset.sort_values(TIME_COL), x=TIME_COL, y=indicator, color="country", markers=True,
            )
            fig.update_layout(height=400, margin=dict(t=10, b=10), **PLOTLY_TRANSPARENT_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)


def render_download(predictions: pd.DataFrame):
    anchor("export")
    st.subheader("Export")
    csv_bytes = predictions.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇ Download Full Predictions (CSV)",
        data=csv_bytes,
        file_name="macro_sentinel_predictions.csv",
        mime="text/csv",
    )


def main():
    apply_custom_theme()
    render_header()
    predictions = load_predictions()
    feature_df = load_feature_dataset()
    bundle = load_bundle()

    if predictions.empty or bundle is None:
        artifacts_missing_notice()
        return

    render_kpi_cards(predictions, bundle)
    st.divider()
    render_world_overview(predictions)
    st.divider()
    render_country_explorer(predictions, feature_df)
    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        render_driver_importance()
    with col_b:
        st.subheader("Risk Category Distribution")
        dist = predictions["risk_category"].value_counts().reset_index()
        dist.columns = ["risk_category", "count"]
        color_map = {c: risk_category_color(c) for c in dist["risk_category"]}
        fig = px.pie(dist, names="risk_category", values="count", color="risk_category",
                     color_discrete_map=color_map, hole=0.45)
        fig.update_layout(height=420, margin=dict(t=10, b=10), **PLOTLY_TRANSPARENT_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)
    st.divider()
    render_indicator_trends(feature_df, predictions)
    st.divider()
    render_download(predictions)

    with st.expander("📋 Full Predictions Table"):
        st.dataframe(predictions, use_container_width=True)


if __name__ == "__main__":
    main()