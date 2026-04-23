"""
app.py – Streamlit main application: Electricity Price Scenarios 2025–2038.
Launch: streamlit run app.py
"""

import os
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ── Page settings (before other st calls) ────────────────────────────────────
st.set_page_config(
    page_title="Electricity Price Scenarios 2025–2038",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Password protection ───────────────────────────────────────────────────────

def check_password() -> bool:
    """Checks the password and returns True if authenticated."""
    if st.session_state.get("authenticated"):
        return True
    st.title("⚡ Electricity Price Scenarios 2025–2038")
    st.markdown("### Sign In")
    password = st.text_input("Password", type="password", key="pw_input")
    if st.button("Sign In"):
        if password == st.secrets.get("APP_PASSWORD", "demo1234"):
            st.session_state["authenticated"] = True
            st.rerun()
        else:
            st.error("Incorrect password")
    st.caption("Hint: default password is `demo1234` (change in .streamlit/secrets.toml)")
    return False

if not check_password():
    st.stop()

# ── Style definitions ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stMetricValue { font-size: 1.8rem !important; }
    .block-container { padding-top: 1rem; }
    div[data-testid="metric-container"] {
        background-color: #F9FBE7;
        border: 1px solid #C5E1A5;
        border-radius: 8px;
        padding: 12px;
    }
    .stAlert { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

st.title("⚡ Electricity Price Scenarios 2025–2038")
st.caption("Finnish electricity market analysis | Monte Carlo simulation | Merit Order | Risk analysis")

# ── Imports from models ───────────────────────────────────────────────────────
from model.data_fetch import load_fundamental_data, load_historical_prices
from model.data_inspect import inspect_excel
from model.scenarios import (
    RegressionResult, ScenarioParams, ScenarioResult,
    NUCLEAR_FI_OPTIONS, NUCLEAR_SE_OPTIONS, HYDRO_OPTIONS,
    INTERCONNECT_FI_EE_OPTIONS, INTERCONNECT_FI_SE_OPTIONS, INTERCONNECT_NO_OPTIONS,
    FI_BASE_CONSUMPTION_TWH, FI_CONSUMPTION_BREAKDOWN_2025, FI_MONTHLY_CONSUMPTION_2025,
    START_YEAR, END_YEAR,
    SCENARIO_NAMES, SCENARIO_LABELS, SCENARIO_COLORS,
    calibrate_regression, run_monte_carlo, scenarios_to_dataframe,
    compute_variable_sensitivities, compute_datacenter_projection,
    compute_consumption_growth, compute_max_hintaero,
    compute_impact_breakdown,
)
from model.risk import (
    HedgeParams, calculate_risk_metrics, calculate_all_hedges,
    calculate_active_hedge, run_stress_tests, get_hedge_recommendation,
    compute_efficient_frontier, build_risk_metrics_table,
)
from model.capacity import (
    CapacityParams, calculate_monthly_capacity, calculate_capacity_margin,
    capacity_time_series, find_critical_months, NUCLEAR_OPTIONS_MW,
)
from model.merit_order import (
    MeritOrderParams, MeritOrderSlice, build_merit_order,
    calculate_market_price, merit_order_time_series, merit_order_to_df,
    SOURCE_COLORS,
)
from ui.charts import (
    correlation_heatmap, datacenter_growth_chart, efficient_frontier_chart,
    fundamental_time_series, hedge_annual_cost_chart, hedge_comparison_chart,
    interconnect_hintaero_chart, monthly_avg_bar, monthly_heatmap,
    price_percentile_paths, price_scenario_chart, regression_coef_chart,
    stress_test_chart, tornado_chart,
)
from ui.report import build_pdf_report, generate_summary_text


# ── Month names ───────────────────────────────────────────────────────────────
MONTH_NAMES = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May",     6: "June",     7: "July",  8: "August",
    9: "September", 10: "October", 11: "November", 12: "December",
}
MONTH_SHORT = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
    7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}


# ── Cache functions ───────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading historical data...")
def get_historical_data() -> pd.DataFrame:
    return load_historical_prices()


@st.cache_data(show_spinner=False, max_entries=30)
def run_scenarios_cached(
    params_key: tuple,
    _params: ScenarioParams,
    _regression: RegressionResult,
    timeout_seconds: float = 30.0,
) -> dict[str, ScenarioResult]:
    """
    Cached Monte Carlo run. Cache invalidates when params_key changes.
    _params and _regression: underscore-prefix → Streamlit does NOT use them for hashing.
    """
    return run_monte_carlo(_params, _regression, timeout_seconds=timeout_seconds)


@st.cache_data(show_spinner="Reading Excel file...", max_entries=3)
def load_excel_cached(file_bytes: bytes, filename: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    suffix = Path(filename).suffix or ".xlsx"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        df, meta = load_fundamental_data(tmp_path)
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
    return df, meta


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("Parameters")

    # ── Data file ─────────────────────────────────────────────────────────────
    st.subheader("Data File")
    uploaded_file = st.file_uploader(
        "Upload Excel file (.xlsx)",
        type=["xlsx", "xls"],
        help="File is processed locally only — your data does not leave your machine.",
    )

    fundamental_df: pd.DataFrame = pd.DataFrame()
    excel_meta: dict[str, Any] = {}
    regression: RegressionResult = RegressionResult()
    has_excel = False

    if uploaded_file is not None:
        file_bytes = uploaded_file.read()
        with st.spinner("Analyzing Excel file..."):
            fundamental_df, excel_meta = load_excel_cached(file_bytes, uploaded_file.name)

        if not fundamental_df.empty:
            has_excel = True
            with st.spinner("Calibrating regression model..."):
                regression = calibrate_regression(fundamental_df)

            with st.expander("Detected data", expanded=True):
                found_cols = excel_meta.get("found_columns", {})
                sheet = excel_meta.get("used_sheet", "?")
                st.caption(f"Sheet: **{sheet}** | {len(fundamental_df)} months")
                col_display = {
                    "price_fi":           "Spot price",
                    "consumption":        "Consumption",
                    "wind_capacity":      "Wind power",
                    "hydro_production":   "Hydro power",
                    "nuclear_production": "Nuclear power",
                    "gas_price":          "Gas price",
                    "co2_price":          "CO₂ price",
                }
                for std, label in col_display.items():
                    if std in found_cols:
                        st.markdown(f"✅ {label}: `{found_cols[std]}`")
                    else:
                        st.markdown(f"🟡 {label}: synthetic default")
                if regression.r2 > 0:
                    r2_pct = regression.r2 * 100
                    color = "green" if r2_pct > 60 else "orange"
                    st.markdown(
                        f"**R² = :{color}[{r2_pct:.1f}%]** "
                        f"({len(regression.used_features)} features)"
                    )
        else:
            st.error(
                "Excel file loading failed. "
                + excel_meta.get("inspect", {}).get("error", "Unknown error.")
            )
    else:
        st.info("No file uploaded — using synthetic data.", icon="ℹ️")

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # FINNISH ELECTRICITY MARKET DEVELOPMENT
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("Finnish Electricity Market Development")

    with st.expander("Consumption growth", expanded=True):
        electrification_twh = st.slider(
            "Base electrification (industry + heat pumps)",
            0, 30, 8, 1,
            help="Consumption growth from electrification by 2035 (TWh)",
        )
        ev_twh = st.slider(
            "Electric vehicles",
            0, 6, 1, 1,
            help="Additional consumption from EVs by 2035 (TWh)",
        )
        total_growth = electrification_twh + ev_twh
        st.caption(
            f"Consumption grows by **{total_growth} TWh** → "
            f"total **{FI_BASE_CONSUMPTION_TWH + total_growth:.0f} TWh** by 2035"
        )

    with st.expander("Datacenters", expanded=False):
        datacenter_base = st.slider(
            "Datacenter consumption 2025 (baseline, TWh)",
            0.5, 10.0, 3.0, 0.5,
        )
        datacenter_growth = st.slider(
            "Datacenter annual growth rate (%/y)",
            0, 50, 32, 1,
        )
        dc_final = datacenter_base * ((1 + datacenter_growth / 100) ** 10)
        dc_capped = dc_final >= 50.0
        dc_final_disp = min(dc_final, 50.0)
        st.caption(
            f"Datacenters grow **{datacenter_base:.1f} TWh → "
            f"{dc_final_disp:.1f} TWh** by 2035"
        )
        if dc_capped:
            st.info(f"Datacenter consumption reaches the 50 TWh cap before 2035.", icon="⚡")

    with st.expander("Wind power and solar energy", expanded=False):
        wind_fi_gw = st.slider(
            "Additional wind power capacity 2025–2035 (GW)",
            0.0, 15.0, 5.0, 0.5,
            help="Finland's current capacity ~7 GW. This is additional build.",
        )
        solar_fi_gw = st.slider(
            "Solar energy growth (GW)",
            0.0, 5.0, 1.5, 0.5,
        )
        new_re_twh = wind_fi_gw * 2.8 + solar_fi_gw * 0.9
        st.caption(
            f"Renewable generation grows by approximately **+{new_re_twh:.0f} TWh/year** by 2035"
        )

    # Quick price impact preview
    _preview_params = ScenarioParams(
        wind_fi_gw=wind_fi_gw,
        solar_fi_gw=solar_fi_gw,
        electrification_twh=electrification_twh,
        ev_twh=ev_twh,
        datacenter_base_twh=datacenter_base,
        datacenter_growth_pct=datacenter_growth,
    )
    _impact = compute_impact_breakdown(_preview_params, ref_years=(2030, 2035))

    with st.expander("Price impacts 2030 / 2035", expanded=False):
        for _yr in (2030, 2035):
            _d = _impact[_yr]
            st.markdown(f"**Year {_yr}** — base scenario P50: **{_d['base_price']:.1f} €/MWh**")

            def _fmt(v: float) -> str:
                sign = "+" if v >= 0 else ""
                return f"{sign}{v:.1f} €/MWh"

            st.caption(
                f"Consumption growth ({_d['total_growth_twh']:.0f} TWh): **{_fmt(_d['consumption_impact'])}**  \n"
                f"Datacenters (+{_d['dc_growth_twh']:.1f} TWh): **{_fmt(_d['datacenter_impact'])}**  \n"
                f"Wind power (+{_d['wind_re_twh']:.0f} TWh/y): **{_fmt(_d['wind_impact'])}**"
            )
            if _yr == 2030:
                st.divider()

    with st.expander("Nuclear power Finland", expanded=False):
        nuclear_fi_keys = list(NUCLEAR_FI_OPTIONS.keys())
        nuclear_fi_labels = [NUCLEAR_FI_OPTIONS[k][0] for k in nuclear_fi_keys]
        nuclear_fi_idx = st.selectbox(
            "Finland nuclear capacity 2035",
            options=range(len(nuclear_fi_keys)),
            format_func=lambda i: nuclear_fi_labels[i],
            index=0,
        )
        nuclear_fi = nuclear_fi_keys[nuclear_fi_idx]

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # NORDIC MARKETS
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("Nordic Markets")

    with st.expander("Hydro power and Sweden", expanded=True):
        hydro_keys = list(HYDRO_OPTIONS.keys())
        hydro_labels = [HYDRO_OPTIONS[k][0] for k in hydro_keys]
        hydro_idx = st.selectbox(
            "Hydro reservoirs (Norway + Sweden)",
            options=range(len(hydro_keys)),
            format_func=lambda i: hydro_labels[i],
        )
        hydro_nordic = hydro_keys[hydro_idx]

        nuclear_se_keys = list(NUCLEAR_SE_OPTIONS.keys())
        nuclear_se_labels = [NUCLEAR_SE_OPTIONS[k][0] for k in nuclear_se_keys]
        nuclear_se_idx = st.selectbox(
            "Swedish nuclear power",
            options=range(len(nuclear_se_keys)),
            format_func=lambda i: nuclear_se_labels[i],
        )
        nuclear_se = nuclear_se_keys[nuclear_se_idx]

    with st.expander("Fuel prices", expanded=True):
        gas_price = st.slider(
            "Gas price (€/MWh)",
            20, 80, 40, 5,
            help="European gas reference price (TTF)",
        )
        co2_price = st.slider(
            "CO₂ emission allowance ETS (€/t CO₂)",
            40, 120, 70, 5,
        )

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # INTERCONNECTIONS
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("Interconnections")

    with st.expander("Interconnection capacity", expanded=False):
        fi_ee_keys = list(INTERCONNECT_FI_EE_OPTIONS.keys())
        fi_ee_labels = [INTERCONNECT_FI_EE_OPTIONS[k][0] for k in fi_ee_keys]
        fi_ee_idx = st.selectbox(
            "EstLink (FI–EE)",
            options=range(len(fi_ee_keys)),
            format_func=lambda i: fi_ee_labels[i],
        )
        interconnect_fi_ee = fi_ee_keys[fi_ee_idx]

        fi_se_keys = list(INTERCONNECT_FI_SE_OPTIONS.keys())
        fi_se_labels = [INTERCONNECT_FI_SE_OPTIONS[k][0] for k in fi_se_keys]
        fi_se_idx = st.selectbox(
            "Fennoscan / FI–SE connection",
            options=range(len(fi_se_keys)),
            format_func=lambda i: fi_se_labels[i],
        )
        interconnect_fi_se = fi_se_keys[fi_se_idx]

        no_keys = list(INTERCONNECT_NO_OPTIONS.keys())
        no_labels = [INTERCONNECT_NO_OPTIONS[k][0] for k in no_keys]
        no_idx = st.selectbox(
            "Norway connection (via Sweden)",
            options=range(len(no_keys)),
            format_func=lambda i: no_labels[i],
        )
        interconnect_no = no_keys[no_idx]

        tmp_params_ic = ScenarioParams(
            interconnect_fi_se=interconnect_fi_se,
            interconnect_fi_ee=interconnect_fi_ee,
        )
        max_price_diff_preview = compute_max_hintaero(tmp_params_ic)
        st.caption(f"Max FI–Nordic price difference: **{max_price_diff_preview:.0f} €/MWh**")

        st.markdown("**Neighboring area price level relative to FI**")
        se3_price_relative = st.slider(
            "SE3 (Stockholm) price % of FI",
            60, 120, 92, 1,
            help="SE3 is historically 5–15% cheaper than FI (FI-SE EPAD). 100% = parity.",
        ) / 100.0
        se1_price_relative = st.slider(
            "SE1 (Luleå) price % of FI",
            40, 110, 78, 1,
            help="Northern Sweden hydro — often 15–30% cheaper than FI.",
        ) / 100.0
        ee_price_relative = st.slider(
            "EE (Estonia) price % of FI",
            70, 130, 100, 1,
            help="Estonia is often near FI parity. Was cheaper with Russian gas.",
        ) / 100.0

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # CAPACITY MODEL
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("Capacity Model")

    with st.expander("Capacity parameters", expanded=False):
        wind_total_gw = st.slider(
            "Total wind power 2025 (GW, installed)",
            3.0, 20.0, 7.0, 0.5,
            help="Finland's total wind power capacity (not additional build)",
        )
        solar_cap_gw = st.slider(
            "Total solar power (GW)",
            0.5, 10.0, 1.5, 0.5,
        )
        _, fi_se_mw_cap = INTERCONNECT_FI_SE_OPTIONS.get(interconnect_fi_se, ("", 2200))
        _, fi_ee_mw_cap = INTERCONNECT_FI_EE_OPTIONS.get(interconnect_fi_ee, ("", 1000))

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # MONTE CARLO
    # ══════════════════════════════════════════════════════════════════════════
    with st.expander("Monte Carlo settings", expanded=False):
        n_simulations = st.selectbox(
            "Simulations",
            options=[100, 500, 1000],
            index=1,
            format_func=lambda x: f"{x} runs",
        )
        crisis_prob = st.slider(
            "Energy crisis probability (%)",
            0, 30, 10, 1,
            help="Crisis event probability in the high scenario",
        ) / 100.0

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # HEDGING STRATEGY
    # ══════════════════════════════════════════════════════════════════════════
    st.subheader("Hedging Strategy")

    hedge_strategy = st.selectbox(
        "Strategy",
        options=["spot", "fixed_100", "partial_5050", "collar", "forward", "combination"],
        format_func=lambda x: {
            "spot":          "Full spot (no hedging)",
            "fixed_100":     "Fixed price 100%",
            "partial_5050":  "Partial fixed 50/50",
            "collar":        "Collar strategy (floor + cap)",
            "forward":       "Forward hedge 12m rolling",
            "combination":   "Combination (spot + collar + forward)",
        }[x],
    )

    fixed_price  = 65.0
    floor_price  = 40.0
    cap_price    = 120.0
    hedge_pct    = 0.70
    fwd_premium  = 0.05
    spot_pct_h   = 0.30
    collar_pct_h = 0.40

    if hedge_strategy in ("fixed_100", "partial_5050"):
        fixed_price = st.number_input(
            "Fixed price (€/MWh)", 20.0, 200.0, 65.0, 1.0
        )

    if hedge_strategy == "collar":
        floor_price = st.slider("Floor price (€/MWh)", 30, 60, 40, 1)
        cap_price   = st.slider("Cap price (€/MWh)", 80, 200, 120, 5)

    if hedge_strategy == "forward":
        hedge_pct   = st.slider("Hedge share (%)", 0, 100, 70, 5) / 100.0
        fwd_premium = st.slider("Forward premium (%)", 0, 15, 5, 1) / 100.0

    if hedge_strategy == "combination":
        spot_pct_h   = st.slider("Spot share (%)", 0, 100, 30, 5) / 100.0
        max_collar   = max(0, int((1 - spot_pct_h) * 100))
        collar_pct_h = st.slider("Collar share (%)", 0, max_collar, min(40, max_collar), 5) / 100.0
        fwd_pct_h    = 1.0 - spot_pct_h - collar_pct_h
        st.caption(f"Forward share: **{fwd_pct_h*100:.0f}%**")
        floor_price  = st.slider("Collar floor (€/MWh)", 30, 60, 40, 1)
        cap_price    = st.slider("Collar cap (€/MWh)", 80, 200, 120, 5)
        fwd_premium  = st.slider("Forward premium (%)", 0, 15, 5, 1) / 100.0

    st.markdown("**Volume**")
    vol_mwh = st.slider(
        "Annual volume (MWh)",
        100, 500_000, 10_000, 500,
        help="Annual electricity volume to hedge",
    )
    dist_type = st.radio(
        "Consumption profile",
        options=["even", "winter", "summer"],
        format_func=lambda x: {
            "even":   "Even throughout the year",
            "winter": "Winter-weighted",
            "summer": "Summer-weighted",
        }[x],
    )

    st.divider()
    st.caption("⚡ Electricity Price Scenarios v4.0")


# ── Assembled parameters ──────────────────────────────────────────────────────

scenario_params = ScenarioParams(
    n_simulations=n_simulations,
    crisis_probability=crisis_prob,
    wind_fi_gw=wind_fi_gw,
    solar_fi_gw=solar_fi_gw,
    nuclear_fi=nuclear_fi,
    hydro_nordic=hydro_nordic,
    nuclear_se=nuclear_se,
    gas_price_mwh=gas_price,
    co2_price_t=co2_price,
    electrification_twh=electrification_twh,
    ev_twh=ev_twh,
    datacenter_base_twh=datacenter_base,
    datacenter_growth_pct=datacenter_growth,
    interconnect_fi_se=interconnect_fi_se,
    interconnect_fi_ee=interconnect_fi_ee,
    interconnect_no=interconnect_no,
    se3_price_relative=se3_price_relative,
    se1_price_relative=se1_price_relative,
    ee_price_relative=ee_price_relative,
)

hedge_params = HedgeParams(
    strategy=hedge_strategy,
    fixed_price=fixed_price,
    floor_price=floor_price,
    cap_price=cap_price,
    hedge_pct=hedge_pct,
    forward_premium=fwd_premium,
    spot_pct=spot_pct_h,
    collar_pct=collar_pct_h,
    vol_mwh=vol_mwh,
    dist_type=dist_type,
)

# Capacity parameters
capacity_params = CapacityParams(
    nuclear_fi=nuclear_fi,
    wind_fi_total_gw=wind_total_gw,
    solar_fi_gw=solar_cap_gw,
    interconnect_fi_se_mw=float(fi_se_mw_cap),
    interconnect_fi_ee_mw=float(fi_ee_mw_cap),
)

max_price_diff = compute_max_hintaero(scenario_params)


# ── Cached computation ────────────────────────────────────────────────────────

params_key = (
    n_simulations, crisis_prob,
    wind_fi_gw, solar_fi_gw, nuclear_fi,
    hydro_nordic, nuclear_se,
    gas_price, co2_price,
    electrification_twh, ev_twh,
    datacenter_base, datacenter_growth,
    interconnect_fi_se, interconnect_fi_ee, interconnect_no,
    tuple(sorted(fundamental_df.columns.tolist())) if not fundamental_df.empty else (),
    len(fundamental_df),
)

if "last_params" not in st.session_state or st.session_state.last_params != params_key:
    hist_df = get_historical_data()

    _progress_bar = st.progress(0.0, text="Starting Monte Carlo simulation...")

    try:
        import time as _t
        _progress_bar.progress(0.15, text="Computing low scenario...")
        _t0 = _t.monotonic()

        scenario_results = run_scenarios_cached(
            params_key,
            scenario_params,
            regression if has_excel else RegressionResult(),
            timeout_seconds=30.0,
        )

        _elapsed = _t.monotonic() - _t0
        _computed = len(scenario_results)

        if _elapsed < 0.05:
            _progress_bar.progress(1.0, text=f"Cache hit ({_computed}/3 scenarios)")
        else:
            _progress_bar.progress(1.0, text=f"Done — {_computed}/3 scenarios ({_elapsed:.2f} s)")

        if _computed < 3:
            st.warning(
                f"Timeout (30 s) exceeded — only {_computed}/3 scenarios computed. "
                "Reduce the number of simulations or simplify parameters.",
            )

    except Exception as e:
        _progress_bar.empty()
        st.error(f"Scenario model failed: {e}")
        st.stop()

    st.session_state.update({
        "last_params":      params_key,
        "hist_df":          hist_df,
        "scenario_results": scenario_results,
    })

hist_df          = st.session_state["hist_df"]
scenario_results = st.session_state["scenario_results"]

# Risk metrics (computed always — fast)
risk_metrics = {
    sc: calculate_risk_metrics(scenario_results[sc], vol_mwh)
    for sc in ["low", "base", "high"]
    if sc in scenario_results
}
risk_summary  = get_hedge_recommendation(risk_metrics)

# Hedge analysis
hedge_results = calculate_all_hedges(scenario_results["base"], hedge_params)
hedge_df      = calculate_active_hedge(scenario_results["base"], hedge_params)

# Strategy key → label
strategy_label_map = {
    "spot":          "Full spot",
    "fixed_100":     "Fixed price 100%",
    "partial_5050":  "Partial fixed 50/50",
    "collar":        "Collar strategy",
    "forward":       "Forward hedge",
    "combination":   "Combination strategy",
}
active_label = strategy_label_map.get(hedge_strategy, hedge_strategy)


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════

_tab_labels = [
    "Price Scenarios",
    "Market Dynamics",
    "Risk Analysis",
    "Monthly Analysis",
    "Report",
]
if has_excel:
    _tab_labels = ["Data Analysis"] + _tab_labels

tabs = st.tabs(_tab_labels)
tab_offset = 1 if has_excel else 0


# ══════════════════════════════════════════════════════════════════════════════
# TAB 0 – DATA ANALYSIS (only when Excel uploaded)
# ══════════════════════════════════════════════════════════════════════════════
if has_excel:
    with tabs[0]:
        st.subheader("Historical fundamental data")
        st.caption(
            f"File: **{uploaded_file.name}** | "
            f"Sheet: **{excel_meta.get('used_sheet', '?')}** | "
            f"{len(fundamental_df)} months"
        )

        try:
            st.plotly_chart(fundamental_time_series(fundamental_df), use_container_width=True)
        except Exception as e:
            st.error(f"Time series chart failed: {e}")

        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("Correlation matrix")
            try:
                st.plotly_chart(correlation_heatmap(fundamental_df), use_container_width=True)
            except Exception as e:
                st.error(f"Correlation matrix failed: {e}")

        with col_right:
            st.subheader("Regression model coefficients")
            if regression.r2 > 0:
                st.success(
                    f"Model explains **{regression.r2*100:.1f}%** "
                    f"of historical price variation (R² = {regression.r2:.3f})"
                )
                try:
                    st.plotly_chart(regression_coef_chart(regression), use_container_width=True)
                except Exception as e:
                    st.error(f"Coefficient chart failed: {e}")

                with st.expander("Regression coefficients in table"):
                    coef_df = pd.DataFrame(
                        {"Feature": list(regression.coef.keys()),
                         "Coefficient": list(regression.coef.values())}
                    ).sort_values("Coefficient", ascending=False)
                    coef_df["Coefficient"] = coef_df["Coefficient"].round(4)
                    st.dataframe(coef_df, use_container_width=True, hide_index=True)
            else:
                st.info(
                    "Regression model could not be calibrated. "
                    "Need at least a price column and one explanatory variable.",
                    icon="ℹ️",
                )

        with st.expander("Show raw data (first 20 rows)"):
            st.dataframe(fundamental_df.head(20), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 – PRICE SCENARIOS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[tab_offset]:
    st.subheader("Electricity price scenario paths 2025–2035")

    if has_excel and regression.r2 > 0:
        st.success(
            f"Scenarios calibrated with your data (R² = {regression.r2:.3f}).",
            icon="✅",
        )
    else:
        st.info(
            "Using market parameter defaults. "
            "Upload an Excel file to activate calibration.",
            icon="ℹ️",
        )

    col_cb1, col_cb2, col_cb3 = st.columns(3)
    show_low  = col_cb1.checkbox("Low (optimistic)", value=True)
    show_base = col_cb2.checkbox("Base (most likely)", value=True)
    show_high = col_cb3.checkbox("High (risk scenario)", value=True)

    visible = (
        (["low"]  if show_low  else []) +
        (["base"] if show_base else []) +
        (["high"] if show_high else [])
    )

    try:
        fig_price = price_scenario_chart(scenario_results, hist_df, visible)
        st.plotly_chart(fig_price, use_container_width=True)
    except Exception as e:
        st.error(f"Chart rendering failed: {e}")

    st.caption(
        "Lines: P50 median. Shaded area: P10–P90 uncertainty band. "
        "Dashed: historical 2015–2024 (synthetic)."
    )

    st.subheader("Annual average prices per scenario (€/MWh, P50)")
    ann_price = {
        sc: result.annual_prices.set_index("year")["p50"]
        for sc, result in scenario_results.items()
    }
    price_table = pd.DataFrame(ann_price).rename(columns={
        "low": "Low", "base": "Base", "high": "High"
    })
    price_table.index.name = "Year"

    try:
        st.dataframe(
            price_table.style.format("{:.1f}").background_gradient(cmap="RdYlGn_r", axis=None),
            use_container_width=True,
        )
    except Exception:
        st.dataframe(price_table.round(1), use_container_width=True)

    # Key metrics
    st.divider()
    st.subheader("Key metrics 2025–2035 (base scenario)")
    base_r = scenario_results.get("base")
    if base_r is not None:
        p_2025 = float(base_r.annual_prices[base_r.annual_prices["year"] == 2025]["p50"].values[0])
        p_2035 = float(base_r.annual_prices[base_r.annual_prices["year"] == 2035]["p50"].values[0])
        p_avg  = float(base_r.annual_prices["p50"].mean())
        p_max  = float(base_r.monthly_prices["p95"].max())

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Price 2025 (P50)", f"{p_2025:.1f} €/MWh")
        c2.metric("Price 2035 (P50)", f"{p_2035:.1f} €/MWh", f"{p_2035 - p_2025:+.1f} €/MWh")
        c3.metric("Average price 2025–35", f"{p_avg:.1f} €/MWh")
        c4.metric("Highest monthly price P95", f"{p_max:.1f} €/MWh")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 – MARKET DYNAMICS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[tab_offset + 1]:
    st.subheader("Market Dynamics – Finland 2025–2035")

    # ── SECTION 1: Tornado + Datacenters ──────────────────────────────────────
    col_t1, col_t2 = st.columns([1.5, 1])

    with col_t1:
        st.markdown("### Variable price impact (tornado)")
        try:
            sens_df = compute_variable_sensitivities(scenario_params)
            st.plotly_chart(tornado_chart(sens_df), use_container_width=True)
            st.caption(
                "Bars show how much extreme values of each variable change "
                "the FI price in the base scenario in 2030 (€/MWh)."
            )
        except Exception as e:
            st.error(f"Tornado chart failed: {e}")

    with col_t2:
        st.markdown("### Datacenter growth")
        try:
            dc_df = compute_datacenter_projection(scenario_params)
            st.plotly_chart(datacenter_growth_chart(dc_df), use_container_width=True)
            dc_2035 = dc_df[dc_df["year"] == 2035]["twh"].values[0]
            st.metric(
                "Datacenters 2035",
                f"{dc_2035:.1f} TWh",
                f"+{dc_2035 - datacenter_base:.1f} TWh from baseline",
            )
        except Exception as e:
            st.error(f"Datacenter chart failed: {e}")

    st.divider()

    # ── SECTION 2: Merit Order chart ──────────────────────────────────────────
    st.markdown("### Merit Order curve")
    st.caption(
        "The merit order curve shows the order of generation sources by marginal cost. "
        "The market price is set by the generation source needed to meet demand."
    )

    mo_col1, mo_col2 = st.columns([3, 1])
    with mo_col2:
        mo_month = st.slider(
            "Month for merit order",
            1, 12, 1,
            format="%d",
            key="mo_month_slider",
        )
        st.caption(f"Selected: **{MONTH_NAMES[mo_month]}**")
        mo_water = st.selectbox(
            "Water reservoirs",
            options=["normal", "dry", "wet"],
            format_func=lambda x: {"normal": "Normal", "dry": "Dry", "wet": "Wet"}[x],
            key="mo_water_select",
        )
        mo_demand_mw = st.slider(
            "Demand (MW)",
            2000, 14000, 8000, 100,
            key="mo_demand_slider",
        )

    with mo_col1:
        try:
            # Build capacity for merit order
            mo_cap_dict = {
                "wind":    wind_total_gw * 1000 * {
                    1:0.35,2:0.35,3:0.35,4:0.28,5:0.28,6:0.22,
                    7:0.22,8:0.22,9:0.32,10:0.32,11:0.32,12:0.35
                }.get(mo_month, 0.28),
                "solar":   solar_cap_gw * 1000 * {
                    1:0.02,2:0.04,3:0.08,4:0.12,5:0.16,6:0.18,
                    7:0.17,8:0.15,9:0.10,10:0.05,11:0.02,12:0.01
                }.get(mo_month, 0.05),
                "nuclear": NUCLEAR_OPTIONS_MW.get(nuclear_fi, 2500) * 0.90,
                "hydro":   {  # Fingrid historical 2015–2024 (MW)
                    1:1484,2:1688,3:1622,4:1589,5:1957,6:1544,
                    7:1594,8:1369,9:1412,10:1588,11:1511,12:1596
                }.get(mo_month, 1500) * 0.6,
                "chp":       {
                    1:1484,2:1688,3:1622,4:1589,5:1957,6:1544,
                    7:1594,8:1369,9:1412,10:1588,11:1511,12:1596
                }.get(mo_month, 1500) * 0.4,
                "import":    float(fi_se_mw_cap + fi_ee_mw_cap) * 0.7,
                "gas":       1500.0,
            }

            mo_params = MeritOrderParams(
                gas_price_mwh=float(gas_price),
                co2_price_t=float(co2_price),
                water_level=mo_water,
                nordpool_ref=55.0,
                month=mo_month,
            )
            mo_slices = build_merit_order(mo_params, mo_cap_dict)
            mo_df = merit_order_to_df(mo_slices)

            # Compute market price
            mo_price, mo_source, mo_surplus = calculate_market_price(
                mo_month, mo_cap_dict, mo_demand_mw,
                mo_water, float(gas_price), float(co2_price),
            )

            # Draw stepwise merit order chart
            fig_mo = go.Figure()

            cum_prev = 0.0
            for _, row in mo_df.iterrows():
                src = row["source"]
                color = SOURCE_COLORS.get(src, "#9E9E9E")
                cum_now = row["cumulative_mw"]
                mc = row["marginal_cost"]

                # Color in rgba format
                hex_c = color.lstrip("#")
                r_c = int(hex_c[0:2], 16)
                g_c = int(hex_c[2:4], 16)
                b_c = int(hex_c[4:6], 16)
                fill_rgba = f"rgba({r_c},{g_c},{b_c},0.7)"

                # Step-bar with two points (left edge to right edge)
                fig_mo.add_trace(go.Scatter(
                    x=[cum_prev, cum_prev, cum_now, cum_now],
                    y=[0, mc, mc, 0],
                    fill="toself",
                    fillcolor=fill_rgba,
                    line=dict(color=color, width=1.5),
                    name=src,
                    hovertemplate=(
                        f"<b>{src}</b><br>"
                        f"Capacity: {row['capacity_mw']:.0f} MW<br>"
                        f"Marginal cost: {mc:.1f} €/MWh<br>"
                        f"Cumulative: {cum_now:.0f} MW<extra></extra>"
                    ),
                    showlegend=True,
                ))
                cum_prev = cum_now

            # Demand line
            fig_mo.add_vline(
                x=mo_demand_mw,
                line_dash="dash",
                line_color="#E53935",
                line_width=2,
                annotation_text=f"Demand {mo_demand_mw:,} MW",
                annotation_position="top left",
            )

            # Market price line
            fig_mo.add_hline(
                y=mo_price,
                line_dash="dot",
                line_color="#FF6F00",
                line_width=2,
                annotation_text=f"Market price {mo_price:.1f} €/MWh ({mo_source})",
                annotation_position="bottom right",
            )

            fig_mo.update_layout(
                title=f"Merit order – {MONTH_NAMES[mo_month]} | Hydro: {mo_water}",
                xaxis_title="Cumulative capacity (MW)",
                yaxis_title="Marginal cost (€/MWh)",
                plot_bgcolor="white",
                paper_bgcolor="white",
                height=420,
                margin=dict(l=60, r=20, t=80, b=60),
                xaxis=dict(gridcolor="#E0E0E0"),
                yaxis=dict(gridcolor="#E0E0E0"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            )
            st.plotly_chart(fig_mo, use_container_width=True)

            # Market price result
            surplus_text = f"+{mo_surplus:.0f} MW surplus" if mo_surplus >= 0 else f"{mo_surplus:.0f} MW deficit"
            col_mo_a, col_mo_b, col_mo_c = st.columns(3)
            col_mo_a.metric("Merit order price", f"{mo_price:.1f} €/MWh")
            col_mo_b.metric("Marginal source", mo_source)
            col_mo_c.metric("Capacity situation", surplus_text)

        except Exception as e:
            st.error(f"Merit order chart failed: {e}")

    st.divider()

    # ── SECTION 3: Capacity model ─────────────────────────────────────────────
    st.markdown("### Capacity vs demand 2025–2035")

    try:
        cap_ts = capacity_time_series(capacity_params, 2025, 2035)

        # Compute monthly demand (MW peak demand)
        base_demand_mw = (FI_BASE_CONSUMPTION_TWH + total_growth / 2) * 1e6 / 8760
        monthly_demand_factor_cap = {
            1:1.35,2:1.30,3:1.10,4:0.95,5:0.85,6:0.80,
            7:0.82,8:0.88,9:0.95,10:1.05,11:1.22,12:1.38
        }
        cap_ts["demand_mw"] = cap_ts["month"].map(
            lambda m: base_demand_mw * monthly_demand_factor_cap.get(m, 1.0) * 1.6
        )
        cap_ts["label"] = cap_ts.apply(
            lambda r: f"{int(r['year'])}-{int(r['month']):02d}", axis=1
        )
        cap_ts["surplus"] = cap_ts["total_mw"] + cap_ts["interconnect_mw"] - cap_ts["demand_mw"]

        fig_cap = go.Figure()

        # Generation sources stacked
        cap_colors = {
            "nuclear_mw":   "rgba(156,39,176,0.8)",
            "wind_mw":      "rgba(33,150,243,0.8)",
            "solar_mw":     "rgba(255,193,7,0.8)",
            "hydro_chp_mw": "rgba(0,188,212,0.8)",
        }
        cap_labels = {
            "nuclear_mw":   "Nuclear power",
            "wind_mw":      "Wind power",
            "solar_mw":     "Solar power",
            "hydro_chp_mw": "Hydro power + CHP",
        }

        for col_key in ["nuclear_mw", "wind_mw", "solar_mw", "hydro_chp_mw"]:
            fig_cap.add_trace(go.Bar(
                x=cap_ts["label"],
                y=cap_ts[col_key],
                name=cap_labels[col_key],
                marker_color=cap_colors[col_key],
                hovertemplate=f"%{{x}}: %{{y:.0f}} MW<extra>{cap_labels[col_key]}</extra>",
            ))

        # Interconnections
        fig_cap.add_trace(go.Scatter(
            x=cap_ts["label"],
            y=cap_ts["total_mw"] + cap_ts["interconnect_mw"],
            name="Generation + interconnections",
            line=dict(color="rgba(0,150,136,1)", width=2, dash="dash"),
            hovertemplate="%{x}: %{y:.0f} MW<extra>Total supply</extra>",
        ))

        # Demand
        fig_cap.add_trace(go.Scatter(
            x=cap_ts["label"],
            y=cap_ts["demand_mw"],
            name="Peak demand",
            line=dict(color="rgba(229,57,53,1)", width=2.5),
            hovertemplate="%{x}: %{y:.0f} MW<extra>Demand</extra>",
        ))

        fig_cap.update_layout(
            title="Capacity vs demand 2025–2035 (monthly)",
            xaxis_title="Month",
            yaxis_title="Power (MW)",
            barmode="stack",
            height=480,
            plot_bgcolor="white",
            paper_bgcolor="white",
            margin=dict(l=60, r=20, t=80, b=60),
            xaxis=dict(
                tickangle=-45,
                tickmode="array",
                tickvals=[f"{y}-01" for y in range(2025, 2036)],
                ticktext=[str(y) for y in range(2025, 2036)],
                gridcolor="#E0E0E0",
            ),
            yaxis=dict(gridcolor="#E0E0E0"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        st.plotly_chart(fig_cap, use_container_width=True)

        # Critical months
        crit_df = find_critical_months(capacity_params, FI_BASE_CONSUMPTION_TWH + total_growth / 2)
        if not crit_df.empty:
            st.warning(
                f"Found {len(crit_df)} capacity deficit months in 2025–2035.",
                icon="⚠️",
            )
            crit_display = crit_df.copy()
            crit_display["Month"] = crit_display["month"].map(
                lambda m: MONTH_NAMES.get(m, str(m))
            )
            crit_display = crit_display.rename(columns={
                "year": "Year",
                "surplus_mw": "Deficit (MW)",
                "utilization_pct": "Utilization (%)",
                "premium_eur_mwh": "Price premium (€/MWh)",
            })
            st.dataframe(
                crit_display[["Year", "Month", "Deficit (MW)", "Utilization (%)", "Price premium (€/MWh)"]],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.success("No capacity deficit months with the selected parameters.", icon="✅")

    except Exception as e:
        st.error(f"Capacity chart failed: {e}")

    st.divider()

    # ── Electricity consumption breakdown 2025 (Fingrid data) ────────────────
    st.markdown("### Electricity consumption breakdown 2025 (Fingrid)")

    col_pie, col_monthly = st.columns(2)

    with col_pie:
        labels = list(FI_CONSUMPTION_BREAKDOWN_2025.keys())
        values = list(FI_CONSUMPTION_BREAKDOWN_2025.values())
        colors = ["#1565C0","#42A5F5","#26C6DA","#66BB6A","#FFA726","#EF5350","#9E9E9E"]
        fig_pie = go.Figure(go.Pie(
            labels=labels, values=values,
            marker=dict(colors=colors),
            textinfo="label+percent",
            hovertemplate="%{label}: %{value:.1f} TWh<extra></extra>",
        ))
        fig_pie.update_layout(
            title=f"Total consumption {sum(values):.1f} TWh",
            height=350, showlegend=False,
            margin=dict(t=40, b=0, l=0, r=0),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_monthly:
        months_en = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        monthly_vals = [FI_MONTHLY_CONSUMPTION_2025[m] for m in range(1, 13)]
        fig_bar = go.Figure(go.Bar(
            x=months_en, y=monthly_vals,
            marker_color="#1565C0",
            hovertemplate="%{x}: %{y:.2f} TWh<extra></extra>",
        ))
        fig_bar.update_layout(
            title="Monthly consumption 2025 (TWh)",
            yaxis_title="TWh", height=350,
            plot_bgcolor="white", paper_bgcolor="white",
            yaxis=dict(gridcolor="#E0E0E0"),
            margin=dict(t=40, b=0, l=0, r=0),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.divider()

    # ── SECTION 4: Consumption growth summary + interconnections ─────────────
    col_m1, col_m2 = st.columns(2)

    with col_m1:
        st.markdown("### Consumption growth summary")
        growth_info = compute_consumption_growth(scenario_params)
        st.metric("Consumption 2025 (current level)", f"{growth_info['current_twh']:.0f} TWh")
        st.metric(
            "Consumption 2035 (forecast)",
            f"{growth_info['forecast_2035_twh']:.0f} TWh",
            f"+{growth_info['growth_twh']:.0f} TWh",
        )
        st.progress(min(growth_info["growth_twh"] / 40.0, 1.0))
        st.caption(
            f"Electrification: **{electrification_twh} TWh** | "
            f"Electric vehicles: **{ev_twh} TWh** | "
            f"Datacenters: **+{dc_final_disp - datacenter_base:.1f} TWh**"
        )

        st.markdown("#### Price impact – parameter effect on base scenario")
        _impact2 = compute_impact_breakdown(scenario_params, ref_years=(2030, 2035))
        _cols_imp = st.columns(2)
        for _ci, _yr in enumerate((2030, 2035)):
            _d = _impact2[_yr]
            with _cols_imp[_ci]:
                st.markdown(f"**Year {_yr}**")
                st.metric("Base scenario P50", f"{_d['base_price']:.1f} €/MWh")

                def _sign_str(v: float) -> str:
                    return f"+{v:.1f}" if v >= 0 else f"{v:.1f}"

                st.markdown(
                    f"| Factor | Impact |\n"
                    f"|--------|--------|\n"
                    f"| Consumption growth ({_d['total_growth_twh']:.0f} TWh) "
                    f"| **{_sign_str(_d['consumption_impact'])} €/MWh** |\n"
                    f"| Datacenters (+{_d['dc_growth_twh']:.1f} TWh) "
                    f"| **{_sign_str(_d['datacenter_impact'])} €/MWh** |\n"
                    f"| Wind power (+{_d['wind_re_twh']:.0f} TWh/y) "
                    f"| **{_sign_str(_d['wind_impact'])} €/MWh** |"
                )

    with col_m2:
        st.markdown("### Interconnections – price difference FI vs Nordic")
        _, fi_se_mw_disp = INTERCONNECT_FI_SE_OPTIONS.get(interconnect_fi_se, ("", 2200))
        _, fi_ee_mw_disp = INTERCONNECT_FI_EE_OPTIONS.get(interconnect_fi_ee, ("", 1000))
        total_ic_gw = (fi_se_mw_disp + fi_ee_mw_disp) / 1000.0
        st.metric(
            "Max allowed price diff FI–Nordic",
            f"{max_price_diff:.0f} €/MWh",
            help="500 / total_capacity_GW",
        )
        st.metric(
            "Total interconnection capacity",
            f"{total_ic_gw:.1f} GW",
            f"FI–SE: {fi_se_mw_disp:,} MW + FI–EE: {fi_ee_mw_disp:,} MW",
        )
        try:
            fig_ic = interconnect_hintaero_chart(scenario_results, max_price_diff)
            st.plotly_chart(fig_ic, use_container_width=True)
        except Exception as e:
            st.error(f"Price difference chart failed: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 – RISK ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[tab_offset + 2]:
    st.subheader("Risk analysis and hedging strategies")

    # ── SECTION 1: Price risk without hedging ─────────────────────────────────
    st.markdown("### Section 1 – Price risk without hedging")

    risk_scenario = st.selectbox(
        "Scenario for risk analysis",
        options=["low", "base", "high"],
        format_func=lambda x: {"low": "Low", "base": "Base", "high": "High"}[x],
        index=1,
        key="risk_scenario_select",
    )

    try:
        fig_pct = price_percentile_paths(scenario_results, risk_scenario)
        st.plotly_chart(fig_pct, use_container_width=True)
    except Exception as e:
        st.error(f"Percentile paths failed: {e}")

    rm = risk_metrics.get(risk_scenario)
    if rm:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("VaR 95% (€/MWh)", f"{rm.var_95:.1f}")
        c2.metric("CVaR 95% (€/MWh)", f"{rm.cvar_95:.1f}")
        c3.metric("Max monthly price (€/MWh)", f"{rm.max_monthly_price:.1f}")
        c4.metric("Volatility (€/MWh)", f"{rm.volatility:.1f}")
        c5.metric(
            "Price spike risk",
            f"{rm.spike_prob * 100:.1f}%",
            help="Probability that some month exceeds 150 €/MWh",
        )

    st.markdown("#### Risk metrics for all scenarios")
    risk_table = build_risk_metrics_table(risk_metrics)
    try:
        st.dataframe(
            risk_table.style.format({
                "VaR 95% (€/MWh)": "{:.1f}",
                "CVaR 95% (€/MWh)": "{:.1f}",
                "Max Monthly Price (€/MWh)": "{:.1f}",
                "Volatility (€/MWh)": "{:.1f}",
                "Price Spike Risk (%)": "{:.1f}",
            }).background_gradient(
                cmap="RdYlGn_r",
                subset=["CVaR 95% (€/MWh)", "Price Spike Risk (%)"],
            ),
            use_container_width=True,
            hide_index=True,
        )
    except Exception:
        st.dataframe(risk_table, use_container_width=True, hide_index=True)

    st.divider()

    # ── SECTION 2: Hedging strategy comparison ────────────────────────────────
    st.markdown("### Section 2 – Hedging strategy comparison")
    st.caption(f"Base scenario | Volume: {vol_mwh:,.0f} MWh/y")

    try:
        fig_hedge = hedge_comparison_chart(hedge_results, vol_mwh)
        st.plotly_chart(fig_hedge, use_container_width=True)
    except Exception as e:
        st.error(f"Hedge comparison failed: {e}")

    hedge_table = pd.DataFrame([{
        "Strategy":                       h.strategy_name,
        "Price P50 (€/MWh)":             h.effective_price_p50,
        "Price P95 (€/MWh)":             h.effective_price_p95,
        "Cost P50 (k€/y)":               round(h.annual_cost_p50 / 1000, 1),
        "Cost P95 (k€/y)":               round(h.annual_cost_p95 / 1000, 1),
        "Additional cost vs spot P50 (k€)": round(h.hedge_cost_vs_spot_p50 / 1000, 1),
        "Savings vs spot P95 (k€)":      round(h.hedge_benefit_vs_spot_p95 / 1000, 1),
        "Risk reduction (%)":            h.risk_reduction_ratio,
    } for h in hedge_results])

    try:
        st.dataframe(
            hedge_table.style
            .highlight_min(subset=["Cost P95 (k€/y)"], color="#C8E6C9")
            .highlight_max(subset=["Risk reduction (%)"], color="#C8E6C9")
            .format({
                "Price P50 (€/MWh)": "{:.1f}",
                "Price P95 (€/MWh)": "{:.1f}",
                "Cost P50 (k€/y)": "{:.1f}",
                "Cost P95 (k€/y)": "{:.1f}",
                "Additional cost vs spot P50 (k€)": "{:+.1f}",
                "Savings vs spot P95 (k€)": "{:+.1f}",
                "Risk reduction (%)": "{:.1f}",
            }),
            use_container_width=True,
            hide_index=True,
        )
    except Exception:
        st.dataframe(hedge_table, use_container_width=True, hide_index=True)

    try:
        fig_h_annual = hedge_annual_cost_chart(hedge_df, active_label, color="#1565C0")
        st.plotly_chart(fig_h_annual, use_container_width=True)
    except Exception as e:
        st.error(f"Effective hedge price chart failed: {e}")

    st.divider()

    # ── SECTION 3: Stress tests ───────────────────────────────────────────────
    st.markdown("### Section 3 – Stress tests")

    with st.spinner("Running stress tests..."):
        stress_tests = run_stress_tests(scenario_params, vol_mwh)

    try:
        fig_stress = stress_test_chart(stress_tests)
        st.plotly_chart(fig_stress, use_container_width=True)
    except Exception as e:
        st.error(f"Stress test chart failed: {e}")

    stress_table = pd.DataFrame([{
        "Scenario":              t.name,
        "Description":           t.description,
        "Price spike (€/MWh)":   t.price_spike,
        "Increase (%)":          t.price_increase_pct,
        "Duration (months)":     t.duration_months,
        "Cost impact (k€)":      round(t.annual_cost_impact / 1000, 1),
        "Best hedge":            t.best_hedge,
    } for t in stress_tests])

    try:
        st.dataframe(
            stress_table.style.format({
                "Price spike (€/MWh)": "{:.1f}",
                "Increase (%)": "{:.1f}",
                "Cost impact (k€)": "{:.1f}",
            }).background_gradient(
                cmap="RdYlGn_r",
                subset=["Price spike (€/MWh)"],
            ),
            use_container_width=True,
            hide_index=True,
        )
    except Exception:
        st.dataframe(stress_table, use_container_width=True, hide_index=True)

    st.divider()

    # ── SECTION 4: Efficient frontier and recommendation ─────────────────────
    st.markdown("### Section 4 – Optimal hedging recommendation")

    st.info(risk_summary.get("recommendation_text", ""), icon="💡")

    col_ef1, col_ef2 = st.columns([2, 1])
    with col_ef1:
        try:
            frontier_df = compute_efficient_frontier(
                scenario_results["base"],
                vol_mwh,
                fixed_price=fixed_price,
                floor_price=floor_price,
                cap_price=cap_price,
                forward_premium=fwd_premium,
            )
            fig_ef = efficient_frontier_chart(frontier_df)
            st.plotly_chart(fig_ef, use_container_width=True)
        except Exception as e:
            st.error(f"Efficient frontier failed: {e}")

    with col_ef2:
        st.markdown("**Risk parameters (base scenario)**")
        pm = risk_metrics.get("base")
        if pm:
            st.metric("Risk level", risk_summary.get("risk_class", "–").capitalize())
            st.metric("VaR 95%", f"{pm.var_95:.1f} €/MWh")
            st.metric("CVaR 95%", f"{pm.cvar_95:.1f} €/MWh")
            st.metric("Volatility", f"{pm.volatility:.1f} €/MWh")
        st.markdown("**Best strategy (CVaR-min)**")
        st.success(risk_summary.get("best_strategy", "–"))


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 – MONTHLY ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[tab_offset + 3]:
    st.subheader("Monthly price analysis")

    hm_scenario = st.selectbox(
        "Scenario for heatmap",
        options=["low", "base", "high"],
        format_func=lambda x: {"low": "Low", "base": "Base", "high": "High"}[x],
        index=1,
        key="hm_scenario_select",
    )

    try:
        st.plotly_chart(monthly_heatmap(scenario_results, hm_scenario), use_container_width=True)
    except Exception as e:
        st.error(f"Heatmap failed: {e}")

    col_l, col_r = st.columns(2)

    with col_l:
        try:
            st.plotly_chart(monthly_avg_bar(scenario_results, hm_scenario), use_container_width=True)
        except Exception as e:
            st.error(f"Monthly bar chart failed: {e}")

    with col_r:
        st.subheader("Seasonal optimization recommendation")
        ref = scenario_results["base"].monthly_prices
        month_avg = ref.groupby("month")["p50"].mean()

        st.write("**Most expensive months (base):**")
        for m, p in month_avg.nlargest(3).items():
            st.write(f"- {MONTH_NAMES[m]}: {p:.1f} €/MWh")

        st.write("**Cheapest months (base):**")
        for m, p in month_avg.nsmallest(3).items():
            st.write(f"- {MONTH_NAMES[m]}: {p:.1f} €/MWh")

        winter_price = float(month_avg[[12, 1, 2]].mean())
        summer_price  = float(month_avg[[6, 7, 8]].mean())
        diff = winter_price - summer_price

        if diff > 5:
            st.success(
                f"Shifting consumption from winter to summer is worthwhile:\n\n"
                f"- Winter P50 price: {winter_price:.1f} €/MWh\n"
                f"- Summer P50 price: {summer_price:.1f} €/MWh\n"
                f"- Price difference: {diff:.1f} €/MWh"
            )
        else:
            st.info("The winter-summer price difference is small with the selected parameters.")

    st.divider()

    # ── Monthly merit order time series ──────────────────────────────────────
    st.markdown("### Monthly merit order price (year 2025)")
    st.caption(
        "Merit order calculation for each month with current capacity and demand parameters."
    )

    try:
        mo_base_cap = {
            "wind":    wind_total_gw * 1000 * 0.28,
            "solar":   solar_cap_gw * 1000 * 0.08,
            "nuclear": NUCLEAR_OPTIONS_MW.get(nuclear_fi, 2500) * 0.90,
            "hydro":   1200.0,
            "chp":     800.0,
            "import":  float(fi_se_mw_cap + fi_ee_mw_cap) * 0.7,
            "gas":     1500.0,
        }
        # Seasonal capacity adjustment
        wind_cf_map = {
            1:0.35,2:0.35,3:0.35,4:0.28,5:0.28,6:0.22,
            7:0.22,8:0.22,9:0.32,10:0.32,11:0.32,12:0.35
        }
        solar_cf_map = {
            1:0.02,2:0.04,3:0.08,4:0.12,5:0.16,6:0.18,
            7:0.17,8:0.15,9:0.10,10:0.05,11:0.02,12:0.01
        }
        hydro_mw_map = {  # Fingrid historical 2015–2024, monthly averages (MW)
            1:1484,2:1688,3:1622,4:1589,5:1957,6:1544,
            7:1594,8:1369,9:1412,10:1588,11:1511,12:1596
        }
        demand_factor_map = {
            1:1.35,2:1.30,3:1.10,4:0.95,5:0.85,6:0.80,
            7:0.82,8:0.88,9:0.95,10:1.05,11:1.22,12:1.38
        }
        dc_twh_2030 = datacenter_base * ((1 + datacenter_growth / 100) ** 5)
        total_consumption_2030 = FI_BASE_CONSUMPTION_TWH + (electrification_twh + ev_twh) * 0.5 + max(dc_twh_2030 - datacenter_base, 0)
        avg_demand_mo = total_consumption_2030 * 1e6 / 8760

        mo_monthly_records = []
        for mo_m in range(1, 13):
            mo_m_cap = {
                "wind":    wind_total_gw * 1000 * wind_cf_map[mo_m],
                "solar":   solar_cap_gw * 1000 * solar_cf_map[mo_m],
                "nuclear": NUCLEAR_OPTIONS_MW.get(nuclear_fi, 2500) * 0.90,
                "hydro":   hydro_mw_map[mo_m] * 0.6,
                "chp":     hydro_mw_map[mo_m] * 0.4,
                "import":  float(fi_se_mw_cap + fi_ee_mw_cap) * 0.7,
                "gas":     1500.0,
            }
            mo_demand_m = avg_demand_mo * demand_factor_map[mo_m] * 1.6
            mo_p, mo_src, mo_sur = calculate_market_price(
                mo_m, mo_m_cap, mo_demand_m,
                hydro_nordic, float(gas_price), float(co2_price),
            )
            mo_monthly_records.append({
                "month": mo_m,
                "month_label": MONTH_SHORT[mo_m],
                "mo_price": mo_p,
                "marginal_source": mo_src,
                "surplus_mw": mo_sur,
            })

        mo_monthly_df = pd.DataFrame(mo_monthly_records)

        # Chart
        bar_colors_mo = [
            SOURCE_COLORS.get(src, "#9E9E9E")
            for src in mo_monthly_df["marginal_source"]
        ]

        fig_mo_monthly = go.Figure()
        fig_mo_monthly.add_trace(go.Bar(
            x=mo_monthly_df["month_label"],
            y=mo_monthly_df["mo_price"],
            marker_color=bar_colors_mo,
            hovertemplate=(
                "%{x}: %{y:.1f} €/MWh"
                "<br>Marginal source: " +
                mo_monthly_df["marginal_source"].astype(str) +
                "<extra></extra>"
            ).tolist(),
            name="Merit order price",
        ))

        # Add spot price P50 for comparison
        base_monthly_avg = (
            scenario_results["base"].monthly_prices
            .groupby("month")["p50"]
            .mean()
            .reset_index()
        )
        fig_mo_monthly.add_trace(go.Scatter(
            x=base_monthly_avg["month"].map(MONTH_SHORT),
            y=base_monthly_avg["p50"],
            name="Spot P50 (base scenario)",
            line=dict(color="#1565C0", width=2, dash="dot"),
            hovertemplate="%{x}: %{y:.1f} €/MWh<extra>Spot P50</extra>",
        ))

        fig_mo_monthly.update_layout(
            title="Monthly merit order price vs spot P50",
            xaxis_title="Month",
            yaxis_title="Price (€/MWh)",
            plot_bgcolor="white",
            paper_bgcolor="white",
            height=380,
            margin=dict(l=60, r=20, t=80, b=60),
            yaxis=dict(gridcolor="#E0E0E0"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_mo_monthly, use_container_width=True)

        # Table
        mo_table = mo_monthly_df.rename(columns={
            "month_label": "Month",
            "mo_price": "Merit order price (€/MWh)",
            "marginal_source": "Marginal source",
            "surplus_mw": "Capacity surplus (MW)",
        })
        st.dataframe(
            mo_table[["Month", "Merit order price (€/MWh)", "Marginal source", "Capacity surplus (MW)"]].style.format({
                "Merit order price (€/MWh)": "{:.1f}",
                "Capacity surplus (MW)": "{:.0f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

    except Exception as e:
        st.error(f"Monthly merit order failed: {e}")

    st.divider()

    # ── Consumption curve 2025–2035 ───────────────────────────────────────────
    st.markdown("### Electricity consumption development 2025–2035")
    st.caption("Consumption forecast based on sidebar parameters.")

    years_range = list(range(2025, 2036))
    base_cons = FI_BASE_CONSUMPTION_TWH
    elec_growth = electrification_twh + ev_twh

    consumption_base  = [base_cons] * len(years_range)
    consumption_elec  = [base_cons + elec_growth * (y - 2025) / 10 for y in years_range]
    consumption_total = [
        base_cons
        + elec_growth * (y - 2025) / 10
        + max(datacenter_base * ((1 + datacenter_growth / 100) ** (y - 2025)) - datacenter_base, 0)
        for y in years_range
    ]

    fig_cons = go.Figure()
    fig_cons.add_trace(go.Scatter(
        x=years_range, y=consumption_base,
        name="Base consumption (no growth)",
        line=dict(color="#9E9E9E", width=1, dash="dot"),
        hovertemplate="%{x}: %{y:.1f} TWh<extra>Base consumption</extra>",
    ))
    fig_cons.add_trace(go.Scatter(
        x=years_range, y=consumption_elec,
        name="+ Electrification & EVs",
        line=dict(color="#1976D2", width=2),
        hovertemplate="%{x}: %{y:.1f} TWh<extra>Electrification</extra>",
    ))
    fig_cons.add_trace(go.Scatter(
        x=years_range, y=consumption_total,
        name="+ Datacenters (total forecast)",
        line=dict(color="#E53935", width=2),
        fill="tonexty",
        fillcolor="rgba(229,57,53,0.10)",
        hovertemplate="%{x}: %{y:.1f} TWh<extra>Total forecast</extra>",
    ))
    fig_cons.update_layout(
        xaxis_title="Year",
        yaxis_title="Consumption (TWh/year)",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=360,
        margin=dict(l=60, r=20, t=40, b=60),
        yaxis=dict(gridcolor="#E0E0E0"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_cons, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 – REPORT
# ══════════════════════════════════════════════════════════════════════════════
with tabs[tab_offset + 4]:
    st.subheader("Analysis summary and downloads")

    data_notes = (
        excel_meta.get("assumptions", [])
        if has_excel
        else ["Used synthetic historical data."]
    )

    try:
        summary = generate_summary_text(
            scenario_results, scenario_params, risk_summary, data_notes
        )
        st.info(summary)
    except Exception as e:
        st.error(f"Summary generation failed: {e}")

    if has_excel:
        st.success(
            f"**Using your data:** {uploaded_file.name} | "
            f"R² = {regression.r2:.3f} | "
            f"Sheet: {excel_meta.get('used_sheet', '?')}",
            icon="✅",
        )
    else:
        st.warning(
            "Scenarios are based on market parameters and synthetic data. "
            "Upload an Excel file from the sidebar to activate calibration.",
            icon="⚠️",
        )

    st.divider()
    st.subheader("Download data and reports")

    col_d1, col_d2, col_d3 = st.columns(3)

    with col_d1:
        try:
            sc_csv = scenarios_to_dataframe(scenario_results)
            st.download_button(
                "Download scenarios_data.csv",
                data=sc_csv.to_csv(index=False, float_format="%.2f"),
                file_name="scenarios_data.csv",
                mime="text/csv",
                help="Monthly prices P5–P95 for all scenarios",
            )
        except Exception as e:
            st.error(f"Scenario file: {e}")

    with col_d2:
        try:
            risk_csv = build_risk_metrics_table(risk_metrics)
            st.download_button(
                "Download risk_analysis.csv",
                data=risk_csv.to_csv(index=False, float_format="%.2f"),
                file_name="risk_analysis.csv",
                mime="text/csv",
                help="Risk metrics for all scenarios",
            )
        except Exception as e:
            st.error(f"Risk file: {e}")

    with col_d3:
        try:
            pdf_bytes = build_pdf_report(
                scenario_results=scenario_results,
                params=scenario_params,
                n_simulations=n_simulations,
                r2=regression.r2,
                risk_summary=risk_summary,
                hedge_results=hedge_results,
                data_notes=data_notes,
            )
            st.download_button(
                "Download PDF report",
                data=pdf_bytes,
                file_name="electricity_scenario_report.pdf",
                mime="application/pdf",
                help="Complete PDF report",
            )
        except Exception as e:
            st.error(f"PDF report: {e}")

    # Merit order summary
    st.divider()
    st.subheader("Merit order price summary (all months)")
    try:
        mo_all_cap_avg = {
            "wind":    wind_total_gw * 1000 * 0.28,
            "solar":   solar_cap_gw * 1000 * 0.06,
            "nuclear": NUCLEAR_OPTIONS_MW.get(nuclear_fi, 2500) * 0.90,
            "hydro":   1200.0,
            "chp":     800.0,
            "import":  float(fi_se_mw_cap + fi_ee_mw_cap) * 0.7,
            "gas":     1500.0,
        }
        mo_ts = merit_order_time_series(
            range(1, 13),
            mo_all_cap_avg,
            FI_BASE_CONSUMPTION_TWH * 1e6 / 8760 * 1.4,
            float(gas_price),
            float(co2_price),
            hydro_nordic,
        )
        mo_ts["month_label"] = mo_ts["month"].map(MONTH_SHORT)
        mo_ts_display = mo_ts.rename(columns={
            "month_label": "Month",
            "price_eur_mwh": "Merit order price (€/MWh)",
            "marginal_source": "Marginal source",
            "surplus_mw": "Surplus (MW)",
        })
        st.dataframe(
            mo_ts_display[["Month", "Merit order price (€/MWh)", "Marginal source", "Surplus (MW)"]].style.format({
                "Merit order price (€/MWh)": "{:.1f}",
                "Surplus (MW)": "{:.0f}",
            }),
            use_container_width=True,
            hide_index=True,
        )
    except Exception as e:
        st.error(f"Merit order summary: {e}")

    st.divider()
    with st.expander("Market parameters"):
        nuclear_fi_label = NUCLEAR_FI_OPTIONS.get(nuclear_fi, ("–",))[0]
        nuclear_se_label = NUCLEAR_SE_OPTIONS.get(nuclear_se, ("–",))[0]
        hydro_label      = HYDRO_OPTIONS.get(hydro_nordic, ("–",))[0]
        fi_se_label      = INTERCONNECT_FI_SE_OPTIONS.get(interconnect_fi_se, ("–",))[0]
        fi_ee_label      = INTERCONNECT_FI_EE_OPTIONS.get(interconnect_fi_ee, ("–",))[0]

        param_info = {
            "Wind power additional capacity":    f"{wind_fi_gw:.1f} GW",
            "Total wind power (capacity)":       f"{wind_total_gw:.1f} GW",
            "Solar energy growth":               f"{solar_fi_gw:.1f} GW",
            "Total solar power":                 f"{solar_cap_gw:.1f} GW",
            "Nuclear FI":                        nuclear_fi_label,
            "Nordic hydro":                      hydro_label,
            "Swedish nuclear":                   nuclear_se_label,
            "Gas price":                         f"{gas_price} €/MWh",
            "CO₂ price":                         f"{co2_price} €/t",
            "Electrification + heat pumps":      f"{electrification_twh} TWh",
            "Electric vehicles":                 f"{ev_twh} TWh",
            "Datacenters (baseline)":            f"{datacenter_base:.1f} TWh",
            "Datacenter growth rate":            f"{datacenter_growth} %/y",
            "FI–SE interconnection":             fi_se_label,
            "FI–EE interconnection":             fi_ee_label,
            "Energy crisis probability":         f"{crisis_prob*100:.0f}%",
            "Monte Carlo runs":                  str(n_simulations),
            "Hedging strategy":                  active_label,
            "Hedge volume":                      f"{vol_mwh:,.0f} MWh/y",
            "Excel uploaded":                    "Yes" if has_excel else "No",
            "Regression R²":                     f"{regression.r2:.3f}" if regression.r2 > 0 else "–",
        }
        param_df = pd.DataFrame.from_dict(param_info, orient="index", columns=["Value"])
        param_df["Value"] = param_df["Value"].astype(str)
        st.table(param_df)
