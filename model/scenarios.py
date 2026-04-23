"""
scenarios.py – Scenario model and Monte Carlo simulation for electricity prices 2025–2035.

Comprehensive market model: FI wind/solar/nuclear, consumption growth, datacenters,
Nordic hydro reserves, Swedish nuclear, interconnections, gas price, CO2.
"""

import copy
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from model.merit_order import chp_marginal, gas_marginal

logger = logging.getLogger(__name__)

# Merit order reference prices (calibration parameters gas=40, CO₂=70)
_MO_REF_GAS  = 40.0
_MO_REF_CO2  = 70.0
_MO_BLENDED_REF = (
    0.60 * chp_marginal(_MO_REF_GAS, _MO_REF_CO2)   # CHP ~60% of gas power
    + 0.40 * gas_marginal(_MO_REF_GAS, _MO_REF_CO2)  # CCGT ~40%
)  # ≈ 0.60*63.5 + 0.40*98.2 ≈ 77.4 €/MWh
# Gas power share of Finland price signal (~35% of hours)
_GAS_MARKET_SHARE = 0.35

# ── Constants ─────────────────────────────────────────────────────────────────

MONTH_FACTORS = {
    1: 1.38, 2: 1.30, 3: 1.12, 4: 0.95, 5: 0.85, 6: 0.80,
    7: 0.82, 8: 0.88, 9: 0.95, 10: 1.05, 11: 1.22, 12: 1.40,
}

SCENARIO_NAMES  = ["low", "base", "high"]
SCENARIO_LABELS = {
    "low":  "Low (optimistic)",
    "base": "Base (most likely)",
    "high": "High (risk scenario)",
}
SCENARIO_COLORS = {
    "low":  "#2E7D32",
    "base": "#1565C0",
    "high": "#B71C1C",
}

START_YEAR = 2025
END_YEAR   = 2038
FI_BASE_CONSUMPTION_TWH = 84.5   # Finland actual consumption 2025 (Fingrid dataset 124)

# Consumption breakdown 2025 (TWh) — Fingrid + Energia.fi + Business Finland
FI_CONSUMPTION_BREAKDOWN_2025 = {
    "Industry":       40.5,   # pulp, paper, metals, chemicals
    "Households":     19.0,
    "Services":       16.5,
    "Heat pumps":      6.0,   # ~1M heat pumps
    "Data centers":    2.5,   # Helsinki datacenter hub
    "Electric vehicles": 0.3, # ~130k EVs @ 2500 kWh/yr
    "Other":           0.7,   # transport, agriculture, losses
}

# Fingrid dataset 124: monthly consumption 2025 (TWh)
FI_MONTHLY_CONSUMPTION_2025 = {
    1: 8.35, 2: 7.72, 3: 7.79, 4: 6.93, 5: 6.66, 6: 5.88,
    7: 6.05, 8: 6.27, 9: 6.05, 10: 7.01, 11: 7.62, 12: 8.22,
}

# Wind power capture rate: wind receives ~70% of average spot price
# (negative correlation between wind output and spot price)
# Used in future investment return calculations (LCOE vs capture price)
WIND_CAPTURE_RATE = 0.70

# ── Market parameter options (key: (UI_name, price_impact)) ──────────────────

NUCLEAR_FI_OPTIONS: dict[str, tuple[str, float]] = {
    "current":       ("Current level (OL1+OL2+OL3 ≈ 4.4 GW)", 0.0),
    "ol3_hanhikivi": ("OL3 full power + Hanhikivi replacement (~5.4 GW)", -0.045),
    "new_plant":     ("New nuclear plant built (~6 GW)", -0.075),
    "smr":           ("SMR reactor (+0.5 GW) ≈ 4.9 GW", -0.022),
}

NUCLEAR_SE_OPTIONS: dict[str, tuple[str, float]] = {
    "normal":      ("Normal", 0.0),
    "one_offline": ("One reactor offline", 0.08),
    "expansions":  ("Expansions underway", -0.05),
}

HYDRO_OPTIONS: dict[str, tuple[str, float]] = {
    "normal": ("Normal", 0.0),
    "dry":    ("Dry year (–20%)", 0.18),
    "wet":    ("Wet year (+15%)", -0.12),
}

INTERCONNECT_FI_EE_OPTIONS: dict[str, tuple[str, int]] = {
    "current":    ("Current level (1,000 MW)", 1000),
    "estlink3":   ("EstLink 3 built (2,000 MW)", 2000),
    "restricted": ("Restricted capacity (500 MW, disruption)", 500),
}

INTERCONNECT_FI_SE_OPTIONS: dict[str, tuple[str, int]] = {
    "current":    ("Current level (~2,200 MW)", 2200),
    "expanded":   ("Expanded (+500 MW = 2,700 MW)", 2700),
    "restricted": ("Restricted (congestion ~1,540 MW)", 1540),
}

INTERCONNECT_NO_OPTIONS: dict[str, tuple[str, float]] = {
    "normal":     ("Normal", 1.0),
    "restricted": ("Restricted", 0.7),
    "expanded":   ("Expanded", 1.3),
}

# ── Forward curve: Finland electricity futures prices (Nasdaq Commodities, 17.4.2026) ──
# Source: Nasdaq Nord Pool Finland power futures, Energia.fi, 10y PPA price
# Values €/MWh at annual level (base scenario = forward curve as-is)
_FORWARD_YEARS  = np.array([2025, 2026, 2027, 2028, 2029, 2031, 2035, 2038], dtype=np.float64)
_FORWARD_PRICES = np.array([42.0, 49.0, 45.28, 42.73, 42.87, 49.0,  56.2,  61.6], dtype=np.float64)
# 2025: estimated annual average (H2 forward + H1 realized ≈ 42 €/MWh)
# 2026: futures price (Apr 57.94, May 37.50, Jun 30.23, Q3 42.40, Q4 64.35 → ~49)
# 2027–2029: Nasdaq futures prices directly
# 2031: user-provided data (49.0 €/MWh)
# 2035: extrapolated (10y PPA baseload ~51 €/MWh, rising trend → 56.2)
# 2036–2038: extrapolated linearly (~+1.8 €/MWh/y)


def _get_forward_base(year: int) -> float:
    """Interpolates the base price from the forward curve for the given year."""
    return float(np.interp(float(year), _FORWARD_YEARS, _FORWARD_PRICES))


# Scenario multipliers on forward curve (base = 1.0 = forward as-is)
# Narrow relative bands, since forward level is lower than old _BASE_PRICES
_SCENARIO_MULT = {
    "low":  0.80,   # −20% from forward (optimistic)
    "base": 1.00,   #  0%  forward as-is
    "high": 1.45,   # +45% from forward (risk scenario)
}
# Relative standard deviation per scenario
_BASE_STDS   = {"low": 0.10, "base": 0.15, "high": 0.22}


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class ScenarioParams:
    """Collection of all market parameters for a scenario run."""
    n_simulations: int = 500
    seed: int = 2025
    crisis_probability: float = 0.10

    # Finland renewable energy
    wind_fi_gw: float = 5.0         # Wind power additional capacity 2025–2035, GW
    solar_fi_gw: float = 1.5        # Solar energy growth, GW

    # Finland nuclear power
    nuclear_fi: str = "current"

    # Nordic markets
    hydro_nordic: str = "normal"
    nuclear_se: str = "normal"

    # Fuel prices
    gas_price_mwh: float = 40.0     # €/MWh
    co2_price_t: float = 70.0       # €/t CO2

    # Consumption growth Finland
    electrification_twh: float = 8.0    # electrification (industry + heat pumps)
    ev_twh: float = 1.0                 # electric vehicles

    # Datacenters
    datacenter_base_twh: float = 2.5   # Fingrid 2025 estimate
    datacenter_growth_pct: float = 8.0  # % per year

    # Interconnections
    interconnect_fi_se: str = "current"
    interconnect_fi_ee: str = "current"
    interconnect_no: str = "normal"


@dataclass
class RegressionResult:
    """Regression model result from fundamental data."""
    r2: float = 0.0
    coef: dict[str, float] = field(default_factory=dict)
    intercept: float = 0.0
    used_features: list[str] = field(default_factory=list)
    base_price_adjustment: float = 0.0
    seasonal_factors: dict[int, float] = field(default_factory=dict)


@dataclass
class ScenarioResult:
    """Results for a single scenario."""
    name: str
    label: str
    color: str
    monthly_prices: pd.DataFrame    # year, month, p5, p10, p25, p50, p75, p90, p95
    annual_prices: pd.DataFrame     # year, p5, p10, p25, p50, p75, p90, p95
    annual_sim_matrix: np.ndarray   # shape (n_sim, n_years): annual avg prices per simulation


# ── Market adjustment calculations ────────────────────────────────────────────

def compute_market_adjustments(params: ScenarioParams, year: int) -> float:
    """
    Computes the total price impact of market parameters as a multiplier
    relative to the base situation (1.0 = no change).

    Public function: also used in risk analysis and sensitivity analysis.
    """
    years = max(year - START_YEAR, 0)  # 0..10
    factor = 1.0

    # 1. Wind + solar (linear growth over 10 years)
    wind_new_twh  = params.wind_fi_gw  * 2.8 * (years / 10.0)
    solar_new_twh = params.solar_fi_gw * 0.9 * (years / 10.0)
    renewable_pct = (wind_new_twh + solar_new_twh) / FI_BASE_CONSUMPTION_TWH
    factor *= max(1.0 + (-0.40) * renewable_pct, 0.50)

    # 2. Finland nuclear (full effect by 2030)
    _, nfi_adj = NUCLEAR_FI_OPTIONS.get(params.nuclear_fi, ("", 0.0))
    ramp_nfi = min(years / 5.0, 1.0)
    factor *= (1.0 + nfi_adj * ramp_nfi)

    # 3. Nordic hydro reserves — immediate effect
    _, hydro_adj = HYDRO_OPTIONS.get(params.hydro_nordic, ("", 0.0))
    # Norwegian transfer potential scales hydro effect
    _, no_factor = INTERCONNECT_NO_OPTIONS.get(params.interconnect_no, ("", 1.0))
    effective_hydro = hydro_adj * no_factor
    factor *= (1.0 + effective_hydro)

    # 4. Swedish nuclear
    _, nse_adj = NUCLEAR_SE_OPTIONS.get(params.nuclear_se, ("", 0.0))
    factor *= (1.0 + nse_adj)

    # 5+6. Combined gas and CO₂ effect via merit order model
    # Weighted marginal cost: CHP ~60% + CCGT ~40% of gas power
    blended_cur = (
        0.60 * chp_marginal(params.gas_price_mwh, params.co2_price_t)
        + 0.40 * gas_marginal(params.gas_price_mwh, params.co2_price_t)
    )
    # Relative change from reference point × gas power market share
    factor *= (1.0 + (blended_cur / _MO_BLENDED_REF - 1.0) * _GAS_MARKET_SHARE)

    # 7. Consumption growth: electrification + EVs (price impact coefficient 0.60)
    # Empirical basis: Aalto 2023 study, demand elasticity -0.16 → supply coefficient ~1.5
    # Using 0.60 (conservative long-run estimate, markets adapt)
    total_growth_twh = params.electrification_twh + params.ev_twh
    consumption_pct = total_growth_twh * (years / 10.0) / FI_BASE_CONSUMPTION_TWH
    factor *= (1.0 + 0.60 * consumption_pct)

    # 8. Datacenters (price impact coefficient 0.30)
    dc_twh = params.datacenter_base_twh * ((1 + params.datacenter_growth_pct / 100) ** years)
    dc_twh = min(dc_twh, 50.0)
    dc_growth_pct = max(dc_twh - params.datacenter_base_twh, 0.0) / FI_BASE_CONSUMPTION_TWH
    factor *= (1.0 + 0.30 * dc_growth_pct)

    # 9. Interconnections: larger capacity → less pressure on prices
    _, fi_se_mw = INTERCONNECT_FI_SE_OPTIONS.get(params.interconnect_fi_se, ("", 2200))
    _, fi_ee_mw = INTERCONNECT_FI_EE_OPTIONS.get(params.interconnect_fi_ee, ("", 1000))
    total_ic_mw = fi_se_mw + fi_ee_mw
    ic_adj = (total_ic_mw - 3200) / 1000.0 * (-0.015)  # +1 GW → -1.5%
    factor *= (1.0 + ic_adj)

    return max(factor, 0.20)


def compute_max_hintaero(params: ScenarioParams) -> float:
    """
    Computes the maximum FI–Nordic price difference (€/MWh)
    based on interconnection capacity.

    max_price_diff = 500 / total_capacity_GW
    """
    _, fi_se_mw = INTERCONNECT_FI_SE_OPTIONS.get(params.interconnect_fi_se, ("", 2200))
    _, fi_ee_mw = INTERCONNECT_FI_EE_OPTIONS.get(params.interconnect_fi_ee, ("", 1000))
    total_gw = (fi_se_mw + fi_ee_mw) / 1000.0
    return 500.0 / max(total_gw, 0.1)


def compute_variable_sensitivities(params: ScenarioParams, base_year: int = 2030) -> pd.DataFrame:
    """
    Computes the price impact of each market variable using the one-at-a-time method
    in the base scenario. Used for the tornado chart.

    Returns DataFrame: variable, impact_low, impact_high,
                        value_low, value_high
    """
    base_price_base = _get_forward_base(base_year)
    base_factor = compute_market_adjustments(params, base_year)
    base_result = base_price_base * base_factor

    def impact(varied: ScenarioParams) -> float:
        return base_price_base * compute_market_adjustments(varied, base_year) - base_result

    def vary(field_name: str, value: Any) -> ScenarioParams:
        p = copy.copy(params)
        setattr(p, field_name, value)
        return p

    rows = [
        {
            "variable":     "Gas Price",
            "impact_low":   impact(vary("gas_price_mwh", 20.0)),
            "impact_high":  impact(vary("gas_price_mwh", 80.0)),
            "value_low":    "20 €/MWh",
            "value_high":   "80 €/MWh",
        },
        {
            "variable":     "CO₂ Price",
            "impact_low":   impact(vary("co2_price_t", 40.0)),
            "impact_high":  impact(vary("co2_price_t", 120.0)),
            "value_low":    "40 €/t",
            "value_high":   "120 €/t",
        },
        {
            "variable":     "Nordic Hydro",
            "impact_low":   impact(vary("hydro_nordic", "wet")),
            "impact_high":  impact(vary("hydro_nordic", "dry")),
            "value_low":    "Wet year",
            "value_high":   "Dry year",
        },
        {
            "variable":     "Swedish Nuclear",
            "impact_low":   impact(vary("nuclear_se", "expansions")),
            "impact_high":  impact(vary("nuclear_se", "one_offline")),
            "value_low":    "Expansions",
            "value_high":   "Reactor offline",
        },
        {
            "variable":     "Wind Power FI (addition)",
            "impact_low":   impact(vary("wind_fi_gw", 0.0)),
            "impact_high":  impact(vary("wind_fi_gw", 15.0)),
            "value_low":    "0 GW",
            "value_high":   "15 GW",
        },
        {
            "variable":     "Nuclear FI",
            "impact_low":   impact(vary("nuclear_fi", "current")),
            "impact_high":  impact(vary("nuclear_fi", "new_plant")),
            "value_low":    "Current level",
            "value_high":   "New plant",
        },
        {
            "variable":     "Solar Energy FI",
            "impact_low":   impact(vary("solar_fi_gw", 0.0)),
            "impact_high":  impact(vary("solar_fi_gw", 5.0)),
            "value_low":    "0 GW",
            "value_high":   "5 GW",
        },
        {
            "variable":     "Datacenter Growth",
            "impact_low":   impact(vary("datacenter_growth_pct", 0.0)),
            "impact_high":  impact(vary("datacenter_growth_pct", 30.0)),
            "value_low":    "0 %/y",
            "value_high":   "30 %/y",
        },
        {
            "variable":     "Electrification + Heat Pumps",
            "impact_low":   impact(vary("electrification_twh", 0.0)),
            "impact_high":  impact(vary("electrification_twh", 30.0)),
            "value_low":    "0 TWh",
            "value_high":   "30 TWh",
        },
        {
            "variable":     "Interconnection FI–SE",
            "impact_low":   impact(vary("interconnect_fi_se", "restricted")),
            "impact_high":  impact(vary("interconnect_fi_se", "expanded")),
            "value_low":    "Restricted",
            "value_high":   "Expanded",
        },
    ]

    df = pd.DataFrame(rows)
    df["spread"] = df["impact_high"] - df["impact_low"]
    # Sort by largest impact (tornado order)
    df = df.reindex(df["spread"].abs().sort_values(ascending=True).index)
    return df.reset_index(drop=True)


def compute_datacenter_projection(params: ScenarioParams) -> pd.DataFrame:
    """Computes the datacenter TWh growth curve annually."""
    rows = []
    for year in range(START_YEAR, END_YEAR + 1):
        years = year - START_YEAR
        twh = params.datacenter_base_twh * ((1 + params.datacenter_growth_pct / 100) ** years)
        capped = twh >= 50.0
        twh = min(twh, 50.0)
        rows.append({"year": year, "twh": round(twh, 2), "capped": capped})
    return pd.DataFrame(rows)


def compute_consumption_growth(params: ScenarioParams) -> dict[str, float]:
    """Computes a consumption growth summary."""
    total_growth = params.electrification_twh + params.ev_twh
    return {
        "current_twh": FI_BASE_CONSUMPTION_TWH,
        "growth_twh": total_growth,
        "forecast_2035_twh": FI_BASE_CONSUMPTION_TWH + total_growth,
    }


def compute_impact_breakdown(
    params: ScenarioParams,
    ref_years: tuple[int, ...] = (2030, 2035),
) -> dict:
    """
    Breaks down the €/MWh impact of three main variables (consumption growth,
    datacenters, wind power) in the base scenario for the given years.

    Uses the one-at-a-time method: sets each variable to zero
    and computes the difference from the full model price.

    Returns a dict:
    {
        year: {
            "base_price":          float,   base scenario with full parameters
            "consumption_impact":  float,   €/MWh (positive = price increase)
            "datacenter_impact":   float,
            "wind_impact":         float,   (negative = price decrease)
            "total_growth_twh":    float,
            "dc_growth_twh":       float,
            "wind_re_twh":         float,
        },
        ...
    }
    """
    result: dict[int, dict] = {}

    for year in ref_years:
        REF_BASE = _get_forward_base(year)  # forward curve price for the given year
        full_factor = compute_market_adjustments(params, year)
        full_price  = REF_BASE * full_factor

        # ── Consumption growth: compare current vs zero ───────────────────
        p0 = copy.copy(params)
        p0.electrification_twh = 0.0
        p0.ev_twh = 0.0
        f0 = compute_market_adjustments(p0, year)
        consumption_impact = (full_factor - f0) * REF_BASE

        # ── Datacenters: zero growth rate (base level stays constant) ─────
        p1 = copy.copy(params)
        p1.datacenter_growth_pct = 0.0
        f1 = compute_market_adjustments(p1, year)
        dc_impact = (full_factor - f1) * REF_BASE

        # ── Wind + solar: zero additional capacity ─────────────────────────
        p2 = copy.copy(params)
        p2.wind_fi_gw  = 0.0
        p2.solar_fi_gw = 0.0
        f2 = compute_market_adjustments(p2, year)
        wind_impact = (full_factor - f2) * REF_BASE  # negative

        # Computed amounts for the given year
        years_from_start = max(year - START_YEAR, 0)
        wind_re_twh = (
            params.wind_fi_gw  * 2.8 * (years_from_start / 10.0)
            + params.solar_fi_gw * 0.9 * (years_from_start / 10.0)
        )
        dc_twh_now = min(
            params.datacenter_base_twh * ((1 + params.datacenter_growth_pct / 100) ** years_from_start),
            50.0,
        )
        dc_growth_twh = max(dc_twh_now - params.datacenter_base_twh, 0.0)

        result[year] = {
            "base_price":          round(full_price, 1),
            "consumption_impact":  round(consumption_impact, 1),
            "datacenter_impact":   round(dc_impact, 1),
            "wind_impact":         round(wind_impact, 1),
            "total_growth_twh":    round(params.electrification_twh + params.ev_twh, 1),
            "dc_growth_twh":       round(dc_growth_twh, 1),
            "wind_re_twh":         round(wind_re_twh, 1),
        }

    return result


# ── Regression model calibration ─────────────────────────────────────────────

def calibrate_regression(fundamental_df: pd.DataFrame) -> RegressionResult:
    """
    Fits a linear regression model to historical fundamental data.
    Returns a RegressionResult object with model coefficients and R².
    If price data is missing or data is too sparse, returns an empty result.
    """
    result = RegressionResult()

    if fundamental_df.empty or "price_fi" not in fundamental_df.columns:
        logger.info("No price data found — skipping calibration")
        return result

    df = fundamental_df.copy().dropna(subset=["price_fi"])
    if len(df) < 10:
        logger.warning("Too little data for calibration (%d rows)", len(df))
        return result

    feature_cols = []
    for col in ["wind_capacity", "hydro_production", "nuclear_production", "gas_price", "co2_price"]:
        if col in df.columns and df[col].notna().sum() > 5:
            feature_cols.append(col)

    if "date" in df.columns:
        df["month_sin"] = np.sin(2 * np.pi * df["date"].dt.month / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["date"].dt.month / 12)
        feature_cols += ["month_sin", "month_cos"]

    if not feature_cols:
        logger.info("No usable features for regression")
        return result

    try:
        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import StandardScaler

        X = df[feature_cols].fillna(df[feature_cols].mean())
        y = df["price_fi"].values

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        model = LinearRegression()
        model.fit(X_scaled, y)

        r2 = model.score(X_scaled, y)
        coef = {feat: float(c) for feat, c in zip(feature_cols, model.coef_)}

        result.r2 = round(r2, 3)
        result.coef = coef
        result.intercept = float(model.intercept_)
        result.used_features = feature_cols
        result.base_price_adjustment = float(y.mean()) - _get_forward_base(2022)

        if "date" in df.columns:
            seasonal = df.groupby(df["date"].dt.month)["price_fi"].mean()
            mean_price = seasonal.mean()
            if mean_price > 0:
                result.seasonal_factors = {m: float(p / mean_price) for m, p in seasonal.items()}

        logger.info("Regression calibrated: R²=%.3f, features=%s", r2, feature_cols)
    except ImportError:
        logger.warning("scikit-learn not installed — skipping calibration")
    except Exception as e:
        logger.warning("Regression failed: %s", e)

    return result


# ── Monte Carlo simulation ────────────────────────────────────────────────────

def run_monte_carlo(
    params: ScenarioParams,
    regression: RegressionResult | None = None,
    progress_callback: Any = None,
    timeout_seconds: float = 30.0,
) -> dict[str, ScenarioResult]:
    """
    Runs a fully vectorized Monte Carlo simulation for three scenarios.

    All nested Python for-loops have been removed:
    - Annual sampling: np.random.normal(means, stds, (n_sim, n_years))
    - Monthly sampling: broadcasting (n_sim, n_years, 12)
    - Percentiles: np.percentile(..., axis=0) for the whole matrix at once

    progress_callback(fraction: float, message: str) – called after each scenario.
    timeout_seconds – if exceeded, partial results are returned with a warning.

    Returns dict {scenario_name: ScenarioResult} where
    monthly_prices contains p5/p10/p25/p50/p75/p90/p95.
    """
    import time as _time

    reg = regression if regression is not None else RegressionResult()
    rng = np.random.default_rng(params.seed)
    results: dict[str, ScenarioResult] = {}

    n_sim   = params.n_simulations
    n_years = END_YEAR - START_YEAR + 1  # 14
    n_months = 12

    years_arr  = np.arange(START_YEAR, END_YEAR + 1, dtype=np.int32)   # (14,)
    months_arr = np.arange(1, 13, dtype=np.int32)                       # (12,)

    # Seasonal factor vector (12,) – from regression or constants
    seasonal = reg.seasonal_factors if reg.seasonal_factors else {}
    mf_arr = np.array(
        [seasonal.get(m, MONTH_FACTORS[m]) for m in range(1, 13)],
        dtype=np.float64,
    )  # (12,)

    pctiles = [5, 10, 25, 50, 75, 90, 95]
    n_pct   = len(pctiles)

    # Indices for DataFrame construction (built once)
    years_rep   = np.repeat(years_arr, n_months)    # (n_years*12,)
    months_tile = np.tile(months_arr, n_years)       # (n_years*12,)

    t_start = _time.monotonic()

    for sc_idx, scenario in enumerate(SCENARIO_NAMES):
        # ── Check timeout before each scenario ──────────────────────────────
        elapsed = _time.monotonic() - t_start
        if elapsed > timeout_seconds and results:
            logger.warning(
                "Monte Carlo timeout %.1fs exceeded in scenario '%s' — "
                "returning partial results (%d/%d scenarios).",
                timeout_seconds, scenario, sc_idx, len(SCENARIO_NAMES),
            )
            break

        sc_mult = _SCENARIO_MULT[scenario]
        std_rel = _BASE_STDS[scenario]
        adj     = reg.base_price_adjustment

        # ── Forward curve base prices per year (calibrated with Nasdaq futures) ──
        forward_bases = np.array(
            [_get_forward_base(int(y)) * sc_mult for y in years_arr],
            dtype=np.float64,
        )  # (n_years,)

        # ── Annual market adjustments (14 scalars — fast) ─────────────────────
        # Only remaining Python loop: just 14 iterations
        market_factors = np.array(
            [compute_market_adjustments(params, int(y)) for y in years_arr],
            dtype=np.float64,
        )  # (n_years,)

        means = np.maximum((forward_bases + adj) * market_factors, 15.0)  # (n_years,)
        stds  = means * std_rel                                    # (n_years,)

        # ── Annual sampling — fully vectorized ────────────────────────────────
        # annual_samples: (n_sim, n_years)
        annual_samples = rng.normal(
            loc=means,                      # broadcast: (n_years,) → (n_sim, n_years)
            scale=stds,
            size=(n_sim, n_years),
        )

        # Crisis events in high scenario — vectorized mask
        if scenario == "high" and params.crisis_probability > 0:
            crisis_mask = rng.random((n_sim, n_years)) < params.crisis_probability
            boosts      = rng.uniform(1.3, 1.8, (n_sim, n_years))
            annual_samples = np.where(crisis_mask, annual_samples * boosts, annual_samples)

        annual_samples = np.maximum(annual_samples, 1.0)  # (n_sim, n_years)

        # ── Monthly sampling — fully vectorized ───────────────────────────────
        # monthly_base: (n_sim, n_years, 12)
        #   annual_samples[:, :, None] * mf_arr[None, None, :]
        monthly_base = annual_samples[:, :, np.newaxis] * mf_arr[np.newaxis, np.newaxis, :]

        # Monthly noise: stds (n_years,) → (1, n_years, 1) broadcast
        noise = rng.normal(
            loc=0.0,
            scale=(stds * 0.12)[np.newaxis, :, np.newaxis],
            size=(n_sim, n_years, n_months),
        )
        monthly_samples = np.maximum(monthly_base + noise, 0.5)  # (n_sim, n_years, 12)

        # ── Percentiles — one numpy call per level ────────────────────────────
        # annual_pcts:  (n_pct, n_years)
        # monthly_pcts: (n_pct, n_years, 12)
        annual_pcts  = np.percentile(annual_samples,  pctiles, axis=0)
        monthly_pcts = np.percentile(monthly_samples, pctiles, axis=0)

        # ── Build DataFrames — no row-level loops ─────────────────────────────
        annual_df = pd.DataFrame(
            {f"p{p}": annual_pcts[i] for i, p in enumerate(pctiles)},
        )
        annual_df.insert(0, "year", years_arr)

        monthly_pcts_flat = monthly_pcts.reshape(n_pct, -1)  # (n_pct, n_years*12)
        monthly_df = pd.DataFrame(
            {f"p{p}": monthly_pcts_flat[i] for i, p in enumerate(pctiles)},
        )
        monthly_df.insert(0, "year",  years_rep)
        monthly_df.insert(1, "month", months_tile)

        results[scenario] = ScenarioResult(
            name=scenario,
            label=SCENARIO_LABELS[scenario],
            color=SCENARIO_COLORS[scenario],
            monthly_prices=monthly_df,
            annual_prices=annual_df,
            annual_sim_matrix=annual_samples.astype(np.float32),
        )

        # ── Progress report ───────────────────────────────────────────────────
        if progress_callback is not None:
            fraction = (sc_idx + 1) / len(SCENARIO_NAMES)
            progress_callback(fraction, SCENARIO_LABELS[scenario])

        logger.debug(
            "Scenario '%s' complete: %.0f ms",
            scenario, (_time.monotonic() - t_start) * 1000,
        )

    return results


def scenarios_to_dataframe(results: dict[str, ScenarioResult]) -> pd.DataFrame:
    """
    Combines monthly results from all scenarios into a single DataFrame.
    Columns: year, month, scenario, p5, p10, p25, p50, p75, p90, p95
    """
    frames = []
    for name, result in results.items():
        df = result.monthly_prices.copy()
        df["scenario"] = name
        frames.append(df)
    return pd.concat(frames, ignore_index=True)
